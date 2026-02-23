# training.py
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import *
from tqdm import tqdm
import time
from collections import defaultdict
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.segmentation import MeanIoU
from metrics import *
from utils import *

def move_to(obj, device):
    """
    递归地将对象及其所有元素移动到指定设备

    支持的数据类型：
    - torch.Tensor / nn.Module：直接调用.to(device)
    - list / tuple / set：递归处理每个元素
    - dict：递归处理每个value
    - 其他类型：原样返回

    :param obj: 要移动的对象（可以是任意嵌套的数据结构）
    :param device: 目标设备，如 'cuda:0' 或 'cpu'
    :return: 移动到指定设备后的对象（与原对象结构相同）
    """
    if isinstance(obj, (torch.Tensor, nn.Module)):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(x, device) for x in obj)
    elif isinstance(obj, set):
        return {move_to(x, device) for x in obj}
    elif isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    else:
        return obj


# noinspection PyUnboundLocalVariable
def run_epoch(model, optimizer, dataloader, loss_fn, device, results,
              score_funcs, task, prefix="", desc=None, **kwargs):
    """
     运行一个训练/验证/测试轮次

     :param model: PyTorch模型
     :param optimizer: 优化器
     :param dataloader: 数据迭代器，提供(inputs, labels)批次
     :param loss_fn: 损失函数，接受(y_pred, y_true)并返回loss
     :param device: 训练设备（'cuda'或'cpu'）
     :param results: 结果字典，格式为defaultdict(list)，用于存储loss和各项指标
     :param score_funcs: 评估函数字典，如{'acc': accuracy_score, 'f1': f1_score}
                        每个函数接受(y_true, y_pred)并返回标量值
     :param task: 任务类型，为'classification' 或 'regression'
     :param prefix: 结果前缀，如'train'、'val'、'test'，用于区分不同阶段
     :param desc: tqdm进度条描述，如'Training'、'Validation'
     :return: 本轮次耗时（秒）
     """
    if task not in ['classification', 'regression']:
        raise ValueError('task must be classification or regression')

    running_loss = []
    y_true = []
    y_pred = []
    # 训练轮次开始
    start = time.time()

    for inputs, labels in tqdm(dataloader, desc=desc):
        # inputs Shape: (B, C, W, H)
        # labels: (B, 1)
        inputs, labels = move_to(inputs, device), move_to(labels, device)

        # 预测, y_hat Shape: (B, num_classes), Binary: logits 正类(1)的置信度，logits > 0 -> sigmoid(logits) > 0.5 -> label = 1
        y_hat = model(inputs)

        loss = loss_fn(y_hat, labels)

        # 无论什么阶段都要记录结果
        running_loss.append(loss.item())

        # 训练阶段更新
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if score_funcs:
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(y_hat)

    # 训练轮次结束
    end = time.time()
    y_pred = np.asarray(y_pred) # logits

    # 如果是多分类任务，预测的是标签，返回argmax；回归任务，直接输出原始值
    if task == 'classification':
        if len(y_pred.shape) == 2:

            # 多分类
            if y_pred.shape[1] > 1:
                y_pred_class = np.argmax(y_pred, axis=1)

            # 二分类
            elif y_pred.shape[1] == 1:
                y_pred_prob = 1 / (1 + np.exp(-y_pred))
                # pred > 0 <=> prob > 0.5 <=> label = 1
                y_pred_class = (y_pred > 0).astype(int)


    # results 是一个列表字典，储存我们想要的数据
    if running_loss:
        results[f'{prefix} loss'].append(np.mean(running_loss))

    for name, score_func in score_funcs.items():
        if name == 'auc':
            results[f'{prefix} {name}'].append(score_func(y_true, y_pred_prob))
        else:
            results[f'{prefix} {name}'].append(score_func(y_true, y_pred_class))

    return end - start


def run_detection_epoch(model, optimizer, dataloader, loss_fn, device, results,
                        score_funcs, task, prefix="", desc=None, **kwargs):
    """
    目标检测专用训练轮次
    """

    # torchmetrics（tensor in -> tensor out）
    evaluator = MeanAveragePrecision()

    running_loss = []
    start = time.time()

    for imgs, targets in tqdm(dataloader, desc=desc, leave=False):
        imgs, targets = move_to(imgs, device), move_to(targets, device)


        # 检测: model(img, target) -> loss_dict (训练时)
        #       model(img) -> predictions (验证时)

        # 训练阶段
        if model.training:
            optimizer.zero_grad()
            # loss_fn 参数是虚拟的，检测模型内部自己算损失
            losses = model(imgs, targets)
            loss = sum(losses.values()) if isinstance(losses, dict) else losses

            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        # 验证/测试阶段
        else:
            with torch.no_grad():
                # model(img) 拿预测框（用于计算mAP）
                predictions = model(imgs)
                # 更新评估器
                evaluator.update(predictions, targets)

    end = time.time()

    if running_loss:
        results[f'{prefix} loss'].append(np.mean(running_loss))

    if not model.training:
        eval_results = evaluator.compute()
        evaluator.reset()

        for name in score_funcs:
            if name in eval_results:
                results[f'{prefix} {name}'].append(eval_results[name].item())

    return end - start


def run_segmentation_epoch(model, optimizer, dataloader, loss_fn, device, results,
                            score_funcs, task='segmentation', prefix="", desc=None, **kwargs):
    """
    分割任务专用训练轮次（支持语义分割和实例分割）

    任务类型通过 loss_fn 自动判断：
        - loss_fn = None      : 实例分割（如 Mask R-CNN），模型内部计算损失，评估用 mask AP
        - loss_fn = 有值       : 语义分割（如 UNet），外部计算损失，评估用 mIoU

    评估指标 score_funcs 说明：
        - 实例分割时，建议传入 ['map', 'map_50', 'map_75']（对应 mask AP）
        - 语义分割时，建议传入 ['mIoU']，如需 pixel acc 可额外添加
        - pixel acc 会单独用 sklearn 计算，不依赖 torchmetrics
    """
    # 根据 loss_fn 选择评估器
    evaluator = MeanAveragePrecision(iou_type='segm') if loss_fn is None else MeanIoU(num_classes=kwargs['num_classes'])

    running_loss = []
    y_true = []  # 用于 pixel acc
    y_pred = []
    start = time.time()

    for imgs, targets in tqdm(dataloader, desc=desc, leave=False):
        imgs, targets = move_to(imgs, device), move_to(targets, device)

        if model.training:
            optimizer.zero_grad()

            if loss_fn is not None:
                y_hat = model(imgs)
                loss = loss_fn(y_hat, targets)
            else:
                losses = model(imgs, targets)
                loss = sum(losses.values()) if isinstance(losses, dict) else losses

            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        else:
            with torch.no_grad():
                # Instance: list(dict of tensor), Semantic: (B,num_classes,H,W)
                predictions = model(imgs)

                if loss_fn is not None:  # 语义分割
                    loss = loss_fn(predictions, targets)
                    running_loss.append(loss.item())
                    # prob -> label
                    predictions = predictions.argmax(dim=1)

                else:
                    # predictions [dict1, dict2, ...]
                     for pred in predictions:
                        if 'masks' in pred:
                            pred['masks'] = (pred['masks'] > 0.5).to(torch.uint8).squeeze(1)


                # 更新评估器 Semantic: predictions [B,H,W]
                evaluator.update(predictions, targets)

                # 收集 pixel acc 数据（只在需要时）
                if 'pixel acc' in score_funcs:
                    y_true.extend(targets.detach().cpu().numpy().ravel())
                    y_pred.extend(predictions.detach().cpu().numpy().ravel())

    end = time.time()

    # 记录 loss
    if running_loss:
        results[f'{prefix} loss'].append(np.mean(running_loss))

    # 计算 pixel acc
    if y_true and y_pred:
        results[f'{prefix} pixel acc'].append(accuracy_score(y_true, y_pred))

    # 计算 torchmetrics 指标
    if not model.training:
        if isinstance(evaluator, MeanIoU):
            # 语义分割：直接取 tensor
            miou = evaluator.compute()
            results[f'{prefix} mIoU'].append(miou.item())
            evaluator.reset()
        else:
            # 实例分割：返回字典
            eval_results = evaluator.compute()
            evaluator.reset()
            for name in score_funcs:
                if name in eval_results and name != 'pixel acc':
                    results[f'{prefix} {name}'].append(eval_results[name].item())

    return end - start


def run_self_supervised_epoch(model, optimizer, dataloader, loss_fn, device, results,
                              score_funcs, task='self_supervised', prefix="", desc=None, **kwargs):
    """
    自监督学习统一训练函数

    :param method: 自监督方法类型
        - 'contrastive': 对比学习 如 SimCLR, MoCo
        - 'asymmetric':  非对称网络 如 BYOL, SimSiam
        - 'masked':      掩码重建 如 MAE)
        - 'clustering':  聚类 如 SwAV
    :param model: 自监督模型，需满足以下接口约定：
        - 对比学习: model(x0, x1) 返回 (z0, z1)
        - 非对称网络: 若模型有 student/teacher 属性则用，否则假设 forward 返回 (student, teacher)
        - 掩码重建: model(x) 返回 (loss, ...) 或直接 loss，loss 必须在第一个元素
        - 聚类: model(x) 返回 (B, num_clusters) 的原始得分
    :param loss_fn: 损失函数，接收模型输出并计算损失
    :param update_teacher: 若模型有 update_teacher 方法，训练后会自动调用（用于动量更新）
    """
    running_loss = []
    start = time.time()

    for batch in tqdm(dataloader, desc=desc, leave=False):
        batch = move_to(batch, device)

        # 自监督数据通常是 (x0, x1) 两个增强，或者 x 单个输入
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                x0, x1 = batch
            else:
                raise ValueError(f'batch must be 2-element tuple/list, got {len(batch)}')
        elif isinstance(batch, torch.Tensor):
            x0 = x1 = batch
        else:
            raise ValueError(f'Unsupported batch type: {type(batch)}')

        method = kwargs['method']
        # 根据方法选择前向逻辑
        if method == 'contrastive':
            # 对比学习: 输入两个增强，输出两个特征
            z0, z1 = model(x0, x1)
            loss = loss_fn(z0, z1)

        elif method == 'asymmetric':
            # 非对称网络: 学生网络 + 教师网络，教师不计算梯度
            if hasattr(model, 'student') and hasattr(model, 'teacher'):
                z_student = model.student(x0)
                with torch.no_grad():
                    z_teacher = model.teacher(x1)
            else:
                # 若模型未分 student/teacher，假设 forward 返回 (student, teacher)
                z_student, z_teacher = model(x0, x1)
                z_teacher = z_teacher.detach()
            loss = loss_fn(z_student, z_teacher)

        elif method == 'masked':
            # 掩码重建: 模型内部计算损失，返回 (loss, ...) 或直接 loss
            output = model(x0)
            loss = output[0] if isinstance(output, (tuple, list)) else output

        elif method == 'clustering':
            # 聚类: 输出原始得分，用 Sinkhorn 均衡后算损失
            features = model(x0)  # (B, num_clusters)
            soft_labels = sinkhorn(features)
            loss = loss_fn(features, soft_labels)

        else:
            raise ValueError(f'Unknown method: {method}')

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 若有动量更新的教师网络，调用 update_teacher（约定方法名）
            if hasattr(model, 'update_teacher'):
                model.update_teacher()

        running_loss.append(loss.item())

    end = time.time()

    if running_loss:
        results[f'{prefix} loss'].append(np.mean(running_loss))

    return end - start

TASK = {
    'classification': run_epoch,
    'regression': run_epoch,
    'detection': run_detection_epoch,
    'segmentation': run_segmentation_epoch,
    'self_supervised': run_self_supervised_epoch,
}

def train_network(model, loss_fn, task, train_dataloader, val_dataloader=None, test_dataloader=None,
                      score_funcs=None,
                      epochs=50, device="cuda" if torch.cuda.is_available() else 'cpu', checkpoint_file=None,
                      lr_schedule=None, optimizer=None, to_df=True, **kwargs
                      ):
    """
    完整的神经网络训练函数

    :param model: 要训练的PyTorch模型
    :param loss_fn: 损失函数，如 nn.CrossEntropyLoss() (分类) 或 nn.MSELoss() (回归)
                    对于检测任务，此参数为占位符（模型内部计算损失）
    :param task: 任务类型，支持 'classification', 'regression', 'detection', 'segmentation'，'self_supervised'
    :param train_dataloader: 训练数据迭代器
    :param val_dataloader: 验证数据迭代器（可选），用于每个epoch评估和保存最佳模型
    :param test_dataloader: 测试数据迭代器（可选），训练结束后用最佳模型评估一次
    :param score_funcs: 评估指标
        - 分类/回归: 字典形式 {'acc': accuracy_score, 'f1': f1_score}
        - 检测: 列表形式 ['mAP', 'mAP_50', 'mAP_75']
    :param epochs: 训练轮数，默认50
    :param device: 训练设备，自动检测cuda可用性
    :param checkpoint_file: 检查点保存路径（可选），每个epoch保存一次
    :param lr_schedule: 学习率调度器（可选），支持 torch.optim.lr_scheduler 下的各类调度器
    :param optimizer: 优化器（可选），默认使用 torch.optim.AdamW
    :param to_df: 转换为pd.DataFrame，默认为True
    :return: (results_df, test_results, best_model_state)
             - results_df: 包含训练和验证指标的DataFrame，可用于可视化
             - test_results: 测试集评估结果字典（若提供了test_dataloader）
             - best_model_state: 最佳模型的状态字典（基于验证集loss），可用于后续加载

    示例:
         # 分类任务
            results, test_results, best_model = train_network(
            model=classifier,
            loss_fn=nn.CrossEntropyLoss(),
            task='classification',
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            score_funcs={'acc': accuracy_score, 'f1': f1_score},
            epochs=50
        )

        # 检测任务
        results, test_results, best_model = train_network(
            model=detector,
            loss_fn=None,  # 占位符
            task='detection',
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            score_funcs=['mAP', 'mAP_50', 'mAP_75'],  # 列表形式
            epochs=50,
        )
    """
    if score_funcs is None:
        score_funcs = {}

    start = time.time()
    # 结果字典
    results = defaultdict(list)
    test_results = defaultdict(list)

    # 总训练时长
    total_train_time =  0

    # 默认 Optimizer 为 AdamW
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters())

    # 最佳模型
    best_val_loss = float('inf')
    best_model_state = None

    # 根据任务类型映射函数
    if task not in TASK.keys():
        raise ValueError(f'task {task} is not supported, task must be one of {TASK}')

    run = TASK[task]

    model.to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} | {epochs}', end='\r', flush=True)

        model.train()

        # 训练时间
        total_train_time += run(model, optimizer, train_dataloader, loss_fn, device, results,
                                score_funcs, task, prefix="train", desc='Training', **kwargs)

        results['epoch'].append(epoch)
        results['total time'].append(total_train_time)

        if val_dataloader is not None:
            model.eval()

            with torch.no_grad():
                run(model, optimizer, val_dataloader, loss_fn, device, results,
                          score_funcs, task, prefix="val", desc='Validation', **kwargs)

            if results['val loss'] and results['val loss'][-1] < best_val_loss:
                best_val_loss = results['val loss'][-1]
                best_model_state = copy.deepcopy(model.state_dict())

        if lr_schedule is not None:
            # 如果是条件调度，要与最后一个 val_loss 比较
            if isinstance(lr_schedule, ReduceLROnPlateau):
                if results['val loss']:
                    lr_schedule.step(results['val loss'][-1])
            else:
                lr_schedule.step()

        # 保存检查点
        if checkpoint_file is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_schedule.state_dict() if lr_schedule is not None else None,
                'results': results
            }, checkpoint_file)

    if best_model_state is not None:
        # deepcopy复制，防止模型参数被覆盖
        best_model = copy.deepcopy(model)
        best_model.load_state_dict(best_model_state)
    else:
        # 没有最佳参数，就用最后一次模型
        best_model = model

    if test_dataloader is not None:
        best_model.eval()

        with torch.no_grad():
            run(best_model, optimizer, test_dataloader, loss_fn, device, test_results,
                score_funcs, task, prefix="test", desc='Testing', **kwargs)

        test_results = {k: v[0] for k, v in test_results.items() if len(v) > 0}

    results = {k: v for k, v in results.items() if len(v) > 0}
    # 返回 pandas表格，方便查看和可视化数据
    if to_df:
        results = pd.DataFrame.from_dict(results)

    end = time.time()
    print(f'\n训练结束，共训练 {epochs} 轮，耗时 {end-start}')
    return results, test_results, best_model_state






