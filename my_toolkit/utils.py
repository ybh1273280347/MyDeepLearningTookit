# utils.py
import numpy as np
import random
import torch
from collections import defaultdict
import albumentations.core.bbox_utils
import albumentations as A
from data import *
import torch.nn as nn

def set_seed(seed=42):
    """固定所有随机种子，保证实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_checkpoint(model, checkpoint_file, device,optimizer=None, lr_schedule=None):
    """
    加载模型检查点

    :param model: 模型实例（必须与保存时结构一致）
    :param checkpoint_file: 检查点文件路径
    :param device: 加载设备，如 'cuda' 或 'cpu'
    :param optimizer: 优化器实例（可选，用于继续训练）
    :param lr_schedule: 学习率调度器实例（可选，用于继续训练）
    :return: (model, start_epoch, results, optimizer, lr_schedule)
            - model: 加载权重后的模型
            - start_epoch: 下一轮开始的epoch编号（保存的epoch+1）
            - results: 训练历史记录字典
            - optimizer: 加载状态后的优化器（如果提供）
            - lr_schedule: 加载状态后的调度器（如果提供）
    """

    checkpoint = torch.load(checkpoint_file, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    start_epoch = checkpoint.get('epoch', 0) + 1
    results = checkpoint.get('results', defaultdict(list))

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if lr_schedule is not None and checkpoint['scheduler_state_dict'] is not None:
        lr_schedule.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f'Checkpoint {checkpoint_file} loaded.')

    return model, start_epoch, results, optimizer, lr_schedule


def relax_check_bboxes(eps=1e-6):
    """
    替换 Albumentations 的 bbox 检查函数，放宽容差并使用 clip_boxes 修正

    :param eps: 容差阈值，默认 1e-6
    """

    def relaxed_check_bboxes(bboxes):
        """带容差的 bbox 检查，用 clip_boxes 修正"""
        # 放宽检查
        valid = (bboxes[:, :4] >= -eps) & (bboxes[:, :4] <= 1 + eps)

        if not valid.all():
            # 找出真正超出的
            serious = (bboxes[:, :4] < -eps) | (bboxes[:, :4] > 1 + eps)

            if serious.any():
                # 真的有严重错误，才报错
                raise ValueError(f"BBox 严重超出范围 ! eps = {eps}")

            # 微小误差，用 clip_boxes 修正（yolo模式）
            bboxes[:, :4] = clip_boxes(bboxes[:, :4], mode='yolo', eps=eps)

    # 动态替换
    A.core.bbox_utils.check_bboxes = relaxed_check_bboxes
    print(f" Albumentations 检查已替换，容差 eps={eps}")


def sinkhorn(features, temperature=0.05, iterations=3):
    """
    Sinkhorn 均衡分配

    :param features: (batch_size, num_clusters) 原始得分
    :param temperature: 温度参数，越小分配越尖锐
    :param iterations: 迭代次数
    :return: (batch_size, num_clusters) 均衡后的软标签
    """
    # 数值稳定的 softmax 预处理
    # 减去每行最大值，防止 exp 溢出
    Q_stable = features / temperature - features.max(dim=1, keepdim=True).values / temperature
    Q = torch.exp(Q_stable)  # (B, K)

    Q /= Q.sum()  # 全局归一化

    # 迭代均衡
    for _ in range(iterations):
        # 列归一化：让聚类的分配归一化，从而分配更均匀
        # keepdim 确保可广播运算， clamp 防止除零
        Q /= Q.sum(dim=0, keepdim=True).clamp(min=1e-8)
        # 行归一化：确保每张图的分配概率和为 1
        Q /= Q.sum(dim=1, keepdim=True).clamp(min=1e-8)

    return Q


class AdaptedModel(nn.Module):
    """
    模型适配器：将各种自监督模型统一成 student/teacher 接口

    :param base_model: 原始模型
    :param student_name: 原始模型中 student 对应的属性名
    :param teacher_name: 原始模型中 teacher 对应的属性名

    示例：
         适配 BYOL
          byol_model = models.BYOL(...)
          adapted_byol = AdaptedModel(byol_model, student_name='online_encoder', teacher_name='target_encoder')
    """

    def __init__(self, base_model, student_name='student', teacher_name='teacher'):
        super().__init__()
        self.base_model = base_model
        self.student = getattr(base_model, student_name)
        self.teacher = getattr(base_model, teacher_name)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


def freeze_weights(layers):
    """
    冻结指定层的权重

    :param layers: nn.Module 或 list/tuple of nn.Module
    """
    if isinstance(layers, (tuple, list)):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
    else:
        for param in layers.parameters():
            param.requires_grad = False


def unfreeze_weights(layers):
    """解冻指定层的权重"""
    if isinstance(layers, (tuple, list)):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        for param in layers.parameters():
            param.requires_grad = True