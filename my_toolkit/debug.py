# debug.py
import random
from PIL import Image
from torchsummary import summary
from data import *
from training import *
from torch.utils.data import *

def inspect(obj, name="obj", depth=0, max_depth=3, show_all=False):
    """
    检查python对象，获取相关属性

    :param obj: 要检查的对象
    :param name: 对象名称
    :param depth: 当前深度（递归用）
    :param max_depth: 最大递归深度
    :param show_all: 是否显示所有元素（默认只显示前3个）
    :return: None, 直接打印信息
    """
    indent = "  " * depth
    prefix = f"{indent} {name}: "

    # 基础类型
    if obj is None:
        print(f"{prefix}None")

    elif isinstance(obj, (int, float, str, bool)):
        print(f"{prefix}{type(obj).__name__}")

    # torch Tensor
    elif isinstance(obj, torch.Tensor):
        print(f"{prefix}torch.Tensor")
        print(f"{indent}  ├─ shape: {tuple(obj.shape)}")
        print(f"{indent}  ├─ dtype: {obj.dtype}")
        print(f"{indent}  ├─ device: {obj.device}")
        print(f"{indent}  ├─ requires_grad: {obj.requires_grad}")
        print(f"{indent}  ├─ min: {obj.min().item():.6f}")
        print(f"{indent}  └─ max: {obj.max().item():.6f}")

    # numpy ndarray
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}numpy.ndarray")
        print(f"{indent}  ├─ shape: {obj.shape}")
        print(f"{indent}  ├─ dtype: {obj.dtype}")
        print(f"{indent}  ├─ min: {obj.min():.6f}")
        print(f"{indent}  └─ max: {obj.max():.6f}")

    # tuple / list
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__}, length={len(obj)}")
        if len(obj) > 0 and depth < max_depth:
            limit_num = min(3, len(obj))
            items = obj if show_all else obj[:limit_num]
            for i, item in enumerate(items):
                inspect(item, f"[{i}]", depth + 1, max_depth, show_all)
            if not show_all and len(obj) > 3:
                print(f"{indent}  ... and {len(obj) - 3} more")

    # dict
    elif isinstance(obj, dict):
        print(f"{prefix}dict, keys={list(obj.keys())}")
        if depth < max_depth:
            for k, v in obj.items():
                inspect(v, f"['{k}']", depth + 1, max_depth, show_all)

    # PIL Image
    elif isinstance(obj, Image.Image):
        print(f"{prefix}PIL.Image")
        print(f"{indent}  ├─ mode: {obj.mode}")
        print(f"{indent}  ├─ size: {obj.size}")
        print(f"{indent}  └─ format: {obj.format}")

    else:
        print(f"{prefix}{type(obj).__name__}")
        if hasattr(obj, '__dict__'):
            attrs = list(obj.__dict__.keys())
            if attrs:
                print(f"{indent}  └─ attributes: {attrs[:5]}{'...' if len(attrs) > 5 else ''}")


def inspect_dataset(dataset, name='dataset', sample_idx=0):
    """
    检查数据集的基本信息和一个样本的结构

    :param dataset: 要检查的 Dataset 对象
    :param name: 数据集的名称，默认为 'dataset'
    :param sample_idx: 要检查的样本索引，默认为 0
    :return: None，直接打印信息
    """
    print(f'检查 {name}')
    print(f'   ├─ 类型: {type(dataset).__name__}')
    print(f'   ├─ 数据集大小: {len(dataset)}')

    if hasattr(dataset, 'transform') and dataset.transform is not None:
        print(f'   └─ transform: {type(dataset.transform).__name__}')

    if len(dataset) > 0:
        print(f'\n第 {sample_idx+1} 个样本:')
        sample = dataset[sample_idx]
        inspect(sample, name='sample', show_all=True)
    else:
        print('数据集为空')

def inspect_dataloader(dataloader, name='dataloader'):
    """
    检查 DataLoader 的基本信息和第一个 batch 的结构

    :param dataloader: 要检查的 DataLoader 对象
    :param name: DataLoader 的名称，默认为 'dataloader'
    :return: None，直接打印信息
    """
    print(f'检查 {name}')
    print(f'   ├─ 数据集大小: {len(dataloader.dataset)}')
    print(f'   ├─ batch 数量: {len(dataloader)}')
    print(f'   ├─ batch_size: {dataloader.batch_size}')
    print(f'   ├─ shuffle: {dataloader.shuffle}')
    print(f'   └─ num_workers: {dataloader.num_workers}')

    if len(dataloader) > 0:
        print(f'\n第一个 batch 结构:')
        batch = next(iter(dataloader))
        inspect(batch, name='batch')
    else:
        print('DataLoader 为空')


def inspect_model(model, input_size=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    查看模型结构和参数信息

    :param model: PyTorch模型
    :param input_size: 输入尺寸，如 (3,224,224)
    :param device: 设备
    """
    print("模型结构摘要")

    # 模型基本信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n参数统计:")
    print(f"   ├─ 总参数: {total_params:,}")
    print(f"   ├─ 可训练参数: {trainable_params:,}")
    print(f"   └─ 不可训练参数: {total_params - trainable_params:,}")

    # torchsummary 输出
    if input_size is not None:
        print("\n各层输出形状:")

        model = model.to(device)
        summary(model, input_size, device=device)
    else:
        print("未提供输入信息，跳过层形状打印")


def inspect_optimizer(optimizer):
    """
    查看优化器信息（学习率、权重衰减等）

    :param optimizer: 优化器对象
    """
    print(f"优化器: {type(optimizer).__name__}")

    # 参数组
    print(f'共 {len(optimizer.param_groups)}个参数组')
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"\n[{i}] param_group")
        print(f"   ├─ 学习率 (lr): {param_group.get('lr', 'N/A')}")
        print(f"   ├─ 权重衰减 (weight_decay): {param_group.get('weight_decay', 'N/A')}")
        print(f"   ├─ 动量 (momentum): {param_group.get('momentum', 'N/A')}")
        print(f"   ├─ betas: {param_group.get('betas', 'N/A')}")
        print(f"   └─ eps: {param_group.get('eps', 'N/A')}")



def inspect_scheduler(scheduler):
    """
    查看学习率调度器信息

    :param scheduler: 学习率调度器对象
    """
    if scheduler is None:
        print("未使用学习率调度器")
        return

    print("学习率调度器")
    print(f"类型: {type(scheduler).__name__}")

    # 当前学习率
    if hasattr(scheduler, 'get_last_lr'):
        current_lr = scheduler.get_last_lr()
        if len(current_lr) == 1:
            print(f"   ├─ 当前学习率: {current_lr[0]:.6f}")
        else:
            print(f"   ├─ 当前学习率: {current_lr}")

    # 步进式调度器 (StepLR, MultiStepLR)
    if hasattr(scheduler, 'step_size'):
        print(f"   ├─ step_size: {scheduler.step_size}")
        print(f"   └─ gamma: {scheduler.gamma}")

    # 自适应调度器 (ReduceLROnPlateau)
    elif hasattr(scheduler, 'mode'):
        print(f"   ├─ mode: {scheduler.mode}")
        print(f"   ├─ factor: {scheduler.factor}")
        print(f"   ├─ patience: {scheduler.patience}")
        print(f"   └─ threshold: {scheduler.threshold}")

    # 余弦退火 (CosineAnnealingLR)
    elif hasattr(scheduler, 'T_max'):
        print(f"   ├─ T_max: {scheduler.T_max}")
        print(f"   └─ eta_min: {scheduler.eta_min}")



def inspect_training_setup(model, optimizer=None, scheduler=None,
                           input_size=None,
                           device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    一站式查看训练设置

    :param model: 模型
    :param optimizer: 优化器（可选）
    :param scheduler: 学习率调度器（可选）
    :param input_size: 输入尺寸
    :param device: 设备
    """

    print("训练设置概览")

    # 设备信息
    print(f"\n设备: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 模型信息
    inspect_model(model, input_size=input_size, device=device)

    # 优化器信息
    if optimizer is not None:
        inspect_optimizer(optimizer)

    # 调度器信息
    if scheduler is not None:
        inspect_scheduler(scheduler)

    print("\n检查完成")


def quick_debug(dataset, batch_size=1, num_train=1, num_val=1, collate_fn=None, **kwargs):
    """
    快速调试训练流程

    :param dataset: 完整数据集
    :param batch_size: 批次大小，默认1
    :param num_train: 训练 batch 数，默认1 → 总训练样本 = batch_size * num_train
    :param num_val: 验证 batch 数，默认1
    :param collate_fn: DataLoader的collate函数
    :param kwargs: 传给 train_network 的参数（model, task, loss_fn, score_funcs 等）
    :return: train_network 的返回结果
    """
    # 强制限制 epochs ≤ 3，防止调试时间过长
    kwargs['epochs'] = min(3, kwargs.get('epochs', 1))

    total_samples = batch_size * (num_train + num_val)
    if total_samples > len(dataset):
        raise ValueError(f"数据集太小 ({len(dataset)})，无法采样 {total_samples} 个样本用于调试")

    indices = random.sample(range(len(dataset)), k=total_samples)
    train_indices = indices[:batch_size * num_train]
    val_indices = indices[batch_size * num_train:]

    train_debug_dataset = Subset(dataset, train_indices)
    val_debug_dataset = Subset(dataset, val_indices)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        train_dataset=train_debug_dataset,
        val_dataset=val_debug_dataset,
        collate_fn=collate_fn
    )

    print(f'Debug 模式启动 | 样本数: {len(train_debug_dataset)} train + {len(val_debug_dataset)} val | epochs = {kwargs["epochs"]}')
    
    results, _, _ = train_network(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        to_df=False,
        **kwargs
    )

    print("\n Debug 完成！Pipeline 验证通过。")
    return results
