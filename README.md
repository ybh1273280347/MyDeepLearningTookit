# MyDeepLearningToolkit

一个轻量级、模块化的深度学习训练工具箱，把那些枯燥的重复劳动打包成几行代码。

✨ 核心特性

📦 一行代码下载并解压数据集  
🔄 一行代码创建 DataLoader（支持自定义 collate_fn）  
🎯 统一接口支持多种任务：图像分类 / 目标检测 / 语义分割 / 实例分割 / 自监督学习 / 回归（通过 task 参数指定）  
📊 训练进度条 + 指标记录 + 可视化一站式集成  
🚀 快速调试模式：10~30 秒验证整个训练 pipeline 是否跑通


📖 快速开始
git clone https://github.com/ybh1273280347/MyDeepLearningToolkit.git

下载数据
```python

from mytoolkit import download_and_extract

url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip'
download_and_extract(url, './dataset')
```
定义数据增强（以目标检测为例）

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomResizedCrop((224, 224), scale=(0.8, 1)),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```
分割数据集

假设 dataset 是一个已加载的 Dataset 对象：
```python
from mytoolkit import split_dataset

train_dataset, val_dataset, _ = split_dataset(
    dataset, 0.8, 0.2, 0,
    train_transform=train_transform,
    val_transform=val_transform
)
```
创建 DataLoader
```python
import torch
from mytoolkit import get_dataloaders

def collate_fn(batch):
    imgs_tuple, targets = tuple(zip(*batch))
    imgs = torch.stack(imgs_tuple, dim=0)
    return imgs, targets

train_loader, val_loader, _ = get_dataloaders(
    batch_size=32,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_workers=4,
    collate_fn=collate_fn
)
```
训练模型
```python
from mytoolkit import train_network, get_detection_metrics
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 示例：使用 torchvision 的 Faster R-CNN
model = fasterrcnn_resnet50_fpn(num_classes=81)

results_df, _, _ = train_network(
    model=model,
    loss_fn=None,  # 检测/分割任务通常由模型内部处理 loss
    task='detection',
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    score_funcs=get_detection_metrics(),
    epochs=20,
    to_df=True
)
```
💡 注：本工具箱聚焦训练流程抽象，模型与评估指标可由用户灵活提供，便于适配不同研究需求。

可视化结果
```python
from mytoolkit import visualize_results

metrics = get_detection_metrics(for_visualization=True)
visualize_results(results_df, metrics, mode='epoch')
```
快速调试（10~30秒验证整个 pipeline）
```python
from mytoolkit import quick_debug

# 只需提供 dataset 和 model，其余自动配置为最小规模
results = quick_debug(
    dataset=dataset,
    model=model,
    task='detection',
    collate_fn=collate_fn
    # 注意：可不指定 epochs！内部已限制 ≤3，确保快速完成
)
```
💡 默认使用 1 个训练 batch + 1 个验证 batch + ≤3 轮训练，通常在 30 秒内完成，适用于：  
验证数据加载是否正常  
检查模型前向/反向是否出错  
确认 loss 和指标计算逻辑正确  
快速测试新任务适配效果

📁 项目结构
```markdown
mytoolkit/
├── data.py          # 数据集下载/解压、通用 Dataset 封装、DataLoader 构建，及其他数据处理函数
├── training.py      # 统一训练循环，通过 task 分发逻辑支持多任务
├── metrics.py       # 任务相关评估指标计算与可视化配置
├── debug.py         # 提供 inspect_* 系列调试函数 + quick_debug 快速验证工具
└── utils.py         # 随机种子固定、加载模型权重、适配器等辅助工具

🎯 支持的任务与评估指标

任务类型           支持指标
图像分类           Accuracy, Precision, Recall, F1

目标检测           mAP, mAP@0.5, mAP@0.75

语义/实例分割      mIoU, Pixel Accuracy 等

......
```
📝 致谢

本工具箱源于我在深度学习自学与项目实践中的工程需求，旨在减少重复性代码编写，提升实验迭代效率。目前仍在持续完善中，如有建议或发现 bug，欢迎提出！

如果你觉得这个项目对你有帮助，欢迎 ⭐ Star 支持！

📬 联系

作者：袁斌皓  
GitHub：https://github.com/ybh1273280347/MyDeepLearningToolkit  
