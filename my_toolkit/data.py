# data.py
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import zipfile
import tarfile
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def download(url, root_dir, file_name=None):
    """
    下载文件到指定目录

    :param url: 文件的下载链接
    :param root_dir: 保存文件的根目录
    :param file_name: 保存的文件名，默认为None，能够自动从url中提取
    :return: 下载文件的完整路径（若文件已存在则直接返回）
    """
    if file_name is None:
        file_name = Path(urlparse(url).path).name
    file_path = Path(root_dir) / file_name

    if file_path.exists():
        print(f'文件 {file_path} 已存在')
        return file_path

    # 创建目录
    Path(root_dir).mkdir(parents=True, exist_ok=True)

    # 下载请求，流式下载
    response = requests.get(url, stream=True)

    # 从请求头中获取文件大小信息
    total_size = int(response.headers.get('content-length'))

    with file_path.open('wb') as f:
        for data in tqdm(response.iter_content(chunk_size=1024*1024), total=total_size//(1024*1024), unit='MB', desc='下载中', leave=False):
            f.write(data)

    print(f'下载结束，文件已保存至{file_path}')

    return file_path


def extract(file_path, extract_to_dir, remove=True):
    """
    解压压缩文件（支持 .zip, .tar, .tgz, .tar.gz 格式）

    :param file_path: 压缩文件路径
    :param extract_to_dir: 解压目标目录
    :param remove: 解压后是否删除原压缩文件，默认True
    :raises FileNotFoundError: 压缩文件不存在时抛出
    """
    file_path = Path(file_path)
    extract_to_dir = Path(extract_to_dir)
    extract_to_dir.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        raise FileNotFoundError(f'文件 {file_path} 不存在')

    if file_path.suffix == '.zip':
        with zipfile.ZipFile(file_path) as zip_file:
            # 逐个解压，显示进度条
            files = zip_file.namelist()
            for file in tqdm(files, total=len(files), unit='个', desc='解压中', leave=False):
                zip_file.extract(file, extract_to_dir)

    elif file_path.suffix in ['.tgz', '.tar'] or str(file_path).endswith('.tar.gz'):
        with tarfile.open(file_path) as tar_file:
            files = tar_file.getnames()
            for file in tqdm(files, total=len(files), unit='个', desc='解压中', leave=False):
                tar_file.extract(file, extract_to_dir)

    print(f'文件 {file_path} 已解压到目录 {extract_to_dir}')

    # 创建空文件用于标记
    marker =  extract_to_dir / f'{file_path.name}-marker'
    marker.touch()

    if remove:
        file_path.unlink()


def download_and_extract(url, root_dir, file_name=None, remove=True):
    """
    下载并解压数据集

    :param url: 文件的下载链接
    :param root_dir: 保存和解压的根目录
    :param file_name: 保存的文件名，默认为None，能够自动从url中提取
    :param remove: 解压后是否删除原压缩文件，默认True
    """
    if file_name is None:
        file_name = Path(urlparse(url).path).name

    marker =  Path(root_dir) / f'{file_name}-marker'
    if marker.exists():
        print(f'文件已解压完成')
        return

    file_path = download(url, root_dir, file_name=file_name)

    if file_path.suffix in ['.zip', '.tgz', '.tar'] or str(file_path).endswith('.tar.gz'):
         extract(file_path, root_dir, remove=remove)


def split_dataset(dataset, train_ratio, val_ratio, test_ratio,
                  train_transform=None, val_transform=None, test_transform=None):
    """
    分割数据集为训练、验证、测试集

    :param dataset: 完整数据集
    :param train_ratio: 训练集比例 (0~1)
    :param val_ratio: 验证集比例 (0~1)
    :param test_ratio: 测试集比例 (0~1)
    :param set_transform: 是否设置transform，默认False
    :param train_transform: 训练集transform
    :param val_transform: 验证集transform
    :param test_transform: 测试集transform
    :return: (train_dataset, val_dataset, test_dataset)
    """
    ratios = [train_ratio, val_ratio, test_ratio]

    # 检查比例范围
    for name, ratio in zip(['train', 'val', 'test'], ratios):
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f'{name} ratio 必须满足 0.0 <= ratio <= 1.0')

    # 检查和是否为1
    if abs(sum(ratios) - 1.0) > 1e-5:
        raise ValueError(f'比例之和必须为1.0，当前为{sum(ratios)}')

    # 计算长度
    lengths = [int(len(dataset) * ratio) for ratio in ratios[:-1]]
    last_length = len(dataset) - sum(lengths)
    lengths.append(last_length)

    # 分割
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths)

    # 设置transform
    if len(train_dataset) > 0 and train_transform is not None:
        # 注意要用 .dataset.transform，因为random_split返回的是Subset对象，其dataset属性指向总数据集
        train_dataset.dataset.transform = train_transform
    if len(val_dataset) > 0 and val_transform is not None:
        val_dataset.dataset.transform = val_transform
    if len(test_dataset) > 0 and test_transform is not None:
        test_dataset.dataset.transform = test_transform

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size, train_dataset, val_dataset=None, test_dataset=None,
                    num_workers=0, pin_memory=True, collate_fn=None):
    """
    创建训练、验证、测试数据加载器

    :param batch_size: 批次大小
    :param train_dataset: 训练数据集
    :param val_dataset: 验证数据集，可选
    :param test_dataset: 测试数据集，可选
    :param num_workers: 数据加载线程数，默认0（主线程）
    :param pin_memory: 是否锁定内存（加速GPU传输），默认True
    :param collate_fn: 自定义collate函数，默认None（使用PyTorch默认的default_collate）
    :return: (train_loader, val_loader, test_loader) 三元组，不存在的返回None
    """
    def create_loader(dataset, shuffle):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=collate_fn  # None -> default_collate
        )

    train_loader = create_loader(train_dataset, shuffle=True)
    val_loader = create_loader(val_dataset, shuffle=False)
    test_loader = create_loader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader


def convert_to_df(file, names=None, show=False):
    """
    将常见数据文件转换为DataFrame

    :param file: 文件路径 (.csv, .tsv, .txt)
    :param names: 列名列表，如 ['image', 'label']
    :param show: 是否显示前几行数据，默认False
    :return: DataFrame

    说明：
        - 适用于深度学习常见的数据文件（默认没有表头）
        - CSV文件：逗号分隔
        - TSV文件：制表符分隔
        - TXT文件：空格分隔
        - 如需其他分隔符，请直接使用 pd.read_csv
    """
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f'文件 {file} 不存在')

    if file.suffix == '.csv':
        df = pd.read_csv(file, header=None, names=names)
    elif file.suffix == '.tsv':
        df = pd.read_csv(file, sep='\t', header=None, names=names)
    elif file.suffix == '.txt':
        df = pd.read_csv(file, sep=' ', header=None, names=names)
    else:
        raise ValueError(f"不支持的文件格式: {file.suffix}，仅支持 .csv, .tsv, .txt")

    if show:
        print(f'文件加载成功: {file}')
        print(f'Shape: {df.shape}')
        print('预览:')
        print(df.head())

    return df

def preview_dataset(dataset, show_picture=False):
    """
    预览数据集

    :param dataset: PyTorch Dataset
    :param show_picture: 是否显示图片（针对3通道图像）
    """
    example = dataset[0]
    print(f'数据集大小 {len(dataset)}')

    for i, item in enumerate(example):
        if isinstance(item, dict):
            print(f'[{i}] dict:')
            for j, (k, v) in enumerate(item.items()):
                print(f'    Key [{j}] Shape: {k}: {v.shape}')
                print(f'              Type: {v.dtype}')
        else:
            print(f'[{i}] Shape: {item.shape} Type: {item.dtype}')

            if show_picture and len(item.shape) == 3 and item.shape[0] == 3:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])

                # 反归一化
                img_np = item.permute(1, 2, 0).cpu().numpy()
                img = img_np * std + mean

                plt.imshow(img)
                plt.title('Image')
                plt.axis('off') # 关闭坐标轴
                plt.show()

    print('数据集构建成功，预览结束')


def clip_boxes(boxes, mode='yolo', img_size=None, eps=1e-6, to_tensor=False):
    """
    将边界框坐标裁剪到有效范围内，并处理微小浮点误差

    :param boxes: 边界框数组，shape (N, 4)
    :param mode: 边界框格式，'yolo' 或 'pascal'，默认 'yolo'
                 - 'yolo': [x_center, y_center, width, height]，值域 [0, 1]
                 - 'pascal': [x_min, y_min, x_max, y_max]，值域 [0, img_size]
    :param img_size: 图像尺寸 (width, height)，pascal 模式必需
    :param eps: 容差阈值，小于此值的微小误差将被修正，默认 1e-6
    :param to_tensor: 是否返回 torch.Tensor，默认 False
    :return: 裁剪后的边界框，格式与原输入相同
    """
    boxes = np.array(boxes, dtype=np.float32)

    if mode == 'yolo':
        # [xc, yc, w, h] 必须在 [0, 1]
        boxes = np.clip(boxes, 0, 1)
        # 处理微小误差（注意括号优先级）
        boxes[(np.abs(boxes) < eps) & (boxes < 0)] = 0
        boxes[(np.abs(boxes - 1) < eps) & (boxes > 1)] = 1

    elif mode == 'pascal':
        if img_size is None:
            raise ValueError("pascal mode requires img_size (width, height)")
        h, w = img_size

        # 分别处理每个坐标
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)  # x_min
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)  # y_min
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)  # x_max
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)  # y_max

        # 处理微小误差
        boxes[(np.abs(boxes[:, 0]) < eps) & (boxes[:, 0] < 0), 0] = 0
        boxes[(np.abs(boxes[:, 1]) < eps) & (boxes[:, 1] < 0), 1] = 0
        boxes[(np.abs(boxes[:, 2] - w) < eps) & (boxes[:, 2] > w), 2] = w
        boxes[(np.abs(boxes[:, 3] - h) < eps) & (boxes[:, 3] > h), 3] = h

    else:
        raise ValueError(f'mode {mode} 未定义, mode must be yolo or pascal')

    if to_tensor:
        boxes = torch.tensor(boxes, dtype=torch.float32)

    return boxes


def shift_box_format(boxes, img_size, to, to_tensor=False):
    """
    在 YOLO 和 Pascal VOC 两种边界框格式之间转换

    :param boxes: 边界框数组，shape (N, 4)
    :param img_size: 图像尺寸 (width, height)
    :param to: 目标格式，'pascal' 或 'yolo'
               - 'pascal': 从 YOLO 转换为 [x_min, y_min, x_max, y_max]
               - 'yolo': 从 Pascal 转换为 [x_center, y_center, width, height]
    :param to_tensor: 是否返回 torch.Tensor，默认 False
    :return: 转换后的边界框
    """
    boxes = np.array(boxes, dtype=np.float32)
    h, w = img_size # (H, W)

    if to == 'pascal':
        # YOLO -> Pascal VOC
        x_min = (boxes[:, 0] - boxes[:, 2] / 2) * w
        y_min = (boxes[:, 1] - boxes[:, 3] / 2) * h
        x_max = (boxes[:, 0] + boxes[:, 2] / 2) * w
        y_max = (boxes[:, 1] + boxes[:, 3] / 2) * h
        boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

    elif to == 'yolo':
        # Pascal VOC -> YOLO
        x_center = (boxes[:, 0] + boxes[:, 2]) / 2 / w
        y_center = (boxes[:, 1] + boxes[:, 3]) / 2 / h
        width = (boxes[:, 2] - boxes[:, 0]) / w
        height = (boxes[:, 3] - boxes[:, 1]) / h
        boxes = np.stack([x_center, y_center, width, height], axis=1)

    else:
        raise ValueError(f'to 必须是 "pascal" 或 "yolo", 得到 {to}')

    if to_tensor:
        boxes = torch.tensor(boxes, dtype=torch.float32)

    return boxes


def polygon_to_mask(polygon, img_size):
    """
    将多边形坐标转换为二值掩码

    :param polygon: numpy.ndarray, shape (N, 2), 归一化的多边形顶点坐标 (范围 0~1)
    :param img_size: tuple, (height, width), 图像尺寸
    :return: numpy.ndarray, shape (height, width), dtype=np.uint8, 二值掩码 (0 或 1)

    示例:
        polygon = np.array([[0.2, 0.3], [0.5, 0.2], [0.8, 0.4], [0.6, 0.7]])
        mask = polygon_to_mask(polygon, (480, 640))
        print(mask.shape)  # (480, 640)
    """
    # 创建空白图像
    h, w = img_size
    # 灰度图(单通道), 背景为 0
    mask = Image.new('L', (w, h), 0)

    # 将归一化坐标转换为像素坐标
    pixel_points = (polygon * [w, h]).astype(np.int32)

    # 画多边形掩码
    ImageDraw.Draw(mask).polygon(
        pixel_points.flatten().tolist(),  # 展平为 [x1, y1, x2, ...]
        outline=1,  # 边框颜色
        fill=1  # 填充颜色
    )

    return np.array(mask, dtype=np.uint8)