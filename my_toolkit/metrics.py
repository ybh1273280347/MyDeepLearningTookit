# metrics.py
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy_score(y_true, y_pred):
    """准确率"""
    return sm.accuracy_score(y_true, y_pred)


def precision_score(y_true, y_pred, average='weighted'):
    """精确率"""
    return sm.precision_score(y_true, y_pred, average=average, zero_division=0)


def recall_score(y_true, y_pred, average='weighted'):
    """召回率"""
    return sm.recall_score(y_true, y_pred, average=average, zero_division=0)


def f1_score(y_true, y_pred, average='weighted'):
    """F1分数"""
    return sm.f1_score(y_true, y_pred, average=average, zero_division=0)


def auc_score(y_true, y_pred_proba):
    """AUC分数（需要预测概率）"""
    return sm.roc_auc_score(y_true, y_pred_proba)


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """绘制混淆矩阵"""
    cm = sm.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def get_classification_metrics(for_visualization=False):
    """获取分类任务指标字典"""
    metrics = {
        'acc': accuracy_score,
        'precision': lambda y, p: precision_score(y, p, average='weighted'),
        'recall': lambda y, p: recall_score(y, p, average='weighted'),
        'f1': lambda y, p: f1_score(y, p, average='weighted'),
    }
    if for_visualization:
        metrics = list(metrics.keys()) + ['loss']

    return metrics


def get_binary_metrics(for_visualization=False):
    """获取二分类任务指标字典（含AUC）"""
    metrics = {
        'acc': accuracy_score,
        'precision': lambda y, p: precision_score(y, p, average='binary'),
        'recall': lambda y, p: recall_score(y, p, average='binary'),
        'f1': lambda y, p: f1_score(y, p, average='binary'),
        'auc': auc_score
    }
    if for_visualization:
        metrics = list(metrics.keys()) + ['loss']

    return metrics


def get_detection_metrics(for_visualization=False):
    """
    获取检测任务最常用评估指标列表

    :return: 指标名称列表，可直接传入 train_network 的 score_funcs 参数
    """
    metrics = ['map',  # mAP@0.5:0.95 (最主要)
            'map_50',  # mAP@0.5 (宽松标准)
            'map_75']  # mAP@0.75 (严格标准
    return metrics if not for_visualization else metrics + ['loss']

def get_semantic_segm_metrics(for_visualization=False):
    """
    语义分割常用评估指标
    - mIoU: 平均交并比（主指标）
    - pixel acc: 像素准确率
    """
    metrics = ['mIoU', 'pixel acc']
    return metrics if not for_visualization else metrics + ['loss']


def get_instance_segm_metrics(for_visualization=False):
    """
    实例分割常用评估指标
    - map: mask AP@0.5:0.95（主指标）
    - map_50: mask AP@0.5
    - map_75: mask AP@0.75
    """
    metrics = ['map', 'map_50', 'map_75']
    return metrics if not for_visualization else metrics + ['loss']


def visualize_metric(results_df, metric, mode='epoch'):
    """
    可视化单个指标的训练曲线

    :param results_df: 训练结果DataFrame，包含epoch、total time及各指标列
    :param metric: 要可视化的指标名称，如 'loss', 'acc', 'score'
    :param mode: x轴模式，可选 'epoch' 或 'total time'，默认 'epoch'
    :return: None，直接显示图表
    """

    if mode not in ['epoch', 'total time']:
        raise ValueError('mode should be "epoch" or "total time"')

    cols = [col for col in results_df.columns if col.endswith(metric)]

    plt.figure(figsize=(10, 6))

    for col in cols:
        sns.lineplot(data=results_df, x=mode, y=col, label=col.split(' ')[0].title())

    plt.title(f'{metric.title()} over {mode.title()}')
    plt.xlabel(mode.title())
    plt.ylabel(metric.title())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# noinspection PyUnboundLocalVariable
def visualize_results(results_df, metrics: list, mode='epoch'):
    """
    可视化多个指标的训练曲线（并排子图）

    :param results_df: 训练结果DataFrame，包含epoch、total time及各指标列
    :param metrics: 要可视化的指标列表，如 ['loss', 'acc', 'score']
    :param mode: x轴模式，可选 'epoch' 或 'total time'，默认 'epoch'
    :return: None，直接显示图表
    """

    if mode not in ['epoch', 'total time']:
        raise ValueError('mode should be "epoch" or "total time"')

    num = len(metrics)

    # 计算行数和列数
    col_num = min(num, 3)
    row_num = (num + col_num - 1) // col_num # 向上取整公式 (a + b - 1) // b

    if num == 1:
        visualize_metric(results_df, metrics[0], mode=mode)

    else:
        _, axes = plt.subplots(row_num, col_num, figsize=(col_num * 6, row_num * 6))
        # 展平，方便索引
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            cols = [col for col in results_df.columns if col.endswith(metric)]

            for col in cols:
                sns.lineplot(data=results_df, x=mode, y=col, label=col.split(' ')[0].title(), ax=axes[i])

            axes[i].set_title(f'{metric.title()} over {mode.title()}')
            axes[i].set_xlabel(mode.title())
            axes[i].set_ylabel(metric.title())
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        for j in range(i, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()