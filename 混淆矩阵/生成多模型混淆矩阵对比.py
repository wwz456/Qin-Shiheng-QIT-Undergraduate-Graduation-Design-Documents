"""生成多模型混淆矩阵对比图

将GAT、Random Forest、XGBoost和K-Nearest Neighbors的混淆矩阵放在同一张图片中进行对比。
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置中文字体
mpl.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
mpl.rcParams['axes.unicode_minus'] = False

# 四个模型的混淆矩阵数据
confusion_matrices = {
    'GAT': np.array([
        [1769, 143, 5],
        [88, 370, 23],
        [0, 17, 75]
    ]),
    'Random Forest': np.array([
        [1733, 180, 4],
        [130, 318, 33],
        [2, 13, 77]
    ]),
    'XGBoost': np.array([
        [1686, 229, 2],
        [119, 336, 26],
        [2, 22, 68]
    ]),
    'K-Nearest Neighbors': np.array([
        [1593, 308, 16],
        [105, 313, 63],
        [2, 11, 79]
    ])
}

class_names = ['Class 0\n(良品)', 'Class 1\n(中等缺陷)', 'Class 2\n(严重缺陷)']
model_names = ['GAT', 'Random Forest', 'XGBoost', 'K-Nearest Neighbors']

def plot_confusion_matrices_comparison(cm_dict, class_names, model_names):
    """绘制多模型混淆矩阵对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        cm = cm_dict[model_name]
        
        # 计算准确率
        accuracy = np.trace(cm) / np.sum(cm)
        
        # 绘制混淆矩阵
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # 设置标签
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=f'{model_name}\n准确率: {accuracy*100:.1f}%')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        
        # 添加数值标签
        fmt = 'd'
        thresh = cm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, format(cm[j, k], fmt),
                        ha="center", va="center",
                        color="white" if cm[j, k] > thresh else "black",
                        fontsize=10, fontweight='bold')
    
    # 添加颜色条
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # 添加整体标题
    fig.suptitle('不同模型的混淆矩阵对比', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('多模型混淆矩阵对比.png', dpi=300, bbox_inches='tight')
    plt.savefig('多模型混淆矩阵对比.svg', dpi=300, bbox_inches='tight')
    plt.savefig('多模型混淆矩阵对比.pdf', dpi=300, bbox_inches='tight')
    print("多模型混淆矩阵对比图已保存")

def plot_normalized_comparison(cm_dict, class_names, model_names):
    """绘制归一化混淆矩阵对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        cm = cm_dict[model_name]
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制归一化混淆矩阵
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        
        # 设置标签
        ax.set(xticks=np.arange(cm_normalized.shape[1]),
               yticks=np.arange(cm_normalized.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=f'{model_name}')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        
        # 添加数值标签
        fmt = '.2f'
        thresh = cm_normalized.max() / 2.
        for j in range(cm_normalized.shape[0]):
            for k in range(cm_normalized.shape[1]):
                ax.text(k, j, format(cm_normalized[j, k], fmt),
                        ha="center", va="center",
                        color="white" if cm_normalized[j, k] > thresh else "black",
                        fontsize=8)
    
    # 添加颜色条
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # 添加整体标题
    fig.suptitle('不同模型的归一化混淆矩阵对比', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('多模型归一化混淆矩阵对比.png', dpi=300, bbox_inches='tight')
    plt.savefig('多模型归一化混淆矩阵对比.svg', dpi=300, bbox_inches='tight')
    print("多模型归一化混淆矩阵对比图已保存")

if __name__ == '__main__':
    print("生成多模型混淆矩阵对比图...")
    plot_confusion_matrices_comparison(confusion_matrices, class_names, model_names)
    plot_normalized_comparison(confusion_matrices, class_names, model_names)
    print("\n所有对比图已生成完成！")
    print("生成的文件：")
    print("- 多模型混淆矩阵对比.png")
    print("- 多模型混淆矩阵对比.svg")
    print("- 多模型混淆矩阵对比.pdf")
    print("- 多模型归一化混淆矩阵对比.png")
    print("- 多模型归一化混淆矩阵对比.svg")
