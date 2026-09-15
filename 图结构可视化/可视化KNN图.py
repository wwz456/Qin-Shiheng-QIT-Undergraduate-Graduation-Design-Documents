"""KNN图可视化：展示不同类别使用不同K值的图结构

本脚本生成KNN图的可视化，展示：
1. 不同类别节点的分布
2. 不同K值对图结构的影响
3. 特征空间中的样本分布
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from sklearn.manifold import TSNE

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 颜色设置
class_colors = {
    0: '#1f77b4',   # 蓝色 - 良品
    1: '#ff7f0e',   # 橙色 - 中等缺陷
    2: '#2ca02c'    # 绿色 - 严重缺陷
}

class_names = {
    0: 'Class 0 (良品)',
    1: 'Class 1 (中等缺陷)',
    2: 'Class 2 (严重缺陷)'
}

def visualize_knn_graph():
    """可视化KNN图结构"""
    # 生成模拟数据
    np.random.seed(42)
    
    # Class 0: 下采样后的核心样本，分布紧凑
    class0 = np.random.normal(loc=[0, 0], scale=[0.8, 0.8], size=(200, 2))
    # Class 1: 中等分布
    class1 = np.random.normal(loc=[3, 0], scale=[1.2, 1.2], size=(200, 2))
    # Class 2: SMOTE生成的样本，分布更弥散
    class2_core = np.random.normal(loc=[1.5, 3], scale=[0.6, 0.6], size=(50, 2))
    class2_smote = np.random.normal(loc=[1.5, 3], scale=[1.5, 1.5], size=(150, 2))
    class2 = np.vstack([class2_core, class2_smote])
    
    # 合并数据
    X = np.vstack([class0, class1, class2])
    y = np.array([0]*200 + [1]*200 + [2]*200)
    
    # 使用t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X)
    
    # 构建KNN图（模拟）
    k_values = {0: 5, 1: 8, 2: 10}
    edges = []
    
    for class_id in [0, 1, 2]:
        class_mask = y == class_id
        class_indices = np.where(class_mask)[0]
        class_points = X_embedded[class_mask]
        
        k = k_values[class_id]
        # 为每个点找最近的k个邻居
        for i, point in enumerate(class_points):
            distances = np.sqrt(np.sum((class_points - point)**2, axis=1))
            nearest_indices = np.argsort(distances)[1:k+1]
            for j in nearest_indices:
                src = class_indices[i]
                dst = class_indices[j]
                edges.append([src, dst])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    
    # 图1: 特征空间分布
    ax1.set_title('特征空间中的样本分布', fontsize=14, fontweight='bold')
    for class_id in [0, 1, 2]:
        mask = y == class_id
        ax1.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                    c=class_colors[class_id], label=class_names[class_id],
                    alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
    
    # 添加分布椭圆
    for class_id in [0, 1, 2]:
        mask = y == class_id
        points = X_embedded[mask]
        mean = points.mean(axis=0)
        cov = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        
        ellipse = patches.Ellipse(mean, width*2, height*2, angle=angle,
                                 edgecolor=class_colors[class_id],
                                 facecolor='none', linestyle='--', linewidth=2)
        ax1.add_patch(ellipse)
    
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('t-SNE维度1', fontsize=12)
    ax1.set_ylabel('t-SNE维度2', fontsize=12)
    
    # 图2: KNN图结构
    ax2.set_title('KNN图结构（不同类别使用不同K值）', fontsize=14, fontweight='bold')
    
    # 绘制节点
    for class_id in [0, 1, 2]:
        mask = y == class_id
        ax2.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                    c=class_colors[class_id], label=f'{class_names[class_id]} (K={k_values[class_id]})',
                    alpha=0.8, s=60, edgecolors='w', linewidth=1)
    
    # 绘制边
    for src, dst in edges:
        ax2.plot([X_embedded[src, 0], X_embedded[dst, 0]],
                 [X_embedded[src, 1], X_embedded[dst, 1]],
                 color='gray', alpha=0.3, linewidth=0.8)
    
    # 添加图例说明K值
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('t-SNE维度1', fontsize=12)
    ax2.set_ylabel('t-SNE维度2', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('KNN图可视化.png', dpi=300, bbox_inches='tight')
    plt.savefig('KNN图可视化.svg', dpi=300, bbox_inches='tight')
    print("KNN图可视化已保存")

def visualize_k_effect():
    """可视化不同K值对连接的影响"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    k_values = [5, 8, 10]
    titles = ['K=5 (Class 0: 多数类)', 'K=8 (Class 1: 中等类)', 'K=10 (Class 2: 少数类)']
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        
        # 生成圆形分布的点
        np.random.seed(42)
        n_points = 50
        angles = np.linspace(0, 2*np.pi, n_points)
        points = np.array([np.cos(angles), np.sin(angles)]).T * 2
        points += np.random.normal(0, 0.1, points.shape)
        
        # 构建KNN连接
        edges = []
        for i, point in enumerate(points):
            distances = np.sqrt(np.sum((points - point)**2, axis=1))
            nearest_indices = np.argsort(distances)[1:k+1]
            for j in nearest_indices:
                edges.append([i, j])
        
        # 绘制节点
        ax.scatter(points[:, 0], points[:, 1], c='blue', s=50, edgecolors='w')
        
        # 绘制边
        for src, dst in edges:
            ax.plot([points[src, 0], points[dst, 0]],
                    [points[src, 1], points[dst, 1]],
                    color='gray', alpha=0.5)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('X', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Y', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('K值影响对比.png', dpi=300, bbox_inches='tight')
    plt.savefig('K值影响对比.svg', dpi=300, bbox_inches='tight')
    print("K值影响对比图已保存")

if __name__ == '__main__':
    print("生成KNN图可视化...")
    visualize_knn_graph()
    visualize_k_effect()
    print("\n所有可视化已完成！")
    print("生成的文件：")
    print("- KNN图可视化.png")
    print("- KNN图可视化.svg")
    print("- K值影响对比.png")
    print("- K值影响对比.svg")
