import torch
import numpy as np
from torch_geometric.data import Data
from dataset import MoldGATDatasetFromRawDataFinalSimple
from collections import Counter

# 加载原始数据集
dataset = MoldGATDatasetFromRawDataFinalSimple(root='.')

# 提取所有样本的特征和标签
all_features = []
all_labels = []

for data in dataset:
    # 将图级特征提取出来（使用全局平均池化）
    graph_feature = data.x.mean(dim=0).numpy()
    all_features.append(graph_feature)
    all_labels.append(data.y.item())

all_features = np.array(all_features)
all_labels = np.array(all_labels)

print(f"原始样本分布: {Counter(all_labels)}")

# 计算每个类别的目标数量
target_count = 500  # 设置目标样本数为500
print(f"目标样本数: {target_count}")

# 为每个类别创建平衡的数据集
balanced_features = []
balanced_labels = []

for class_label in [0, 1, 2]:
    # 提取该类别的原始样本
    class_features = all_features[all_labels == class_label]
    class_count = len(class_features)
    
    # 添加原始样本
    balanced_features.extend(class_features)
    balanced_labels.extend([class_label] * class_count)
    
    # 生成新样本
    if class_count < target_count:
        needed_count = target_count - class_count
        print(f"为类别 {class_label} 生成 {needed_count} 个新样本...")
        
        # 使用随机噪声生成新特征
        for i in range(needed_count):
            # 随机选择一个原始样本
            random_idx = np.random.choice(class_count)
            base_feature = class_features[random_idx]
            
            # 添加随机噪声生成新特征
            noise = np.random.normal(0, 0.05, size=base_feature.shape)
            new_feature = base_feature + noise
            
            balanced_features.append(new_feature)
            balanced_labels.append(class_label)

balanced_features = np.array(balanced_features)
balanced_labels = np.array(balanced_labels)

print(f"平衡后的样本数: {len(balanced_features)}")
print(f"平衡后的样本分布: {Counter(balanced_labels)}")

# 保存平衡后的特征和标签
np.save('balanced_features.npy', balanced_features)
np.save('balanced_labels.npy', balanced_labels)
print("平衡后的特征和标签已保存为 balanced_features.npy 和 balanced_labels.npy")