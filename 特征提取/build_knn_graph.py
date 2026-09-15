"""
第二步：图构建前的相似性计算（KNN Graph）
功能：
1. 以产品实例（批次）为节点
2. 计算节点特征（工艺参数 + 设备/材料ID Embedding）
3. 构建K近邻图：同模具下最相似的K个产品相连
4. 可选：时间相邻边（前后相邻时间的产品相连）
5. 生成图数据文件供GAT模型训练使用
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
import pickle
import os
from collections import defaultdict

print("=" * 60)
print("第二步：KNN图构建")
print("=" * 60)

# 1. 加载预处理后的数据
print("\n[1/6] 加载预处理数据...")
df = pd.read_csv('processed_raw_data.csv')
print(f"数据形状: {df.shape}")
print(f"列数: {len(df.columns)}")

# 2. 加载标签数据
print("\n[2/6] 加载标签数据...")
y_train = pd.read_csv('raw/y_train.csv')
print(f"标签数据: {y_train.shape}")
print(f"标签分布:\n{y_train['label'].value_counts().sort_index()}")

# 合并标签 - 使用行索引对齐
print("\n[3/6] 合并特征和标签...")
# y_train的Batch_ID是从0开始的索引，对应processed_raw_data的行顺序
df = df.reset_index(drop=True)
df['label'] = y_train['label'].values
print(f"合并后数据: {df.shape}")
print(f"有标签的数据分布:\n{df['label'].value_counts().sort_index()}")

# 3. 定义节点特征
print("\n[4/6] 构建节点特征...")

# 连续变量特征（统计特征）
continuous_features = [col for col in df.columns if any(x in col for x in ['_mean', '_std', '_max', '_min', '_median'])]
print(f"连续变量特征数: {len(continuous_features)}")

# 类别变量（已编码）
categorical_features = ['Mold_ID_encoded', 'Phase_mode', 'MouldFlow1_mode', 'MouldFlow2_mode', 'MouldFlow3_mode']
categorical_features = [f for f in categorical_features if f in df.columns]
print(f"类别变量特征数: {len(categorical_features)}")

# 时间特征
time_features = ['SampleTime_Range']
print(f"时间特征数: {len(time_features)}")

# 组合所有特征
feature_cols = continuous_features + categorical_features + time_features
print(f"总特征数: {len(feature_cols)}")

# 提取特征矩阵
X = df[feature_cols].values
y = df['label'].values
batch_ids = df['Batch_ID'].values
mold_ids = df['Mold_ID'].values

print(f"\n特征矩阵形状: {X.shape}")
print(f"标签形状: {y.shape}")

# 标准化特征
print("\n标准化特征...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 构建KNN图
print("\n[5/6] 构建KNN图...")

K = 10  # 每个节点连接最近的K个邻居
edge_list = []
edge_weights = []

# 方法1: 全局KNN图（所有样本之间找最近邻）
print("构建全局KNN图...")
knn = NearestNeighbors(n_neighbors=K+1, metric='euclidean')  # +1 因为会包含自己
knn.fit(X_scaled)
distances, indices = knn.kneighbors(X_scaled)

# 构建边
for i, (dist, idx) in enumerate(zip(distances, indices)):
    for d, neighbor_idx in zip(dist[1:], idx[1:]):  # 跳过第一个（自己）
        # 添加双向边
        edge_list.append([i, neighbor_idx])
        # 使用距离倒数作为边权重（距离越近权重越大）
        weight = 1.0 / (1.0 + d) if d > 0 else 1.0
        edge_weights.append(weight)

print(f"全局KNN边数: {len(edge_list)}")

# 方法2: 按模具分组构建KNN图（同模具下的样本相连）
print("\n构建同模具KNN图...")
mold_edge_count = 0
unique_molds = df['Mold_ID'].unique()
print(f"模具数量: {len(unique_molds)}")

for mold_id in unique_molds:
    # 获取该模具下的所有样本索引
    mold_mask = df['Mold_ID'] == mold_id
    mold_indices = df[mold_mask].index.tolist()
    
    if len(mold_indices) > 1:
        # 如果同模具下有多个样本，互相连接
        for i in range(len(mold_indices)):
            for j in range(i+1, len(mold_indices)):
                edge_list.append([mold_indices[i], mold_indices[j]])
                edge_list.append([mold_indices[j], mold_indices[i]])
                edge_weights.append(2.0)  # 同模具边权重更高
                edge_weights.append(2.0)
                mold_edge_count += 2

print(f"同模具边数: {mold_edge_count}")
print(f"总KNN边数: {len(edge_list)}")

# 5. 可选：添加时间相邻边
print("\n添加时间相邻边...")

# 按Batch_ID排序，添加相邻批次之间的边
df_sorted = df.sort_values('Batch_ID').reset_index()
time_edges = 0

for i in range(len(df_sorted) - 1):
    current_idx = df_sorted.loc[i, 'index']
    next_idx = df_sorted.loc[i + 1, 'index']
    
    # 添加时间相邻边
    edge_list.append([current_idx, next_idx])
    edge_list.append([next_idx, current_idx])
    edge_weights.append(0.5)  # 时间边的权重可以设置小一些
    edge_weights.append(0.5)
    time_edges += 2

print(f"时间相邻边数: {time_edges}")
print(f"总边数: {len(edge_list)}")

# 6. 转换为PyG格式
print("\n[6/6] 转换为PyG图数据...")

# 创建边索引矩阵
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

# 节点特征
x = torch.tensor(X_scaled, dtype=torch.float)

# 标签
y_tensor = torch.tensor(y, dtype=torch.long)

# 创建PyG Data对象
data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y_tensor,
    num_nodes=len(df)
)

print(f"\n图数据信息:")
print(f"  节点数: {data.num_nodes}")
print(f"  边数: {data.num_edges}")
print(f"  节点特征维度: {data.num_features}")
print(f"  类别数: {data.y.max().item() + 1}")
print(f"  标签分布: {torch.bincount(data.y)}")

# 7. 保存图数据
print("\n保存图数据...")

# 保存PyG数据
torch.save(data, 'knn_graph_data.pt')
print("[OK] 图数据已保存: knn_graph_data.pt")

# 保存节点信息供后续使用
node_info = pd.DataFrame({
    'index': range(len(df)),
    'Batch_ID': batch_ids,
    'Mold_ID': mold_ids,
    'label': y
})
node_info.to_csv('node_info.csv', index=False)
print("[OK] 节点信息已保存: node_info.csv")

# 保存特征名称
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("[OK] 特征名称已保存: feature_names.pkl")

# 保存scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("[OK] Scaler已保存: scaler.pkl")

print("\n" + "=" * 60)
print("KNN图构建完成！")
print("=" * 60)
print(f"\n生成的文件:")
print(f"  1. knn_graph_data.pt - PyG图数据")
print(f"  2. node_info.csv - 节点信息")
print(f"  3. feature_names.pkl - 特征名称")
print(f"  4. scaler.pkl - 标准化器")
print(f"\n图结构统计:")
print(f"  - 节点数: {data.num_nodes}")
print(f"  - 边数: {data.num_edges}")
print(f"  - 平均度数: {data.num_edges / data.num_nodes:.2f}")
