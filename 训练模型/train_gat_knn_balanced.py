"""
使用组合策略解决极度不平衡问题
功能：
1. 实现重采样（SMOTE/ADASYN）
2. 使用 Focal Loss + Class-Balanced Loss
3. 改进评估指标（F1-score, Recall per class）
4. 图结构增强少数类
5. 按时间划分数据集
6. GAT模型微调
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
import pickle
from imblearn.over_sampling import SMOTE, ADASYN

print("=" * 60)
print("GAT模型训练（组合策略解决类别不平衡）")
print("=" * 60)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 1. 加载原始数据（未构建图之前的预处理数据）
print("\n[1/6] 加载预处理数据...")
df = pd.read_csv('processed_raw_data.csv')
y_train = pd.read_csv('raw/y_train.csv')

# 合并标签
df = df.reset_index(drop=True)
df['label'] = y_train['label'].values

print(f"数据形状: {df.shape}")
print(f"标签分布: {dict(Counter(df['label']))}")

# 2. 特征提取
print("\n[2/6] 提取特征...")

# 连续变量特征（统计特征）
continuous_features = [col for col in df.columns if any(x in col for x in ['_mean', '_std', '_max', '_min', '_median'])]
categorical_features = ['Mold_ID_encoded', 'Phase_mode', 'MouldFlow1_mode', 'MouldFlow2_mode', 'MouldFlow3_mode']
categorical_features = [f for f in categorical_features if f in df.columns]
time_features = ['SampleTime_Range']

feature_cols = continuous_features + categorical_features + time_features
print(f"总特征数: {len(feature_cols)}")

X = df[feature_cols].values
y = df['label'].values
batch_ids = df['Batch_ID'].values
mold_ids = df['Mold_ID'].values

# 3. 分层随机划分数据集
print("\n[3/6] 分层随机划分数据集...")

from sklearn.model_selection import StratifiedShuffleSplit

# 分层随机划分，保持类别比例
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_val_idx = next(sss.split(X, y))

# 再将验证集和测试集分开
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss2.split(X[test_val_idx], y[test_val_idx]))
val_idx = test_val_idx[val_idx]
test_idx = test_val_idx[test_idx]

# 按Batch_ID排序（用于时间相邻边）
sorted_indices = np.argsort(df['Batch_ID'].values)
X_sorted = X[sorted_indices]
y_sorted = y[sorted_indices]
batch_ids_sorted = batch_ids[sorted_indices]
mold_ids_sorted = mold_ids[sorted_indices]

print(f"训练集: {len(train_idx)} 样本")
print(f"  标签分布: {dict(Counter(y[train_idx]))}")
print(f"验证集: {len(val_idx)} 样本")
print(f"  标签分布: {dict(Counter(y[val_idx]))}")
print(f"测试集: {len(test_idx)} 样本")
print(f"  标签分布: {dict(Counter(y[test_idx]))}")

# 4. 重采样少数类（SMOTE）
print("\n[4/6] 重采样少数类（SMOTE）...")

X_train = X[train_idx]
y_train = y[train_idx]

# 使用SMOTE过采样，更平衡的策略
smote = SMOTE(sampling_strategy={1: 4000, 2: 2000}, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"重采样前: {dict(Counter(y_train))}")
print(f"重采样后: {dict(Counter(y_train_resampled))}")

# 5. 构建KNN图（增强少数类）
print("\n[5/6] 构建KNN图（增强少数类）...")

# 合并重采样后的数据
X_combined = np.vstack([X_train_resampled, X[val_idx], X[test_idx]])
y_combined = np.hstack([y_train_resampled, y[val_idx], y[test_idx]])

# 标准化特征
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# 构建KNN图
K = 5  # 基础K值
edge_list = []
edge_weights = []

# 按类别设置不同的K值
class_K = {
    0: 5,  # 多数类
    1: 8,  # 中等类
    2: 10  # 少数类
}

print("构建KNN图...")
from sklearn.neighbors import NearestNeighbors

for class_id in [0, 1, 2]:
    class_mask = y_combined == class_id
    class_indices = np.where(class_mask)[0]
    if len(class_indices) == 0:
        continue
    
    k = class_K[class_id]
    print(f"  Class {class_id}: K={k}, 样本数={len(class_indices)}")
    
    class_features = X_scaled[class_mask]
    knn = NearestNeighbors(n_neighbors=min(k+1, len(class_indices)), metric='euclidean')
    knn.fit(class_features)
    distances, indices = knn.kneighbors(class_features)
    
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        src_idx = class_indices[i]
        for d, neighbor_local_idx in zip(dist[1:], idx[1:]):
            dst_idx = class_indices[neighbor_local_idx]
            edge_list.append([src_idx, dst_idx])
            weight = 1.0 / (1.0 + d) if d > 0 else 1.0
            edge_weights.append(weight)

# 添加时间相邻边
time_edges = 0

# 为重采样数据创建临时Batch_ID
train_batch_ids = np.zeros(len(X_train_resampled), dtype=int)
val_batch_ids = batch_ids[val_idx]
test_batch_ids = batch_ids[test_idx]
combined_batch_ids = np.hstack([train_batch_ids, val_batch_ids, test_batch_ids])

# 按Batch_ID排序
sorted_indices_combined = np.argsort(combined_batch_ids)

for i in range(len(sorted_indices_combined) - 1):
    current_idx = sorted_indices_combined[i]
    next_idx = sorted_indices_combined[i+1]
    edge_list.append([current_idx, next_idx])
    edge_list.append([next_idx, current_idx])
    edge_weights.append(0.5)
    edge_weights.append(0.5)
    time_edges += 2

print(f"总边数: {len(edge_list)}")

# 转换为PyG格式
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
x = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y_combined, dtype=torch.long)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y_tensor,
    num_nodes=len(X_combined)
)

# 创建mask
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_mask[:len(X_train_resampled)] = True
val_mask[len(X_train_resampled):len(X_train_resampled)+len(val_idx)] = True
test_mask[len(X_train_resampled)+len(val_idx):] = True

data.train_mask = train_mask.to(device)
data.val_mask = val_mask.to(device)
data.test_mask = test_mask.to(device)
data = data.to(device)

print(f"\n图数据信息:")
print(f"  节点数: {data.num_nodes}")
print(f"  边数: {data.num_edges}")
print(f"  特征维度: {data.num_features}")
print(f"  类别数: {data.y.max().item() + 1}")

# 6. 定义Focal Loss
print("\n[6/6] 定义模型和训练...")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            alpha = self.alpha[targets]
            F_loss = alpha * F_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# 定义GAT模型
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        
        # 第一层GAT
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # 第二层GAT（输出层）
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

# 初始化模型
model = GAT(
    in_channels=data.num_features,
    hidden_channels=64,
    out_channels=3,
    heads=8,
    dropout=0.5
).to(device)

print(f"模型结构:")
print(f"  输入维度: {data.num_features}")
print(f"  隐藏层维度: 64")
print(f"  注意力头数: 8")
print(f"  输出维度: 3")

# 类别权重
class_counts = Counter(y_train_resampled)
total = sum(class_counts.values())
class_weights = torch.tensor([
    0.1,  # Class 0
    0.3,  # Class 1
    0.6   # Class 2
], dtype=torch.float).to(device)

print(f"类别权重: {class_weights.cpu().numpy()}")

# 使用Focal Loss
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_attr)
    pred = out.argmax(dim=1)
    
    correct = (pred[mask] == data.y[mask]).sum().item()
    total = mask.sum().item()
    acc = correct / total
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    f1_macro = f1_score(y_true, y_pred, average='macro')
    class_recall = []
    for i in range(3):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_recall.append((i, (y_pred[class_mask] == i).sum() / class_mask.sum()))
        else:
            class_recall.append((i, 0.0))
    
    return acc, f1_macro, y_pred, y_true, class_recall

# 训练循环
best_val_f1 = 0
best_model_state = None
patience = 20
patience_counter = 0

print("\n开始训练...")
print("-" * 60)

for epoch in range(1, 201):
    loss = train()
    train_acc, train_f1, _, _, _ = evaluate(data.train_mask)
    val_acc, val_f1, val_pred, val_true, val_recall = evaluate(data.val_mask)
    
    # 学习率调度
    scheduler.step(val_f1)
    
    # 保存最佳模型
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        print(f"  Class 2 Recall: {val_recall[2][1]:.4f}")
    
    # 早停
    if patience_counter >= patience:
        print(f"\n早停于 Epoch {epoch}")
        break

print("-" * 60)

# 加载最佳模型
model.load_state_dict(best_model_state)

# 评估模型
print("\n" + "=" * 60)
print("模型评估")
print("=" * 60)

test_acc, test_f1, test_pred, test_true, test_recall = evaluate(data.test_mask)
print(f"\n测试集准确率: {test_acc:.4f}")
print(f"测试集Macro F1: {test_f1:.4f}")

# 详细分类报告
print("\n分类报告:")
print(classification_report(test_true, test_pred, target_names=['Class 0', 'Class 1', 'Class 2']))

# 混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(test_true, test_pred)
print(cm)

# 计算每个类别的召回率
print("\n各类别召回率:")
for class_id, recall in test_recall:
    print(f"  Class {class_id}: {recall:.4f}")

# 保存模型
print("\n保存模型...")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_f1': best_val_f1,
    'test_f1': test_f1,
    'class_weights': class_weights.cpu()
}, 'gat_knn_balanced_model.pt')
print("[OK] 模型已保存: gat_knn_balanced_model.pt")

print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)
