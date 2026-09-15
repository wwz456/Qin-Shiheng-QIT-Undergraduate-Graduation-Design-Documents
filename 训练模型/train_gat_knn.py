"""
第三步：使用KNN图训练GAT模型
功能：
1. 加载KNN图数据
2. 实现类别平衡的数据划分
3. 训练GAT模型
4. 评估模型性能（特别关注类别1和2）
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import pickle

print("=" * 60)
print("第三步：GAT模型训练（基于KNN图）")
print("=" * 60)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 1. 加载图数据
print("\n[1/5] 加载KNN图数据...")
data = torch.load('knn_graph_data.pt', map_location=device, weights_only=False)
print(f"图数据信息:")
print(f"  节点数: {data.num_nodes}")
print(f"  边数: {data.num_edges}")
print(f"  特征维度: {data.num_features}")
print(f"  类别数: {data.y.max().item() + 1}")

# 将数据移到设备
data = data.to(device)

# 2. 类别平衡的数据划分
print("\n[2/5] 类别平衡的数据划分...")

# 获取标签分布
labels = data.y.cpu().numpy()
print(f"原始标签分布: {dict(Counter(labels))}")

# 为每个类别创建平衡的训练/验证/测试集
def create_balanced_split(labels, train_ratio=0.7, val_ratio=0.15):
    """创建类别平衡的数据划分"""
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_id in np.unique(labels):
        class_indices = indices[labels == class_id]
        n_class = len(class_indices)
        
        # 随机打乱
        np.random.shuffle(class_indices)
        
        # 计算划分点
        n_train = int(n_class * train_ratio)
        n_val = int(n_class * val_ratio)
        
        # 划分
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train+n_val])
        test_indices.extend(class_indices[n_train+n_val:])
    
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)

train_idx, val_idx, test_idx = create_balanced_split(labels)

print(f"训练集: {len(train_idx)} 样本")
print(f"  标签分布: {dict(Counter(labels[train_idx]))}")
print(f"验证集: {len(val_idx)} 样本")
print(f"  标签分布: {dict(Counter(labels[val_idx]))}")
print(f"测试集: {len(test_idx)} 样本")
print(f"  标签分布: {dict(Counter(labels[test_idx]))}")

# 创建mask
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask.to(device)
data.val_mask = val_mask.to(device)
data.test_mask = test_mask.to(device)

# 3. 定义GAT模型
print("\n[3/5] 定义GAT模型...")

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.3):
        super(GAT, self).__init__()
        
        # 第一层GAT
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # 第二层GAT
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # 第三层GAT（输出层）
        self.conv3 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_attr=None):
        # 第一层
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第三层
        x = self.conv3(x, edge_index, edge_attr)
        
        return x

# 初始化模型
model = GAT(
    in_channels=data.num_features,
    hidden_channels=64,
    out_channels=3,  # 3个类别
    heads=8,
    dropout=0.3
).to(device)

print(f"模型结构:")
print(f"  输入维度: {data.num_features}")
print(f"  隐藏层维度: 64")
print(f"  注意力头数: 8")
print(f"  输出维度: 3")

# 4. 定义损失函数和优化器
print("\n[4/5] 定义损失函数和优化器...")

# 类别权重（处理类别不平衡）
class_counts = Counter(labels[train_idx])
total = sum(class_counts.values())
class_weights = torch.tensor([
    total / (3 * class_counts[0]),
    total / (3 * class_counts[1]),
    total / (3 * class_counts[2])
], dtype=torch.float).to(device)

print(f"类别权重: {class_weights.cpu().numpy()}")

# 使用加权交叉熵损失
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

# 5. 训练模型
print("\n[5/5] 训练模型...")

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
    
    return acc, pred[mask].cpu().numpy(), data.y[mask].cpu().numpy()

# 训练循环
best_val_acc = 0
best_model_state = None
patience = 20
patience_counter = 0

print("\n开始训练...")
print("-" * 60)

for epoch in range(1, 201):
    loss = train()
    train_acc, _, _ = evaluate(data.train_mask)
    val_acc, val_pred, val_true = evaluate(data.val_mask)
    
    # 学习率调度
    scheduler.step(val_acc)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # 早停
    if patience_counter >= patience:
        print(f"\n早停于 Epoch {epoch}")
        break

print("-" * 60)

# 加载最佳模型
model.load_state_dict(best_model_state)

# 6. 评估模型
print("\n" + "=" * 60)
print("模型评估")
print("=" * 60)

# 测试集评估
test_acc, test_pred, test_true = evaluate(data.test_mask)
print(f"\n测试集准确率: {test_acc:.4f}")

# 详细分类报告
print("\n分类报告:")
print(classification_report(test_true, test_pred, target_names=['Class 0', 'Class 1', 'Class 2']))

# 混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(test_true, test_pred)
print(cm)

# 计算每个类别的召回率
print("\n各类别召回率:")
for i in range(3):
    class_mask = test_true == i
    if class_mask.sum() > 0:
        class_acc = (test_pred[class_mask] == i).sum() / class_mask.sum()
        print(f"  Class {i}: {class_acc:.4f}")

# 7. 保存模型
print("\n保存模型...")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'class_weights': class_weights.cpu()
}, 'gat_knn_model.pt')
print("[OK] 模型已保存: gat_knn_model.pt")

print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)
