"""消融实验：测试各组件对模型性能的贡献
组件：
1. 重采样策略（SMOTE上采样 + RandomUnderSampler下采样）
2. Focal Loss
3. 类别感知K值（图结构增强）
4. 时间相邻边
5. GAT层数
6. Dropout率
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

print("=" * 60)
print("消融实验：测试各组件的贡献")
print("=" * 60)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 1. 加载数据
print("\n[1/7] 加载数据...")
df = pd.read_csv('processed_raw_data.csv')
y_train = pd.read_csv('raw/y_train.csv')
df = df.reset_index(drop=True)
df['label'] = y_train['label'].values

continuous_features = [col for col in df.columns if any(x in col for x in ['_mean', '_std', '_max', '_min', '_median'])]
categorical_features = ['Mold_ID_encoded', 'Phase_mode', 'MouldFlow1_mode', 'MouldFlow2_mode', 'MouldFlow3_mode']
categorical_features = [f for f in categorical_features if f in df.columns]
time_features = ['SampleTime_Range']
feature_cols = continuous_features + categorical_features + time_features

X = df[feature_cols].values
y = df['label'].values
batch_ids = df['Batch_ID'].values

# 2. 划分数据集
print("\n[2/7] 划分数据集...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_val_idx = next(sss.split(X, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss2.split(X[test_val_idx], y[test_val_idx]))
val_idx = test_val_idx[val_idx]
test_idx = test_val_idx[test_idx]

X_train_full = X[train_idx]
y_train_full = y[train_idx]

print(f"训练集: {len(train_idx)} 样本, 分布: {dict(Counter(y_train_full))}")
print(f"验证集: {len(val_idx)} 样本")
print(f"测试集: {len(test_idx)} 样本")

# Focal Loss定义
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

# GAT模型定义
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5, num_layers=2):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True))
        self.convs.append(GATConv(hidden_channels * heads if num_layers > 2 else hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        return x

def build_knn_graph(X_scaled, y_combined, class_K, use_time_edges=True, train_batch_ids=None, batch_ids=None, val_idx=None, test_idx=None):
    edge_list = []
    edge_weights = []

    for class_id in [0, 1, 2]:
        class_mask = y_combined == class_id
        class_indices = np.where(class_mask)[0]
        if len(class_indices) == 0:
            continue
        k = class_K[class_id]
        class_features = X_scaled[class_mask]
        knn = NearestNeighbors(n_neighbors=min(k + 1, len(class_indices)), metric='euclidean')
        knn.fit(class_features)
        distances, indices = knn.kneighbors(class_features)
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            src_idx = class_indices[i]
            for d, neighbor_local_idx in zip(dist[1:], idx[1:]):
                dst_idx = class_indices[neighbor_local_idx]
                edge_list.append([src_idx, dst_idx])
                weight = 1.0 / (1.0 + d) if d > 0 else 1.0
                edge_weights.append(weight)

    if use_time_edges:
        time_edges = 0
        combined_batch_ids = np.hstack([train_batch_ids, batch_ids[val_idx], batch_ids[test_idx]])
        sorted_indices_combined = np.argsort(combined_batch_ids)
        for i in range(len(sorted_indices_combined) - 1):
            current_idx = sorted_indices_combined[i]
            next_idx = sorted_indices_combined[i + 1]
            edge_list.append([current_idx, next_idx])
            edge_list.append([next_idx, current_idx])
            edge_weights.append(0.5)
            edge_weights.append(0.5)
            time_edges += 2

    return edge_list, edge_weights

def prepare_data(X_train_resampled, y_train_resampled, X, y, batch_ids, train_idx, val_idx, test_idx, class_K, use_time_edges=True):
    X_combined = np.vstack([X_train_resampled, X[val_idx], X[test_idx]])
    y_combined = np.hstack([y_train_resampled, y[val_idx], y[test_idx]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    train_batch_ids = np.zeros(len(X_train_resampled), dtype=int)
    edge_list, edge_weights = build_knn_graph(X_scaled, y_combined, class_K, use_time_edges, train_batch_ids, batch_ids, val_idx, test_idx)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    x = torch.tensor(X_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y_combined, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor, num_nodes=len(X_combined))

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[:len(X_train_resampled)] = True
    val_mask[len(X_train_resampled):len(X_train_resampled) + len(val_idx)] = True
    test_mask[len(X_train_resampled) + len(val_idx):] = True

    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)
    data = data.to(device)
    return data

def train_and_evaluate(data, class_weights, use_focal_loss=True, hidden_channels=64, heads=8, dropout=0.5, num_layers=2, epochs=100):
    model = GAT(in_channels=data.num_features, hidden_channels=hidden_channels, out_channels=3, heads=heads, dropout=dropout, num_layers=num_layers).to(device)

    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

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

    best_val_f1 = 0
    best_model_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        loss = train()
        train_acc, train_f1, _, _, _ = evaluate(data.train_mask)
        val_acc, val_f1, val_pred, val_true, val_recall = evaluate(data.val_mask)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    test_acc, test_f1, test_pred, test_true, test_recall = evaluate(data.test_mask)

    return {
        'test_acc': test_acc,
        'test_f1': test_f1,
        'class_recall': dict(test_recall),
        'cm': confusion_matrix(test_true, test_pred)
    }

# ============================================================
# 消融实验
# ============================================================
results = {}

# 准备基础训练数据
print("\n[3/7] 准备基础训练数据...")

# Resampling策略：下采样 + 上采样
undersampler = RandomUnderSampler(sampling_strategy={0: 2000}, random_state=42)
X_train_temp, y_train_temp = undersampler.fit_resample(X_train_full, y_train_full)
undersampler2 = RandomUnderSampler(sampling_strategy={1: 2000}, random_state=42)
X_train_temp2, y_train_temp2 = undersampler2.fit_resample(X_train_temp, y_train_temp)
if dict(Counter(y_train_temp2))[2] < 2000:
    smote = SMOTE(sampling_strategy={2: 2000}, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_temp2, y_train_temp2)
else:
    undersampler3 = RandomUnderSampler(sampling_strategy={2: 2000}, random_state=42)
    X_train_resampled, y_train_resampled = undersampler3.fit_resample(X_train_temp2, y_train_temp2)

print(f"平衡后训练集: {dict(Counter(y_train_resampled))}")

# 类别权重
class_weights_balanced = torch.tensor([0.33, 0.33, 0.34], dtype=torch.float).to(device)

# Experiment 1: 完整模型（基线）
print("\n" + "=" * 60)
print("Experiment 1: 完整模型（基线）")
print("  - 重采样: 是")
print("  - Focal Loss: 是")
print("  - 类别感知K: 是")
print("  - 时间边: 是")
print("  - GAT层数: 2")
print("  - Dropout: 0.5")
print("=" * 60)

data_full = prepare_data(X_train_resampled, y_train_resampled, X, y, batch_ids, train_idx, val_idx, test_idx, {0: 5, 1: 8, 2: 10}, True)
results['full_model'] = train_and_evaluate(data_full, class_weights_balanced, True, 64, 8, 0.5, 2)
print(f"\n结果: Acc={results['full_model']['test_acc']:.4f}, F1={results['full_model']['test_f1']:.4f}")
print(f"Class Recall: {results['full_model']['class_recall']}")

# Experiment 2: 无重采样
print("\n" + "=" * 60)
print("Experiment 2: 无重采样")
print("=" * 60)

data_no_resample = prepare_data(X_train_full, y_train_full, X, y, batch_ids, train_idx, val_idx, test_idx, {0: 5, 1: 8, 2: 10}, True)
results['no_resample'] = train_and_evaluate(data_no_resample, class_weights_balanced, True, 64, 8, 0.5, 2)
print(f"\n结果: Acc={results['no_resample']['test_acc']:.4f}, F1={results['no_resample']['test_f1']:.4f}")
print(f"Class Recall: {results['no_resample']['class_recall']}")

# Experiment 3: 无Focal Loss
print("\n" + "=" * 60)
print("Experiment 3: 无Focal Loss (使用标准CrossEntropy)")
print("=" * 60)

results['no_focal'] = train_and_evaluate(data_full, class_weights_balanced, False, 64, 8, 0.5, 2)
print(f"\n结果: Acc={results['no_focal']['test_acc']:.4f}, F1={results['no_focal']['test_f1']:.4f}")
print(f"Class Recall: {results['no_focal']['class_recall']}")

# Experiment 4: 均匀K值（无类别感知K）
print("\n" + "=" * 60)
print("Experiment 4: 均匀K值（K=5，无类别感知）")
print("=" * 60)

data_uniform_k = prepare_data(X_train_resampled, y_train_resampled, X, y, batch_ids, train_idx, val_idx, test_idx, {0: 5, 1: 5, 2: 5}, True)
results['uniform_k'] = train_and_evaluate(data_uniform_k, class_weights_balanced, True, 64, 8, 0.5, 2)
print(f"\n结果: Acc={results['uniform_k']['test_acc']:.4f}, F1={results['uniform_k']['test_f1']:.4f}")
print(f"Class Recall: {results['uniform_k']['class_recall']}")

# Experiment 5: 无时间边
print("\n" + "=" * 60)
print("Experiment 5: 无时间相邻边")
print("=" * 60)

data_no_time = prepare_data(X_train_resampled, y_train_resampled, X, y, batch_ids, train_idx, val_idx, test_idx, {0: 5, 1: 8, 2: 10}, False)
results['no_time_edges'] = train_and_evaluate(data_no_time, class_weights_balanced, True, 64, 8, 0.5, 2)
print(f"\n结果: Acc={results['no_time_edges']['test_acc']:.4f}, F1={results['no_time_edges']['test_f1']:.4f}")
print(f"Class Recall: {results['no_time_edges']['class_recall']}")

# Experiment 6: 1层GAT
print("\n" + "=" * 60)
print("Experiment 6: 1层GAT（vs 2层）")
print("=" * 60)

results['gat_1layer'] = train_and_evaluate(data_full, class_weights_balanced, True, 64, 8, 0.5, 1)
print(f"\n结果: Acc={results['gat_1layer']['test_acc']:.4f}, F1={results['gat_1layer']['test_f1']:.4f}")
print(f"Class Recall: {results['gat_1layer']['class_recall']}")

# Experiment 7: 3层GAT
print("\n" + "=" * 60)
print("Experiment 7: 3层GAT（vs 2层）")
print("=" * 60)

results['gat_3layer'] = train_and_evaluate(data_full, class_weights_balanced, True, 64, 8, 0.5, 3)
print(f"\n结果: Acc={results['gat_3layer']['test_acc']:.4f}, F1={results['gat_3layer']['test_f1']:.4f}")
print(f"Class Recall: {results['gat_3layer']['class_recall']}")

# Experiment 8: 不同Dropout
print("\n" + "=" * 60)
print("Experiment 8: Dropout=0.3（vs 0.5）")
print("=" * 60)

results['dropout_03'] = train_and_evaluate(data_full, class_weights_balanced, True, 64, 8, 0.3, 2)
print(f"\n结果: Acc={results['dropout_03']['test_acc']:.4f}, F1={results['dropout_03']['test_f1']:.4f}")
print(f"Class Recall: {results['dropout_03']['class_recall']}")

print("\n" + "=" * 60)
print("Experiment 9: Dropout=0.7")
print("=" * 60)

results['dropout_07'] = train_and_evaluate(data_full, class_weights_balanced, True, 64, 8, 0.7, 2)
print(f"\n结果: Acc={results['dropout_07']['test_acc']:.4f}, F1={results['dropout_07']['test_f1']:.4f}")
print(f"Class Recall: {results['dropout_07']['class_recall']}")

# ============================================================
# 汇总结果
# ============================================================
print("\n" + "=" * 80)
print("消融实验结果汇总")
print("=" * 80)
print(f"\n{'实验名称':<25} {'测试Acc':<10} {'测试F1':<10} {'Class 0':<10} {'Class 1':<10} {'Class 2':<10}")
print("-" * 80)

baseline_f1 = results['full_model']['test_f1']
for name, res in results.items():
    cr = res['class_recall']
    delta_f1 = res['test_f1'] - baseline_f1 if name != 'full_model' else 0
    print(f"{name:<25} {res['test_acc']:.4f}     {res['test_f1']:.4f}     {cr[0]:.4f}     {cr[1]:.4f}     {cr[2]:.4f}" + (f"     ({delta_f1:+.4f})" if name != 'full_model' else ""))

print("\n" + "=" * 80)
print("结论分析")
print("=" * 80)

print(f"\n基线模型 (完整) F1: {baseline_f1:.4f}")

print("\n各组件贡献:")
print(f"  1. 重采样: F1变化 {results['no_resample']['test_f1'] - baseline_f1:+.4f}")
print(f"  2. Focal Loss: F1变化 {results['no_focal']['test_f1'] - baseline_f1:+.4f}")
print(f"  3. 类别感知K: F1变化 {results['uniform_k']['test_f1'] - baseline_f1:+.4f}")
print(f"  4. 时间边: F1变化 {results['no_time_edges']['test_f1'] - baseline_f1:+.4f}")
print(f"  5. GAT层数 (1层): F1变化 {results['gat_1layer']['test_f1'] - baseline_f1:+.4f}")
print(f"  6. GAT层数 (3层): F1变化 {results['gat_3layer']['test_f1'] - baseline_f1:+.4f}")
print(f"  7. Dropout 0.3: F1变化 {results['dropout_03']['test_f1'] - baseline_f1:+.4f}")
print(f"  8. Dropout 0.7: F1变化 {results['dropout_07']['test_f1'] - baseline_f1:+.4f}")

# 保存结果
import pickle
with open('ablation_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\n[OK] 结果已保存: ablation_results.pkl")

print("\n" + "=" * 60)
print("消融实验完成！")
print("=" * 60)
