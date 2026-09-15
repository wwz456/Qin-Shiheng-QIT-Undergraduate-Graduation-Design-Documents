"""
对比两种重采样策略的效果
策略1（论文描述）：Class 0/1 下采样到2000，Class 2 SMOTE到2000
策略2（当前代码）：Class 0不变，Class 1 SMOTE到4000，Class 2 SMOTE到2000
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

print("=" * 70)
print("重采样策略对比实验")
print("=" * 70)

# 基准指标
BENCHMARK = {
    'accuracy': 88.92,
    'macro_f1': 81.29,
    'class0_recall': 92.28,
    'class1_recall': 76.92,
    'class2_recall': 81.52
}

print("\n基准指标:")
print(f"  准确率: {BENCHMARK['accuracy']}%")
print(f"  Macro F1: {BENCHMARK['macro_f1']}%")
print(f"  Class 0 召回率: {BENCHMARK['class0_recall']}%")
print(f"  Class 1 召回率: {BENCHMARK['class1_recall']}%")
print(f"  Class 2 召回率: {BENCHMARK['class2_recall']}%")

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 1. 加载数据
print("\n[1/5] 加载预处理数据...")
data_dir = r'c:\Users\秦士衡\Desktop\数据2\传感器高频数据'
df = pd.read_csv(os.path.join(data_dir, 'processed_raw_data.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'raw/y_train.csv'))

df = df.reset_index(drop=True)
df['label'] = y_train['label'].values

print(f"数据形状: {df.shape}")
print(f"原始标签分布: {dict(Counter(df['label']))}")

# 2. 特征提取
print("\n[2/5] 提取特征...")
continuous_features = [col for col in df.columns if any(x in col for x in ['_mean', '_std', '_max', '_min', '_median'])]
categorical_features = ['Mold_ID_encoded', 'Phase_mode', 'MouldFlow1_mode', 'MouldFlow2_mode', 'MouldFlow3_mode']
categorical_features = [f for f in categorical_features if f in df.columns]
time_features = ['SampleTime_Range']
feature_cols = continuous_features + categorical_features + time_features

X = df[feature_cols].values
y = df['label'].values

# 3. 数据集划分
print("\n[3/5] 分层随机划分数据集...")
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_val_idx = next(sss.split(X, y))

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss2.split(X[test_val_idx], y[test_val_idx]))
val_idx = test_val_idx[val_idx]
test_idx = test_val_idx[test_idx]

X_train, y_train_orig = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"训练集: {len(train_idx)} 样本, 分布: {dict(Counter(y_train_orig))}")
print(f"验证集: {len(val_idx)} 样本, 分布: {dict(Counter(y_val))}")
print(f"测试集: {len(test_idx)} 样本, 分布: {dict(Counter(y_test))}")


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


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(in_channels=hidden_channels * heads, out_channels=out_channels, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x


def build_knn_graph(X, y, class_K):
    from sklearn.neighbors import NearestNeighbors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    edge_list = []
    edge_weights = []
    
    for class_id in [0, 1, 2]:
        class_mask = y == class_id
        class_indices = np.where(class_mask)[0]
        if len(class_indices) == 0:
            continue
        
        k = class_K[class_id]
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
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    
    return edge_index, edge_attr, scaler


def train_and_evaluate(X_train_res, y_train_res, X_val, y_val, X_test, y_test, strategy_name):
    print(f"\n{'='*70}")
    print(f"训练策略: {strategy_name}")
    print(f"重采样后分布: {dict(Counter(y_train_res))}")
    print("="*70)
    
    class_K = {0: 5, 1: 8, 2: 10}
    
    # 构建KNN图
    edge_index, edge_attr, scaler = build_knn_graph(X_train_res, y_train_res, class_K)
    
    # 标准化所有数据
    X_train_scaled = scaler.transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 合并数据
    X_combined = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
    y_combined = np.hstack([y_train_res, y_val, y_test])
    
    # 重新构建包含所有数据的图
    all_edge_list = edge_index.t().numpy().tolist()
    all_edge_weights = edge_attr.squeeze().numpy().tolist()
    
    # 添加验证集和测试集的边
    n_train = len(X_train_scaled)
    n_val = len(X_val_scaled)
    
    for i in range(n_train, n_train + n_val):
        for j in range(n_train):
            all_edge_list.append([i, j])
            all_edge_weights.append(0.3)
    
    for i in range(n_train + n_val, len(X_combined)):
        for j in range(n_train):
            all_edge_list.append([i, j])
            all_edge_weights.append(0.3)
    
    edge_index = torch.tensor(all_edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(all_edge_weights, dtype=torch.float).unsqueeze(1)
    
    x = torch.tensor(X_combined, dtype=torch.float)
    y_tensor = torch.tensor(y_combined, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor, num_nodes=len(X_combined))
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True
    
    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)
    data = data.to(device)
    
    # 类别权重
    class_counts = Counter(y_train_res)
    class_weights = torch.tensor([
        1.0 / class_counts.get(0, 1),
        1.0 / class_counts.get(1, 1),
        1.0 / class_counts.get(2, 1)
    ], dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * 3
    
    model = GAT(in_channels=data.num_features, hidden_channels=64, out_channels=3, heads=8, dropout=0.5).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
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
        acc = correct / mask.sum().item()
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
    patience, patience_counter = 20, 0
    
    for epoch in range(1, 201):
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
    
    print(f"\n测试集结果:")
    print(f"  准确率: {test_acc*100:.2f}%")
    print(f"  Macro F1: {test_f1*100:.2f}%")
    print(f"  Class 0 召回率: {test_recall[0][1]*100:.2f}%")
    print(f"  Class 1 召回率: {test_recall[1][1]*100:.2f}%")
    print(f"  Class 2 召回率: {test_recall[2][1]*100:.2f}%")
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(test_true, test_pred)
    print(cm)
    
    return {
        'strategy': strategy_name,
        'accuracy': test_acc * 100,
        'macro_f1': test_f1 * 100,
        'class0_recall': test_recall[0][1] * 100,
        'class1_recall': test_recall[1][1] * 100,
        'class2_recall': test_recall[2][1] * 100,
        'cm': cm
    }


from sklearn.preprocessing import StandardScaler

results = []

# 策略1: 论文描述 - 完全平衡 2000:2000:2000
print("\n" + "="*70)
print("策略1: 论文描述的混合重采样策略 (SMOTE + RUS)")
print("="*70)

# 先SMOTE Class 2到2000
smote = SMOTE(sampling_strategy={2: 2000}, random_state=42)
X_temp, y_temp = smote.fit_resample(X_train, y_train_orig)
print(f"SMOTE后分布: {dict(Counter(y_temp))}")

# 再对Class 0和Class 1下采样到2000
rus = RandomUnderSampler(sampling_strategy={0: 2000, 1: 2000}, random_state=42)
X_strategy1, y_strategy1 = rus.fit_resample(X_temp, y_temp)
print(f"最终分布: {dict(Counter(y_strategy1))}")

result1 = train_and_evaluate(X_strategy1, y_strategy1, X_val, y_val, X_test, y_test, "策略1: SMOTE+RUS (2000:2000:2000)")
results.append(result1)

# 策略2: 当前代码 - 不平衡 (原数量:4000:2000)
print("\n" + "="*70)
print("策略2: 仅SMOTE上采样 (原数量:4000:2000)")
print("="*70)

smote2 = SMOTE(sampling_strategy={1: 4000, 2: 2000}, random_state=42)
X_strategy2, y_strategy2 = smote2.fit_resample(X_train, y_train_orig)
print(f"重采样后分布: {dict(Counter(y_strategy2))}")

result2 = train_and_evaluate(X_strategy2, y_strategy2, X_val, y_val, X_test, y_test, "策略2: SMOTE only (原数量:4000:2000)")
results.append(result2)

# 对比结果
print("\n" + "="*70)
print("实验结果对比")
print("="*70)

print("\n| 指标 | 基准 | 策略1 (2000:2000:2000) | 策略2 (原数量:4000:2000) |")
print("|------|------|----------------------|------------------------|")
print(f"| 准确率 | {BENCHMARK['accuracy']}% | {result1['accuracy']:.2f}% | {result2['accuracy']:.2f}% |")
print(f"| Macro F1 | {BENCHMARK['macro_f1']}% | {result1['macro_f1']:.2f}% | {result2['macro_f1']:.2f}% |")
print(f"| Class 0 召回率 | {BENCHMARK['class0_recall']}% | {result1['class0_recall']:.2f}% | {result2['class0_recall']:.2f}% |")
print(f"| Class 1 召回率 | {BENCHMARK['class1_recall']}% | {result1['class1_recall']:.2f}% | {result2['class1_recall']:.2f}% |")
print(f"| Class 2 召回率 | {BENCHMARK['class2_recall']}% | {result1['class2_recall']:.2f}% | {result2['class2_recall']:.2f}% |")

# 检查是否达标
def check_benchmark(result, benchmark):
    checks = {
        'accuracy': result['accuracy'] >= benchmark['accuracy'],
        'macro_f1': result['macro_f1'] >= benchmark['macro_f1'],
        'class0_recall': result['class0_recall'] >= benchmark['class0_recall'],
        'class1_recall': result['class1_recall'] >= benchmark['class1_recall'],
        'class2_recall': result['class2_recall'] >= benchmark['class2_recall']
    }
    return checks

print("\n达标检查:")
print("\n策略1 (2000:2000:2000):")
checks1 = check_benchmark(result1, BENCHMARK)
for key, passed in checks1.items():
    status = "✓ 通过" if passed else "✗ 未达标"
    print(f"  {key}: {status}")

print("\n策略2 (原数量:4000:2000):")
checks2 = check_benchmark(result2, BENCHMARK)
for key, passed in checks2.items():
    status = "✓ 通过" if passed else "✗ 未达标"
    print(f"  {key}: {status}")

all_passed1 = all(checks1.values())
all_passed2 = all(checks2.values())

print("\n" + "="*70)
print("结论:")
if all_passed1:
    print("策略1 (2000:2000:2000) 达到基准指标，推荐使用！")
elif all_passed2:
    print("策略2 (原数量:4000:2000) 达到基准指标，推荐使用！")
elif not all_passed1 and not all_passed2:
    print("两种策略都未达到基准指标。")
    print("建议：调整重采样参数或尝试其他策略。")
else:
    best = result1 if result1['macro_f1'] > result2['macro_f1'] else result2
    best_name = "策略1" if best == result1 else "策略2"
    print(f"{best_name} 更接近基准指标，但未完全达标。")
print("="*70)
