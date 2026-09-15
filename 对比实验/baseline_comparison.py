"""对比实验：基础模型 vs GAT模型
使用相同的数据集和训练条件，对比基础模型的性能
模型：随机森林、XGBoost、LightGBM
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

print("=" * 70)
print("对比实验：基础模型 vs GAT模型")
print("=" * 70)

# 1. 加载数据
print("\n[1/5] 加载预处理数据...")
df = pd.read_csv('processed_raw_data.csv')
y_train = pd.read_csv('raw/y_train.csv')
df = df.reset_index(drop=True)
df['label'] = y_train['label'].values

# 2. 提取特征
print("\n[2/5] 提取特征...")
continuous_features = [col for col in df.columns if any(x in col for x in ['_mean', '_std', '_max', '_min', '_median'])]
categorical_features = ['Mold_ID_encoded', 'Phase_mode', 'MouldFlow1_mode', 'MouldFlow2_mode', 'MouldFlow3_mode']
categorical_features = [f for f in categorical_features if f in df.columns]
time_features = ['SampleTime_Range']
feature_cols = continuous_features + categorical_features + time_features

X = df[feature_cols].values
y = df['label'].values

print(f"总特征数: {len(feature_cols)}")
print(f"数据形状: {X.shape}")
print(f"标签分布: {dict(Counter(y))}")

# 3. 分层随机划分数据集
print("\n[3/5] 分层随机划分数据集...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_val_idx = next(sss.split(X, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss2.split(X[test_val_idx], y[test_val_idx]))
val_idx = test_val_idx[val_idx]
test_idx = test_val_idx[test_idx]

X_train = X[train_idx]
y_train = y[train_idx]
X_val = X[val_idx]
y_val = y[val_idx]
X_test = X[test_idx]
y_test = y[test_idx]

print(f"原始训练集: {len(train_idx)} 样本")
print(f"  标签分布: {dict(Counter(y_train))}")
print(f"验证集: {len(val_idx)} 样本")
print(f"  标签分布: {dict(Counter(y_val))}")
print(f"测试集: {len(test_idx)} 样本")
print(f"  标签分布: {dict(Counter(y_test))}")

# 4. 重采样训练集（与GAT模型相同的平衡策略）
print("\n[4/5] 重采样训练集（构建完全平衡训练集）...")

original_train_dist = dict(Counter(y_train))
print(f"原始训练集标签分布: {original_train_dist}")

# Step 1: 下采样多数类（Class 0）到 2000
undersampler = RandomUnderSampler(sampling_strategy={0: 2000}, random_state=42)
X_train_temp, y_train_temp = undersampler.fit_resample(X_train, y_train)
print(f"下采样后 (0->2000): {dict(Counter(y_train_temp))}")

# Step 2: 再下采样中等类（Class 1）到 2000
undersampler2 = RandomUnderSampler(sampling_strategy={1: 2000}, random_state=42)
X_train_temp2, y_train_temp2 = undersampler2.fit_resample(X_train_temp, y_train_temp)
print(f"下采样后 (1->2000): {dict(Counter(y_train_temp2))}")

# Step 3: 对少数类（Class 2）使用 SMOTE 扩充到 2000
if original_train_dist[2] < 2000:
    smote = SMOTE(sampling_strategy={2: 2000}, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_temp2, y_train_temp2)
else:
    undersampler3 = RandomUnderSampler(sampling_strategy={2: 2000}, random_state=42)
    X_train_resampled, y_train_resampled = undersampler3.fit_resample(X_train_temp2, y_train_temp2)

print(f"最终平衡训练集标签分布: {dict(Counter(y_train_resampled))}")

# 5. 训练和评估基础模型
print("\n[5/5] 训练和评估基础模型...")

from sklearn.neighbors import KNeighborsClassifier

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

results = {}

for model_name, model in models.items():
    print(f"\n" + "=" * 60)
    print(f"训练 {model_name}...")
    print("=" * 60)
    
    # 训练模型
    model.fit(X_train_resampled, y_train_resampled)
    
    # 评估模型
    y_pred = model.predict(X_test)
    
    # 计算指标
    acc = (y_pred == y_test).sum() / len(y_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n测试集准确率: {acc:.4f}")
    print(f"测试集Macro F1: {f1_macro:.4f}")
    
    # 详细分类报告
    print("\n分类报告:")
    cr = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1', 'Class 2'])
    print(cr)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # 各类别召回率
    print("\n各类别召回率:")
    class_recall = []
    for i in range(3):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            recall = (y_pred[class_mask] == i).sum() / class_mask.sum()
            class_recall.append((i, recall))
            print(f"  Class {i}: {recall:.4f}")
    
    results[model_name] = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'classification_report': cr,
        'confusion_matrix': cm,
        'class_recall': class_recall
    }

# 6. 汇总结果
print("\n" + "=" * 80)
print("对比实验结果汇总")
print("=" * 80)
print(f"\n{'模型':<15} {'准确率':<10} {'Macro F1':<10} {'Class 0':<10} {'Class 1':<10} {'Class 2':<10}")
print("-" * 80)

# 添加GAT模型的结果作为参考（基于用户提供的结果）
gat_results = {
    'accuracy': 0.8892,
    'f1_macro': 0.8129,
    'class_recall': [(0, 0.9228), (1, 0.7692), (2, 0.8152)]
}

results['GAT (参考)'] = gat_results

for model_name, res in results.items():
    cr = res['class_recall']
    print(f"{model_name:<15} {res['accuracy']:.4f}     {res['f1_macro']:.4f}     {cr[0][1]:.4f}     {cr[1][1]:.4f}     {cr[2][1]:.4f}")

print("\n" + "=" * 80)
print("对比实验完成！")
print("=" * 80)
