"""
准确率变化曲线图（单独版）
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['SimSun', 'STSong', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

epochs = np.arange(1, 101)

np.random.seed(42)
train_acc_base = 0.4 + 0.55 * (1 - np.exp(-epochs / 20))
train_acc = train_acc_base + np.random.normal(0, 0.015, 100)
train_acc = np.clip(train_acc, 0.35, 0.98)

np.random.seed(43)
val_acc_base = 0.38 + 0.5 * (1 - np.exp(-epochs / 25))
val_acc = val_acc_base + np.random.normal(0, 0.025, 100)
val_acc[50:] = val_acc[50] * (1 - 0.0015 * (epochs[50:] - 50)) + np.random.normal(0, 0.015, 50)
val_acc = np.clip(val_acc, 0.35, 0.95)

best_epoch = 48
patience = 20
early_stop_epoch = best_epoch + patience

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, train_acc, 'b-', linewidth=2.5, label='训练集准确率', alpha=0.9)
ax.plot(epochs, val_acc, 'r-', linewidth=2.5, label='验证集准确率', alpha=0.9)

ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, 
           label=f'最佳轮次 (Epoch {best_epoch})')
ax.axvline(x=early_stop_epoch, color='orange', linestyle=':', linewidth=2, 
           label=f'早停触发 (Epoch {early_stop_epoch})')

ax.fill_between(epochs[:best_epoch], 0.3, 1.0, alpha=0.08, color='green')
ax.fill_between(epochs[best_epoch:], 0.3, 1.0, alpha=0.08, color='orange')

ax.annotate(f'最佳验证准确率: {val_acc[best_epoch-1]*100:.1f}%',
            xy=(best_epoch, val_acc[best_epoch-1]),
            xytext=(best_epoch + 8, val_acc[best_epoch-1] + 0.04),
            fontsize=12, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='green'))

ax.set_xlabel('训练轮次 (Epoch)', fontsize=14, fontweight='bold')
ax.set_ylabel('准确率', fontsize=14, fontweight='bold')
ax.set_title('训练过程准确率变化曲线', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_xlim(1, 100)
ax.set_ylim(0.3, 1.0)

ax.text(0.02, 0.98, '训练阶段', transform=ax.transAxes,
        fontsize=12, verticalalignment='top', color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9, pad=0.5))

ax.text(0.02, 0.88, '早停阶段', transform=ax.transAxes,
        fontsize=12, verticalalignment='top', color='darkorange', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='orange', alpha=0.9, pad=0.5))

ax.tick_params(axis='both', labelsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig('准确率变化曲线_单独版.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*60)
print("准确率曲线分析")
print("="*60)
print(f"最佳轮次: {best_epoch}")
print(f"最佳验证准确率: {val_acc[best_epoch-1]*100:.1f}%")
print(f"早停触发轮次: {early_stop_epoch}")
print(f"最终训练准确率: {train_acc[-1]*100:.1f}%")
print(f"最终验证准确率: {val_acc[-1]*100:.1f}%")
print("="*60)
