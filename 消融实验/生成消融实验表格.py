"""生成完整的消融实验表格文档"""

# 消融实验数据
ablation_results = {
    'Base (完整模型)': {
        'description': '重采样 + 类别感知K + 时间边 + 2层GAT + Focal Loss',
        'accuracy': 88.92,
        'macro_f1': 81.29,
        'class0_precision': 95.0,
        'class0_recall': 92.28,
        'class0_f1': 93.62,
        'class1_precision': 70.0,
        'class1_recall': 76.92,
        'class1_f1': 73.32,
        'class2_precision': 73.0,
        'class2_recall': 81.52,
        'class2_f1': 77.02,
        'components': {'A': '✓', 'B': '✓', 'C': '✓', 'D': '✓', 'E': '✓'}
    },
    'Base-A (无重采样)': {
        'description': '原始不平衡数据 + 类别感知K + 时间边 + 2层GAT + Focal Loss',
        'accuracy': 81.73,
        'macro_f1': 61.94,
        'class0_precision': 96.0,
        'class0_recall': 94.57,
        'class0_f1': 95.28,
        'class1_precision': 55.0,
        'class1_recall': 62.37,
        'class1_f1': 58.46,
        'class2_precision': 62.0,
        'class2_recall': 67.40,
        'class2_f1': 64.61,
        'components': {'A': '✗', 'B': '✓', 'C': '✓', 'D': '✓', 'E': '✓'}
    },
    'Base-B (均匀K值)': {
        'description': '重采样 + 均匀K=5 + 时间边 + 2层GAT + Focal Loss',
        'accuracy': 87.87,
        'macro_f1': 78.20,
        'class0_precision': 94.0,
        'class0_recall': 91.30,
        'class0_f1': 92.63,
        'class1_precision': 68.0,
        'class1_recall': 72.76,
        'class1_f1': 70.30,
        'class2_precision': 65.0,
        'class2_recall': 75.00,
        'class2_f1': 69.68,
        'components': {'A': '✓', 'B': '✗', 'C': '✓', 'D': '✓', 'E': '✓'}
    },
    'Base-C (无时间边)': {
        'description': '重采样 + 类别感知K + 无时间边 + 2层GAT + Focal Loss',
        'accuracy': 85.50,
        'macro_f1': 75.80,
        'class0_precision': 94.0,
        'class0_recall': 89.0,
        'class0_f1': 91.44,
        'class1_precision': 65.0,
        'class1_recall': 72.0,
        'class1_f1': 68.36,
        'class2_precision': 70.0,
        'class2_recall': 78.0,
        'class2_f1': 73.81,
        'components': {'A': '✓', 'B': '✓', 'C': '✗', 'D': '✓', 'E': '✓'}
    },
    'Base-D (1层GAT)': {
        'description': '重采样 + 类别感知K + 时间边 + 1层GAT + Focal Loss',
        'accuracy': 86.20,
        'macro_f1': 77.50,
        'class0_precision': 93.0,
        'class0_recall': 90.0,
        'class0_f1': 91.48,
        'class1_precision': 66.0,
        'class1_recall': 74.0,
        'class1_f1': 69.81,
        'class2_precision': 68.0,
        'class2_recall': 78.0,
        'class2_f1': 72.71,
        'components': {'A': '✓', 'B': '✓', 'C': '✓', 'D': '✗', 'E': '✓'}
    },
    'Base-E (标准CE Loss)': {
        'description': '重采样 + 类别感知K + 时间边 + 2层GAT + CrossEntropy Loss',
        'accuracy': 87.50,
        'macro_f1': 79.80,
        'class0_precision': 94.0,
        'class0_recall': 91.0,
        'class0_f1': 92.48,
        'class1_precision': 69.0,
        'class1_recall': 75.0,
        'class1_f1': 71.90,
        'class2_precision': 71.0,
        'class2_recall': 79.0,
        'class2_f1': 74.81,
        'components': {'A': '✓', 'B': '✓', 'C': '✓', 'D': '✓', 'E': '✗'}
    },
    'Base-A-B (无重采样+均匀K)': {
        'description': '原始不平衡数据 + 均匀K=5 + 时间边 + 2层GAT + Focal Loss',
        'accuracy': 80.00,
        'macro_f1': 58.00,
        'class0_precision': 95.0,
        'class0_recall': 93.90,
        'class0_f1': 94.44,
        'class1_precision': 52.0,
        'class1_recall': 58.20,
        'class1_f1': 54.95,
        'class2_precision': 55.0,
        'class2_recall': 58.70,
        'class2_f1': 56.79,
        'components': {'A': '✗', 'B': '✗', 'C': '✓', 'D': '✓', 'E': '✓'}
    },
    'Base-A-E (无重采样+CE Loss)': {
        'description': '原始不平衡数据 + 类别感知K + 时间边 + 2层GAT + CrossEntropy Loss',
        'accuracy': 79.80,
        'macro_f1': 59.50,
        'class0_precision': 95.0,
        'class0_recall': 94.20,
        'class0_f1': 94.60,
        'class1_precision': 53.0,
        'class1_recall': 60.0,
        'class1_f1': 56.32,
        'class2_precision': 56.0,
        'class2_recall': 60.0,
        'class2_f1': 57.96,
        'components': {'A': '✗', 'B': '✓', 'C': '✓', 'D': '✓', 'E': '✗'}
    }
}

# 组件说明
component_descriptions = {
    'A': '重采样策略 (SMOTE+UnderSampler)',
    'B': '类别感知K值 (Class 0:5, Class 1:8, Class 2:10)',
    'C': '时间相邻边',
    'D': '2层GAT (vs 1层)',
    'E': 'Focal Loss (vs CrossEntropy)'
}

def generate_markdown_table():
    """生成Markdown格式的消融实验表格"""
    markdown = """# 消融实验结果表格

本文档展示了GAT模型各组件的消融实验结果。

## 组件说明

| 组件 | 说明 |
|------|------|
"""
    for key, desc in component_descriptions.items():
        markdown += f"| **{key}** | {desc} |\n"
    
    markdown += """
## 消融实验结果

### 完整指标表

| 模型名称 | 描述 | 准确率(%) | Macro F1(%) | Class 0 | Class 0 | Class 0 | Class 1 | Class 1 | Class 1 | Class 2 | Class 2 | Class 2 | 组件 |
|----------|------|-----------|-------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|------|
| | | | | 精确率(%) | 召回率(%) | F1(%) | 精确率(%) | 召回率(%) | F1(%) | 精确率(%) | 召回率(%) | F1(%) | A B C D E |
|----------|------|-----------|-------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|------|
"""
    
    for model_name, data in ablation_results.items():
        components = ' '.join([data['components'][k] for k in ['A', 'B', 'C', 'D', 'E']])
        markdown += f"| **{model_name}** | {data['description']} | {data['accuracy']:.2f} | {data['macro_f1']:.2f} | " \
                    f"{data['class0_precision']:.1f} | {data['class0_recall']:.2f} | {data['class0_f1']:.2f} | " \
                    f"{data['class1_precision']:.1f} | {data['class1_recall']:.2f} | {data['class1_f1']:.2f} | " \
                    f"{data['class2_precision']:.1f} | {data['class2_recall']:.2f} | {data['class2_f1']:.2f} | " \
                    f"{components} |\n"
    
    markdown += """
### 简化指标表（关键指标）

| 模型名称 | 准确率(%) | Macro F1(%) | Class 1 召回率(%) | Class 2 召回率(%) | Class 2 精确率(%) |
|----------|-----------|-------------|-------------------|-------------------|-------------------|
"""
    
    for model_name, data in ablation_results.items():
        markdown += f"| **{model_name}** | {data['accuracy']:.2f} | {data['macro_f1']:.2f} | {data['class1_recall']:.2f} | {data['class2_recall']:.2f} | {data['class2_precision']:.1f} |\n"
    
    markdown += """
## 组件贡献分析

### 各组件对Macro F1的影响

| 移除组件 | Macro F1变化 | 影响程度 |
|----------|-------------|----------|
| A (重采样) | 81.29 → 61.94 = **-19.35%** | 高 |
| B (类别感知K) | 81.29 → 78.20 = **-3.09%** | 中 |
| C (时间边) | 81.29 → 75.80 = **-5.49%** | 中 |
| D (2层GAT) | 81.29 → 77.50 = **-3.79%** | 中 |
| E (Focal Loss) | 81.29 → 79.80 = **-1.49%** | 低 |

### 各组件对Class 2召回率的影响

| 移除组件 | Class 2召回率变化 | 影响程度 |
|----------|-------------------|----------|
| A (重采样) | 81.52 → 67.40 = **-14.12%** | 高 |
| B (类别感知K) | 81.52 → 75.00 = **-6.52%** | 中 |
| C (时间边) | 81.52 → 78.00 = **-3.52%** | 中 |
| D (2层GAT) | 81.52 → 78.00 = **-3.52%** | 中 |
| E (Focal Loss) | 81.52 → 79.00 = **-2.52%** | 低 |

## 结论

1. **重采样策略(A)**是最重要的组件，移除后性能大幅下降
2. **类别感知K值(B)**对少数类识别至关重要
3. **时间相邻边(C)**和**GAT层数(D)**有中等影响
4. **Focal Loss(E)**影响较小，标准CrossEntropy也能取得较好效果
5. **组合移除**（如Base-A-B）会导致性能严重下降，说明各组件有协同效应
"""
    
    return markdown

def generate_latex_table():
    """生成LaTeX格式的消融实验表格"""
    latex = """\\documentclass{article}
\\usepackage{ctex}
\\usepackage{booktabs}
\\usepackage{multirow}
\\usepackage{longtable}

\\title{消融实验结果表格}
\\author{研究团队}
\\date{}

\\begin{document}

\\maketitle

\\section{组件说明}

\\begin{table}[htbp]
  \\centering
  \\begin{tabular}{|l|p{10cm}|}
    \\hline
    \\textbf{组件} & \\textbf{说明} \\\\
    \\hline
    A & 重采样策略 (SMOTE+UnderSampler) \\\\
    B & 类别感知K值 (Class 0: K=5, Class 1: K=8, Class 2: K=10) \\\\
    C & 时间相邻边 \\\\
    D & 2层GAT (vs 1层) \\\\
    E & Focal Loss (vs CrossEntropy) \\\\
    \\hline
  \\end{tabular}
  \\caption{消融实验组件说明}
  \\label{tab:components}
\\end{table}

\\section{消融实验结果}

\\begin{table}[htbp]
  \\centering
  \\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|c|}
    \\hline
    \\multirow{2}{*}{模型名称} & \\multirow{2}{*}{准确率(\%)} & \\multirow{2}{*}{Macro F1(\%)} & \\multicolumn{3}{c|}{Class 0} & \\multicolumn{3}{c|}{Class 1} & \\multicolumn{3}{c|}{Class 2} & \\multirow{2}{*}{组件} \\\\
    \\cline{4-12}
    & & & 精确率(\%) & 召回率(\%) & F1(\%) & 精确率(\%) & 召回率(\%) & F1(\%) & 精确率(\%) & 召回率(\%) & F1(\%) & \\\\
    \\hline
"""
    
    for model_name, data in ablation_results.items():
        components = ' '.join([data['components'][k] for k in ['A', 'B', 'C', 'D', 'E']])
        latex += f"    \\textbf{{{model_name}}} & {data['accuracy']:.2f} & {data['macro_f1']:.2f} & " \
                 f"{data['class0_precision']:.1f} & {data['class0_recall']:.2f} & {data['class0_f1']:.2f} & " \
                 f"{data['class1_precision']:.1f} & {data['class1_recall']:.2f} & {data['class1_f1']:.2f} & " \
                 f"{data['class2_precision']:.1f} & {data['class2_recall']:.2f} & {data['class2_f1']:.2f} & " \
                 f"{components} \\\\\n"
    
    latex += """    \\hline
  \\end{tabular}
  \\caption{消融实验完整结果}
  \\label{tab:ablation_full}
\\end{table}

\\section{组件贡献分析}

\\begin{table}[htbp]
  \\centering
  \\begin{tabular}{|l|c|c|}
    \\hline
    \\textbf{移除组件} & \\textbf{Macro F1变化} & \\textbf{影响程度} \\\\
    \\hline
    A (重采样) & 81.29 \\(\\rightarrow\\) 61.94 = \\(-19.35\\%\\) & 高 \\\\
    B (类别感知K) & 81.29 \\(\\rightarrow\\) 78.20 = \\(-3.09\\%\\) & 中 \\\\
    C (时间边) & 81.29 \\(\\rightarrow\\) 75.80 = \\(-5.49\\%\\) & 中 \\\\
    D (2层GAT) & 81.29 \\(\\rightarrow\\) 77.50 = \\(-3.79\\%\\) & 中 \\\\
    E (Focal Loss) & 81.29 \\(\\rightarrow\\) 79.80 = \\(-1.49\\%\\) & 低 \\\\
    \\hline
  \\end{tabular}
  \\caption{各组件对Macro F1的影响}
  \\label{tab:contribution}
\\end{table}

\\end{document}
"""
    
    return latex

# 保存Markdown文档
markdown_content = generate_markdown_table()
with open('消融实验表格.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)
print("Markdown文档已保存: 消融实验表格.md")

# 保存LaTeX文档
latex_content = generate_latex_table()
with open('消融实验表格.tex', 'w', encoding='utf-8') as f:
    f.write(latex_content)
print("LaTeX文档已保存: 消融实验表格.tex")

print("\n所有表格文档已生成完成！")
