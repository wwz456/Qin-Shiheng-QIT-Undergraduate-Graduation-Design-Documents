import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 获取所有原始数据文件
data_dir = 'c:\\Users\\秦士衡\\Desktop\\数据2\\传感器高频数据'
raw_files = [f for f in os.listdir(data_dir) if f.startswith('mold_data_') and f.endswith('.csv')]

# 存储处理后的数据
processed_data = []

# 遍历所有原始数据文件
for file in raw_files:
    # 提取模具ID和批次信息
    file_parts = file.split('_')
    mold_id = file_parts[2]
    batch_id = file_parts[3].split('.')[0]
    
    # 读取数据
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)
    
    # 添加模具ID和批次ID作为特征
    df['Mold_ID'] = mold_id
    df['Batch_ID'] = batch_id
    
    # 时间特征提取 - 由于SampleTime是相对时间，我们可以计算一些统计特征
    df['SampleTime_Max'] = df['SampleTime'].max()
    df['SampleTime_Min'] = df['SampleTime'].min()
    df['SampleTime_Range'] = df['SampleTime_Max'] - df['SampleTime_Min']
    
    # 识别连续变量和类别变量
    continuous_vars = ['Sensor1', 'Sensor2', 'Sensor3', 'IJ', 'Sensor5', 'Sensor6', 
                      'MouldTemp1', 'MouldTemp2', 'MouldTemp3', 'MouldTemp4', 'MouldTemp5', 
                      'MouldTemp9', 'MouldTemp10', 'MouldTemp11', 'MouldTemp12', 
                      'MouldTemp13', 'MouldTemp14', 'Sensor8', 'SP']
    
    categorical_vars = ['Phase', 'Mold_ID', 'Batch_ID', 'MouldFlow1', 'MouldFlow2', 'MouldFlow3']
    
    # 处理缺失值
    for var in continuous_vars:
        if df[var].isnull().any():
            df[var].fillna(df[var].mean(), inplace=True)
    
    for var in categorical_vars:
        if df[var].isnull().any():
            df[var].fillna(df[var].mode()[0], inplace=True)
    
    # 按模具分组进行标准化
    scaler = StandardScaler()
    for var in continuous_vars:
        # 按模具ID分组标准化
        grouped = df.groupby('Mold_ID')[var]
        df[var + '_scaled'] = grouped.transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
    
    # 类别变量编码
    for var in categorical_vars:
        if var not in ['Mold_ID', 'Batch_ID']:
            # 整数编码
            df[var + '_encoded'] = pd.factorize(df[var])[0]
    
    # 模具ID和批次ID也进行整数编码
    df['Mold_ID_encoded'] = pd.factorize(df['Mold_ID'])[0]
    df['Batch_ID_encoded'] = pd.factorize(df['Batch_ID'])[0]
    
    # 计算统计特征
    batch_stats = {
        'Mold_ID': mold_id,
        'Batch_ID': batch_id
    }
    
    # 添加连续变量的统计特征
    for var in continuous_vars:
        batch_stats[f'{var}_mean'] = df[var].mean()
        batch_stats[f'{var}_std'] = df[var].std()
        batch_stats[f'{var}_max'] = df[var].max()
        batch_stats[f'{var}_min'] = df[var].min()
        batch_stats[f'{var}_median'] = df[var].median()
    
    # 添加类别特征
    batch_stats['Phase_mode'] = df['Phase'].mode()[0]
    batch_stats['MouldFlow1_mode'] = df['MouldFlow1'].mode()[0]
    batch_stats['MouldFlow2_mode'] = df['MouldFlow2'].mode()[0]
    batch_stats['MouldFlow3_mode'] = df['MouldFlow3'].mode()[0]
    
    # 添加时间特征
    batch_stats['SampleTime_Range'] = df['SampleTime_Range'].iloc[0]
    
    # 添加编码后的类别特征
    batch_stats['Mold_ID_encoded'] = df['Mold_ID_encoded'].iloc[0]
    batch_stats['Batch_ID_encoded'] = df['Batch_ID_encoded'].iloc[0]
    
    # 将批次统计特征转换为DataFrame
    stats_df = pd.DataFrame([batch_stats])
    
    # 添加到处理后的数据中
    processed_data.append(stats_df)

# 合并所有处理后的数据
final_data = pd.concat(processed_data, ignore_index=True)

# 保存处理后的数据
output_path = os.path.join(data_dir, 'processed_raw_data.csv')
final_data.to_csv(output_path, index=False)

print(f"预处理完成，数据已保存到 {output_path}")
print(f"处理后的数据形状: {final_data.shape}")
print("处理后的数据列:")
print(final_data.columns.tolist())
