import pandas as pd

# 检查单个 mold_data 文件
file_path = './raw/mold_data_611028906_33722.csv' # 根据您提供的路径
df = pd.read_csv(file_path)
print(f"File: {file_path}")
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

print("\nFirst few rows:")
print(df.head())

# 提取 prod_id
filename = 'mold_data_611028906_33722.csv'
parts = filename.replace('.csv', '').split('_')
file_prod_id = int(parts[3]) # mold_data_{mold_id}_{prod_id}
print(f"\nExtracted prod_id from filename: {file_prod_id}")

# 检查文件内是否有 'Id' 列
if 'Id' in df.columns:
    print(f"Ids found inside the file: {df['Id'].unique()}")
    # 检查 file_prod_id 是否与文件内的 Id 列有关
    if file_prod_id in df['Id'].values:
        print(f"✅ Success! Filename's prod_id ({file_prod_id}) is present in the file's 'Id' column.")
    else:
        print(f"⚠️  Filename's prod_id ({file_prod_id}) is NOT in the file's 'Id' column.")
else:
    print("⚠️  No 'Id' column found inside the file.")