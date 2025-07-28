import pandas as pd

# 读取原始文件（假设没有真正的列名，只有一列数据）
df_raw = pd.read_csv('student-mat.csv', header=None)

# 将第一列按分号分割为多个列
df_split = df_raw[0].str.split(';', expand=True)

# 第一行是列名，重新设置为列名
df_split.columns = df_split.iloc[0]

# 删除原本的列名那一行（现在是数据）
df_split = df_split.drop(index=0).reset_index(drop=True)
df_split = df_split.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
# 如果需要，将某些列转换为数值（例如 age, G1, G2, G3）
# 可选：df_split = df_split.apply(pd.to_numeric, errors='ignore')

# 保存为新CSV文件（可选）
df_split.to_csv('cleaned_data.csv', index=False)

# 打印前几行查看效果
print(df_split.head())
