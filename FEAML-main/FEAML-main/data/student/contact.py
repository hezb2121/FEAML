import pandas as pd

# 读取清洗后的数据
df = pd.read_csv('cleaned_data.csv')

# 标签列
label_columns = ['higher', 'romantic', 'schoolsup', 'activities', 'internet', 'famrel', 'G3']

# famrel 二值化处理（>=4 为良好家庭关系，记作 1）
df['famrel'] = df['famrel'].astype(int)  # 转为整数再处理
df['famrel'] = (df['famrel'] >= 4).astype(int)

# 将标签列中非 numeric 的 yes/no 转为 1/0
for col in ['higher', 'romantic', 'schoolsup', 'activities', 'internet']:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# G3 作为一个多标签任务的目标之一，直接保留（也可以分段映射成分类）

# 构造标签数据集
Y = df[label_columns]

# 构造输入特征（除标签列之外的所有）
X = df.drop(columns=label_columns)

# 检查结果
print("标签 Y 示例：")
print(Y.head())

print("\n输入特征 X 示例：")
print(X.head())

df_final = pd.concat([X, Y], axis=1)
df_final.to_csv('student.csv', index=False)
