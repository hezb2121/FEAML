import pandas as pd

df = pd.read_csv("adult.csv")

# 生成4个新标签
high_education = ['Bachelors', 'Masters', 'Doctorate']
married_status = ['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent']


df['education_high'] = df['education'].apply(lambda x: 1 if x in high_education else 0)
df['married'] = df['marital-status'].apply(lambda x: 1 if x in married_status else 0)
df['gender_male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# 删除原始单标签和用作新标签的特征
df.drop(columns=['education', 'marital-status', 'gender'], inplace=True)

# 保存处理后数据到本地 CSV 文件
df.to_csv("adult_multilabel.csv", index=False)

print("多标签数据保存成功：adult_multilabel.csv")
