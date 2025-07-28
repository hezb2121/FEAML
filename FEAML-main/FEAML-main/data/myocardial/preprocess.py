import pandas as pd

# 读取原始数据
df = pd.read_csv('myocardial.csv')

# 这里用你的DataFrame代替
# df = your_dataframe

# 选择做多标签的字段
target_labels = [
    "nr_04",         # 持续性房颤
    "np_04",         # 三度房室传导阻滞
    "endocr_01",     # 糖尿病
    "zab_leg_03",    # 哮喘
    "O_L_POST",      # 肺水肿
    "K_SH_POST",     # 心源性休克
    "FIB_G_POST"     # 心室颤动
]

# 检查是否缺失标签列
missing_labels = [label for label in target_labels if label not in df.columns]
if missing_labels:
    raise ValueError(f"缺失标签字段: {missing_labels}")

# 抽取标签列Y
Y = df[target_labels].copy()

# 取剩余列作为输入特征X
X = df.drop(columns=target_labels)

# 合并X和Y，保存成新的CSV文件（输入特征+多标签）
df_multilabel = pd.concat([X, Y], axis=1)

# 保存
df_multilabel.to_csv('myocardial_multilabel.csv', index=False)

print("多标签格式数据已保存到 myocardial_multilabel.csv")
