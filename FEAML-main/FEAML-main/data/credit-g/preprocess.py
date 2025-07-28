import pandas as pd

# 读取原始数据
df = pd.read_csv("credit-g.csv")  # 请将路径替换为你本地的路径

# ===== employment 字段转换 =====
def convert_employment_text(x):
    if isinstance(x, str):
        x = x.strip()
        if '<1' in x:
            return 0.5
        elif '1<=X<4' in x:
            return 2.5
        elif '4<=X<7' in x:
            return 5.5
        elif '>=7' in x:
            return 10
        elif 'unemployed' in x.lower():
            return 0
    return None

df['employment_num'] = df['employment'].apply(convert_employment_text)

# ===== 构造标签 =====

# 标签1：高信用额度（credit_amount > 5000）
df['label_high_credit'] = (df['credit_amount'] > 5000).astype(int)


# 标签3：年轻且无依赖人群（age < 30 且 num_dependents == 1）
df['label_young_independent'] = ((df['age'] < 30) & (df['num_dependents'] == 1)).astype(int)

# 标签4：稳定就业人群（employment 年限 >= 4 年）
df['label_stable_job'] = (df['employment_num'] >= 4).astype(int)

# 标签5：有房地产资产（property_magnitude 含 real estate）
df['label_has_property'] = df['property_magnitude'].str.contains("real estate", case=False).astype(int)

# 标签6：外籍工人中高信用者（foreign_worker 是 yes 且 credit_amount > 4000）
df['label_foreign_good_credit'] = ((df['foreign_worker'].str.lower() == 'yes') &
                                   (df['credit_amount'] > 4000)).astype(int)

# ===== 删除原始标签字段 =====
df.drop(columns=['employment', 'employment_num'], inplace=True)

# ===== 保存新数据集 =====
df.to_csv("credit-g_multilabel.csv", index=False)
print("✅ 已保存多标签数据集为 credit-g_multilabel.csv")
