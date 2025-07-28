import pandas as pd
import torch
import numpy as np
import copy
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def set_seed(seed): #设置种子，便于复现
    random.seed(seed)
    np.random.seed(seed)

def multilabel_few_shot_by_labelset(X, Y, shot, seed=0):
    """
    从多标签数据中采样 shot 条样本，确保每个标签组合至少出现一次。
    - X: 特征 DataFrame
    - Y: 多标签 DataFrame 或 numpy 2D array
    - shot: 采样总数
    """
    np.random.seed(seed)

    if isinstance(Y, pd.DataFrame):
        Y_array = Y.values
    else:
        Y_array = Y

    # 统计每种独特的标签组合
    label_sets = {}
    for i, label in enumerate(Y_array):
        key = tuple(label)
        if key not in label_sets:
            label_sets[key] = []
        label_sets[key].append(i)

    num_combinations = len(label_sets)
    if shot < num_combinations:
        raise ValueError(
            f"给定 shot={shot} 太小，无法覆盖全部 {num_combinations} 种不同标签组合。请至少设置 shot >= {num_combinations}"
        )

    # 每种标签组合选一个
    selected_indices = []
    for key, indices in label_sets.items():
        chosen = np.random.choice(indices, size=1)[0]
        selected_indices.append(chosen)

    # 不足部分，随机补齐
    remaining_pool = list(set(range(len(X))) - set(selected_indices))
    if len(selected_indices) < shot:
        n_extra = shot - len(selected_indices)
        extra_indices = np.random.choice(remaining_pool, size=n_extra, replace=False)
        selected_indices.extend(extra_indices)

    # 返回结果
    X_selected = X.iloc[selected_indices].reset_index(drop=True)
    Y_selected = (
        Y.iloc[selected_indices].reset_index(drop=True)
        if isinstance(Y, pd.DataFrame)
        else Y_array[selected_indices]
    )

    return X_selected, Y_selected


def get_dataset(data_name, label_column, seed):
    file_name = f"./data/{data_name}/{data_name+'_multilabel'}.csv"
    print(file_name)
    df = pd.read_csv(file_name)

    # 提取标签列名和特征列名
    default_target_attribute = df.columns[-label_column:]#特征列名
    attribute_names = df.columns[:-label_column].tolist()#标签列名

    # 判断特征中哪些是分类变量
    categorical_indicator = [
        True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False
        for dt in df.dtypes.tolist()[:-label_column]
    ]

    # 分离特征和标签
    X = df[attribute_names].copy()
    y_df = df[default_target_attribute].copy()

    # 对特征中的分类变量进行编码
    if any(categorical_indicator):
        encoder = OrdinalEncoder()
        X.loc[:, [name for i, name in enumerate(attribute_names) if categorical_indicator[i]]] = \
            encoder.fit_transform(X.loc[:, [name for i, name in enumerate(attribute_names) if categorical_indicator[i]]])
    if data_name=="car":
        X = X.astype(float)
    #X = X.astype(float)
    # 对每一列标签分别使用 LabelEncoder
    label_encoders = {}
    for col in default_target_attribute:
        le = LabelEncoder()
        y_df[col] = le.fit_transform(y_df[col])
        label_encoders[col] = le  # 如果后续需要 inverse_transform 可用

    column_names = df.columns.tolist()
    new_df = pd.DataFrame(data=np.concatenate([X, y_df], -1),columns=column_names)#进行编码
    #print(new_df)
    # 转为 numpy 格式
    y = y_df

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    X_few, Y_few = multilabel_few_shot_by_labelset(X_train, y_train, shot=200)

    return new_df, X_few, X_test, Y_few, y_test, default_target_attribute, categorical_indicator, label_encoders

#get_dataset("credit-g",6,0)
