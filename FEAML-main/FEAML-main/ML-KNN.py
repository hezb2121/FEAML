import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.neighbors import NearestNeighbors
from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import newdata  # 你自己的数据处理模块

# ===================== 参数设置 ============================
dict = {
    'adult': 2,
    'bank': 2,
    'blood': 1,
    'car': 1,
    'communities': 2,
    'credit-g': 2,
    'diabetes': 1,
    'heart': 2,
    'myocardial': 3,
    'student': 7
}

data_name = "credit-g"
_CLASS = dict[data_name]
_SEED = 0
newdata.set_seed(_SEED)

# ===================== 获取数据 ============================
df, X_train, X_test, y_train, y_test, target_attr, is_cat, label_encoders = \
    newdata.get_dataset(data_name, _CLASS, _SEED)

# 特殊处理 student 数据集（最后一列 G3 分组）
def map_g3(g):
    return 0 if g <= 10 else 1

if data_name == "student":
    y_train.iloc[:, 6] = np.vectorize(map_g3)(y_train.iloc[:, 6])
    y_test.iloc[:, 6] = np.vectorize(map_g3)(y_test.iloc[:, 6])

# ===================== 数据标准化 ============================
# 特征归一化/标准化（ML-KNN 依赖距离计算）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转为稀疏矩阵（ML-KNN 要求稀疏输入）
X_train_sparse = csr_matrix(X_train_scaled)
X_test_sparse = csr_matrix(X_test_scaled)

# 标签转为布尔型并转换为稀疏矩阵
y_train_bool = y_train.astype(int)
y_test_bool = y_test.astype(int)
y_train_sparse = csr_matrix(y_train_bool.values)
y_test_sparse = csr_matrix(y_test_bool.values)

# ===================== 定义并训练模型 ============================
mlknn = MLkNN(k=10)

# 拟合模型
mlknn.fit(X_train_sparse, y_train_sparse)

# 预测
y_pred = mlknn.predict(X_test_sparse)

# ===================== 评估指标 ============================
print("=== ML-KNN 多标签分类结果 ===")
acc = accuracy_score(y_test_sparse.toarray(), y_pred.toarray())
print(f"Multilabel Accuracy:{acc:.4f}")

hamming = hamming_loss(y_test_sparse.toarray(), y_pred.toarray())
print(f"Hamming Loss:{hamming:.4f}")
