import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

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

data_name = "myocardial"
_CLASS = dict[data_name]
_SEED = 0
newdata.set_seed(_SEED)

# ===================== 获取数据 =============================
df, X_train, X_test, y_train, y_test, target_attr, is_cat, label_encoders = \
    newdata.get_dataset(data_name, _CLASS, _SEED)

# 特殊处理 student 数据集（最后一列 G3 分组）
def map_g3(g):
    return 0 if g <= 10 else 1

if data_name == "student":
    y_train.iloc[:, 6] = np.vectorize(map_g3)(y_train.iloc[:, 6])
    y_test.iloc[:, 6] = np.vectorize(map_g3)(y_test.iloc[:, 6])

# ===================== 数据标准化 =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 标签转为整数类型（如果不是整数）
y_train_int = y_train.astype(int)
y_test_int = y_test.astype(int)

# ===================== 定义并训练多标签SVM模型 ====================
svm_base = SVC(kernel='rbf', probability=False, random_state=_SEED)
multi_svm = MultiOutputClassifier(svm_base, n_jobs=-1)

multi_svm.fit(X_train_scaled, y_train_int)

# ===================== 预测 ====================================
y_pred = multi_svm.predict(X_test_scaled)

# ===================== 评估指标 ================================
print("=== Multi-label SVM 分类结果 ===")
acc = accuracy_score(y_test_int, y_pred)
print(f"Multilabel Accuracy:{acc:.4f}")

hamming = hamming_loss(y_test_int, y_pred)
print(f"Hamming Loss:{hamming:.4f}")
