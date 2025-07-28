import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss
import newdata
import mafe

# ==== 1. 数据加载 ====
dict={
    'adult':4,
    'bank':2,
    'communities':10,#?
    'credit-g':6,#√
    #'diabetes': 1,
    'heart': 2,
    'myocardial': 16,
    'student':7 #√
}

data_name = "adult"
num_labels = dict[data_name]
_SEED = 0
newdata.set_seed(_SEED)
df, X_train, X_test, y_train, y_test, target_attr, is_cat, label_encoders = \
    newdata.get_dataset(data_name, num_labels, _SEED)

y_train = np.array(y_train)
y_test = np.array(y_test)

# 确保X_train和X_test是DataFrame格式，方便后续操作
if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)

print("Start LDA feature transformation...")

# LDA拟合训练集特征并转换训练和测试集
lda = LatentDirichletAllocation(n_components=5, random_state=_SEED)
lda.fit(X_train)
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# 将LDA转换后的特征转换为DataFrame，方便拼接或查看
X_train_lda_df = pd.DataFrame(X_train_lda, index=X_train.index, columns=[f"lda_{i}" for i in range(5)])
X_test_lda_df = pd.DataFrame(X_test_lda, index=X_test.index, columns=[f"lda_{i}" for i in range(5)])

print("LDA feature transformation done.")

# 如果想拼接原始特征和LDA新特征，可以如下操作（可选）
X_train_enhanced = pd.concat([X_train, X_train_lda_df], axis=1)
X_test_enhanced = pd.concat([X_test, X_test_lda_df], axis=1)

print("Start multilabel classification...")
base_clf = RandomForestClassifier(random_state=_SEED)
multi_clf = MultiOutputClassifier(base_clf)
multi_clf.fit(X_train_enhanced, y_train)

y_pred = multi_clf.predict(X_test_enhanced)
print("Prediction done.")

# ==== 评估指标 ====
acc = accuracy_score(y_test, y_pred)
hamming_loss_value = hamming_loss(y_test, y_pred)
print("Multilabel Accuracy:", acc)
print("Hamming Loss:", hamming_loss_value)
