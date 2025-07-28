import numpy as np
import pandas as pd
from autofeat import AutoFeatClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss
from xgboost import XGBClassifier

import newdata
import mafe
import xgboost as xgb
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

# 如果X_train不是DataFrame，转换为DataFrame方便concat
if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)

print("Start feature engineering per label...")
X_train_enhanced = X_train.copy()
X_test_enhanced = X_test.copy()

# 逐标签调用AutoFeatClassifier
for i in range(num_labels):
    print(f"Processing label {i+1}/{num_labels} ...")
    af_model = AutoFeatClassifier(n_jobs=8, verbose=1, feateng_steps=1, featsel_runs=2)
    # fit_transform只接受一维y，所以传入单个标签列
    X_train_new = af_model.fit_transform(X_train, y_train[:, i])
    X_test_new = af_model.transform(X_test)

    # 拼接新特征
    X_train_enhanced = pd.concat([X_train_enhanced, pd.DataFrame(X_train_new.values, index=X_train.index)], axis=1)
    X_test_enhanced = pd.concat([X_test_enhanced, pd.DataFrame(X_test_new.values, index=X_test.index)], axis=1)

print("Feature engineering done.")

# 拼接特征后，统一转换列名为字符串类型
X_train_enhanced.columns = X_train_enhanced.columns.astype(str)
X_test_enhanced.columns = X_test_enhanced.columns.astype(str)



# ==== 多标签模型训练和预测 ====
print("Start multilabel classification with XGBoost...")

# ==== 3. 多标签建模 ====
print("Start multilabel classification...")
#base_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
base_clf = RandomForestClassifier(random_state=0)
multi_clf = MultiOutputClassifier(base_clf)
multi_clf.fit(X_train_enhanced, y_train)
y_pred = multi_clf.predict(X_test_enhanced)
print("Prediction done.")

# ==== 评估指标 ====
acc = accuracy_score(y_test, y_pred)
hamming_loss_value = hamming_loss(y_test, y_pred)
print("Multilabel Accuracy:", acc)
print("Hamming Loss:", hamming_loss_value)