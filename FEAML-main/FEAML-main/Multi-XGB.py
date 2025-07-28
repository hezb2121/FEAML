import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, hamming_loss, label_ranking_loss
import newdata , mafe
import numpy as np

dict={
    'adult':4,
    'bank':2,
    'communities':9,#?
    'credit-g':6,#√
    #'diabetes': 1,
    'heart': 2,
    'myocardial': 16,
    'student':7 #√
}
data_name="student"
_CLASS = dict[data_name]  # 多目标数量
_SEED=0
newdata.set_seed(_SEED)
df, X_train, X_test, y_train, y_test, target_attr, is_cat, label_encoders= \
    newdata.get_dataset(data_name,_CLASS,_SEED)
#x,y都是dataframe
#print(X_train)
#print(y_train)

#print(X_train.dtypes)

# 1. 定义基础模型
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',  # 每个标签是二分类任务
    #use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)


base_clf = RandomForestClassifier(random_state=0)
multi_model = MultiOutputClassifier(base_clf)
# 2. 封装成多输出模型
#multi_model = MultiOutputClassifier(xgb_base)

def map_g3(g):
    if g <= 10:
        return 0
    else:
        return 1

# 处理 student数据集
if data_name== "student":
    y_train.iloc[:, 6] = np.vectorize(map_g3)(y_train.iloc[:, 6])
    y_test.iloc[:, 6] = np.vectorize(map_g3)(y_test.iloc[:, 6])


# 3. 模型训练
multi_model.fit(X_train, y_train)

# 4. 模型预测
y_pred = multi_model.predict(X_test)

# 5. 评估指标
# Accuracy（平均准确率）
acc = accuracy_score(y_test, y_pred)
print("Multilabel Accuracy:", acc)

# 计算汉明损失
hamming_loss_value = hamming_loss(y_test, y_pred)
print("Hamming Loss:", hamming_loss_value)

#MAFFE
llm_model = 'gpt-4'
code, prompt, messages = mafe.generate_features(data_name,df,
                                      X_train,y_train,target_attr,is_cat,
                                      model=llm_model,
                                      iterative=3,
                                      )
print(code)

X_train_extended=mafe.run_llm_code(code,X_train)
X_test_extended=mafe.run_llm_code(code,X_test)

#print(X_train_extended)
#print((X_test_extended))

X_train_extended = mafe.convert_category_to_int(X_train_extended)
X_test_extended = mafe.convert_category_to_int(X_test_extended)


# 3. 模型训练
multi_model.fit(X_train_extended, y_train)

# 4. 模型预测
y_pred = multi_model.predict(X_test_extended)
#y_prob = multi_model.predict_proba(X_test_extended)  # 用于 AUC

# 5. 评估指标
# Accuracy（平均准确率）
acc = accuracy_score(y_test, y_pred)
print("Multilabel Accuracy:", acc)

hamming_loss_value = hamming_loss(y_test, y_pred)
print("Hamming Loss:", hamming_loss_value)

