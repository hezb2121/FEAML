import numpy as np
import openai

from run_llm_code import run_llm_code
from sklearn.model_selection import train_test_split
import pandas as pd
import openai
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

TASK_DICT={
    'adult':"This is a multi-label classification task with two labels to predict. The first prediction is whether the individual is male or female, and the second prediction is whether the individual's income exceeds $50K per year.",
    'bank':"This is a multi-label classification task with two labels to predict. The first prediction is whether the individual has a housing loan (yes or no), and the second prediction is whether the individual has subscribed to a term deposit (yes or no). ",
    'communities': "How high will the rate of violent crimes per 100K population be in this area. Low, medium, or high? ",
    'credit-g': "This is a multi-label classification task with two labels to predict. The first prediction is whether the individual is a foreign worker (yes or no), and the second prediction is whether the individual has good credit (good or bad).",
    'heart': "This is a multi-label classification task with one label to predict. The prediction is the diagnosis of heart disease for the individual. ",
    'myocardial': "Does the myocardial infarction complications data of this patient show chronic heart failure? Yes or no? ",
    'student': "This is a multi-label classification task with several labels to predict. "
               "The goal is to predict whether an individual has access to higher education, "
               "whether they are in a romantic relationship, "
               "whether they receive additional educational support, "
               "whether they participate in extracurricular activities, "
               "the quality of their family relationships, "
               "whether they have internet access at home, and their academic performance (good or bad). "
               "These labels are binary."
}
#提示词模板，prompt工程！加个分类特征的平衡性
def get_prompt(data_name, df, X_train, y_train, target_attr, is_cat, iterative=1):
    how_many =  "exactly one useful column"
    X_all = df.drop(target_attr, axis=1)
    meta_data_name = f"./data/{data_name}/{data_name}-metadata.json"
    selected_column = X_all.columns.tolist()
    features = get_feature_desc(X_all, selected_column, is_cat, meta_data_name)
    TASK = TASK_DICT[data_name]
    samples = X_all.head(10)#df以及编码

    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
{TASK}
The detailed information of the dataset features are as follows:
{features}
The following are the first ten rows of the dataset:
{samples}

In addition to feature semantics, a label co-occurrence matrix has been computed to quantify statistical dependencies among target labels. 
This matrix captures both joint occurrence probabilities and conditional likelihoods between label pairs.

The following are selected high-correlation label pairs that exhibit strong co-occurrence patterns:
This code creates extra columns to help a multi-label classifier predict the target labels. Besides standard feature engineering, consider relationships between labels. 
If some label combinations occur often, generate features capturing these dependencies using only input features—not labels. 
New features can include combinations, transformations, or aggregations of existing columns. Column scale and offset don’t matter, but ensure all columns exist and data types are correct. 
Avoid creating object-type columns that may cause issues with downstream models.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Why this helps classify multi-label targets "{', '.join(target_attr)}")
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ... )
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end

Each codeblock generates {how_many} new columns and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""

def generate_code(model, messages): #得到LLM回复
    openai.api_key = "sk-BeMohrCnfiKUOR9YB552Dd2d3dCe43459b0b2eD35aCf49Ee"
    openai.base_url = "https://api.gpt.ge/v1/"
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        stop=["```end"],
        temperature=0.5,
        max_tokens=1000,
    )
    code = completion.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").replace("<end>", "")
    return code

def convert_category_to_int(df):
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.codes
    return df

def execute_and_evaluate_code_block(df, code, target_attr, X_train, y_train, test_size=0.2, random_state=42):
    # 使用 LLM 生成的代码增强训练特征
    X_train_extended = run_llm_code(code, X_train)

    X_train_extended = convert_category_to_int(X_train_extended)
    # 划分训练集和测试集（在增强后的特征上进行划分）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_extended, y_train, test_size=test_size, random_state=random_state
    )

    # 模型构建与训练
    model = MultiOutputClassifier(
        xgb.XGBClassifier(eval_metric='logloss')
    )
    model.fit(X_tr, y_tr)

    # 标签预测
    y_pred = model.predict(X_te)

    # 预测概率用于 ROC AUC
    try:
        y_prob = np.stack([est.predict_proba(X_te)[:, 1] for est in model.estimators_], axis=1)
        roc = roc_auc_score(y_te, y_prob, average='macro')
    except Exception as e:
        print("AUC failed:", e)
        roc = 0.0

    acc = accuracy_score(y_te, y_pred)

    return [roc], [acc]


def generate_features(data_name, df, X_train, y_train, target_attr, is_cat, model="gpt-3.5-turbo", iterative=1,):
    prompt = get_prompt(data_name, df, X_train, y_train, target_attr, is_cat, iterative=iterative)
    #print(prompt)

    messages = [
        {"role": "system", "content": "You are a datascience assistant. Output code only."},
        {"role": "user", "content": prompt},
    ]

    base_acc, base_roc = 0, 0

    full_code = ""
    n_iter = iterative#迭代次数

    i = 0
    while i < n_iter:
        code = generate_code(model, messages)
        #print(code)
        #print(f"\n[💡 Iteration {i + 1}]\n{code}")
        i = i + 1

        rocs, accs = execute_and_evaluate_code_block(df, code, target_attr, X_train, y_train)
        #print(accs)
        #根据ROC和ACC是否增加决定是否添加该特征
        '''
        if accs and accs[0] > base_acc:
            full_code += code
            base_acc = accs[0]
            base_roc = rocs[0]
            print("✅ Feature accepted.")
            add_feature_sentence="Feature accepted."
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Performance after adding feature ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}. {add_feature_sentence}
            Next codeblock:
            """,
                },
            ]
        else:
            print("❌ Feature rejected.")
        '''
        full_code += code
        #print("✅ Feature accepted.")
        messages += [
            {"role": "assistant", "content": code},
            {"role": "user","content": f"""Next codeblock:""",
            },
        ]

    return full_code, prompt, messages


def get_feature_desc(df_all, selected_column, is_cat, meta_file_name):#生成特征描述，包括
    try:
        with open(meta_file_name, "r") as f:  # 读取数据集json文件，关于特征的描述
            meta_data = json.load(f)
    except:
        meta_data = {}

    feature_name_list = []
    sel_cat_idx = [df_all.columns.tolist().index(col_name) for col_name in selected_column]
    is_cat_sel = np.array(is_cat)[sel_cat_idx]  # 选中的分类特征

    # 遍历选定的列，生成特征描述
    for cidx, cname in enumerate(selected_column):
        if is_cat_sel[cidx]:  # 是分类变量
            clist = df_all[cname].unique().tolist()
            if len(clist) > 20:
                clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
            else:
                clist_str = ", ".join(map(str, clist))
            desc = meta_data.get(cname, "")
            feature_name_list.append(f"- {cname}: {desc} (categorical variable with categories [{clist_str}])")
        else:  # 是数值型变量
            desc = meta_data.get(cname, "")
            feature_name_list.append(f"- {cname}: {desc} (numerical variable)")

    # 将特征描述转换为字符串
    feature_desc = "\n".join(feature_name_list)

    return feature_desc

