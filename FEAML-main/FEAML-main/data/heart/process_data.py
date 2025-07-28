import pandas as pd
from scipy.io import arff


def convert_arff_to_csv(input_file, output_file):
    # 读取 .arff 文件
    data, meta = arff.loadarff(input_file)

    # 将字节对象转换为字符串（如有必要）
    df = pd.DataFrame(data)
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.decode('utf-8')

    # 保存为 .csv 文件
    df.to_csv(output_file, index=False)
    print(f"File saved to {output_file}")


# 读取 CSV 文件
file_path = 'heart/heart.csv'  # 请替换为你的 CSV 文件路径
df = pd.read_csv(file_path)

# 指定要交换的两列
col1 = 'Sex'  # 替换为你要交换的第一列的列名
col2 = 'ST_Slope'  # 替换为你要交换的第二列的列名

# 交换两列
df[col1], df[col2] = df[col2], df[col1]

# 交换列名
df = df.rename(columns={col1: col2, col2: col1})

# 将修改后的 DataFrame 覆盖保存为原文件
df.to_csv(file_path, index=False)

print(f"Successfully swapped columns '{col1}' and '{col2}' and saved the file.")

