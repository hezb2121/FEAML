import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter  # 添加这行

# Data

df= {
    'datasets': ['bank'],
    'FEAML': [0.2021],
    'FEAML+': [0.2145],
    'FEAML-': [0.2268]
}

colors = ['#00008B',  # 深蓝 (Deep Blue)
          '#0000FF',  # 蓝 (Blue)
          '#ADD8E6']  # 浅蓝 (Light Blue)

# Set bar width and positions
bar_width = 0.25
x = np.arange(len(df["datasets"]))

# Create a grouped bar chart
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, df["FEAML"], bar_width-0.05, label="FEAML", color=colors[0])
plt.bar(x, df["FEAML+"], bar_width-0.05, label="FEAML+", color=colors[1])
plt.bar(x + bar_width, df["FEAML-"], bar_width-0.05, label="FEAML-", color=colors[2])

plt.ylim(0.18, auto=True)

plt.ylabel("Hamming Loss", fontsize=36, fontstyle='italic')
plt.tick_params(axis='y', which='major', labelsize=36)  #y轴字体大小
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks([])
plt.legend(fontsize=25)

dataset_name = df['datasets'][0]
plt.tight_layout()
plt.savefig(f"{dataset_name}36.png", dpi=300)
plt.show()
