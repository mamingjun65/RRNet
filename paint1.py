import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2", "AVG"]
likelihood = [0.044, 0.0139, 0.0601, 0.061, 0.0692, 0.04964]
probability = [0.2088, 0.3148, 0.1348, 0.1351, 0.4033, 0.23936]

x = np.arange(len(categories))  # 横坐标的位置
width = 0.35  # 柱形的宽度

# 创建图形和子图
fig, ax = plt.subplots()

# 绘制柱形图
bars1 = ax.bar(x - width/2, likelihood, width, label=r"$\mathit{A}_{L}$",alpha=0.5)
bars2 = ax.bar(x + width/2, probability, width, label=r"$\mathit{A}_{P}$",alpha=0.5)
# 添加文本标签、标题及坐标轴标签
ax.set_xlabel('Scene', fontsize=14)
ax.set_ylabel('Value', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=14)
# 调整横轴和纵轴刻度的字体大小
plt.xticks(fontsize=14)  # 12可以根据需要调整
plt.yticks(fontsize=14)
# 显示图形
plt.show()
