import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2", "AVG"]
BiTrap = [0.668, 0.209, 0.388, 0.336, 0.233, 0.3668]
Net = [0.61, 0.20, 0.36, 0.32, 0.22, 0.342]
# 9 4 7 5 6
x = np.arange(len(categories))  # 横坐标的位置
width = 0.35  # 柱形的宽度

# 创建图形和子图
fig, ax = plt.subplots()

# 绘制柱形图
bars1 = ax.bar(x - width/2, BiTrap, width, label=r"$\mathit{R}_{a}$"+'=0', alpha=0.4, color="purple")
bars2 = ax.bar(x + width/2, Net, width, label=r"$\mathit{R}_{a}$"+'=1', alpha=0.4, color="green")

# 添加文本标签、标题及坐标轴标签
ax.set_xlabel('Scene', fontsize=14)
ax.set_ylabel('FDE', fontsize=14)
#ax.set_title('Ablation study on Randomization')
ax.set_xticks(x)
ax.set_xticklabels(categories)

y_ticks = np.arange(0, 0.7, 0.1)
ax.set_yticks(y_ticks)

ax.legend(fontsize=14)

# 调整横轴和纵轴刻度的字体大小
plt.xticks(fontsize=14)  # 12可以根据需要调整
plt.yticks(fontsize=14)

# 显示图形
plt.show()
