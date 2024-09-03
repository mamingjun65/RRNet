import matplotlib.pyplot as plt

# Data
x = [20, 40, 60, 80, 100]
y1 = [0.668, 0.621, 0.678, 0.677, 0.643]
y2 = [0.209, 0.188, 0.176, 0.178, 0.168]
y3 = [0.388, 0.403, 0.407, 0.433, 0.438]
y4 = [0.336, 0.347, 0.352, 0.363, 0.360]
y5 = [0.233, 0.255, 0.260, 0.277, 0.273]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='ETH', marker='o')
plt.plot(x, y2, label='HOTEL', marker='o')
plt.plot(x, y3, label='UNIV', marker='o')
plt.plot(x, y4, label='ZARA1', marker='o')
plt.plot(x, y5, label='ZARA2', marker='o')

# Adding titles and labels
#plt.title('Multi-Line Plot for Different Datasets')
plt.xlabel(r"$\mathit{R}_{e}$", fontsize=16)
plt.ylabel('FDE', fontsize=16)
plt.legend(fontsize=16)  # Show legend

# Show the plot
plt.grid(True)

# 调整横轴和纵轴刻度的字体大小
plt.xticks(fontsize=16)  # 12可以根据需要调整
plt.yticks(fontsize=16)
plt.show()