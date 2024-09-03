import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def draw_density(data1):
    # 计算第一组数据的核密度估计
    kde1 = gaussian_kde(data1.T)


    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 生成第一组数据的二维网格
    x1, y1 = np.meshgrid(np.linspace(data1[:,0].min(), data1[:,0].max(), 100),
                         np.linspace(data1[:,1].min(), data1[:,1].max(), 100))
    grid_coords1 = np.vstack([x1.ravel(), y1.ravel()])

    # 计算第一组数据每个网格点的核密度估计值
    z1 = kde1(grid_coords1)
    z1 = z1.reshape(x1.shape)

    # 绘制第一组数据的密度图
    contour1 = ax.contourf(x1, y1, z1, cmap='Blues', levels=20, alpha=0.6)



    # 添加原始数据点的散点图（第一组数据）
    ax.scatter(data1[:, 0], data1[:, 1], color='k', s=5, alpha=0.5)

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Density Plot with KDE')


    # 显示图形
    plt.show()
