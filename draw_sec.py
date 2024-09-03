import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc

def draw_sector(points_o,points):
    # 圆心和半径
    center = (0, 0)
    radius = 8.5
    theta = 90
    # 设置角度范围
    angles = np.arange(-75, 75, 30)

    # 统计每个扇形区域内的散点数量
    counts = []
    for start_angle in angles:
        end_angle = start_angle + 30
        count = 0
        for point in points:
            angle = np.degrees(np.arctan2(point[1], point[0]))
            if start_angle <= angle < end_angle:
                count += 1
        counts.append(count)

    # 计算占比
    #total_points = sum(counts)
    total_points = points.shape[0]
    ratios = [count / total_points for count in counts]
    print(ratios)
    # 颜色渐变
    base_color = 'green'  # 基础颜色
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', base_color])  # 创建自定义渐变色彩映射

    # 绘制图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_aspect('equal')

    ax.scatter(points[:, 0], points[:, 1], color='lightcoral')
    ax.scatter(points_o[:, 0], points_o[:, 1], color='blue', alpha=0.3)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-10, 10)

    arc = Arc(center, 2 * radius, 2 * radius, angle=0, theta1=-theta, theta2=theta, edgecolor='black', facecolor='none')
    ax.add_patch(arc)  # 添加半圆到子图

    # 画径向线和扇形区域
    for i, (start_angle, end_angle) in enumerate(zip(angles, angles + [30])):
        # 画径向线
        x_start, y_start = center
        x_end = x_start + radius * np.cos(np.radians(start_angle))
        y_end = y_start + radius * np.sin(np.radians(start_angle))
        ax.plot([x_start, x_end], [y_start, y_end], color='black', linestyle='--')

        # 画扇形区域
        color = cmap(ratios[i])
        wedge = Wedge(center, radius, start_angle, end_angle, facecolor=color, alpha=0.5)
        ax.add_patch(wedge)

        # 标注占比
        angle_mid = (start_angle + end_angle) / 2
        x_text = x_start + ((radius+5) / 2) * np.cos(np.radians(angle_mid))
        y_text = y_start + ((radius+5) / 2) * np.sin(np.radians(angle_mid))

        print(ratios[i])
        ax.text(x_text, y_text, f'{ratios[i]:.2%}', ha='center', va='center',fontsize=20)

    x_end = x_start + radius * np.cos(np.radians(75))
    y_end = y_start + radius * np.sin(np.radians(75))
    ax.plot([x_start, x_end], [y_start, y_end], color='black', linestyle='--')

    # 圆心和半径
    center = (0, 0)
    radius = 8.5
    width = -1.5

    # 设置角度范围
    angles = np.arange(-75, 75, 5)

    # 统计每个扇形区域内的散点数量
    counts = []
    for start_angle in angles:
        end_angle = start_angle + 5
        count = 0
        for point in points:
            angle = np.degrees(np.arctan2(point[1], point[0]))
            if start_angle <= angle < end_angle:
                count += 1
        counts.append(count)

    # 计算占比
    count_max = max(counts)
    ratios = [count / count_max for count in counts]
    print(ratios)
    # 颜色渐变
    base_color = 'skyblue'  # 基础颜色
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', base_color])  # 创建自定义渐变色彩映射

    # 画径向线和扇形区域
    for i, (start_angle, end_angle) in enumerate(zip(angles, angles + [5])):
        # 画径向线
        if i % 6 == 0:
            radius2 = radius + 1.5
        else:
            radius2 = radius + 1
        x_start, y_start = center
        x_start2 = x_start + radius * np.cos(np.radians(start_angle))
        y_start2 = y_start + radius * np.sin(np.radians(start_angle))
        x_end = x_start + radius2 * np.cos(np.radians(start_angle))
        y_end = y_start + radius2 * np.sin(np.radians(start_angle))
        ax.plot([x_start2, x_end], [y_start2, y_end], color='grey', linestyle='--')

        # 画扇形区域
        color = cmap(ratios[i])
        wedge = Wedge(center, radius, start_angle, end_angle, facecolor=color, alpha=0.5, width=width)
        ax.add_patch(wedge)

    x_start, y_start = center
    x_start2 = x_start + radius * np.cos(np.radians(75))
    y_start2 = y_start + radius * np.sin(np.radians(75))
    x_end = x_start + (radius + 1.5) * np.cos(np.radians(75))
    y_end = y_start + (radius + 1.5) * np.sin(np.radians(75))
    ax.plot([x_start2, x_end], [y_start2, y_end], color='grey', linestyle='--')

    ax.set_title('final points distribution')

    plt.show()
