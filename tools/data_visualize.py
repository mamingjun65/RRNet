import os
import sys

sys.path.append(os.path.realpath('.'))
import numpy as np
import torch

from datasets import make_dataloader

import argparse
from configs import cfg
import os
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from draw_sec import draw_sector
import math
from draw_den import draw_density

def main():
    # 设置gpu和配置文件
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)  # cfg为配置文件
    cfg.merge_from_list(args.opts)
    os.environ['CUDA_VISIBLE_cfg.DEVICES'] = args.gpu


    # get dataloaders
    train_dataloader = make_dataloader(cfg, 'train')
    print('Dataloader built!')

    torch.set_printoptions(threshold=torch.inf)
    np.set_printoptions(threshold=1e6)


    for iters, batch in enumerate(tqdm(train_dataloader), start=1):
        x = np.round(batch['input_x_st'].numpy()[:,:,:2],decimals=3)
        y = np.round(batch['target_y_st'].numpy(),decimals=3)

        top_sequences_indices_x = find_linear_sequences(x, num_sequences=10)
        concatenated_array = np.concatenate((x[top_sequences_indices_x], y[top_sequences_indices_x]), axis=1)
        if iters==1:
            straight_trajectory = concatenated_array
        else:
            straight_trajectory=np.concatenate((straight_trajectory,concatenated_array),axis=0)

    straight_trajectory = np.round(straight_trajectory, decimals=3)


    rotated_data = np.array([rotate_sequence(seq) for seq in straight_trajectory])
    results=np.array([seq for seq in rotated_data if np.all(seq[6,:]!=0) and np.arctan2(math.fabs(seq[2,1]), math.fabs(seq[2,0]))<(math.pi/12)])
    print(results.shape)

    observation = results[:, :8, :]
    future = results[:, 8:, :]
    # 创建一个新的图形
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 10))

    # 绘制第一类点
    for i in range(observation.shape[0]):
        axes[0].scatter(observation[i, :, 0], observation[i, :, 1], color='blue', label='observation' if i == 0 else None)
    axes[0].set_title('observation')
    axes[0].set_xlabel('x/m')
    axes[0].set_ylabel('y/m')
    axes[0].set_xlim(-2,0)
    axes[0].set_ylim(-4.5, 4.5)
    axes[0].legend()
    # 绘制第二类点
    for i in range(future.shape[0]):
        axes[1].scatter(future[i, :, 0], future[i, :, 1], color='red', label='future' if i == 0 else None)
    axes[1].set_title('future')
    axes[1].set_xlabel('x/m')
    axes[1].set_ylabel('y/m')
    axes[1].set_xlim(-3.5, 8.5)
    axes[1].set_ylim(-4.5, 4.5)
    axes[1].legend()

    for i in range(observation.shape[0]):
        axes[2].scatter(observation[i, :, 0], observation[i, :, 1], color='blue', label='observation' if i == 0 else None)
    for i in range(future.shape[0]):
        axes[2].scatter(future[i, :, 0], future[i, :, 1], color='red', label='future' if i == 0 else None)
    axes[2].set_title('observation+future')
    axes[2].set_xlabel('x/m')
    axes[2].set_ylabel('y/m')
    axes[2].legend()


    # 显示图形
    plt.grid(True)
    plt.show()
    observation2 = observation.reshape(-1, 2)

    print(observation2.shape)

    draw_sector(observation2,np.array([seq[-1,:] for seq in future]).reshape(-1,2))
    draw_density(np.array([seq[-1,:] for seq in future]).reshape(-1,2))

def is_linearly_distributed(points):
    # 提取x和y坐标
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    # 线性回归拟合
    reg = LinearRegression().fit(X, y)

    # 预测y值
    y_pred = reg.predict(X)

    # 计算均方误差
    mse = mean_squared_error(y, y_pred)

    return mse


def find_linear_sequences(data, num_sequences=10):
    linearities = []

    for i in range(data.shape[0]):
        sequence = data[i, -5:, :]  # 提取后5个点的前2个特征
        mse = is_linearly_distributed(sequence)
        linearities.append((i, mse))

    # 根据线性度排序，选择前num_sequences个序列
    linearities.sort(key=lambda x: x[1])
    top_sequences = [index for index, mse in linearities[:num_sequences]]

    return top_sequences

def rotate_sequence(sequence):
    # 第8个点是原点
    origin = sequence[7]

    assert np.all(origin == 0), "The 8th point is not the origin."

    # 第7个点的坐标
    point_7 = sequence[6]

    # 计算旋转角度，使得第7个点的y坐标为0，且x坐标小于0
    angle = np.arctan2(point_7[1], point_7[0])

    # 构建旋转矩阵
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # 旋转序列
    rotated_sequence = sequence @ rotation_matrix.T

    # 检查第7个点的x坐标是否小于0，否则旋转180度
    if rotated_sequence[6, 0] > 0:
        rotation_matrix_180 = np.array([[-1, 0], [0, -1]])
        rotated_sequence = rotated_sequence @ rotation_matrix_180.T

    return rotated_sequence

def draw(data,label_pred,centers,k,axes3):
    cmap = plt.get_cmap('viridis')

    cluster = []
    for i in range(k):
        cluster.append(data[label_pred == i])


    for i in range(k):
        axes3.scatter(cluster[i][:, 0], cluster[i][:, 1], color=cmap(i / k))

    for i in range(k):
        axes3.scatter(centers[i][0], centers[i][1], color=cmap(i / k), marker='*')

    axes3.set_title("k-means")
    axes3.set_xlabel('x/m')
    axes3.set_ylabel('y/m')



if __name__ == '__main__':
    main()



