import torch
import torch.nn.functional as F
import pdb

def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()

def cvae_loss_revised(pred_goal, revised_goal, pred_traj, target, best_of_many=True):
        '''
        CVAE loss use best-of-many
        Params:
            pred_goal: (Batch, K, pred_dim)
            pred_traj: (Batch, T, K, pred_dim)
            target: (Batch, T, pred_dim)
            best_of_many: whether use best of many loss or not
        Returns:

        '''
        K = pred_goal.shape[1]  #60
        K2 = revised_goal.shape[1]  #20
        target1 = target.unsqueeze(2).repeat(1, 1, K, 1) #(128,12,60,2)
        target2 = target.unsqueeze(2).repeat(1, 1, K2, 1)  # (128,12,20,2)
    
        # select bom based on  goal_rmse
        goal_rmse = torch.sqrt(torch.sum((pred_goal - target1[:, -1, :, :])**2, dim=-1)) #(128,60)
        revised_goal_rmse = torch.sqrt(torch.sum((revised_goal - target2[:, -1, :, :]) ** 2, dim=-1))  # (128,20)
        traj_rmse = torch.sqrt(torch.sum((pred_traj - target2)**2, dim=-1)).sum(dim=1)   #(128,20)
        #生成20个终点，及其对应轨迹，挑选最接近的终点来确定要最小化的损失值（最接近终点+其对应轨迹）
        if best_of_many:
            best_idx = torch.argmin(goal_rmse, dim=1)   #tensor:128行，每行存着最佳（终点生成）的索引
            loss_goal = goal_rmse[range(len(best_idx)), best_idx].mean()

            best_revised_idx = torch.argmin(revised_goal_rmse, dim=1)   #tensor:128行，每行存着最佳（终点生成）的索引
            loss_revised_goal = revised_goal_rmse[range(len(best_revised_idx)), best_revised_idx].mean()

            loss_traj = traj_rmse[range(len(best_revised_idx)), best_revised_idx].mean()
        else:
            loss_goal = goal_rmse.mean()
            loss_traj = traj_rmse.mean()
        
        return loss_goal, loss_revised_goal, loss_traj
        
def bom_traj_loss(pred, target):
    '''
    pred: (B, T, K, dim)
    target: (B, T, dim)
    '''
    K = pred.shape[2]
    target = target.unsqueeze(2).repeat(1, 1, K, 1)
    traj_rmse = torch.sqrt(torch.sum((pred - target)**2, dim=-1)).sum(dim=1)
    best_idx = torch.argmin(traj_rmse, dim=1)
    loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
    return loss_traj

def fol_rmse(x_true, x_pred):
    '''
    Params:
        x_pred: (batch, T, pred_dim) or (batch, T, K, pred_dim)
        x_true: (batch, T, pred_dim) or (batch, T, K, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''

    L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=-1))#
    L2_diff = torch.sum(L2_diff, dim=-1).mean()

    return L2_diff


def conidence_loss(con_traj, pred_goal, target):  #[128,20,1] [128,20,2]  [128,2]
    K = pred_goal.shape[1]
    target = target.unsqueeze(1).repeat(1, K, 1)  # (128,20,2)
    goal_distance = torch.sqrt(torch.sum((pred_goal - target) ** 2, dim=-1))    # [128,20]
    con_traj=con_traj.view(-1)
    goal_distance=goal_distance.view(-1)
    processed_goal_distance = custom_function(goal_distance)

    # 计算处理后的goal_distance与con_traj的差值
    diff = torch.abs(processed_goal_distance - con_traj)

    # 计算差值的均值
    mean_diff = torch.mean(diff)
    return mean_diff

# 定义自定义函数
def custom_function(x):
    return 1 / (1 + x)



def cvae_loss(pred_goal, revised_goal, pred_traj, target, best_of_many=True):
    '''
    CVAE loss use best-of-many
    Params:
        pred_goal: (Batch, K, pred_dim)
        pred_traj: (Batch, T, K, pred_dim)
        target: (Batch, T, pred_dim)
        best_of_many: whether use best of many loss or not
    Returns:

    '''

    K = pred_goal.shape[1]  # 60
    K2 = revised_goal.shape[1]  # 20
    target1 = target.unsqueeze(2).repeat(1, 1, K, 1)  # (128,12,60,2)
    target2 = target.unsqueeze(2).repeat(1, 1, K2, 1)  # (128,12,20,2)

    # select bom based on  goal_rmse
    goal_rmse = torch.sqrt(torch.sum((pred_goal - target1[:, -1, :, :]) ** 2, dim=-1))  # (128,20)
    revised_goal_rmse = torch.sqrt(torch.sum((revised_goal - target2[:, -1, :, :]) ** 2, dim=-1))  # (128,20)
    traj_rmse = torch.sqrt(torch.sum((pred_traj - target2) ** 2, dim=-1)).sum(dim=1)  # (128,20)
    # 生成20个终点，及其对应轨迹，挑选最接近的终点来确定要最小化的损失值（最接近终点+其对应轨迹）
    if best_of_many:
        best_idx = torch.argmin(goal_rmse, dim=1)  # tensor:128行，每行存着最佳（终点生成）的索引
        loss_goal = goal_rmse[range(len(best_idx)), best_idx].mean()


        best_revised_idx = torch.argmin(revised_goal_rmse, dim=1)  # tensor:128行，每行存着最佳（终点生成）的索引
        loss_traj = traj_rmse[range(len(best_revised_idx)), best_revised_idx].mean()
    else:
        loss_goal = goal_rmse.mean()
        loss_traj = traj_rmse.mean()

    return loss_goal, loss_traj