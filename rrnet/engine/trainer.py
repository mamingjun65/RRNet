import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rrnet.utils.visualization import Visualizer
from rrnet.utils.box_utils import cxcywh_to_x1y1x2y2
from rrnet.utils.dataset_utils import restore
from .evaluate import evaluate_multimodal, compute_kde_nll
from .utils import print_info, viz_results, post_process

from tqdm import tqdm
import pickle as pkl
import pdb
import logging
import matplotlib.pyplot as plt
type_result = 0


def do_train(cfg, epoch, model, optimizer, dataloader, device, logger=None, lr_scheduler=None):
    model.train()
    max_iters = len(dataloader)

    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')

    torch.set_printoptions(threshold=torch.inf)
    with torch.set_grad_enabled(True):
        for iters, batch in enumerate(tqdm(dataloader), start=1):

            np.set_printoptions(threshold=1e6)

            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()

            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
                input_y = batch['target_y_st'].to(device)
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None

            pred_goal, pred_traj, con_traj, _, loss_dict, dist_goal, dist_traj, _, prob_stat, _ = model(input_x,
                                                                                          input_y,
                                                                                          neighbors_st=neighbors_st,
                                                                                          adjacency=adjacency,
                                                                                          cur_pos=X_global[:, -1,
                                                                                                  :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                                          first_history_indices=first_history_indices)



            if cfg.MODEL.LATENT_DIST == 'categorical':  # gaussian
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + \
                       loss_dict['loss_confidence'] + \
                       model.param_scheduler.kld_weight * loss_dict['loss_kld'] - \
                       1. * loss_dict['mutual_info_p']
            else:
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + \
                       loss_dict['loss_confidence'] + \
                       model.param_scheduler.kld_weight * loss_dict['loss_kld']
                if cfg.MODEL.ATTENTION:
                    loss+=loss_dict['loss_revised_goal']
            model.param_scheduler.step()
            loss_dict = {k: v.item() for k, v in loss_dict.items()}
            loss_dict['lr'] = optimizer.param_groups[0]['lr']
            # optimize
            optimizer.zero_grad()  # avoid gradient accumulate from loss.backward()
            loss.backward()

            # loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            if cfg.SOLVER.scheduler == 'exp':
                lr_scheduler.step()
            # 打印损失
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info(epoch, model, optimizer, loss_dict, logger, cfg)
            # 轨迹可视化
            if cfg.VISUALIZE and iters % max(int(len(dataloader) / 5), 1) == 0:
                ret = post_process(cfg, X_global, y_global, pred_traj, con_traj, pred_goal=pred_goal,
                                   dist_goal=dist_goal)
                X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal, con_traj = ret
                viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj,
                            bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_train')




def do_val(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    loss_goal_val = 0.0
    loss_revised_goal_val = 0.0
    loss_traj_val = 0.0
    loss_KLD_val = 0.0
    loss_con_val = 0.0
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            input_y = batch['target_y_st'].to(device)

            img_path = batch['cur_image_file']
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None

            pred_goal, pred_traj, con_traj, _, loss_dict, _, _, _, _, _ = model(input_x,
                                                                          input_y,
                                                                          neighbors_st=neighbors_st,
                                                                          adjacency=adjacency,
                                                                          cur_pos=X_global[:, -1,
                                                                                  :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                          first_history_indices=first_history_indices)

            # compute loss
            loss = loss_dict['loss_goal'] + loss_dict['loss_traj'] + loss_dict['loss_kld'] + loss_dict[
                'loss_confidence']
            if cfg.MODEL.ATTENTION:
                loss += loss_dict['loss_revised_goal']
                loss_revised_goal_val += loss_dict['loss_revised_goal'].item()
            loss_goal_val += loss_dict['loss_goal'].item()
            loss_traj_val += loss_dict['loss_traj'].item()
            loss_KLD_val += loss_dict['loss_kld'].item()
            loss_con_val += loss_dict['loss_confidence'].item()

    loss_goal_val /= (iters + 1)

    loss_traj_val /= (iters + 1)
    loss_KLD_val /= (iters + 1)
    loss_con_val /= (iters + 1)
    loss_val = loss_goal_val + loss_traj_val + loss_KLD_val + loss_con_val
    if cfg.MODEL.ATTENTION:

        loss_revised_goal_val /= (iters + 1)
        loss_val += loss_revised_goal_val
        info = "loss_val:{:.4f}, \
                loss_revised_val:{:.4f}, \
                loss_goal_val:{:.4f}, \
                loss_traj_val:{:.4f}, \
                loss_con_val:{:.4f}, \
                loss_kld_val:{:.4f}".format(loss_val, loss_revised_goal_val, loss_goal_val, loss_traj_val, loss_con_val,
                                            loss_KLD_val)

        if hasattr(logger, 'log_values'):
            logger.info(info)
            logger.log_values({'loss_val': loss_val,
                               'loss_revised_val': loss_revised_goal_val,
                               'loss_goal_val': loss_goal_val,
                               'loss_traj_val': loss_traj_val,
                               'loss_con_val': loss_con_val,
                               'loss_kld_val': loss_KLD_val})  # , step=epoch)
        else:
            print(info)
    else:
        info = "loss_val:{:.4f}, \
                        loss_goal_val:{:.4f}, \
                        loss_traj_val:{:.4f}, \
                        loss_con_val:{:.4f}, \
                        loss_kld_val:{:.4f}".format(loss_val, loss_goal_val, loss_traj_val,
                                                    loss_con_val, loss_KLD_val)

        if hasattr(logger, 'log_values'):
            logger.info(info)
            logger.log_values({'loss_val': loss_val,
                               'loss_goal_val': loss_goal_val,
                               'loss_traj_val': loss_traj_val,
                               'loss_con_val': loss_con_val,
                               'loss_kld_val': loss_KLD_val})  # , step=epoch)
        else:
            print(info)
    return loss_val


def inference(cfg, epoch, model, dataloader, device, logger=None, eval_kde_nll=False, test_mode=False):
    model.eval()
    all_img_paths = []
    all_X_globals = []
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_distributions = []
    all_timesteps = []
    data_num=0
    recon_num=0
    prob_num=0
    total_time=0

    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')
    np.set_printoptions(threshold=1e6)
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y']
            img_path = batch['cur_image_file']
            input_y = batch['target_y_st']

            resolution = batch['pred_resolution'].numpy()

            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None


            pred_goal, pred_traj, con_traj, reconstruct_prob_sm, _, dist_goal, dist_traj, revised_goal, prob_stat, time = model(input_x,
                                                                                                               neighbors_st=neighbors_st,
                                                                                                               adjacency=adjacency,
                                                                                                               z_mode=False,
                                                                                                               cur_pos=X_global[
                                                                                                                       :,
                                                                                                                       -1,
                                                                                                                       :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                                                               first_history_indices=first_history_indices, stat_y=input_y.to(device))

            data_num += prob_stat["data_num"]
            recon_num += prob_stat["recon_num"]
            prob_num += prob_stat["prob_num"]

            total_time += time
            # transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj, con_traj, pred_goal=pred_goal, dist_traj=dist_traj,
                               dist_goal=dist_goal)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal, con_traj = ret  # 格式变为numpy，其他不改变
            all_img_paths.extend(img_path)  # 无
            all_X_globals.append(X_global)
            all_pred_goals.append(pred_goal)
            all_pred_trajs.append(pred_traj)
            all_gt_goals.append(input_y[:, -1])
            all_gt_trajs.append(input_y)
            all_timesteps.append(batch['timestep'].numpy())
            if dist_traj is not None:
                all_distributions.append(dist_traj)
            else:
                all_distributions.append(dist_goal)
            if cfg.VISUALIZE and iters % max(int(len(dataloader) / 5), 1) == 0:
                viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj,
                            bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_test')


        recon_accu=recon_num/data_num
        prob_accu=prob_num/data_num
        time_cost = total_time/data_num
        print("recon_accu: "+str(recon_accu))
        print("prob_accu: " + str(prob_accu))
        print(f'Time cost: {(1E3 * time_cost):.2f}ms')
        # Evaluate
        all_X_globals = np.concatenate(all_X_globals, axis=0)
        all_pred_goals = np.concatenate(all_pred_goals, axis=0)
        all_pred_trajs = np.concatenate(all_pred_trajs, axis=0)
        all_gt_goals = np.concatenate(all_gt_goals, axis=0)
        all_gt_trajs = np.concatenate(all_gt_trajs, axis=0)
        all_timesteps = np.concatenate(all_timesteps, axis=0)
        if hasattr(all_distributions[0], 'mus'):
            distribution = model.GMM(torch.cat([d.input_log_pis for d in all_distributions], axis=0),
                                     torch.cat([d.mus for d in all_distributions], axis=0),
                                     torch.cat([d.log_sigmas for d in all_distributions], axis=0),
                                     torch.cat([d.corrs for d in all_distributions], axis=0))
        else:
            distribution = None
            # eval_pred_results = evaluate(all_pred_goals, all_gt_goals)
        mode = 'bbox' if all_gt_trajs.shape[-1] == 4 else 'point'
        eval_results = evaluate_multimodal(all_pred_trajs, all_gt_trajs, mode=mode, distribution=distribution,
                                           bbox_type=cfg.DATASET.BBOX_TYPE)
        for key, value in eval_results.items():
            info = "Testing prediction {}:{}".format(key, str(np.around(value, decimals=3)))
            if hasattr(logger, 'log_values'):
                logger.info(info)
            else:
                print(info)

        if hasattr(logger, 'log_values'):
            logger.log_values(eval_results)

        if test_mode:
            # save inputs, redictions and targets for test mode
            outputs = {'img_path': all_img_paths, 'X_global': all_X_globals, 'timestep': all_timesteps,
                       'pred_trajs': all_pred_trajs, 'gt_trajs': all_gt_trajs, 'distributions': distribution}

            if not os.path.exists(cfg.OUT_DIR):
                os.makedirs(cfg.OUT_DIR)
            output_file = os.path.join(cfg.OUT_DIR, '{}_{}.pkl'.format(cfg.MODEL.LATENT_DIST, cfg.DATASET.NAME))
            print("Writing outputs to: ", output_file)
            pkl.dump(outputs, open(output_file, 'wb'))

    # Mevaluate KDE NLL, since we sample 2000, need to use a smaller batchsize
    if eval_kde_nll:
        dataloader_params = {
            "batch_size": cfg.TEST.KDE_BATCH_SIZE,
            "shuffle": False,
            "num_workers": cfg.DATALOADER.NUM_WORKERS,
            "collate_fn": dataloader.collate_fn,
        }
        kde_nll_dataloader = DataLoader(dataloader.dataset, **dataloader_params)
        inference_kde_nll(cfg, epoch, model, kde_nll_dataloader, device, logger)




def inference_kde_nll(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_kde_nll = []
    all_per_step_kde_nll = []
    num_samples_K2 = model.K2
    model.K2 = 2000
    num_samples = model.K
    model.K = 2000
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y_st']
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            pred_goal, pred_traj, con_traj, _, _, _, _, _, _, _ = model(input_x,
                                                          neighbors_st=neighbors_st,
                                                          adjacency=adjacency,
                                                          z_mode=False,
                                                          cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                          first_history_indices=first_history_indices)
            '''
            plt.figure()
            plt.scatter(pred_traj[0,-1, :,0].detach().to('cpu').numpy(), pred_traj[0, -1,:,1].detach().to('cpu').numpy(), color='blue', marker='o')
            plt.scatter(y_global[0,-1,0].detach().to('cpu').numpy(), y_global[0,-1,1].detach().to('cpu').numpy(), color='red', marker='o')
            print(y_global[0,-1])
            plt.show()
            '''
            # transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj, con_traj, pred_goal=pred_goal, dist_traj=None, dist_goal=None)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal, con_traj = ret
            for i in range(len(pred_traj)):
                KDE_NLL, KDE_NLL_PER_STEP = compute_kde_nll(pred_traj[i:i + 1], y_global[i:i + 1])
                all_kde_nll.append(KDE_NLL)
                all_per_step_kde_nll.append(KDE_NLL_PER_STEP)
        KDE_NLL = np.array(all_kde_nll).mean()
        KDE_NLL_PER_STEP = np.stack(all_per_step_kde_nll, axis=0).mean(axis=0)
        # Evaluate
        Goal_NLL = KDE_NLL_PER_STEP[-1]
        nll_dict = {'KDE_NLL': KDE_NLL} if cfg.MODEL.LATENT_DIST == 'categorical' else {'KDE_NLL': KDE_NLL,
                                                                                        'Goal_NLL': Goal_NLL}
        info = "Testing prediction KDE_NLL:{:.4f}, per step NLL:{}".format(KDE_NLL, KDE_NLL_PER_STEP)
        if hasattr(logger, 'log_values'):
            logger.info(info)
        else:
            print(info)
        if hasattr(logger, 'log_values'):
            logger.log_values(nll_dict)

    # reset model.K back to 20
    model.K = num_samples
    model.K2 = num_samples_K2
    return KDE_NLL


