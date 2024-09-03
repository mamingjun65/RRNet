'''
Defined classes:
    class BiTraPNP()
Some utilities are cited from Trajectron++
'''
import sys
import numpy as np
import copy
from collections import defaultdict
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn

from .latent_net import CategoricalLatent, kl_q_p
from rrnet.layers.loss import cvae_loss, mutual_inf_mc, conidence_loss, cvae_loss_revised
import logging
from torch.distributions import Normal
import time

# 配置日志记录


# 观测8×0.4=3.2s，预测12×0.4=4.8s

class RRNet(nn.Module):
    def __init__(self, cfg, dataset_name=None): # cfg配置文件
        super(RRNet, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.K = self.cfg.K
        self.K2 = self.cfg.K2
        self.STO = self.cfg.STO
        self.mu = 0.0
        self.sigma = 1.0
        self.attention= self.cfg.ATTENTION
        self.param_scheduler = None
        # encoder
        self.box_embed = nn.Sequential(nn.Linear(self.cfg.GLOBAL_INPUT_DIM, self.cfg.INPUT_EMBED_SIZE), # 6,256
                                        nn.ReLU())
        self.traj_embed = nn.Sequential(nn.Linear(self.cfg.GLOBAL_INPUT_DIM, self.cfg.INPUT_EMBED_SIZE), # 6,256
                                        nn.ReLU())
        self.box_encoder = nn.GRU(input_size=self.cfg.INPUT_EMBED_SIZE, # 256
                                hidden_size=self.cfg.ENC_HIDDEN_SIZE, # 256
                                batch_first=True)
        self.box_encoder_traj = nn.GRU(input_size=self.cfg.INPUT_EMBED_SIZE, # 256
                                hidden_size=self.cfg.ENC_HIDDEN_SIZE, # 256
                                batch_first=True)

        #encoder for future trajectory
        # self.gt_goal_encoder = nn.Sequential(nn.Linear(self.cfg.DEC_OUTPUT_DIM, 16),  #2
        #                                         nn.ReLU(),
        #                                         nn.Linear(16, 32),
        #                                         nn.ReLU(),
        #                                         nn.Linear(32, self.cfg.GOAL_HIDDEN_SIZE),
        #                                         nn.ReLU())
        self.node_future_encoder_h = nn.Linear(self.cfg.GLOBAL_INPUT_DIM, 32)   # 6
        self.gt_goal_encoder = nn.GRU(input_size=self.cfg.DEC_OUTPUT_DIM,   # 2
                                        hidden_size=32,
                                        bidirectional=True,
                                        batch_first=True)
        
            
        self.hidden_size = self.cfg.ENC_HIDDEN_SIZE        #256
        self.p_z_x = nn.Sequential(nn.Linear(self.hidden_size,  #256
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.cfg.LATENT_DIM*2))   #256——>32×2
        # posterior
        self.q_z_xy = nn.Sequential(nn.Linear(self.hidden_size + self.cfg.GOAL_HIDDEN_SIZE,     #256+64(configs/defaults)
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.cfg.LATENT_DIM*2))      #(256+64)——>32*2

        # goal predictor
        self.goal_decoder = nn.Sequential(nn.Linear(self.hidden_size + self.cfg.LATENT_DIM,     #256+32
                                                    128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.cfg.DEC_OUTPUT_DIM))     #(256+32)——>2
        #  add bidirectional predictor
        self.dec_init_hidden_size = self.hidden_size + self.cfg.LATENT_DIM if self.cfg.DEC_WITH_Z else self.hidden_size
                                        # 256
        self.enc_h_to_forward_h = nn.Sequential(nn.Linear( self.dec_init_hidden_size,   #256
                                                      self.cfg.DEC_HIDDEN_SIZE),        #256
                                                nn.ReLU(),
                                                )
        self.traj_dec_input_forward = nn.Sequential(nn.Linear(self.cfg.DEC_HIDDEN_SIZE, #256
                                                              self.cfg.DEC_INPUT_SIZE), #256
                                                    nn.ReLU(),
                                                    )
        self.traj_dec_forward = nn.GRUCell(input_size=self.cfg.DEC_INPUT_SIZE,      #256
                                            hidden_size=self.cfg.DEC_HIDDEN_SIZE)   #256
        
        self.enc_h_to_back_h = nn.Sequential(nn.Linear( self.dec_init_hidden_size,  #256
                                                      self.cfg.DEC_HIDDEN_SIZE),    #256
                                            nn.ReLU(),
                                            )
        
        self.traj_dec_input_backward = nn.Sequential(nn.Linear(self.cfg.DEC_OUTPUT_DIM, # 2 or 4 
                                                                self.cfg.DEC_INPUT_SIZE),   #256
                                                        nn.ReLU(),
                                                        )
        self.traj_dec_backward = nn.GRUCell(input_size=self.cfg.DEC_INPUT_SIZE,     #256
                                            hidden_size=self.cfg.DEC_HIDDEN_SIZE)   #256

        self.traj_output = nn.Linear(self.cfg.DEC_HIDDEN_SIZE * 2, # merged forward and backward 
                                     self.cfg.DEC_OUTPUT_DIM)       #2
        self.confidence_encoder = nn.GRU(input_size=self.cfg.DEC_OUTPUT_DIM,   # 2
                                        hidden_size=32,
                                        bidirectional=True,
                                        batch_first=True)

        # goal predictor
        self.confidence_decoder = nn.Sequential(nn.Linear(64,     #256+32
                                                    32),
                                            nn.ReLU(),
                                            nn.Linear(32, 16),
                                            nn.ReLU(),
                                            nn.Linear(16, 1))     #(256+32)——>2

        self.partial_encoder = nn.GRU(6, 256, batch_first=True)
        self.attention_gru= nn.GRU(256, 32, batch_first=True)
        self.goal_attention_decoder = nn.Sequential(nn.Linear(self.hidden_size + self.cfg.LATENT_DIM+32,     #256+32+32
                                                    128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.cfg.DEC_OUTPUT_DIM))
        self.cat_dec_h_and_attention=nn.Linear(256+32,
                                     256)
        self.query = nn.Linear(2, 32)
        self.key = nn.Linear(2, 32)
        self.value = nn.Linear(2, 32)
        self.attention_goal_revise=nn.Linear(32,
                                     2)
        # 将需要冻结的层放入ModuleList中
        self.parts_to_freeze_goal1 = nn.ModuleList([
            self.box_embed,
            self.box_encoder,
            self.node_future_encoder_h,
            self.gt_goal_encoder,
            self.p_z_x,
            self.q_z_xy,
            self.goal_decoder,
            self.confidence_encoder,
            self.confidence_decoder,
        ])
        self.parts_to_freeze_goal2 = nn.ModuleList([
            self.query,
            self.key,
            self.value,
            self.attention_goal_revise
        ])
    def gaussian_latent_net(self, enc_h, cur_state, target=None, z_mode=None):  # (编码h_x, 最后一个观测点input_x[:, -1, :],预测标签target_y, z_mode=False)
        # get mu, sigma
        # 1. sample z from piror
        z_mu_logvar_p = self.p_z_x(enc_h) #行为batch_size，列为32个均值、32个方差

        z_mu_p = z_mu_logvar_p[:, :self.cfg.LATENT_DIM]
        z_logvar_p = z_mu_logvar_p[:, self.cfg.LATENT_DIM:]
        if target is not None:
            # 2. sample z from posterior, for training only
            initial_h = self.node_future_encoder_h(cur_state)   # 6-->32
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)   #加一行0
            _, target_h = self.gt_goal_encoder(target, initial_h)  #预测标签双向gru，用当前点初始化前向gru，返回最后一层hidden——layer
            target_h = target_h.permute(1,0,2)  #换位
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])  #3维->2维

            target_h = F.dropout(target_h,
                                p=0.25,
                                training=self.training)

            z_mu_logvar_q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))   #特征维度拼接：history编码+（future+current）编码    256+32*2——>32*2
            z_mu_q = z_mu_logvar_q[:, :self.cfg.LATENT_DIM]
            z_logvar_q = z_mu_logvar_q[:, self.cfg.LATENT_DIM:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            # 3. compute KL(q_z_xy||p_z_x)
            KLD = 0.5 * ((z_logvar_p.exp()/z_logvar_q.exp()) + \
                        (z_mu_p - z_mu_q).pow(2)/z_logvar_p.exp() - \
                        1 + \
                        (z_logvar_q-z_logvar_p))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = 0.0

        # 4. Draw sample
        #K_samples = torch.randn(enc_h.shape[0], self.K, self.cfg.LATENT_DIM).cuda()     #(128,20,32)
        with torch.set_grad_enabled(False):
            K_samples = torch.normal(self.mu, self.sigma, size = (enc_h.shape[0], self.K2, self.cfg.LATENT_DIM)).cuda()
        probability = self.reconstructed_probability(K_samples)
        Z_std = torch.exp(0.5 * Z_logvar)   #标准差
        Z = Z_mu.unsqueeze(1).repeat(1, self.K2, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, self.K2, 1)    #重参数化(128,20,32)

        if z_mode:
            Z = torch.cat((Z_mu.unsqueeze(1), Z), dim=1)
        return Z, KLD, probability

    def encode_variable_length_seqs(self, original_seqs, lower_indices=None, upper_indices=None, total_length=None,traj=False):
        '''
        take the input_x, pack it to remove NaN, embed, and run GRU
        '''
        # 一个batch中，一组8个点会出现缺失，因此要
        bs, tf = original_seqs.shape[:2] #[128,8,6]取出batch_size=128,timestep=8
        if lower_indices is None:   #tensor([0, 2, 0, 0,......,0, 1, 0, 0], dtype=torch.int32)
            lower_indices = torch.zeros(bs, dtype=torch.int)
        if upper_indices is None:   # 8-1=7
            upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
        if total_length is None:    #7+1=8
            total_length = max(upper_indices) + 1
        # This is done so that we can just pass in self.prediction_timesteps
        # (which we want to INCLUDE, so this will exclude the next timestep).
        inclusive_break_indices = upper_indices + 1     # 7+1=8
        pad_list = []
        length_per_batch = []
        for i, seq_len in enumerate(inclusive_break_indices): #batch中的每个，（128）
            pad_list.append(original_seqs[i, lower_indices[i]:seq_len]) #取出一个观测点序列的有效点，加入大列表pad_list
            length_per_batch.append(seq_len-lower_indices[i]) #记录一个batch里每个有效点序列的长度
        # pad_list里有128个(seq_len-lower_indices[i])×6维张量
        # 1. embed and convert back to pad_list
        if traj:
            x = self.traj_embed(torch.cat(pad_list, dim=0))
        else:
            x = self.box_embed(torch.cat(pad_list, dim=0))# sum×6——>sum×256
        pad_list = torch.split(x, length_per_batch) #恢复为128个(seq_len-lower_indices[i])×256维张量

        # 2. run temporal
        packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False) # 打包成特殊类
        if traj:
            packed_output, h_x = self.box_encoder_traj(packed_seqs)
        else:
            packed_output, h_x = self.box_encoder(packed_seqs)

        # pad zeros to the end so that the last non zero value
        output, _ = rnn.pad_packed_sequence(packed_output,      #解包、最后补0
                                            batch_first=True,
                                            total_length=total_length)


        return output, h_x

    def encoder(self, x, first_history_indices=None, traj=False):
        '''
        x: encoder inputs
        '''
        outputs, _ = self.encode_variable_length_seqs(x,
                                                      lower_indices=first_history_indices,traj=traj)  #[128,8,256]
        outputs = F.dropout(outputs,
                            p=self.cfg.DROPOUT,  # 0.25
                            training=self.training)
        if first_history_indices is not None:
            last_index_per_sequence = -(first_history_indices + 1).to(torch.long)
            return outputs[torch.arange(first_history_indices.shape[0]).to(torch.long), last_index_per_sequence]
        else:
            # if no first_history_indices, all sequences are full length
            return outputs[:, -1, :]

    def forward(self, input_x,
                target_y=None,
                neighbors_st=None,
                adjacency=None,
                z_mode=False,
                cur_pos=None,
                first_history_indices=None, stat_y=None):


        '''
        Params:
            input_x: (batch_size, segment_len, dim =2 or 4)
            target_y: (batch_size, pred_len, dim = 2 or 4)
        Returns:
            pred_traj: (batch_size, K, pred_len, 2 or 4)
        '''
        if target_y is None:
            test=True
        else:
            test=False

        #input_x:[128,8,6]      target_y:[128,12,2]
        gt_goal = target_y[:, -1] if target_y is not None else None  # 预测轨迹的最后一个点为目标终点
        cur_pos = input_x[:, -1, :] if cur_pos is None else cur_pos # 全局观测轨迹的最后一个点为当前位置
        batch_size, seg_len, _ = input_x.shape
        # 1. encoder
        start_time1 = time.time()
        h_x = self.encoder(input_x, first_history_indices)  # [128,256]:1个序列中的8个点经过gru后的最后一个output=[256]

        # 2-3. latent net and goal decoder
        Z, KLD, reconstruct_prob = self.gaussian_latent_net(h_x, input_x[:, -1, :], target_y, z_mode=False)   #返回采样和kl散度
        enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1) #(128,20,256+32)/(128,60,256+32)

        # coarse goal+input_hidden_info
        pred_goal = self.goal_decoder(enc_h_and_z)  #->2
        dec_h = enc_h_and_z if self.cfg.DEC_WITH_Z else h_x #false


        # confidence-----------start
        con_input_x = input_x.unsqueeze(1).repeat(1, pred_goal.shape[1], 1, 1)[:,:,:,:2] #(128,20,8,2)
        # 将 goal 的形状变换为 [128, 20, 1, 2]
        con_pred_goal = pred_goal.unsqueeze(2)

        # 沿着维度 1 拼接
        con_input = torch.cat((con_input_x, con_pred_goal), dim=2)  #[128,20,9,2]
        con_input = con_input.view(-1, con_input.shape[2],con_input.shape[3])  # (128×20,9,2)

        _, con_hn = self.confidence_encoder(con_input)  #[1,128×20,64]
        con_hn = con_hn.permute(1, 0, 2)  # 换位
        con_hn = con_hn.reshape(-1, con_hn.shape[1] * con_hn.shape[2])  # 3维->2维    [128×20,64]

        con_traj=self.confidence_decoder(con_hn)    #[128×20,1]
        con_traj_reshaped = con_traj.view(-1, pred_goal.shape[1], 1)  # (128, 20, 1)
        end_time1 = time.time()
        con_traj_sm = F.softmax(con_traj_reshaped, dim=1)
        # confidence-----------finish

        # 似然概率
        reconstruct_prob_sm = F.softmax(reconstruct_prob, dim=-1)

        start_time2 = time.time()
        if self.attention:
            revised_goal = self.goal_revise_attention(pred_goal, con_traj_reshaped,test, self.K)  # (128,20,2) (128,20,1)
        else:
            revised_goal, _=self.goal_revise(pred_goal, con_traj_reshaped, self.K)
        # trajectory network
        # h_x_traj = self.encoder(input_x, first_history_indices,traj=True)  # [128,256]:1个序列中的8个点经过gru后的最后一个output=[256]




        pred_traj = self.pred_future_traj(dec_h, revised_goal)  # (128,12,20,2)
        end_time2 = time.time()
        # 5. compute loss
        if target_y is not None:
            # train and val
            if self.attention:
                loss_goal, loss_revised_goal, loss_traj = cvae_loss_revised(pred_goal,
                                             revised_goal,
                                            pred_traj, 
                                            target_y, 
                                            best_of_many=self.cfg.BEST_OF_MANY
                                            )
                loss_confidence = conidence_loss(con_traj, pred_goal, target_y[:,-1,:])
                loss_dict = {'loss_goal': loss_goal, 'loss_revised_goal': loss_revised_goal, 'loss_traj': loss_traj,
                             'loss_kld': KLD, 'loss_confidence': loss_confidence}

            else:
                loss_goal, loss_traj = cvae_loss(pred_goal,
                                                 revised_goal,
                                                pred_traj,
                                                target_y,
                                                best_of_many=self.cfg.BEST_OF_MANY
                                                )
                loss_confidence = conidence_loss(con_traj, pred_goal, target_y[:, -1, :])
                loss_dict = {'loss_goal': loss_goal, 'loss_traj': loss_traj,
                         'loss_kld': KLD, 'loss_confidence': loss_confidence}
        else:
            # test
            loss_dict = {}
            #pred_traj=self.traj_revise(pred_traj,con_traj_sm)
        prob_stat={}
        if stat_y is not None:
            prob_stat=self.stat(stat_y[:,-1,:], pred_goal, con_traj_sm, reconstruct_prob_sm)


        total_time = end_time2 - start_time2 + end_time1 - start_time1

        return pred_goal, pred_traj, con_traj_sm, reconstruct_prob_sm, loss_dict, None, None, revised_goal, prob_stat, total_time  #(128,20,2) (128,12,20,2) (128,20,1) (128,20)

    def pred_future_traj(self, dec_h, G):   #(128,256)  (128,20,2)
        '''
        use a bidirectional GRU decoder to plan the path.
        Params:
            dec_h: (Batch, hidden_dim) if not using Z in decoding, otherwise (Batch, K, dim) 
            G: (Batch, K, pred_dim)
        Returns:
            backward_outputs: (Batch, T, K, pred_dim)
        '''
        pred_len = self.cfg.PRED_LEN    #12

        K = G.shape[1]
        # 1. run forward
        forward_outputs = []
        forward_h = self.enc_h_to_forward_h(dec_h)  #256
        if len(forward_h.shape) == 2:   #2
            forward_h = forward_h.unsqueeze(1).repeat(1, K, 1)  #(128,20,256)
        forward_h = forward_h.view(-1, forward_h.shape[-1]) #(128×20,256)
        forward_input = self.traj_dec_input_forward(forward_h)
        for t in range(pred_len): # the last step is the goal, no need to predict
            forward_h = self.traj_dec_forward(forward_input, forward_h)
            forward_input = self.traj_dec_input_forward(forward_h)
            forward_outputs.append(forward_h)
        
        forward_outputs = torch.stack(forward_outputs, dim=1)   #(20×128,12,256)
        
        # 2. run backward on all samples
        backward_outputs = []
        backward_h = self.enc_h_to_back_h(dec_h)
        if len(dec_h.shape) == 2:
            backward_h = backward_h.unsqueeze(1).repeat(1, K, 1)
        backward_h = backward_h.view(-1, backward_h.shape[-1])  #(20×128,256)
        backward_input = self.traj_dec_input_backward(G)#torch.cat([G]) (128,20,256)
        backward_input = backward_input.view(-1, backward_input.shape[-1])  #(20×128,256)
        
        for t in range(pred_len-1, -1, -1):
            backward_h = self.traj_dec_backward(backward_input, backward_h)
            output = self.traj_output(torch.cat([backward_h, forward_outputs[:, t]], dim=-1))   #(20×128,256+256->2)
            backward_input = self.traj_dec_input_backward(output)
            backward_outputs.append(output.view(-1, K, output.shape[-1]))   #(128,20,2)
        
        # inverse because this is backward 
        backward_outputs = backward_outputs[::-1]
        backward_outputs = torch.stack(backward_outputs, dim=1) #(128,12,20,2)
        
        return backward_outputs



    def traj_revise(self, pred_traj, con_traj): #[128,12,20,2]  [128,20,1]
        # 选择概率最高的3条预测轨迹和概率最低的11条预测轨迹
        top5_idx = torch.topk(con_traj, 5, dim=1, largest=True).indices.squeeze(-1) #[128,5]
        top17_idx = torch.topk(con_traj, 17, dim=1, largest=True).indices.squeeze(-1)  # [128,17]
        bottom3_idx = torch.topk(con_traj, 3, dim=1, largest=False).indices.squeeze(-1)   #[128,11]

        batch_size = pred_traj.shape[0]

        # 根据选择的索引从 pred_traj 中提取相应的轨迹
        top5_traj = torch.stack([pred_traj[i, :, top5_idx[i]] for i in range(batch_size)])  #[128,12,5,2]
        top17_traj = torch.stack([pred_traj[i, :, top17_idx[i]] for i in range(batch_size)])  # [128,12,17,2]
        bottom3_traj = torch.stack([pred_traj[i, :, bottom3_idx[i]] for i in range(batch_size)])  #[128,12,3,2]


        # 计算每个batch的3个概率最高的预测轨迹的最后一个时间步的平均中心位置
        mu = top5_traj[:, -1, :, :].mean(dim=1) #[128,2]

        # 对概率最低的11条预测轨迹进行调整
        for i in range(batch_size):
            for j in range(3):
                last_pos = bottom3_traj[i, -1, j]  #[2]
                p, q = mu[i]
                s, t = last_pos
                if abs(s - p) > abs(t - q):
                    offset_direction = torch.tensor([(s - p) / abs(s - p), -t+q], device='cuda')
                else:
                    offset_direction = torch.tensor([-s+p, (t - q) / abs(t - q)], device='cuda')

                offsets = offset_direction * (torch.arange(1, 13, device='cuda').unsqueeze(-1) / 12.0)
                bottom3_traj[i, :, j] += offsets

        # 将基准预测轨迹和新的预测轨迹结合在一起
        result_traj = torch.cat((top17_traj,bottom3_traj), dim=2)  # 形状 [128, 12, 20, 2]

        return result_traj


    def attention_encoder(self, input_x,enc_h_and_z):
        batch_size, seq_len, _ = input_x.size()
        encodings = []
        for i in range(seq_len):
            _, encoding = self.partial_encoder(input_x[:, i:, :])   #[1,128,256]
            encoding = encoding.squeeze(0)  # 去掉第一个维度，形状变为 [128, 256]
            encodings.append(encoding)

        encodings = torch.stack(encodings, dim=1)   #[128,8,256]



        # 计算 query, key 和 value，实际上在简化版本中它们可以是相同的
        query = encodings
        key = encodings
        value = encodings

        # 计算注意力权重
        # QK^T
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(key.shape[-1], dtype=torch.float32))

        # 对每个时间步应用 softmax
        attention_weights = F.softmax(scores, dim=-1)

        # 使用注意力权重加权求和 value
        attention_output = torch.matmul(attention_weights, value)   # [128, 8, 256]
        _, attention_h=self.attention_gru(attention_output)  # [128,32]
        attention_h = attention_h.squeeze(0)
        enc_h_and_z_and_attention = torch.cat([attention_h.unsqueeze(1).repeat(1, enc_h_and_z.shape[1], 1), enc_h_and_z], dim=-1)  # (128,20,256+32+32)
        attention_goal=self.goal_attention_decoder(enc_h_and_z_and_attention)
        return attention_goal, attention_h

    def reconstructed_probability(self,sample):
        recon_dist = Normal(self.mu, self.sigma)
        p = recon_dist.log_prob(sample).exp().mean(dim=-1)  # [batch_size, K]
        return p

    def goal_revise_attention(self, pred_goal, prob, test, K):   # (128,20,2) (128,20,1)
        top20_goal, top20_prob = self.goal_revise(pred_goal, prob, K)  # (128,20,2) (128,20,1)
        #input = torch.cat((top20_goal*0.3, top20_prob), dim=-1)
        input = top20_goal*0.3
        Q = self.query(input)
        K = self.key(input)
        V = self.value(input)


        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(32, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)

        # 使用注意力权重加权求和 value
        attention_output = torch.matmul(attention_weights, V)   #[128,20,32]
        revised_pred_goal_bias=self.attention_goal_revise(attention_output)
        if test:
            revised_pred_goal = top20_goal + revised_pred_goal_bias * self.STO
        else:
            revised_pred_goal = top20_goal + revised_pred_goal_bias

        return revised_pred_goal

    def goal_revise(self,pred_goal, prob, K):  # (128,20,2) (128,20,1) 第二梯队取两个，总的取6个，3*6+2
        # 选择概率最高的3条预测轨迹和概率最低的11条预测轨迹
        top20_prob, top20_idx = torch.topk(prob, K, dim=1, largest=True)  # [128,6]
        top20_idx = top20_idx.squeeze(-1)
        batch_size = pred_goal.shape[0]

        # 根据选择的索引从 pred_traj 中提取相应的轨迹
        top20_goal = torch.stack([pred_goal[i, top20_idx[i]] for i in range(batch_size)])  # [128,20,2]

        return top20_goal, top20_prob

    def stat(self, gt_goal, pred_goal, con_traj_sm, reconstruct_prob_sm):  #[128,2],[128,20,2],[128,20,1],[128,20]
        con_traj_sm=con_traj_sm.squeeze(-1)
        prob1=con_traj_sm
        prob2=reconstruct_prob_sm
        batchsize = gt_goal.size(0)

        # 计算距离真实点最近的点
        distances = torch.norm(pred_goal - gt_goal[:, None, :], dim=2)
        min_distance_indices = torch.argmin(distances, dim=1)  # shape: [batchsize]

        # 找到最高的概率
        max_prob1_indices = torch.argmax(prob1, dim=1)  # shape: [batchsize]
        max_prob2_indices = torch.argmax(prob2, dim=1)  # shape: [batchsize]

        # 计算prob1_num和prob2_num
        prob1_num = torch.sum(min_distance_indices == max_prob1_indices).item()
        prob2_num = torch.sum(min_distance_indices == max_prob2_indices).item()
        prob_stat={"data_num": batchsize, "recon_num": prob2_num, "prob_num": prob1_num}
        return prob_stat

