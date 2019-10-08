# author: @wangyunbo, @liubo

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from configs import *
from utils import *

#########################
# Training Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, s_t, a_t, r_t, s_tp1, done, obs, curr_ps, mean_state, hidden, cell, pf_sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s_t, a_t, r_t, s_tp1, done, obs, curr_ps, mean_state, hidden, cell, pf_sample)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, obs, curr_ps, mean_state, hidden, cell, pf_sample = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done, obs, curr_ps, mean_state, hidden, cell, pf_sample

    def __len__(self):
        return len(self.buffer)

#########################
# Planning Network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, mean_state, par_states):
        state = torch.cat((mean_state, par_states), -1)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + const)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def get_action(self, mean_state, par_states):
        a, log_prob, _ = self.sample(mean_state, par_states)
        return a, log_prob[0]

#########################
# Transition model
class DynamicNetwork(nn.Module):
    def __init__(self):
        super(DynamicNetwork, self).__init__()
        self.t_enc = nn.Sequential(
            nn.Linear(DIM_STATE + DIM_ACTION, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_STATE * 2)
        )

    def t_model(self, state, action):
        if len(action.shape) == 1:
            action = action.unsqueeze(0).repeat(state.shape[0], 1)
        x = torch.cat([state, action], -1)
        x = self.t_enc(x)
        mean = x[:, :DIM_STATE]
        std = x[:, DIM_STATE:].exp()
        delta = torch.randn_like(state) * std + mean
        next_state = state + action + delta
        return next_state


#########################
# Particle Proposer
class ProposerNetwork(nn.Module):
    def __init__(self):
        super(ProposerNetwork, self).__init__()
        self.dim = 64
        self.obs_encode = nn.Sequential(
            nn.Linear(DIM_OBS, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, self.dim),
            nn.ReLU()
        )
        self.p_net = nn.Sequential(
            nn.Linear(self.dim * 2, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, 2)
        )

    def forward(self, obs, num_par=NUM_PAR_PF):
        obs_enc = self.obs_encode(obs)  # (B, C)
        obs_enc = obs_enc.repeat(num_par, 1)  # (B * num_par, C)
        z = torch.randn_like(obs_enc)  # (B * num_par, C)
        x = torch.cat([obs_enc, z], -1)  # (B * num_par, 2C)
        proposal = self.p_net(x)  # [B * num_par, 2]
        return proposal

class MeasureNetwork(nn.Module):
    def __init__(self):
        super(MeasureNetwork, self).__init__()
        self.dim_m = 16
        self.obs_encode = nn.Sequential(
            nn.Linear(DIM_OBS, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(DIM_HIDDEN, DIM_LSTM_HIDDEN, NUM_LSTM_LAYER)
        self.lstm_out = nn.Sequential(
            nn.Linear(DIM_LSTM_HIDDEN, self.dim_m),
            nn.ReLU()
        )
        self.m_net = nn.Sequential(
            nn.Linear(self.dim_m + DIM_STATE, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, 1),
            nn.Sigmoid()
        )

    def m_model(self, state, obs, hidden, cell, num_par=NUM_PAR_PF):
        # state: (B * K, dim_s)
        # obs: (B, dim_s)
        obs_enc = self.obs_encode(obs)  # (batch, dim_m)
        x = obs_enc.unsqueeze(0)  # -> [1, batch_size, dim_obs]
        x, (h, c) = self.lstm(x, (hidden, cell))
        x = self.lstm_out(x[0])  # (batch, dim_m)
        x = x.repeat(num_par, 1)  # (batch * num_par, dim_m)
        x = torch.cat((x, state), -1)  # (batch * num_par, dim_m + 2)
        lik = self.m_net(x).view(-1, num_par)  # (batch, num_par)
        return lik, h, c

#########################
# Training Process
class DUAL_SMC:
    def __init__(self):
        self.replay_buffer = ReplayMemory(replay_buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.MSE_criterion = nn.MSELoss()
        self.BCE_criterion = nn.BCELoss()
        # Filtering
        self.dynamic_net = DynamicNetwork().to(device)
        self.measure_net = MeasureNetwork().to(device)
        self.pp_net = ProposerNetwork().to(device)
        self.dynamic_optimizer = Adam(self.dynamic_net.parameters(), lr=FIL_LR)
        self.measure_optimizer = Adam(self.measure_net.parameters(), lr=FIL_LR)
        self.pp_optimizer = Adam(self.pp_net.parameters(), lr=FIL_LR)
        # Planning
        self.critic = QNetwork(DIM_STATE, DIM_ACTION, DIM_HIDDEN).to(device=device)
        self.critic_optim = Adam(self.critic.parameters(), lr=PLA_LR)
        self.critic_target = QNetwork(DIM_STATE, DIM_ACTION, DIM_HIDDEN).to(device)
        hard_update(self.critic_target, self.critic)
        self.target_entropy = -torch.prod(torch.Tensor(DIM_ACTION).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = Adam([self.log_alpha], lr=PLA_LR)
        self.policy = GaussianPolicy(DIM_STATE * (NUM_PAR_SMC_INIT + 1), DIM_ACTION, DIM_HIDDEN).to(device)
        self.policy_optim = Adam(self.policy.parameters(), lr=PLA_LR)

    def save_model(self, path):
        stats = {}
        stats['p_net'] = self.policy.state_dict()
        stats['c_net'] = self.critic.state_dict()
        stats['d_net'] = self.dynamic_net.state_dict()
        stats['m_net'] = self.measure_net.state_dict()
        stats['pp_net'] = self.pp_net.state_dict()
        torch.save(stats, path)

    def load_model(self, path):
        stats = torch.load(path)
        # Filtering
        self.dynamic_net.load_state_dict(stats['d_net'])
        self.measure_net.load_state_dict(stats['m_net'])
        self.pp_net.load_state_dict(stats['pp_net'])
        # Planning
        self.policy.load_state_dict(stats['p_net'])
        self.critic.load_state_dict(stats['c_net'])
        self.dynamic_net.load_state_dict(stats['d_net'])

    def get_mean_state(self, state, weight):
        if len(state.shape) == 2:
            # states: [num_particles, dim_state]
            # weights: [num_particles]
            state = torch.FloatTensor(state).to(device)
            weight = weight.unsqueeze(1).to(device)
            mean_state = torch.sum(state * weight, 0)
        elif len(state.shape) == 3:
            # states: torch.Size([batch, num_particles, dim_state])
            # weights: torch.Size([batch, num_particles])
            # return: torch.Size([batch, dim_state])
            weight = weight.unsqueeze(2).to(device)
            mean_state = torch.sum(state * weight, 1).view(state.shape[0], state.shape[2])
        return mean_state

    def density_loss(self, p, w, s):
        # p: [B * K, dim_s]
        # w: [B, K]
        # s: [B, dim_s]
        s = s.unsqueeze(1).repeat(1, NUM_PAR_PF, 1)  # [B, K, dim_s]
        x = torch.exp(-(p - s).pow(2).sum(-1))  # [B, K]
        x = (w * x).sum(-1)  # [B]
        loss = -torch.log(const + x)
        return loss

    def par_weighted_var(self, par_states, par_weight, mean_state):
        # par_states: [B, K, dim_s]
        # par_weight: [B, K]
        # mean_state: [B, dim_s]
        num_par = par_states.shape[1]
        mean_state = mean_state.unsqueeze(1).repeat(1, num_par, 1)  # [B, K, dim_s]
        x = par_weight * (par_states - mean_state).abs().sum(-1)  # [B, K]
        return x.sum(-1)  # [B]

    def par_var(self, par_states):
        # par_states: [B, K, dim_s]
        mean_state = par_states.mean(1).unsqueeze(1).repeat(1, NUM_PAR_PF, 1)  # mean_state: [B, K, dim_s]
        x = (par_states - mean_state).pow(2).sum(-1)  # [B, K]
        return x.mean(-1)  # [B]

    def get_q(self, state, action):
        qf1, qf2 = self.critic(state, action)
        q = torch.min(qf1, qf2)
        return q

    def soft_q_update(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, \
            obs, curr_par, mean_state, hidden, cell, pf_sample = self.replay_buffer.sample(BATCH_SIZE)
        state_batch = torch.FloatTensor(state_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)  # (B, 1)
        mask_batch = torch.FloatTensor(np.float32(1 - done_batch)).unsqueeze(1).to(device)
        curr_obs = torch.FloatTensor(obs).to(device)
        curr_par = torch.FloatTensor(curr_par).to(device)  # (B, K, dim_s)
        mean_state = torch.FloatTensor(mean_state).to(device) # (B, dim_s)
        curr_par_sample = torch.FloatTensor(pf_sample).to(device) # (B, M, 2)
        hidden = torch.FloatTensor(hidden).to(device)  # [128, NUM_LSTM_LAYER, 1, DIM_LSTM_HIDDEN]
        hidden = torch.transpose(torch.squeeze(hidden), 0, 1).contiguous()
        cell = torch.FloatTensor(cell).to(device)
        cell = torch.transpose(torch.squeeze(cell), 0, 1).contiguous()

        # ------------------------
        #  Train Particle Proposer
        # ------------------------
        if PP_EXIST:
            self.pp_optimizer.zero_grad()
            state_propose = self.pp_net(curr_obs, NUM_PAR_PF)
            PP_loss = 0
            if 'mse' in PP_LOSS_TYPE:
                PP_loss += self.MSE_criterion(state_batch.repeat(NUM_PAR_PF, 1), state_propose)
            if 'adv' in PP_LOSS_TYPE:
                fake_logit, _, _ = self.measure_net.m_model(state_propose, curr_obs, hidden, cell, NUM_PAR_PF)  # (B, K)
                real_target = torch.ones_like(fake_logit)
                PP_loss += self.BCE_criterion(fake_logit, real_target)
            if 'density' in PP_LOSS_TYPE:
                std = 0.1
                DEN_COEF = 1
                std_scale = torch.FloatTensor(np.array([2, 1])).to(device)
                par_s = state_propose.view(BATCH_SIZE, -1, DIM_STATE) # [B * K, 2] -> [B, K, 2]
                true_s = state_batch.unsqueeze(1).repeat(1, NUM_PAR_PF, 1) # [B, 2] -> [B, K, 2]
                square_distance = ((par_s - true_s) * std_scale).pow(2).sum(-1)  # [B, K] scale all dimension to -1, +1
                true_state_lik = 1. / (2 * np.pi * std ** 2) * (-square_distance / (2 * std ** 2)).exp()
                pp_nll = -(const + true_state_lik.mean(1)).log().mean()
                PP_loss += DEN_COEF * pp_nll
            PP_loss.backward()
            self.pp_optimizer.step()

        # ------------------------
        #  Train Observation Model
        # ------------------------
        self.measure_optimizer.zero_grad()
        fake_logit, next_hidden, next_cell = self.measure_net.m_model(curr_par.view(-1, DIM_STATE),
                                                                      curr_obs, hidden, cell, NUM_PAR_PF)  # (B, K)
        if PP_EXIST:
            fake_logit_pp, _, _ = self.measure_net.m_model(state_propose.detach(),
                                                           curr_obs, hidden, cell, NUM_PAR_PF)  # (B, K)
            fake_logit = torch.cat((fake_logit, fake_logit_pp), -1)  # (B, 2K)
        fake_target = torch.zeros_like(fake_logit)
        fake_loss = self.BCE_criterion(fake_logit, fake_target)
        real_logit, _, _ = self.measure_net.m_model(state_batch, curr_obs, hidden, cell, 1)  # (batch, num_pars)
        real_target = torch.ones_like(real_logit)
        real_loss = self.BCE_criterion(real_logit, real_target)
        OM_loss = real_loss + fake_loss
        OM_loss.backward()
        self.measure_optimizer.step()

        # ------------------------
        #  Train Transition Model
        # ------------------------
        self.dynamic_optimizer.zero_grad()
        state_predict = self.dynamic_net.t_model(state_batch, action_batch * STEP_RANGE)
        TM_loss = self.MSE_criterion(state_predict, next_state_batch)
        TM_loss.backward()
        self.dynamic_optimizer.step()

        # ------------------------
        #  Train SAC
        # ------------------------
        next_mean_state = self.dynamic_net.t_model(mean_state, action_batch * STEP_RANGE)
        next_par_sample = self.dynamic_net.t_model(
            curr_par_sample.view(-1, DIM_STATE),
            action_batch.repeat(NUM_PAR_SMC_INIT, 1) * STEP_RANGE)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_mean_state, next_par_sample.view(BATCH_SIZE, -1))
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(mean_state, curr_par_sample.view(BATCH_SIZE, -1))
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        soft_update(self.critic_target, self.critic, self.tau)
