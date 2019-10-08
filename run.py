# author: @wangyunbo, @liubo

import sys
import os
import numpy as np
import shutil
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from dual_smc import DUAL_SMC
from configs import *
from utils import *
from env import *

def dualsmc():
    model = DUAL_SMC()
    step_list = []
    dist_list = []
    rmse_per_step = np.zeros((MAX_STEPS))

    if len(sys.argv) > 1:
        load_path = sys.argv[1]
        model.load_model(load_path)

    experiment_id = "dualsmc" + get_datetime()
    save_path = CKPT + experiment_id
    img_path = IMG + experiment_id
    check_path(save_path)
    check_path(img_path)

    for episode in range(MAX_EPISODES):
        episode += 1
        env = Environment()
        filter_dist = 0
        trajectory = []

        hidden = np.zeros((NUM_LSTM_LAYER, 1, DIM_LSTM_HIDDEN))
        cell = np.zeros((NUM_LSTM_LAYER, 1, DIM_LSTM_HIDDEN))

        curr_state = env.state
        curr_obs = env.get_observation()
        trajectory.append(curr_state)

        par_states = np.random.rand(NUM_PAR_PF, 2)
        par_states[:, 0] = par_states[:, 0] * 0.4 + 0.8
        par_states[:, 1] = par_states[:, 1] * 0.3 + 0.1 + np.random.randint(2, size=NUM_PAR_PF) * 0.5
        par_weight = torch.log(torch.ones((NUM_PAR_PF)).to(device) * (1.0 / float(NUM_PAR_PF)))
        normalized_weights = torch.softmax(par_weight, -1)
        mean_state = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()

        if SHOW_TRAJ and episode % DISPLAY_ITER == 0:
            traj_dir = img_path + "/iter-" + str(episode)
            if os.path.exists(traj_dir):
                shutil.rmtree(traj_dir)
            os.mkdir(traj_dir)

        num_par_propose = int(NUM_PAR_PF * PP_RATIO)
        for step in range(MAX_STEPS):
            # 1. observation model
            # 2. planning
            # 3. re-sample
            # 4. transition model

            #######################################
            # Observation model
            lik, next_hidden, next_cell = model.measure_net.m_model(
                torch.FloatTensor(par_states).to(device),
                torch.FloatTensor(curr_obs).unsqueeze(0).to(device),
                torch.FloatTensor(hidden).to(device),
                torch.FloatTensor(cell).to(device))
            par_weight += lik.squeeze()  # (NUM_PAR_PF)
            normalized_weights = torch.softmax(par_weight, -1)

            if SHOW_DISTR and episode % DISPLAY_ITER == 0:
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                frm_name = traj_dir + '/' + file_name + '_distr' + FIG_FORMAT
                weights = normalized_weights.detach().cpu().numpy()
                fig1, ax1 = plt.subplots()
                plt.hist(weights, bins=np.logspace(-5, 0, 50))
                ax1.set_xscale("log")
                ax1.set_xlim(1e-5, 1e0)
                plt.savefig(frm_name)
                plt.close()

            curr_s = par_states.copy()

            #######################################
            # Planning
            if SMCP_MODE == 'topk':
                weight_init, idx = torch.topk(par_weight, NUM_PAR_SMC_INIT)
                idx = idx.detach().cpu().numpy()
            elif SMCP_MODE == 'samp':
                idx = torch.multinomial(normalized_weights, NUM_PAR_SMC_INIT, replacement=True).detach().cpu().numpy()
                weight_init = par_weight[idx]
            weight_init = torch.softmax(weight_init, -1).unsqueeze(1).repeat(1, NUM_PAR_SMC)  # [M, N]
            states_init = par_states[idx]  # [K, C] -> [M, C]
            states_init_ = np.reshape(states_init, (1, NUM_PAR_SMC_INIT, 1, DIM_STATE))  # [1, M, 1, C]
            smc_states = np.tile(states_init_, (HORIZON, 1, NUM_PAR_SMC, 1))  # [1, M, 1, C] -> [T, M, N, C]
            smc_action = np.zeros((HORIZON, NUM_PAR_SMC, DIM_ACTION))  # [T, N, dim_a]
            smc_weight = torch.log(torch.ones((NUM_PAR_SMC)).to(device) * (1.0 / float(NUM_PAR_SMC)))  # [N]
            mean_state = np.reshape(mean_state, (1, 1, DIM_STATE))  # [1, 1, C]
            smc_mean_state = np.tile(mean_state, (HORIZON, NUM_PAR_SMC, 1))  # [T, N, C]
            prev_q = 0

            for i in range(HORIZON):
                curr_smc_state = torch.FloatTensor(smc_states[i]).to(device) # [M, N, C]
                action, log_prob = model.policy.get_action(
                    torch.FloatTensor(smc_mean_state[i]).to(device), # [N, C]
                    torch.transpose(curr_smc_state, 0, 1).contiguous().view(NUM_PAR_SMC, -1)) # [N, M * C]
                action_tile = action.unsqueeze(0).repeat(NUM_PAR_SMC_INIT, 1, 1).view(-1, DIM_ACTION)

                next_smc_state = model.dynamic_net.t_model(
                    torch.FloatTensor(smc_states[i]).to(device).view(-1, DIM_STATE),  action_tile * STEP_RANGE)
                next_smc_state[:, 0] = torch.clamp(next_smc_state[:, 0], 0, 2)
                next_smc_state[:, 1] = torch.clamp(next_smc_state[:, 1], 0, 1)
                next_smc_state = next_smc_state.view(NUM_PAR_SMC_INIT, NUM_PAR_SMC, DIM_STATE)

                mean_par = model.dynamic_net.t_model(
                    torch.FloatTensor(smc_mean_state[i]).to(device), action * STEP_RANGE)
                mean_par[:, 0] = torch.clamp(mean_par[:, 0], 0, 2)
                mean_par[:, 1] = torch.clamp(mean_par[:, 1], 0, 1)

                if i < HORIZON - 1:
                    smc_action[i] = action.detach().cpu().numpy()
                    smc_states[i + 1] = next_smc_state.detach().cpu().numpy()
                    smc_mean_state[i + 1] = mean_par.detach().cpu().numpy()

                q = model.get_q(curr_smc_state.view(-1, DIM_STATE), action_tile).view(NUM_PAR_SMC_INIT, -1)
                advantage = q - prev_q - log_prob.unsqueeze(0).repeat(NUM_PAR_SMC_INIT, 1) # [M, N]
                advantage = torch.sum(weight_init * advantage, 0).squeeze()  # [N]
                smc_weight += advantage
                prev_q = q
                normalized_smc_weight = F.softmax(smc_weight, -1)  # [N]

                if SMCP_RESAMPLE and (i % SMCP_RESAMPLE_STEP == 1):
                    idx = torch.multinomial(normalized_smc_weight, NUM_PAR_SMC, replacement=True).detach().cpu().numpy()
                    smc_action = smc_action[:, idx, :]
                    smc_states = smc_states[:, :, idx, :]
                    smc_mean_state = smc_mean_state[:, idx, :]
                    smc_weight = torch.log(torch.ones((NUM_PAR_SMC)).to(device) * (1.0 / float(NUM_PAR_SMC)))
                    normalized_smc_weight = F.softmax(smc_weight, -1)  # [N]

            smc_xy = np.reshape(smc_states[:, :, :, :2], (-1, NUM_PAR_SMC_INIT * NUM_PAR_SMC, 2))

            if SMCP_RESAMPLE and (HORIZON % SMCP_RESAMPLE_STEP == 0):
                n = np.random.randint(NUM_PAR_SMC, size=1)[0]
            else:
                n = Categorical(normalized_smc_weight).sample().detach().cpu().item()
            action = smc_action[0, n, :]

            #######################################
            if step % PF_RESAMPLE_STEP == 0:
                if PP_EXIST:
                    idx = torch.multinomial(normalized_weights, NUM_PAR_PF - num_par_propose,
                                            replacement=True).detach().cpu().numpy()
                    resample_state = par_states[idx]
                    proposal_state = model.pp_net(torch.FloatTensor(curr_obs).unsqueeze(0).to(device), num_par_propose)
                    proposal_state[:, 0] = torch.clamp(proposal_state[:, 0], 0, 2)
                    proposal_state[:, 1] = torch.clamp(proposal_state[:, 1], 0, 1)
                    proposal_state = proposal_state.detach().cpu().numpy()
                    par_states = np.concatenate((resample_state, proposal_state), 0)
                else:
                    idx = torch.multinomial(normalized_weights, NUM_PAR_PF, replacement=True).detach().cpu().numpy()
                    par_states = par_states[idx]

                par_weight = torch.log(torch.ones((NUM_PAR_PF)).to(device) * (1.0 / float(NUM_PAR_PF)))
                normalized_weights = torch.softmax(par_weight, -1)

            mean_state = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()
            filter_rmse = math.sqrt(pow(mean_state[0] - curr_state[0], 2) + pow(mean_state[1] - curr_state[1], 2))
            rmse_per_step[step] += filter_rmse
            filter_dist += filter_rmse

            #######################################
            if SHOW_TRAJ and episode % DISPLAY_ITER == 0:
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                frm_name = traj_dir + '/' + file_name + '_par' + FIG_FORMAT

                if PP_EXIST and step % PF_RESAMPLE_STEP == 0:
                    plot_par(frm_name, curr_state, mean_state, resample_state, proposal_state, smc_xy)

            #######################################
            # Update the environment
            reward = env.step(action * STEP_RANGE)
            next_state = env.state
            next_obs = env.get_observation()

            #######################################
            if TRAIN:
                model.replay_buffer.push(curr_state, action, reward, next_state, env.done, curr_obs,
                                         curr_s, mean_state, hidden, cell, states_init)
                if len(model.replay_buffer) > BATCH_SIZE:
                    model.soft_q_update()

            #######################################
            # Transition Model
            par_states = model.dynamic_net.t_model(torch.FloatTensor(par_states).to(device),
                                                   torch.FloatTensor(action * STEP_RANGE).to(device))
            par_states[:, 0] = torch.clamp(par_states[:, 0], 0, 2)
            par_states[:, 1] = torch.clamp(par_states[:, 1], 0, 1)
            par_states = par_states.detach().cpu().numpy()

            #######################################
            curr_state = next_state
            curr_obs = next_obs
            hidden = next_hidden.detach().cpu().numpy()
            cell = next_cell.detach().cpu().numpy()
            trajectory.append(next_state)
            if env.done:
                break

        filter_dist = filter_dist / (step + 1)
        dist_list.append(filter_dist)
        step_list.append(step)

        if episode >= SUMMARY_ITER:
            step_list.pop(0)
            dist_list.pop(0)

        if episode % SAVE_ITER == 0:
            model_path = save_path + '_' + str(episode)
            model.save_model(model_path)
            print("save model to %s" % model_path)

        if episode % DISPLAY_ITER == 0:
            st = img_path + "/" + str(episode) + "-trj" + FIG_FORMAT
            print("plotting ... save to %s" % st)
            plot_maze(figure_name=st, states=np.array(trajectory))

            if episode >= SUMMARY_ITER:
                total_iter = SUMMARY_ITER
            else:
                total_iter = episode
            reach = np.array(step_list) < (MAX_STEPS - 1)
            num_reach = sum(reach)
            step_reach = step_list * reach
            interaction = 'Episode %s: steps = %s, success = %s, avg_steps = %s, avg_dist = %s' % (
                episode, step, num_reach / total_iter, sum(step_reach) / (num_reach + const), sum(dist_list) / total_iter)
            print('\r{}'.format(interaction))

    rmse_per_step = rmse_per_step / MAX_EPISODES
    print(rmse_per_step)

if __name__ == "__main__":
    if MODEL_NAME == 'dualsmc':
        dualsmc()