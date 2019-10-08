# author: @wangyunbo, @liubo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from configs import *

def check_path(path):
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)

def get_datetime():
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%m%d%H%M")

def detect_collison(curr_state, next_state):
    if len(curr_state.shape) == 1:
        cx = curr_state[0]
        cy = curr_state[1]
        nx = next_state[0]
        ny = next_state[1]
    elif len(curr_state.shape) == 2:
        cx = curr_state[:, 0]
        cy = curr_state[:, 1]
        nx = next_state[:, 0]
        ny = next_state[:, 1]
    cond_hit = (nx < 0) + (nx > 2) + (ny < 0) + (ny > 1)  # hit the surrounding walls
    cond_hit += (cy <= 0.5) * (ny > 0.5)  # cross the middle wall
    cond_hit += (cy >= 0.5) * (ny < 0.5)  # cross the middle wall
    cond_hit += (cx <= 1.2) * (nx > 1.2) * ((cy > 0.9) * (cy < 1) + (ny > 0.9) * (ny < 1))
    cond_hit += (cx >= 1.2) * (nx < 1.2) * ((cy > 0.9) * (cy < 1) + (ny > 0.9) * (ny < 1))
    cond_hit += (cx <= 0.4) * (nx > 0.4) * ((cy > 0.4) * (cy < 0.5) + (ny > 0.4) * (ny < 0.5))
    cond_hit += (cx >= 0.4) * (nx < 0.4) * ((cy > 0.4) * (cy < 0.5) + (ny > 0.4) * (ny < 0.5))
    cond_hit += (cx <= 1.2) * (nx > 1.2) * ((cy > 0.5) * (cy < 0.6) + (ny > 0.5) * (ny < 0.6))
    cond_hit += (cx >= 1.2) * (nx < 1.2) * ((cy > 0.5) * (cy < 0.6) + (ny > 0.5) * (ny < 0.6))
    cond_hit += (cx <= 0.4) * (nx > 0.4) * ((cy > 0.0) * (cy < 0.1) + (ny > 0.0) * (ny < 0.1))
    cond_hit += (cx >= 0.4) * (nx < 0.4) * ((cy > 0.0) * (cy < 0.1) + (ny > 0.0) * (ny < 0.1))
    return cond_hit

def l2_distance(state, goal):
    if len(state.shape) == 1:
        dist = np.power((state[0] - goal[0]), 2) + np.power((state[1] - goal[1]), 2) + const
    elif len(state.shape) == 2:
        dist = (state[:, 0] - goal[:, 0]).pow(2) + (state[:, 1] - goal[:, 1]).pow(2) + const
    elif len(state.shape) == 3:
        dist = (state[:, :, 0] - goal[:, :, 0]).pow(2) + (state[:, :, 1] - goal[:, :, 1]).pow(2) + const
    return dist

def plot_maze(figure_name='default', states=None):
    plt.figure(figure_name)
    ax = plt.axes()

    ax.set_xlim([0, 2])
    ax.set_ylim([0, 1])

    # goals
    if states[0, 1] <= 0.5:
        cir = plt.Circle((2, 0.25), 0.07, color='orange')
    else:
        cir = plt.Circle((0, 0.75), 0.07, color='orange')
    ax.add_artist(cir)

    walls = np.array([
        # horizontal
        [[0, 0], [2, 0]],
        [[0, 0.5], [2, 0.5]],
        [[0, 1], [2, 1]],
        # vertical
        [[0, 0], [0, 1]],
        [[2, 0], [2, 1]],
        [[0.4, 0.4], [0.4, 0.5]],
        [[1.2, 0.9], [1.2, 1]],
        [[0.4, 0.0], [0.4, 0.1]],
        [[1.2, 0.5], [1.2, 0.6]],
    ])
    walls_dotted = np.array([
        [[0, 0.4], [2, 0.4]],
        [[0, 0.9], [2, 0.9]],
        [[0, 0.6], [2, 0.6]],
        [[0, 0.1], [2, 0.1]],
    ])

    color = (0, 0, 0)
    ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color=color, linewidth=1.0)

    color = (0, 0, 1)
    ax.plot(walls_dotted[:, :, 0].T, walls_dotted[:, :, 1].T, color=color, linewidth=1.0, linestyle='--')

    if type(states) is np.ndarray:
        xy = states[:,:2]
        x, y = zip(*xy)
        ax.plot(x, y, 'ro')
        ax.plot(x[0],y[0], 'bo')

    ax.set_aspect('equal')
    plt.savefig(figure_name)
    plt.close()


def plot_par(figure_name='default', true_state=None, mean_state=None, pf_state=None,
             pp_state=None, smc_traj=None):
    plt.figure(figure_name)
    ax = plt.axes()
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 1])

    # goals
    if true_state[1] <= 0.5:
        cir = plt.Circle((2, 0.25), 0.07, color='orange')
    else:
        cir = plt.Circle((0, 0.75), 0.07, color='orange')
    ax.add_artist(cir)

    walls = np.array([
        # horizontal
        [[0, 0], [2, 0]],
        [[0, 0.5], [2, 0.5]],
        [[0, 1], [2, 1]],
        # vertical
        [[0, 0], [0, 1]],
        [[2, 0], [2, 1]],
        [[0.4, 0.4], [0.4, 0.5]],
        [[1.2, 0.9], [1.2, 1]],
        [[0.4, 0.0], [0.4, 0.1]],
        [[1.2, 0.5], [1.2, 0.6]],
    ])
    walls_dotted = np.array([
        [[0, 0.4], [2, 0.4]],
        [[0, 0.9], [2, 0.9]],
        [[0, 0.6], [2, 0.6]],
        [[0, 0.1], [2, 0.1]],
    ])

    color = (0, 0, 0)
    ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color=color, linewidth=1.0)

    color = (0, 0, 1)
    ax.plot(walls_dotted[:, :, 0].T, walls_dotted[:, :, 1].T, color=color, linewidth=1.0, linestyle='--')

    # planning trajectories
    num_par_smc = smc_traj.shape[1]
    for k in range(num_par_smc):
        points = smc_traj[:, k, :]
        ax.plot(*points.T, lw=1, color=(0.5, 0.5, 0.5))  # RGB

    ax.plot(mean_state[0], mean_state[1], 'ko')
    ax.plot(true_state[0], true_state[1], 'ro')

    xy = pf_state[:, :2]
    x, y = zip(*xy)
    ax.plot(x, y, 'gx')

    xy = pp_state[:, :2]
    x, y = zip(*xy)
    ax.plot(x, y, 'bx')

    ax.set_aspect('equal')
    plt.savefig(figure_name)
    plt.close()