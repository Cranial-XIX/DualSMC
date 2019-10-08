# author: @wangyunbo

######################
# Environment
EPI_REWARD = 100
DIM_ACTION = 2
DIM_STATE = 2
DIM_OBS = 4
END_RANGE = 0.005
STEP_RANGE = 0.05
MAX_STEPS = 200
FIG_FORMAT = '.png'
IMG = 'img/'
CKPT = 'ckpt/'
device = 'cuda'

######################
# Network
MODEL_NAME = 'dualsmc'
DIM_HIDDEN = 256
DIM_LSTM_HIDDEN = 128
NUM_LSTM_LAYER = 2

######################
# Training
TRAIN = True
MAX_EPISODES = 20000
BATCH_SIZE  = 64
FIL_LR = 1e-3 # filtering
PLA_LR = 1e-3 # planning
SAVE_ITER = 1000
SUMMARY_ITER = 1000
DISPLAY_ITER = 10
SHOW_TRAJ = True
SHOW_DISTR = False

######################
# Filtering
PF_RESAMPLE_STEP = 3
NUM_PAR_PF = 100
PP_EXIST = True
PP_RATIO = 0.3
PP_LOSS_TYPE = 'adv' # 'mse', 'adv', 'density'

######################
# Planning
NUM_PAR_SMC_INIT = 3
NUM_PAR_SMC = 30
HORIZON = 10
SMCP_MODE = 'topk' # 'samp', 'topk'
SMCP_RESAMPLE = True
SMCP_RESAMPLE_STEP = 3

######################
# SAC
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
replay_buffer_size = 100000
alpha = 1.0
gamma = 0.95
tau = 0.005
const = 1e-6

