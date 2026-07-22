from Constellation import *
import os

STARLINK_S2 =   Const_Param(p=12, s=10, t_max=90, ) # p = 36, 20
STARLINK_NO_ISL_ =   Const_Param(p=36, s=10, t_max=300,  n_neighbor=0) # p = 36, 20
STARLINK_NO_RLNC_ =   Const_Param(p=36, s=10, t_max=300,  enable_RLNC=False) # p = 36, 20
# TELESAT =   Const_Param(alt=1000, inc=99, p=27, s=13,  t_max=90, ) # K = 10, T = 40 (too high)
AMAZON =        Const_Param(alt=630, inc=51.9, p=6, s=15, t_max=90, )  # p=17, s=15,
AMAZON_NO_ISL_ =        Const_Param(alt=630, inc=51.9, p=17, s=15, t_max=90,  n_neighbor=0)
AMAZON_NO_RLNC_ =        Const_Param(alt=630, inc=51.9, p=17, s=15, t_max=300,  enable_RLNC=False) 

TEST_ =         Const_Param(p=3, s=10,  t_max=200, ) # p = 13, s = 10
TEST_W3_ =         Const_Param(p=3, s=10,  t_max=200,  Tw=3) # p = 13, s = 10
TEST_W4_ =         Const_Param(p=3, s=10,  t_max=200,  Tw=4) # p = 13, s = 10
TEST_NO_ISL_ =     Const_Param(p=3, s=10,  t_max=200,  n_neighbor=0) # p = 13, s = 10
TEST_NO_RLNC_ =     Const_Param(p=3, s=10, t_max=300,  enable_RLNC=False) # time is too long...
TEST_FIELD =         Const_Param(p=3, s=10, t_max=200,  enable_field=True, q=2) 

TEST4_ =        Const_Param(p=6, s=10,  t_max=90, ) # p = 13, s = 10
TEST4_NO_ISL_ =        Const_Param(p=6, s=10,  t_max=90,  n_neighbor=0)
TEST4_NO_RLNC_ =        Const_Param(p=6, s=10,  t_max=300,  enable_RLNC=False)
TEST_DENSE_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=600, )
TEST_DENSE_W4_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=600,  Tw=4) 
TEST_DENSE_W8_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=600,  Tw=8) 
TEST_DENSE_W10_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=600,  Tw=10) 
TEST_DENSE_NO_ISL_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=1000,  n_neighbor=0)
TEST_DENSE_NO_RLNC_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=300,  enable_RLNC=False)
TEST_DENSE_FIELD_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=300,  enable_field=True, q=2)
TEST_DENSE_FIELD_Q4_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=300,  enable_field=True, q=4)
# TEST_GRID_ =   Const_Param(alt=550, inc=53, p=3, s=22,  t_max=90,  grid_scale=15) # MAPPO train better

TEST_ERASURE_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=90) # dl_cp should smaller, e = 0.1, lambda = 0.01

MY_CONST_NAME = "test_dense"

# Pareto 掃描的權重組合 [omega_t (時間), omega_c (能量)]
PARETO_CONFIGS = [
    # {"omega_t": 1.0, "omega_c": 0.0}, # 極端求快
    # {"omega_t": 0.9, "omega_c": 0.1},   # dense
    {"omega_t": 0.8, "omega_c": 0.2},  # dense settings (TW)
    # {"omega_t": 0.6, "omega_c": 0.4}, # Exp1. setting
    # {"omega_t": 0.7, "omega_c": 0.3},
    # {"omega_t": 0.95, "omega_c": 0.05}, 
    # {"omega_t": 0.85, "omega_c": 0.15}, 
    # {"omega_t": 0.75, "omega_c": 0.25},
    # {"omega_t": 0.65, "omega_c": 0.35},
    # {"omega_t": 0.5, "omega_c": 0.5}, # 平衡
    # {"omega_t": 0.4, "omega_c": 0.6}, # fail compeltely..
    # {"omega_t": 0.3, "omega_c": 0.7}, 
    # {"omega_t": 0.2, "omega_c": 0.8},
    # {"omega_t": 0.1, "omega_c": 0.9},  
    # {"omega_t": 0.0, "omega_c": 1.0},  # 極端省電
]

############# training setting ############
IS_MYOTIC = False
N_TRAIN_ITER = 50
N_USER = 40 # for training
ERASURE = 0.1
DO_TEST_LOG = True

###########################################

# TEST_MODES = ["MAPPO"] # "MAPPO" , "MYOTIC"
TEST_MODES = ["MAPPO"] # "GREEDY", "ERNC" , "STATIC_R"
USER_NUMBERS = [1, 40, 80, 120, 160]
ERASURES = [0.1] #[0.1, 0.2, 0.3, 0.4]
THETA_THES = [15]
MAX_BUFS = [30] # [5, 10, 15, 20, 25, 30]
TARGET_KS = [30] #[10, 20, 30]

# set to True if checkpoint it stored in checkpoints/WTX_WCX
TEST_PARETO = True

SEED_LIST = [
    1, 12, 123, 1234, 12345 #1235, 777, 1, 12, 1234
] 
# general: 123,     1234, 1235,1236, 1237
# X 777, 30, 222
# TW candidate: 222, 777
#######################################################
if IS_MYOTIC:       _path = f"./satellite_{MY_CONST_NAME}_myotic_checkpoints/" # f"./satellite_test_dense_myotic_checkpoints/" | None
else:               _path = f"./satellite_{MY_CONST_NAME}_checkpoints/"
_path = None

USE_DEFICIT = True
if _path is not None: RESTORE_CHECKPOINT_PATH = os.path.abspath(_path)
else:                 RESTORE_CHECKPOINT_PATH = None

# TEST_CHECKPOINT_PATH = f"./satellite_{MY_CONST_NAME}_checkpoints"
if TEST_PARETO:
    if IS_MYOTIC:   TEST_CHECKPOINT_PATH = f"./{MY_CONST_NAME}_myotic_checkpoints" # f"./{MY_CONST_NAME}_myotic_checkpoints"
    else:           TEST_CHECKPOINT_PATH = f"./{MY_CONST_NAME}_checkpoints" # f"./{MY_CONST_NAME}_checkpoints"
else:
    if IS_MYOTIC:   TEST_CHECKPOINT_PATH = f"./satellite_{MY_CONST_NAME}_myotic_checkpoints"
    else:           TEST_CHECKPOINT_PATH = f"./satellite_test_dense_checkpoints" # f"./satellite_{MY_CONST_NAME}_checkpoints"

IS_TEST_MODE = True # extra test mode for env
PLOT_USER_NUM = 400
# if MY_CONST_NAME == "telesat":       CONST_PARAM = TELESAT
if MY_CONST_NAME == "starlink":   CONST_PARAM = STARLINK_S2
elif MY_CONST_NAME == "starlink_no_isl":   CONST_PARAM = STARLINK_NO_ISL_
elif MY_CONST_NAME == "starlink_no_rlnc":   CONST_PARAM = STARLINK_NO_RLNC_
elif MY_CONST_NAME == 'amazon':    CONST_PARAM = AMAZON
elif MY_CONST_NAME == 'amazon_no_isl':    CONST_PARAM = AMAZON_NO_ISL_
elif MY_CONST_NAME == 'amazon_no_rlnc':    CONST_PARAM = AMAZON_NO_RLNC_
# elif MY_CONST_NAME == 'test_grid':      CONST_PARAM = TEST_GRID_
elif MY_CONST_NAME == 'test4':      CONST_PARAM = TEST4_
elif MY_CONST_NAME == 'test4_no_isl':      CONST_PARAM = TEST4_NO_ISL_
elif MY_CONST_NAME == 'test4_no_rlnc':      CONST_PARAM = TEST4_NO_RLNC_
elif MY_CONST_NAME == 'test_w3':      CONST_PARAM = TEST_W3_
elif MY_CONST_NAME == 'test_w4':      CONST_PARAM = TEST_W4_
elif MY_CONST_NAME == 'test_no_isl':      CONST_PARAM = TEST_NO_ISL_
elif MY_CONST_NAME == 'test_no_rlnc':      CONST_PARAM = TEST_NO_RLNC_
elif MY_CONST_NAME == 'test_field':      CONST_PARAM = TEST_FIELD
# elif MY_CONST_NAME == 'test_dense_buf':      CONST_PARAM = TEST_DENSE_BUF_
elif MY_CONST_NAME == 'test_dense':         CONST_PARAM = TEST_DENSE_
elif MY_CONST_NAME == 'test_dense_w4':         CONST_PARAM = TEST_DENSE_W4_
elif MY_CONST_NAME == 'test_dense_w8':         CONST_PARAM = TEST_DENSE_W8_
elif MY_CONST_NAME == 'test_dense_w10':         CONST_PARAM = TEST_DENSE_W10_
elif MY_CONST_NAME == 'test_dense_no_isl':         CONST_PARAM = TEST_DENSE_NO_ISL_
elif MY_CONST_NAME == 'test_dense_no_rlnc':         CONST_PARAM = TEST_DENSE_NO_RLNC_
elif MY_CONST_NAME == 'test_dense_field':         CONST_PARAM = TEST_DENSE_FIELD_
elif MY_CONST_NAME == 'test_dense_field_q4':      CONST_PARAM = TEST_DENSE_FIELD_Q4_
elif MY_CONST_NAME == 'test_erasure':         CONST_PARAM = TEST_ERASURE_
elif MY_CONST_NAME == 'test':               CONST_PARAM = TEST_
else:                                raise NameError(f"Not known Const Name: {MY_CONST_NAME}")

TEST_ID = 'Starlink_Shell2_0_2'


######## plot settings ########
# 定義每條線的點樣式(Marker)與顏色
MARKERS = ['x', 'x', 'x', 'o', 's', 'v', 'o', '^']
COLORS = ["#7a7a7a", "#b81dff", "#b41f21", '#1f77b4', "#070400", '#ff7f0e', '#2ca02c', "#f8f012"] 
LINESTYLES = ['dotted', 'dotted', 'dashed', 'solid', 'dashdot', 'dashdot', 'dashed', 'solid']
ALGO_TILTE = ["No-RLNC", "No-ISL", "Myopic", "PACE", "Greedy", "ERNC", "Static Redundancy", "Offline"]

# MARKERS = ['x', 'x', 'x', 'o', 's', 'v', 'o', '^']
# COLORS = ["#7a7a7a", "#b81dff", "#b41f21", '#1f77b4', "#070400", '#ff7f0e', '#2ca02c', "#f8f012"] 
# LINESTYLES = ['dotted', 'dotted', 'dashed', 'solid', 'dashdot', 'dashdot', 'dashed', 'solid']
# # ALGO_TILTE = ["Myopic", "PACE", "PACE (Tw = 3)", "PACE (Tw = 4)"]
# ALGO_TILTE = ["PACE (Tw = 2)", "PACE (Tw = 4)", "PACE (Tw = 8)", "PACE (Tw = 10)"]


# Thm 1
# MARKERS = ['x', 'o', 's', 'v', 'o', '^']
# COLORS = ["#1f77b4", '#1f77b4', "#1f77b4", "#a80fef", "#a80fef", "#a80fef"] # 經典的藍、橘、綠
# LINESTYLES = ['solid', 'solid', 'solid', 'dashdot', 'dashdot', 'dashdot']
# ALGO_TILTE = ["proposed", "proposed1", "proposed2", "lower bound", "lower bound1", "lower bound2"] #, "Greedy", "ERNC", "Offline", "Static Redundancy"]

# Thm 2
# MARKERS = ['x', 'o']
# COLORS = ["#b41f21", '#1f77b4'] # 經典的藍、橘、綠
# LINESTYLES = ['dashed', 'solid',]
# ALGO_TILTE = ["Myopic", "PACE"] #, "Greedy", "ERNC", "Offline", "Static Redundancy"]
