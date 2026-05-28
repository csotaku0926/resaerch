from Constellation import *
import os

STARLINK_S2 =   Const_Param(alt=540.0, inc=53.2, p=12, s=10, t_max=90, target_k=50) # p = 36, 20
STARLINK_NO_ISL_ =   Const_Param(alt=540.0, inc=53.2, p=36, s=10, t_max=300, target_k=50, n_neighbor=0) # p = 36, 20
STARLINK_NO_RLNC_ =   Const_Param(alt=540.0, inc=53.2, p=36, s=10, t_max=300, target_k=20, enable_RLNC=False) # p = 36, 20
# TELESAT =   Const_Param(alt=1000, inc=99, p=27, s=13,  t_max=90, target_k=40) # K = 10, T = 40 (too high)
AMAZON =        Const_Param(alt=630, inc=51.9, p=6, s=15, t_max=90, target_k=50)  # p=17, s=15,
AMAZON_NO_ISL_ =        Const_Param(alt=630, inc=51.9, p=17, s=15, t_max=90, target_k=50, n_neighbor=0)
AMAZON_NO_RLNC_ =        Const_Param(alt=630, inc=51.9, p=17, s=15, t_max=300, target_k=20, enable_RLNC=False) 

TEST_ =         Const_Param(alt=540.0, inc=53.2, p=3, s=10,  t_max=200, target_k=30) # p = 13, s = 10
TEST_W3_ =         Const_Param(alt=540.0, inc=53.2, p=3, s=10,  t_max=200, target_k=30, Tw=3) # p = 13, s = 10
TEST_W4_ =         Const_Param(alt=540.0, inc=53.2, p=3, s=10,  t_max=200, target_k=30, Tw=4) # p = 13, s = 10
TEST_NO_ISL_ =     Const_Param(alt=540.0, inc=53.2, p=3, s=10,  t_max=200, target_k=30, n_neighbor=0) # p = 13, s = 10
TEST_NO_RLNC_ =     Const_Param(alt=540.0, inc=53.2, p=3, s=10, t_max=300, target_k=20, enable_RLNC=False) # time is too long...

# TEST2_ =        Const_Param(alt=540.0, inc=53.2, p=6, s=10,  t_max=90, target_k=60) # fail to converge..
# TEST3_ =        Const_Param(alt=540.0, inc=53.2, p=3, s=10,  t_max=90, target_k=30) # MAPPO failed 200 iter
TEST4_ =        Const_Param(alt=540.0, inc=53.2, p=6, s=10,  t_max=90, target_k=30) # p = 13, s = 10
TEST4_NO_ISL_ =        Const_Param(alt=540.0, inc=53.2, p=6, s=10,  t_max=90, target_k=30, n_neighbor=0)
TEST4_NO_RLNC_ =        Const_Param(alt=540.0, inc=53.2, p=6, s=10,  t_max=300, target_k=20, enable_RLNC=False)
# TEST_H550_ =   Const_Param(alt=550, inc=53, p=3, s=22,  t_max=90, target_k=30) # MAPPO train better
TEST_DENSE_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=300, target_k=50)
TEST_DENSE_W4_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=300, target_k=50, Tw=4) 
TEST_DENSE_W8_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=200, target_k=50, Tw=8) 
TEST_DENSE_W10_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=200, target_k=50, Tw=10) 
TEST_DENSE_NO_ISL_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=500, target_k=50, n_neighbor=0)
TEST_DENSE_NO_RLNC_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=300, target_k=20, enable_RLNC=False)
TEST_GRID_ =   Const_Param(alt=550, inc=53, p=3, s=22,  t_max=90, target_k=30, grid_scale=15) # MAPPO train better
TEST_HARD_ =   Const_Param(alt=550, inc=53, p=3, s=22,  t_max=90, max_buf=10, target_k=40) # max_buf has no effect..

TEST_ERASURE_ =   Const_Param(alt=550, inc=53, p=10, s=22,  t_max=90, target_k=120) # dl_cp should smaller, e = 0.1, lambda = 0.01
TEST_DEFICIT_     =  Const_Param(alt=540.0, inc=53.2, p=18, s=5,  t_max=90, target_k=70)
TEST_DEFICIT_W3_  =  Const_Param(alt=540.0, inc=53.2, p=18, s=5,  t_max=90, target_k=70, Tw=3)
TEST_DEFICIT_W4_  =  Const_Param(alt=540.0, inc=53.2, p=18, s=5,  t_max=90, target_k=70, Tw=4)

TEST_DEFICIT_FUL_     =  Const_Param(alt=540.0, inc=53.2, p=3, s=5,  t_max=90, target_k=40)
TEST_DEFICIT_FUL_W3_  =  Const_Param(alt=540.0, inc=53.2, p=3, s=5,  t_max=90, target_k=40, Tw=3)

MY_CONST_NAME = "amazon"

# Pareto 掃描的權重組合 [omega_t (時間), omega_c (能量)]
PARETO_CONFIGS = [
    # {"omega_t": 1.0, "omega_c": 0.0}, # 極端求快
    # {"omega_t": 0.9, "omega_c": 0.1},   # dense
    {"omega_t": 0.6, "omega_c": 0.4}, # Exp1. setting
    # {"omega_t": 0.8, "omega_c": 0.2},  # dense settings
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

TEST_MODES = ["MAPPO"] # "MAPPO" , "MYOTIC"
# TEST_MODES = ["GREEDY", "ERNC" , "STATIC_R"] # "GREEDY", "ERNC" , "STATIC_R"

# set to True if checkpoint it stored in checkpoints/WTX_WCX
TEST_PARETO = True
TEST_ERASURE = False

SEED_LIST = [1, 12, 123, 1234]
ERASURES = [0.1, 0.2, 0.3, 0.4]
#######################################################
if IS_MYOTIC:       _path = f"./satellite_{MY_CONST_NAME}_myotic_checkpoints/" # f"./satellite_test_dense_myotic_checkpoints/" | None
else:               _path = f"./satellite_{MY_CONST_NAME}_checkpoints/"
_path = None

USE_DEFICIT = True
if _path is not None: RESTORE_CHECKPOINT_PATH = os.path.abspath(_path)
else:                 RESTORE_CHECKPOINT_PATH = None

# TEST_CHECKPOINT_PATH = f"./satellite_{MY_CONST_NAME}_checkpoints"
if TEST_PARETO:
    if IS_MYOTIC:   TEST_CHECKPOINT_PATH = f"./{MY_CONST_NAME}_myotic_checkpoints"
    else:           TEST_CHECKPOINT_PATH = f"./{MY_CONST_NAME}_checkpoints" # f"./satellite_{MY_CONST_NAME}_checkpoints"
else:
    if IS_MYOTIC:   TEST_CHECKPOINT_PATH = f"./satellite_{MY_CONST_NAME}_myotic_checkpoints"
    else:           TEST_CHECKPOINT_PATH = f"./satellite_{MY_CONST_NAME}_checkpoints" # f"./satellite_{MY_CONST_NAME}_checkpoints"

IS_TEST_MODE = True # extra test mode for env
PLOT_USER_NUM = 400
# if MY_CONST_NAME == "telesat":       CONST_PARAM = TELESAT
if MY_CONST_NAME == "starlink":   CONST_PARAM = STARLINK_S2
elif MY_CONST_NAME == "starlink_no_isl":   CONST_PARAM = STARLINK_NO_ISL_
elif MY_CONST_NAME == "starlink_no_rlnc":   CONST_PARAM = STARLINK_NO_RLNC_
elif MY_CONST_NAME == 'amazon':    CONST_PARAM = AMAZON
elif MY_CONST_NAME == 'amazon_no_isl':    CONST_PARAM = AMAZON_NO_ISL_
elif MY_CONST_NAME == 'amazon_no_rlnc':    CONST_PARAM = AMAZON_NO_RLNC_
elif MY_CONST_NAME == 'test_grid':      CONST_PARAM = TEST_GRID_
elif MY_CONST_NAME == 'test4':      CONST_PARAM = TEST4_
elif MY_CONST_NAME == 'test4_no_isl':      CONST_PARAM = TEST4_NO_ISL_
elif MY_CONST_NAME == 'test4_no_rlnc':      CONST_PARAM = TEST4_NO_RLNC_
elif MY_CONST_NAME == 'test_w3':      CONST_PARAM = TEST_W3_
elif MY_CONST_NAME == 'test_w4':      CONST_PARAM = TEST_W4_
elif MY_CONST_NAME == 'test_no_isl':      CONST_PARAM = TEST_NO_ISL_
elif MY_CONST_NAME == 'test_no_rlnc':      CONST_PARAM = TEST_NO_RLNC_
elif MY_CONST_NAME == 'test_hard':      CONST_PARAM = TEST_HARD_
elif MY_CONST_NAME == 'test_dense':         CONST_PARAM = TEST_DENSE_
elif MY_CONST_NAME == 'test_dense_w4':         CONST_PARAM = TEST_DENSE_W4_
elif MY_CONST_NAME == 'test_dense_w8':         CONST_PARAM = TEST_DENSE_W8_
elif MY_CONST_NAME == 'test_dense_w10':         CONST_PARAM = TEST_DENSE_W10_
elif MY_CONST_NAME == 'test_dense_no_isl':         CONST_PARAM = TEST_DENSE_NO_ISL_
elif MY_CONST_NAME == 'test_dense_no_rlnc':         CONST_PARAM = TEST_DENSE_NO_RLNC_
elif MY_CONST_NAME == 'test_erasure':         CONST_PARAM = TEST_ERASURE_
elif MY_CONST_NAME == 'test_deficit':         CONST_PARAM = TEST_DEFICIT_
elif MY_CONST_NAME == 'test_deficit_w3':         CONST_PARAM = TEST_DEFICIT_W3_
elif MY_CONST_NAME == 'test_deficit_w4':         CONST_PARAM = TEST_DEFICIT_W4_
elif MY_CONST_NAME == 'test_deficit_ful':         CONST_PARAM = TEST_DEFICIT_FUL_
elif MY_CONST_NAME == 'test_deficit_ful_w3':         CONST_PARAM = TEST_DEFICIT_FUL_W3_
elif MY_CONST_NAME == 'test':               CONST_PARAM = TEST_
else:                                raise NameError(f"Not known Const Name: {MY_CONST_NAME}")

TEST_ID = 'Starlink_Shell2_0_2'


######## plot settings ########
# 定義每條線的點樣式(Marker)與顏色
MARKERS = ['x', 'x', 'x', 'o', 's', 'v', 'o', '^']
COLORS = ["#7a7a7a", "#b81dff", "#b41f21", '#1f77b4', "#070400", '#ff7f0e', "#f8f012", '#2ca02c'] # 經典的藍、橘、綠
LINESTYLES = ['dotted', 'dotted', 'dashed', 'solid', 'dashdot', 'dashdot', 'solid', 'dashed']
ALGO_TILTE = ["No-RLNC", "No-ISL", "Myopic", "Proposed", "Greedy", "ERNC", "Offline", "Static Redundancy"]
