from Constellation import *
# import sys

STARLINK_S2 =   Const_Param(alt=540.0, inc=53.2, p=10, s=10, f=17, t_max=90, target_k=20) # p = 36, 20
ONEWEB_GEN1 =   Const_Param(alt=1200, inc=88, p=18, s=20, f=17, t_max=40, target_k=10) # K = 10, T = 40
TELESAT_P1 =    Const_Param(alt=1325, inc=50.88, p=20, s=11, f=17, t_max=50, target_k=10) # 
TEST_ =         Const_Param(alt=540.0, inc=53.2, p=3, s=10, f=17, t_max=90, target_k=30) # p = 13, s = 10
# TEST2_ =        Const_Param(alt=540.0, inc=53.2, p=6, s=10, f=17, t_max=90, target_k=60) # fail to converge..
# TEST3_ =        Const_Param(alt=540.0, inc=53.2, p=3, s=10, f=17, t_max=90, target_k=30) # MAPPO failed 200 iter
TEST4_ =        Const_Param(alt=540.0, inc=53.2, p=6, s=10, f=17, t_max=90, target_k=30) # p = 13, s = 10
TEST_H550_ =   Const_Param(alt=550, inc=53, p=3, s=22, f=17, t_max=90, target_k=30) # MAPPO train better
TEST_DENSE_ =   Const_Param(alt=550, inc=53, p=10, s=22, f=17, t_max=90, target_k=30) # MYOTIC better..
TEST_GRID_ =   Const_Param(alt=550, inc=53, p=3, s=22, f=17, t_max=90, target_k=30, n_neighbor=2, grid_scale=15) # MAPPO train better
TEST_HARD_ =   Const_Param(alt=550, inc=53, p=3, s=22, f=17, t_max=90, max_buf=10, target_k=40) # max_buf has no effect..

MY_CONST_NAME = "test_dense"
IS_MYOTIC = False
N_TRAIN_ITER = 300
N_USER = 80 # for training
PLOT_USER_NUM = 120
# TEST_MODES = ["MAPPO", "MYOTIC"] # "MAPPO" , "MYOTIC"
TEST_MODES = ["GREEDY", "ERNC" , "STATIC_R"] # "ERNC" , "STATIC_R", "GREEDY"
IS_TEST_MODE = True # extra test mode for env

if MY_CONST_NAME == "oneweb":       CONST_PARAM = ONEWEB_GEN1
elif MY_CONST_NAME == "starlink":   CONST_PARAM = STARLINK_S2
elif MY_CONST_NAME == 'telesat':    CONST_PARAM = TELESAT_P1
elif MY_CONST_NAME == 'test_grid':      CONST_PARAM = TEST_GRID_
# elif MY_CONST_NAME == 'test3':      CONST_PARAM = TEST3_
elif MY_CONST_NAME == 'test4':      CONST_PARAM = TEST4_
elif MY_CONST_NAME == 'test_h550':      CONST_PARAM = TEST_H550_
elif MY_CONST_NAME == 'test_hard':      CONST_PARAM = TEST_HARD_
elif MY_CONST_NAME == 'test_dense':         CONST_PARAM = TEST_DENSE_
elif MY_CONST_NAME == 'test':               CONST_PARAM = TEST_
else:                                raise NameError(f"Not known Const Name: {MY_CONST_NAME}")

TEST_ID = 'Starlink_Shell2_0_2'
