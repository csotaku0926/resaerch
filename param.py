from Constellation import *
# import sys

STARLINK_S2 = Const_Param(alt=540.0, inc=53.2, p=10, s=10, f=17, t_max=90, target_k=20) # step = 10
ONEWEB_GEN1 = Const_Param(alt=1200, inc=88, p=18, s=20, f=17, t_max=40, target_k=10) # K = 10, T = 40
TELESAT_P1 = Const_Param(alt=1325, inc=50.88, p=20, s=11, f=17, t_max=50, target_k=10) # 
TEST_ = Const_Param(alt=540.0, inc=53.2, p=3, s=10, f=17, t_max=90, target_k=30) # p = 13, s = 10
TEST2_ = Const_Param(alt=540.0, inc=53.2, p=6, s=10, f=17, t_max=90, target_k=60) # p = 13, s = 10
TEST3_ = Const_Param(alt=540.0, inc=53.2, p=3, s=10, f=17, t_max=90, target_k=30) # MAPPO failed 200 iter
TEST200_ = Const_Param(alt=540.0, inc=53.2, p=3, s=10, f=17, t_max=90, target_k=30) # MAPPO failed 200 iter

MY_CONST_NAME = "test"
IS_MYOTIC = True
N_TRAIN_ITER = 300
N_USER = 100
TEST_MODES = ["MYOTIC"] # "MAPPO" , "MYOTIC"
# TEST_MODES = ["ERNC" , "STATIC_R", "GREEDY"] # "ERNC" , "STATIC_R", "GREEDY"
IS_TEST_MODE = True # extra test mode for env

if MY_CONST_NAME == "oneweb":       CONST_PARAM = ONEWEB_GEN1
elif MY_CONST_NAME == "starlink":   CONST_PARAM = STARLINK_S2
elif MY_CONST_NAME == 'telesat':    CONST_PARAM = TELESAT_P1
elif MY_CONST_NAME == 'test2':      CONST_PARAM = TEST2_
elif MY_CONST_NAME == 'test3':      CONST_PARAM = TEST3_
elif MY_CONST_NAME == 'test200':      CONST_PARAM = TEST200_
else:                               CONST_PARAM = TEST_

TEST_ID = 'Starlink_Shell2_0_2'
