from Constellation import *
# import sys

STARLINK_S2 = Const_Param(alt=540.0, inc=53.2, p=10, s=10, f=17, t_max=90, target_k=20) # step = 10
ONEWEB_GEN1 = Const_Param(alt=1200, inc=88, p=18, s=20, f=17, t_max=40, target_k=10) # K = 10, T = 40
TELESAT_P1 = Const_Param(alt=1325, inc=50.88, p=20, s=11, f=17, t_max=50, target_k=10) # 
TEST_ = Const_Param(alt=540.0, inc=53.2, p=1, s=10, f=17, t_max=90, target_k=15) # p = 13, s = 10

MY_CONST_NAME = "test"
IS_MYOTIC = True

if MY_CONST_NAME == "oneweb":
    CONST_PARAM = ONEWEB_GEN1
elif MY_CONST_NAME == "starlink":
    CONST_PARAM = STARLINK_S2
elif MY_CONST_NAME == 'telesat':
    CONST_PARAM = TELESAT_P1
else:
    CONST_PARAM = TEST_

TEST_MODES = ["MAPPO"] # "MAPPO" , "MYOTIC", "GREEDY" , "ERNC" , "STATIC_R"

TEST_ID = 'Starlink_Shell2_0_0'

# print(sys.argv)