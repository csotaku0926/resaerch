from pettingzoo.test import parallel_api_test
from SatelliteDataDisseminationEnv import SatelliteDataDisseminationEnv

import numpy as np
from datetime import timedelta, datetime
from skyfield.api import load
from Constellation import Constellation
from param import *

# N_USER = 50
CONST_ = CONST_PARAM
TMAX = CONST_.t_max
TARGET_K = CONST_.target_k
N_NEIGHBOR = 2

def check_roi_coverage(T_max=100, step_second=10):
    print("初始化星系與網格 (這會花幾秒鐘)...")
    # env = Constellation(param=CONST_, num_users=N_USER)
    env = SatelliteDataDisseminationEnv(const_param=CONST_, num_users=N_USER)
    
    ts = load.timescale()
    start_dt = env.constellation.agents[0].skyfield_sat.epoch.utc_datetime()
    
    print(f"模擬開始時間: {start_dt}")
    print(f"尋找涵蓋 RoI (台灣) 的衛星...")

    # 模擬未來 2 個小時 (LEO 繞地球超過一圈)
    # 2 小時 = 7200 秒 = 720 個 step (每步 10 秒)
    
    for step in range(T_max):
        current_dt = start_dt + timedelta(seconds=step * step_second)
        current_time = ts.utc(current_dt.year, current_dt.month, current_dt.day, 
                              current_dt.hour, current_dt.minute, current_dt.second)
        
        # 檢查每顆衛星是否看到任何 Ground Grid
        for i, sat in enumerate(env.constellation.agents):
            grids = env.constellation.get_visible_grids(i, current_time)
            # target_name = "Starlink_Shell2_61_0"
            # _id = env.get_id_by_name(target_name)
            if len(grids) > 0:
                print(f"[Step {step:03d} | {current_dt.strftime('%H:%M:%S')}] 衛星 {sat.name} 進入 RoI ! 可視網格數: {len(grids)}\
                CV: {env.constellation.get_teg_downlink_volume(i, 2, current_time)} buf: {sat.get_buffer()}")
                visible_found = True

                
def run_diagnostic(step_second=10, is_all_in=True, do_log=False, n_user=100):
    print("=== 衛星環境物理參數診斷開始 ===")
    
    # 1. 初始化環境
    env = SatelliteDataDisseminationEnv(const_param=CONST_, num_users=n_user, step_seconds=step_second, test_mode=True)
    obs, info = env.reset()
    
    # 獲取初始參數
    T_max = env.T_max
    n_agents = len(env.possible_agents)
    total_grids = len(env.constellation.user_grids)
    
    # 取得當前總需求 (以所有網格的 K 總和為準)
    # 假設 get_user_fulfill_percent 是根據完成度比例
    initial_fulfill = env.constellation.get_user_fulfill_percent()
    
    ts = load.timescale()
    start_dt = datetime(2026, 4, 1, 0, 0, 0) #env.constellation.agents[0].skyfield_sat.epoch.utc_datetime()
    
    print(f"模擬開始時間: {start_dt}")

    print(f"[參數確認]")
    print(f"- 衛星數量: {n_agents}")
    print(f"- 網格數量: {total_grids}")
    print(f"- user number: {n_user}")
    print(f"- 最大步數 (T_max): {T_max}")
    print(f"- action is all ones: {is_all_in}")
    print("-" * 30)

    # 2. 測試：如果所有衛星「完全躺平」(零動作)
    # print("[測試 1: 零動作測試 (純觀察 MEO 補給與時間流逝)]")
    # for s in range(5):
    #     # 建立全 0 的動作 (不傳輸任何資料)
    #     actions = {agent: np.zeros(env.action_spaces[agent].shape, dtype=np.float32) 
    #                for agent in env.agents}
    #     obs, rewards, terms, truncs, infos = env.step(actions)
        
    #     fulfill = env.constellation.get_user_fulfill_percent()
    #     # 看看 MEO 有沒有把 LEO 的 Buffer 填滿
    #     total_buffer = sum([env.constellation.get_leo_buffer(i) for i in range(n_agents)])
        
    #     print(f"Step {s+1} | 全網總 Buffer: {total_buffer} packet | 完成度: {fulfill:.2%}")

    # print("-" * 30)

    # 3. 測試：如果所有衛星「全力輸出」(動作設為 1)
    print("[測試 2: 最大輸出測試 (確認產力上限)]")
    test_id = env.constellation.get_id_by_name(TEST_ID)
    env.reset()
    for s in range(T_max):
        # 動作設為 1，代表衛星嘗試把所有 Buffer 往外丟
        # 實際上會被 env 裡的 min(desired, capacity) 擋住

        current_dt = start_dt + timedelta(seconds=s * step_second)
        current_time = ts.utc(current_dt.year, current_dt.month, current_dt.day, 
                              current_dt.hour, current_dt.minute, current_dt.second)
        

        if is_all_in:
            actions = {agent: np.zeros(env.action_spaces[agent].shape, dtype=np.float32)
                        for agent in env.agents}
            for agent in env.agents: actions[agent][N_NEIGHBOR] = 1.0

        else:
            arr = [0.1] * N_NEIGHBOR + [1.0 - 0.1 * N_NEIGHBOR]
            actions = {agent: np.array(arr, dtype=np.float32)
                        for agent in env.agents}

        # example toy strategy perform better than "baseline"
        # CV_0 = env.constellation.get_teg_downlink_volume(test_id, 2, current_time) 
        # if (CV_0[0] > 0.002):
        #     actions = {agent: np.ones(env.action_spaces[agent].shape, dtype=np.float32)
        #             for agent in env.agents}
        # else:
        #     actions = {agent: np.zeros(env.action_spaces[agent].shape, dtype=np.float32)
        #             for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if (do_log): print("[reward]:", rewards[TEST_ID])
        if (do_log): print("[OBS]: ", obs[TEST_ID]["local_obs"])
        # print("[INFO]: ", infos[TEST_ID]["sent_user_count"]) <-- problematic
        
         # 檢查每顆衛星是否看到任何 Ground Grid
        for i, sat in enumerate(env.constellation.agents):
            grids = env.constellation.get_visible_grids(i, current_time)
            # target_name = "Starlink_Shell2_61_0"
            # _id = env.get_id_by_name(target_name)
            CV_vec = env.constellation.get_teg_downlink_volume(i, 2, current_time)
            if len(grids) > 0 and do_log:
                # print("TEST:", env.constellation.get_teg_downlink_volume(test_id, 2, current_time))
                print(f"[Step {s:03d} | {current_dt.strftime('%H:%M:%S')}] 衛星 {sat.name} 進入 RoI ! 可視網格: {grids}\
                CV: {CV_vec} buf: {sat.get_buffer()}")
                
        fulfill = env.constellation.get_user_fulfill_percent()
        recvd = env.constellation.get_user_received_percent()
        violated = infos[TEST_ID]["is_violation"]        
        # each 10 step
        if (s+1) % 10 == 0:
            print(f"Step {s+1} | 完成度: {fulfill:.2%} | 已收到 avg: {recvd}")

        # terminate
        if fulfill >= (1 - env.e) or (s+1) == T_max:
            print(f">>> 最終結果 @ Step {s+1}: 完成度 {fulfill:.2%} | 是否結束: {not violated}")
            break

    # print(infos)
    print("\n=== 診斷結論 ===")
    if fulfill >= (1 - env.e) and (s+1) < 5:
        print("[警告] 任務太簡單了！衛星在 5 步內就傳完了。這會導致 Cost 永遠是 0。")
        print("建議：增加網格需求量 K，或減少 broadcast_rate_bps。")
    elif fulfill < 0.01:
        print("[警告] 任務太難或物理參數斷開了！跑滿 90 步完成度幾乎沒動。")
        print("建議：檢查 download_to_grid 邏輯，或增加傳輸速率。")
    else:
        print("[正常] 物理參數看起來在合理範圍，任務需要一段時間才能完成。")

def basic_test():
    env = SatelliteDataDisseminationEnv()
    # 這行會自動用隨機動作幫你跑過幾百個 step，檢查有沒有任何格式、維度不合的 bug
    parallel_api_test(env, num_cycles=1000)
    print("環境測試完美通過！可以開始訓練了！")

def reset_test():
    env = SatelliteDataDisseminationEnv(const_param=CONST_, num_users=N_USER, step_seconds=10, test_mode=True, num_grids=10)
    
    ts = load.timescale()
    start_dt = datetime(2026, 4, 1, 0, 0, 0) #env.constellation.agents[0].skyfield_sat.epoch.utc_datetime()
    
    current_dt = start_dt
    current_time = ts.utc(current_dt.year, current_dt.month, current_dt.day, 
                            current_dt.hour, current_dt.minute, current_dt.second)
    
    print("[INFO] satellite const pos")
    print([s.skyfield_sat.at(current_time).subpoint().latitude.degrees for s in env.constellation.agents])
    print("[INFO] user lat")
    print([u.lat for u in env.constellation.user_grids[0].users])

    obs, info = env.reset()
    print("[INFO] satellite const pos")
    print([s.skyfield_sat.at(current_time).subpoint().latitude.degrees for s in env.constellation.agents])
    print("[INFO] user lat")
    print([u.lat for u in env.constellation.user_grids[0].users])

def main():
    for n_user in [40]:
        run_diagnostic(is_all_in=True, n_user=n_user)
        run_diagnostic(is_all_in=False, n_user=n_user)
   

if __name__ == '__main__':
    main()