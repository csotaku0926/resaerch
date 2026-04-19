from pettingzoo.test import parallel_api_test
from SatelliteDataDisseminationEnv import SatelliteDataDisseminationEnv

import numpy as np
from datetime import timedelta
from skyfield.api import load
from Constellation import Constellation
from param import *

N_USER = 50
CONST_ = CONST_PARAM
TMAX = CONST_.t_max
TARGET_K = CONST_.target_k

def check_roi_coverage(T_max=100, step_second=10):
    print("初始化星系與網格 (這會花幾秒鐘)...")
    env = Constellation(param=CONST_, num_users=N_USER)
    
    ts = load.timescale()
    start_dt = env.agents[0].skyfield_sat.epoch.utc_datetime()
    
    print(f"模擬開始時間: {start_dt}")
    print(f"尋找涵蓋 RoI (台灣) 的衛星...")

    # 模擬未來 2 個小時 (LEO 繞地球超過一圈)
    # 2 小時 = 7200 秒 = 720 個 step (每步 10 秒)
    
    for step in range(T_max):
        current_dt = start_dt + timedelta(seconds=step * step_second)
        current_time = ts.utc(current_dt.year, current_dt.month, current_dt.day, 
                              current_dt.hour, current_dt.minute, current_dt.second)
        
        # 檢查每顆衛星是否看到任何 Ground Grid
        visible_found = False
        for i, sat in enumerate(env.agents):
            grids = env.get_visible_grids(i, current_time)
            # target_name = "Starlink_Shell2_61_0"
            # _id = env.get_id_by_name(target_name)
            if len(grids) > 0:
                print(f"[Step {step:03d} | {current_dt.strftime('%H:%M:%S')}] 衛星 {sat.name} 進入 RoI！可視網格數: {len(grids)}\
                CV: {env.get_teg_downlink_volume(i, 2, current_time)}")
                visible_found = True
                
        # 如果你想讓畫面乾淨一點，可以把下面這行註解掉
        # if not visible_found:
        #     print(f"[Step {step:03d} | {current_dt.strftime('%H:%M:%S')}] 無衛星涵蓋")

def run_diagnostic(T_max=100):
    print("=== 衛星環境物理參數診斷開始 ===")
    
    # 1. 初始化環境
    env = SatelliteDataDisseminationEnv(const_param=CONST_, num_users=N_USER)
    obs, info = env.reset()
    
    # 獲取初始參數
    T_max = env.T_max
    n_agents = len(env.possible_agents)
    total_grids = env.constellation.n_grids
    
    # 取得當前總需求 (以所有網格的 K 總和為準)
    # 假設 get_user_fulfill_percent 是根據完成度比例
    initial_fulfill = env.constellation.get_user_fulfill_percent()
    
    print(f"[參數確認]")
    print(f"- 衛星數量: {n_agents}")
    print(f"- 網格數量: {total_grids}")
    print(f"- 最大步數 (T_max): {T_max}")
    print(f"- 初始完成度: {initial_fulfill:.2%}")
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
    env.reset()
    for s in range(T_max):
        # 動作設為 1，代表衛星嘗試把所有 Buffer 往外丟
        # 實際上會被 env 裡的 min(desired, capacity) 擋住
        actions = {agent: np.ones(env.action_spaces[agent].shape, dtype=np.float32)
                   for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # print("TEG:")
        # for agent in env.agents:
        #     print(env.constellation.get_teg_downlink_volume(env.constellation.get_id_by_name(agent), 2, s))
        fulfill = env.constellation.get_user_fulfill_percent()
        recvd = env.constellation.get_user_received_percent()
        violated = infos['Starlink_Shell2_0_0']["is_violation"]        
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

def main():
    check_roi_coverage(T_max=TMAX)
    run_diagnostic(T_max=TMAX)


if __name__ == '__main__':
    main()