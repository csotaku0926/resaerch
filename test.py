from pettingzoo.test import parallel_api_test
from SatelliteDataDisseminationEnv import SatelliteDataDisseminationEnv

from datetime import timedelta
from skyfield.api import load
from Constellation import Constellation

def check_roi_coverage(T_max=900):
    print("初始化星系與網格 (這會花幾秒鐘)...")
    env = Constellation(step_seconds=10)
    
    ts = load.timescale()
    start_dt = env.agents[0].skyfield_sat.epoch.utc_datetime()
    
    print(f"模擬開始時間: {start_dt}")
    print(f"尋找涵蓋 RoI (台灣) 的衛星...")

    # 模擬未來 2 個小時 (LEO 繞地球超過一圈)
    # 2 小時 = 7200 秒 = 720 個 step (每步 10 秒)
    
    for step in range(T_max):
        current_dt = start_dt + timedelta(seconds=step * 10)
        current_time = ts.utc(current_dt.year, current_dt.month, current_dt.day, 
                              current_dt.hour, current_dt.minute, current_dt.second)
        
        # 檢查每顆衛星是否看到任何 Ground Grid
        visible_found = False
        for i, sat in enumerate(env.agents):
            grids = env.get_visible_grids(i, current_time)
            target_name = "Starlink_Shell2_61_0"
            _id = env.get_id_by_name(target_name)
            if len(grids) > 0 and sat.name == target_name:
                print(f"[Step {step:03d} | {current_dt.strftime('%H:%M:%S')}] 衛星 {sat.name} 進入 RoI！可視網格數: {len(grids)}\
                CV: {env.get_teg_downlink_volume(_id, grids, 2, current_time)}")
                visible_found = True
                
        # 如果你想讓畫面乾淨一點，可以把下面這行註解掉
        # if not visible_found:
        #     print(f"[Step {step:03d} | {current_dt.strftime('%H:%M:%S')}] 無衛星涵蓋")

def main():
    env = SatelliteDataDisseminationEnv()
    # 這行會自動用隨機動作幫你跑過幾百個 step，檢查有沒有任何格式、維度不合的 bug
    parallel_api_test(env, num_cycles=1000)
    print("環境測試完美通過！可以開始訓練了！")
    # check_roi_coverage()


if __name__ == '__main__':
    main()