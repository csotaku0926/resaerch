import os
import numpy as np
import math
import ray
import matplotlib.pyplot as plt
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from datetime import timedelta

# 引入你的環境與自定義模型
from SatelliteDataDisseminationEnv import SatelliteDataDisseminationEnv
from train import MAPPO_CTDE_Model, T_MAX  

### 測試開關設定 ####
IS_MINE = False
IS_ERNC = True    # 開啟 ERNC 真實模擬
IS_ONC = False    # 開啟 ONC 真實模擬
IS_MYOTIC = False
##===============###

def plot(user_numbers, avg_tx_costs):
    plt.figure(figsize=(8, 6))
    
    label_name = 'MAPPO (CTDE)'
    if IS_ERNC: label_name = 'ERNC (Simulation Baseline)'
    elif IS_ONC: label_name = 'ONC (Simulation Baseline)'
    
    plt.plot(user_numbers, avg_tx_costs, marker='o', linestyle='-', color='b', label=label_name)
    plt.title('Transmission Cost vs User Number', fontsize=14)
    plt.xlabel('Number of Users', fontsize=12)
    plt.ylabel('Average Transmission Cost (Tx Cost)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('tx_cost_vs_users_sim.png', dpi=300)
    plt.show()

def main():
    ray.init()

    # 1. 註冊環境與模型
    ModelCatalog.register_custom_model("my_ctde_model", MAPPO_CTDE_Model)
    
    def env_creator(config):
        num_users = config.get("num_users", 10)
        env = SatelliteDataDisseminationEnv(
            T_max=T_MAX, 
            num_users=num_users, 
            is_ERNC=False, # 關閉環境內的舊版硬編碼
            is_ORNC=False, 
            is_myotic=IS_MYOTIC
        )
        return ParallelPettingZooEnv(env)
        
    register_env("satellite_nc_env", env_creator)

    # 2. 載入模型權重
    algo = None
    if IS_MINE:
        checkpoint_path = "./satellite_checkpoints" 
        print(f"正在從 {checkpoint_path} 載入模型...")
        algo = Algorithm.from_checkpoint(checkpoint_path)
    elif IS_MYOTIC:
        checkpoint_path = "./satellite_myotic_checkpoints" 
        print(f"正在從 {checkpoint_path} 載入模型...")
        algo = Algorithm.from_checkpoint(checkpoint_path)

    # 3. 測試迴圈：Tx Cost v.s. User Number
    user_numbers = [10, 20, 30, 40, 50]
    num_episodes = 10 # 跑 10 局來讓機率分佈 (Monte Carlo) 更平滑
    avg_tx_costs = []

    for n_users in user_numbers:
        print(f"\n==============================")
        print(f"開始測試 User Number = {n_users}")
        
        env = env_creator({"num_users": n_users})
        episode_tx_costs = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            terminations = {"__all__": False}
            truncations = {"__all__": False}
            
            actual_env = env.par_env if hasattr(env, "par_env") else env.unwrapped
            TARGET_K = actual_env.constellation.target_k

            final_episode_cost = 0.0

            while not terminations["__all__"] and not truncations["__all__"]:
                actions = {}
                
                # 計算時間
                current_dt = actual_env.start_dt + timedelta(seconds=actual_env.current_step * actual_env.step_seconds)
                current_time = actual_env.ts.utc(current_dt.year, current_dt.month, current_dt.day,
                                                 current_dt.hour, current_dt.minute, current_dt.second)

                for agent_id, agent_obs in obs.items():
                    real_id = actual_env.constellation.get_id_by_name(agent_id)
                    action_shape = (actual_env.M + 1,)
                    
                    if IS_MINE or IS_MYOTIC:
                        actions[agent_id] = algo.compute_single_action(
                            observation=agent_obs,
                            policy_id="shared_policy",
                            explore=False 
                        )
                        
                    elif IS_ERNC or IS_ONC:
                        action = np.zeros(action_shape, dtype=np.float32)
                        
                        # 取得衛星自己目前的總 Buffer
                        current_leo_buffer = actual_env.constellation.get_leo_buffer(real_id)
                        
                        # ISL 給鄰居：有頻寬就填滿 (依照你原本設定)
                        for idx, neighbor_id in enumerate(actual_env.constellation.get_neighbors(real_id)):
                            action[idx] = float(actual_env.constellation.get_ISL_capacity(real_id, neighbor_id, current_time))
                        
                        # ---------------------------------------------------------
                        # 【核心模擬邏輯】：尋找地面最缺水的人，精準控制水閥
                        # ---------------------------------------------------------
                        visible_grids = actual_env.constellation.get_visible_grids(real_id, current_time)
                        if len(visible_grids) > 0 and current_leo_buffer > 0:
                            
                            max_intended_tx = 0.0
                            
                            for gi in visible_grids:
                                grid = actual_env.constellation.user_grids[gi]
                                # 找出這個網格裡，收最少封包的那個衰鬼
                                recv_list = grid.get_user_total_recv()
                                min_recv = min(recv_list)
                                
                                # 計算缺口
                                deficit = TARGET_K - min_recv
                                
                                if deficit > 0:
                                    # 找出這個網格最差的掉包率，用來估算需要補發的量
                                    worst_p = max([actual_env.constellation.calculate_erasure_rate(real_id, u, current_time) for u in grid.users])
                                    
                                    # 為了填補 deficit，預期需要發送的量
                                    intended = deficit / (1.0 - worst_p)
                                    
                                    # 如果是 ONC (配對效率差)，需要更多的盲發才能湊到有用的 XOR
                                    if IS_ONC: 
                                        intended *= (1.0 + 0.2 * math.log(n_users)) 
                                        
                                    if intended > max_intended_tx:
                                        max_intended_tx = intended
                            
                            # 物理限制：不能超過下行最大頻寬
                            cap = actual_env.constellation.get_downlink_capacity()
                            actual_target_flow = min(max_intended_tx, cap)
                            
                            # 將目標流量轉換為 action 的比例 (因為環境是 action_prob * leo_buffer)
                            # 如果 target_flow 是 3，自己 buffer 是 10，action 比例就是 0.3
                            action[actual_env.M] = actual_target_flow
                        
                        # 正規化輸出給環境
                        total_cap = np.sum(action)
                        if total_cap > current_leo_buffer:
                            actions[agent_id] = action / total_cap  # 按比例縮放
                        else:
                            actions[agent_id] = action / current_leo_buffer if current_leo_buffer > 0 else action

                # 環境推進下一步 (真實丟骰子 np.random.binomial)
                obs, rewards, terminations, truncations, infos = env.step(actions)
                
                # 抓取這局累積的真實 Tx Cost
                if len(infos) > 0:
                    first_agent = list(infos.keys())[0]
                    final_episode_cost = infos[first_agent].get("tx_cost", 0.0)

            episode_tx_costs.append(final_episode_cost)
            
        mean_cost = np.mean(episode_tx_costs)
        avg_tx_costs.append(mean_cost)
        print(f"User Number: {n_users} | 平均真實 Tx Cost: {mean_cost:.2f}")
        
    # 畫圖
    plot(user_numbers, avg_tx_costs)
    ray.shutdown()

if __name__ == "__main__":
    main()