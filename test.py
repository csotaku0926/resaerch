import os
import numpy as np
import ray
import matplotlib.pyplot as plt
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# 引入你的環境與自定義模型 (與 train.py 相同)
from SatelliteDataDisseminationEnv import SatelliteDataDisseminationEnv
from train import MAPPO_CTDE_Model, T_MAX  # 假設你把前面的程式碼存成 train.py

from datetime import timedelta

### test settings ####
IS_MINE = False
IS_ERNC = False
IS_ONC = True
IS_MYOTIC = False
##===============###

def plot(user_numbers, avg_tx_costs):
        # ==========================================
    # 4. 畫圖 (Tx Cost v.s. User Number)
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.plot(user_numbers, avg_tx_costs, marker='o', linestyle='-', color='b', label='MAPPO (CTDE)')
    
    plt.title('Transmission Cost vs User Number', fontsize=14)
    plt.xlabel('Number of Users', fontsize=12)
    plt.ylabel('Average Transmission Cost (Tx Cost)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 儲存圖片或顯示出來
    plt.savefig('tx_cost_vs_users.png', dpi=300)
    plt.show()

def main():
    ray.init()

    # ==========================================
    # 1. 重新註冊環境與神經網路模型 (必須與訓練時一致)
    # ==========================================
    ModelCatalog.register_custom_model("my_ctde_model", MAPPO_CTDE_Model)
    
    # 這裡你需要確保你的環境支援動態調整使用者/節點數量 (例如傳入 num_users)
    def env_creator(config):
        # config 會帶入我們想要測試的使用者數量
        num_users = config.get("num_users", 10)
        env = SatelliteDataDisseminationEnv(T_max=T_MAX, num_users=num_users, 
                                            is_ERNC=IS_ERNC, is_ORNC=IS_ONC, is_myotic=IS_MYOTIC)
        return ParallelPettingZooEnv(env)
        
    register_env("satellite_nc_env", env_creator)

    # ==========================================
    # 2. 載入訓練好的權重 (Checkpoint)
    # ==========================================
    # 請將下方路徑替換成你實際的 checkpoint 資料夾路徑
    # 通常在 ./satellite_checkpoints/checkpoint_000XXX 裡面
    
    if (IS_MINE):
        checkpoint_path = "./satellite_checkpoints" 
        print(f"正在從 {checkpoint_path} 載入模型...")
        algo = Algorithm.from_checkpoint(checkpoint_path)
    elif (IS_MYOTIC):
        checkpoint_path = "./satellite_myotic_checkpoints" 
        print(f"正在從 {checkpoint_path} 載入模型...")
        algo = Algorithm.from_checkpoint(checkpoint_path)

    # ==========================================
    # 3. 測試迴圈：Tx Cost v.s. User Number
    # ==========================================
    user_numbers = [10, 20, 30, 40, 50]
    num_episodes = 1 # 每個設定跑 50 回合做蒙地卡羅平均
    
    avg_tx_costs = []

    for n_users in user_numbers:
        print(f"\n開始測試 User Number = {n_users}")
        
        # 建立指定使用者數量的環境
        env = env_creator({"num_users": n_users})
        episode_tx_costs = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            terminations = {"__all__": False}
            truncations = {"__all__": False}
            
            # 【關鍵定義 1】: 脫掉 Ray 的防護衣，取得最底層的原始環境
            actual_env = env.par_env if hasattr(env, "par_env") else env.unwrapped
            
            # 準備記錄這一局最後的成本 (不用每個 step 累加，因為環境裡面已經幫你累加了)
            final_episode_cost = 0.0

            while not terminations["__all__"] and not truncations["__all__"]:
                actions = {}
                
                # 【關鍵定義 2】: 計算當下真實時間 (Rule-based 需要看時間算頻寬)
                current_dt = actual_env.start_dt + timedelta(seconds=actual_env.current_step * actual_env.step_seconds)
                current_time = actual_env.ts.utc(current_dt.year, current_dt.month, current_dt.day,
                                                 current_dt.hour, current_dt.minute, current_dt.second)

                for agent_id, agent_obs in obs.items():
                    # 【關鍵定義 3】: 把字串 ID 轉成整數 ID
                    real_id = actual_env.constellation.get_id_by_name(agent_id)
                    # 【關鍵定義 4】: 定義輸出的 action 陣列形狀 (M個鄰居 + 1個對地)
                    action_shape = (actual_env.M + 1,)

                    # ==========================================
                    # 決策樹：根據開關選擇大腦
                    # ==========================================
                    if IS_MINE or IS_MYOTIC:
                        actions[agent_id] = algo.compute_single_action(
                            observation=agent_obs,
                            policy_id="shared_policy",
                            explore=False 
                        )
                        
                    elif IS_ERNC:
                        # ER-NC: 無腦廣播，有連線就填 1.0
                        action = np.zeros(action_shape, dtype=np.float32)
                        for idx, neighbor_id in enumerate(actual_env.constellation.get_neighbors(real_id)):
                            if actual_env.constellation.get_ISL_capacity(real_id, neighbor_id, current_time) > 0:
                                action[idx] = 1.0
                                
                        if len(actual_env.constellation.get_visible_grids(real_id, current_time)) > 0:
                            if actual_env.constellation.get_downlink_capacity() > 0:
                                action[actual_env.M] = 1.0
                                
                        actions[agent_id] = action
                        
                    elif IS_ONC:
                        # ONC: 按頻寬比例分配
                        action = np.zeros(action_shape, dtype=np.float32)
                        for idx, neighbor_id in enumerate(actual_env.constellation.get_neighbors(real_id)):
                            action[idx] = float(actual_env.constellation.get_ISL_capacity(real_id, neighbor_id, current_time))
                            
                        if len(actual_env.constellation.get_visible_grids(real_id, current_time)) > 0:
                            action[actual_env.M] = float(actual_env.constellation.get_downlink_capacity())
                        
                        total_cap = np.sum(action)
                        actions[agent_id] = (action / total_cap) if total_cap > 0 else action

                # 環境推進下一步
                obs, rewards, terminations, truncations, infos = env.step(actions)
                
                # 【修復邏輯 Bug】：你的環境 episode_tx_cost 是全域累加的
                # 所以我們只需要抓「任何一個 agent」的 info 紀錄，並在迴圈結束時存下來就好
                if len(infos) > 0:
                    first_agent = list(infos.keys())[0]
                    final_episode_cost = infos[first_agent].get("tx_cost", 0.0)

            # 迴圈結束，把這局最後的總成本加進 list
            episode_tx_costs.append(final_episode_cost)
            
        # 計算該 User Number 下的平均傳輸成本
        mean_cost = np.mean(episode_tx_costs)
        avg_tx_costs.append(mean_cost)
        print(f"User Number: {n_users} | 平均 Tx Cost: {mean_cost:.2f}")

    ray.shutdown()

if __name__ == "__main__":
    main()