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


### test settings ####
IS_ERNC = True
IS_ONC = False
IS_MYOTIC = False
##===============###

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
    checkpoint_path = "./satellite_checkpoints/checkpoint_000300" 
    
    print(f"正在從 {checkpoint_path} 載入模型...")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # ==========================================
    # 3. 測試迴圈：Tx Cost v.s. User Number
    # ==========================================
    user_numbers = [10, 20, 30, 40, 50]
    num_episodes = 50 # 每個設定跑 50 回合做蒙地卡羅平均
    
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
            total_tx_cost = 0.0

            while not terminations["__all__"] and not truncations["__all__"]:
                # decentralized execution: 每個 agent 獨立給動作
                actions = {}
                for agent_id, agent_obs in obs.items():
                    # compute_single_action 會自動呼叫你模型裡的 forward 函數
                    # explore=False 確保 Actor 只輸出機率最高的 Deterministic Action
                    actions[agent_id] = algo.compute_single_action(
                        observation=agent_obs,
                        policy_id="shared_policy",
                        explore=False 
                    )
                
                # 環境推進下一步
                obs, rewards, terminations, truncations, infos = env.step(actions)
                
                # 從 infos 中擷取你的 tx_cost
                # 假設 PettingZoo 環境的 info 字典結構是 info[agent_id]["tx_cost"]
                step_tx_cost = 0.0
                for agent_id, agent_info in infos.items():
                    step_tx_cost += agent_info.get("tx_cost", 0.0)
                
                total_tx_cost += step_tx_cost

            episode_tx_costs.append(total_tx_cost)
            
        # 計算該 User Number 下的平均傳輸成本
        mean_cost = np.mean(episode_tx_costs)
        avg_tx_costs.append(mean_cost)
        print(f"User Number: {n_users} | 平均 Tx Cost: {mean_cost:.2f}")

    ray.shutdown()

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

if __name__ == "__main__":
    main()