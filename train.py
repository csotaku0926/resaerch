import os
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# 引入你的環境
from SatelliteDataDisseminationEnv import SatelliteDataDisseminationEnv

# =====================================================================
# 1. 神經網路大腦：CTDE 模型 (Local Actor + Global Critic)
# =====================================================================
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MAPPO_CTDE_Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 動態計算輸入維度 (攤平後的長度)
        local_obs_space = obs_space.original_space["local_obs"]
        global_state_space = obs_space.original_space["global_state"]
        
        local_dim = np.prod(local_obs_space["buffers"].shape) + np.prod(local_obs_space["contact_volumes"].shape)
        global_dim = np.prod(global_state_space["buffers"].shape) + np.prod(global_state_space["contact_volumes"].shape)

        # 【核心 1】: Local Actor 網路 (只吃 local_dim)
        self.actor = nn.Sequential(
            nn.Linear(local_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs) # 輸出流量分配比例
        )

        # 【核心 2】: Global Critic 網路 (只吃 global_dim)
        self.critic = nn.Sequential(
            nn.Linear(global_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 輸出全局 Value 評分
        )
        
        self._last_global_state = None # 暫存區，用來給 Critic 訓練

    def forward(self, input_dict, state, seq_lens):
        """執行階段：衛星呼叫 Actor 做出決策"""
        # 1. 提取 Local Obs 並攤平成 1D 向量
        local_buf = input_dict["obs"]["local_obs"]["buffers"]
        local_cv = input_dict["obs"]["local_obs"]["contact_volumes"]
        
        local_buf_flat = local_buf.reshape(local_buf.shape[0], -1)
        local_cv_flat = local_cv.reshape(local_cv.shape[0], -1)
        local_features = torch.cat([local_buf_flat, local_cv_flat], dim=1)

        # 2. 偷偷把 Global State 存起來，供等一下 Critic 訓練使用
        self._last_global_state = input_dict["obs"]["global_state"]

        # 3. Actor 僅憑 Local 資訊給出動作
        action_logits = self.actor(local_features)
        return action_logits, state

    def value_function(self):
        """訓練階段：Critic 拿上帝視角評估剛剛的表現"""
        global_buf = self._last_global_state["buffers"]
        global_cv = self._last_global_state["contact_volumes"]
        
        global_buf_flat = global_buf.reshape(global_buf.shape[0], -1)
        global_cv_flat = global_cv.reshape(global_cv.shape[0], -1)
        global_features = torch.cat([global_buf_flat, global_cv_flat], dim=1)

        # Critic 憑藉 Global 資訊給出分數
        return self.critic(global_features).squeeze(-1)

# =====================================================================
# 2. 拉格朗日回呼函數：實作 CMARL 約束
# =====================================================================
class CMARL_LagrangianCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.lambda_weight = 0.0  
        self.target_e = 0.2       # 超時率必須 <= 20%
        self.lr_lambda = 0.01     

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # 如果因為 current_step >= T_max 而結束，代表有人超時了 (Truncated)
        # 你可以根據 infos 或是 step 的回傳來更精確判定，這裡簡單判定回合長度
        is_timeout = 1.0 if episode.length >= 90 else 0.0
        episode.custom_metrics["episode_cost"] = is_timeout

    def on_train_result(self, *, algorithm, result, **kwargs):
        if "custom_metrics" in result and "episode_cost_mean" in result["custom_metrics"]:
            avg_cost = result["custom_metrics"]["episode_cost_mean"]
            
            # 動態調整懲罰權重
            constraint_violation = avg_cost - self.target_e
            self.lambda_weight = max(0.0, self.lambda_weight + self.lr_lambda * constraint_violation)
            result["custom_metrics"]["lambda_weight"] = self.lambda_weight

            # 【核心 3】: Global Reward 扣除懲罰
            result["env_runners"]["episode_reward_mean"] -= (self.lambda_weight * avg_cost)

# =====================================================================
# 3. 主程式：設定與啟動訓練
# =====================================================================
def env_creator(args):
    env = SatelliteDataDisseminationEnv()
    return PettingZooEnv(env)

def main():
    ray.init()

    # 註冊環境與模型
    env_name = "satellite_nc_env"
    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model("my_ctde_model", MAPPO_CTDE_Model)

    # 取得空間大小
    dummy_env = SatelliteDataDisseminationEnv()
    sample_agent = dummy_env.possible_agents[0]
    obs_space = dummy_env.observation_space(sample_agent)
    act_space = dummy_env.action_space(sample_agent)

    # 所有衛星共用這一個大腦 (包含 Actor 和 Critic)
    policies = {"shared_policy": (None, obs_space, act_space, {})}
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=2, rollout_fragment_length="auto") 
        .resources(num_gpus=1) # 根據硬體調整
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .callbacks(CMARL_LagrangianCallback) # 掛載拉格朗日懲罰
        .training(
            gamma=0.99,            
            lr=1e-4,               
            train_batch_size=4000, 
            clip_param=0.2,        
            model={
                # 告訴 RLlib：不要用預設網路，用我寫好的 CTDE 模型！
                "custom_model": "my_ctde_model",
            }
        )
        .debugging(log_level="WARN")
    )

    algo = config.build()
    print("神經網路 (CTDE) 建構完成，開始訓練！")

    checkpoint_dir = "./satellite_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for i in range(100):
        result = algo.train()
        reward_mean = result["env_runners"]["episode_reward_mean"]
        cost_mean = result["custom_metrics"].get("episode_cost_mean", 0.0)
        lam = result["custom_metrics"].get("lambda_weight", 0.0)
        
        print(f"Iter {i:03d} | 全局 Reward: {reward_mean:.2f} | 超時率(Cost): {cost_mean*100:.1f}% | 懲罰權重(Lambda): {lam:.3f}")

        if i % 10 == 0:
            algo.save(checkpoint_dir)

    print("訓練結束！")
    ray.shutdown()

if __name__ == "__main__":
    main()