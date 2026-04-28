import os
import numpy as np
import sys
import csv
from param import *
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
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


class MAPPO_LSTM_Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 抓取特徵維度
        local_obs_space = obs_space.original_space["local_obs"]
        global_state_space = obs_space.original_space["global_state"]
        
        self.Tw = local_obs_space["contact_volumes"].shape[1]
        self.num_local_links = local_obs_space["contact_volumes"].shape[0] # 通常是 M+1 (鄰居+地面)
        self.num_global_links = global_state_space["contact_volumes"].shape[0] # N (全網衛星數)
        
        buf_local_dim = np.prod(local_obs_space["buffers"].shape)
        buf_global_dim = np.prod(global_state_space["buffers"].shape)
        mask_local_dim = np.prod(local_obs_space["action_mask"].shape)

        # ==========================================
        # 【核心 1】: Local Actor (LSTM + MLP 雙流架構)
        # ==========================================
        self.lstm_hidden_dim = 64
        
        # 專門處理 TEG 時間序列的 LSTM
        self.local_teg_lstm = nn.LSTM(
            input_size=self.num_local_links, 
            hidden_size=self.lstm_hidden_dim, 
            batch_first=True
        )
        
        # 融合靜態 Buffer 與動態 TEG 的決策層
        self.actor_mlp = nn.Sequential(
            nn.Linear(buf_local_dim + self.lstm_hidden_dim + mask_local_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

        # ==========================================
        # 【核心 2】: Global Critic (上帝視角的 LSTM)
        # ==========================================
        self.global_teg_lstm = nn.LSTM(
            input_size=self.num_global_links, 
            hidden_size=self.lstm_hidden_dim, 
            batch_first=True
        )
        
        self.critic_mlp = nn.Sequential(
            nn.Linear(buf_global_dim + self.lstm_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 輸出 Value
        )
        
        self._last_global_state = None

    def forward(self, input_dict, state, seq_lens):
        """Actor 決策前向傳播"""
        local_buf = input_dict["obs"]["local_obs"]["buffers"] # 形狀: [Batch, M]
        local_cv = input_dict["obs"]["local_obs"]["contact_volumes"] # 形狀: [Batch, M, Tw]
        local_action_mask = input_dict["obs"]["local_obs"]["action_mask"] # [B, M]
        self._last_global_state = input_dict["obs"]["global_state"]

        # 將 [Batch, Links, Time] 轉成 [Batch, Time, Links]
        local_cv_seq = local_cv.transpose(1, 2) 
        
        # 通過 LSTM，提取時間趨勢
        _, (h_n, _) = self.local_teg_lstm(local_cv_seq)
        cv_features = h_n.squeeze(0) # 變成 [Batch, 64]

        # 特徵融合與決策
        combined_features = torch.cat([local_buf, cv_features, local_action_mask], dim=1)
        action_logits = self.actor_mlp(combined_features)
        
        return action_logits, state

    def value_function(self):
        """Critic 價值評估前向傳播"""
        global_buf = self._last_global_state["buffers"]
        global_cv = self._last_global_state["contact_volumes"]

        global_cv_seq = global_cv.transpose(1, 2)
        _, (h_n, _) = self.global_teg_lstm(global_cv_seq)
        global_cv_features = h_n.squeeze(0)

        global_features = torch.cat([global_buf, global_cv_features], dim=1)
        return self.critic_mlp(global_features).squeeze(-1)
    
class MAPPO_CTDE_Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        local_obs_space = obs_space.original_space["local_obs"]
        global_state_space = obs_space.original_space["global_state"]

        self.Tw = local_obs_space["contact_volumes"].shape[1]
        self.num_local_links = local_obs_space["contact_volumes"].shape[0] 
        self.num_global_links = global_state_space["contact_volumes"].shape[0] 
        
        buf_local_dim = np.prod(local_obs_space["buffers"].shape)
        buf_global_dim = np.prod(global_state_space["buffers"].shape)
        mask_local_dim = np.prod(local_obs_space["action_mask"].shape)
        
        local_dim = buf_local_dim + np.prod(local_obs_space["contact_volumes"].shape) + mask_local_dim
        global_dim = np.prod(global_state_space["buffers"].shape) + np.prod(global_state_space["contact_volumes"].shape)

        self.actor = nn.Sequential(
            nn.Linear(local_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs) 
        )

        self.critic = nn.Sequential(
            nn.Linear(global_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1) 
        )
        
        self._last_global_state = None 

    def forward(self, input_dict, state, seq_lens):
        local_buf = input_dict["obs"]["local_obs"]["buffers"]
        local_cv = input_dict["obs"]["local_obs"]["contact_volumes"]
        local_action_mask = input_dict["obs"]["local_obs"]["action_mask"]
        
        local_buf_flat = local_buf.reshape(local_buf.shape[0], -1)
        local_cv_flat = local_cv.reshape(local_cv.shape[0], -1)
        local_features = torch.cat([local_buf_flat, local_cv_flat, local_action_mask], dim=1)

        self._last_global_state = input_dict["obs"]["global_state"]

        action_logits = self.actor(local_features)
        return action_logits, state

    def value_function(self):
        global_buf = self._last_global_state["buffers"]
        global_cv = self._last_global_state["contact_volumes"]
        
        global_buf_flat = global_buf.reshape(global_buf.shape[0], -1)
        global_cv_flat = global_cv.reshape(global_cv.shape[0], -1)
        global_features = torch.cat([global_buf_flat, global_cv_flat], dim=1)

        return self.critic(global_features).squeeze(-1)


# =====================================================================
# 2. 拉格朗日回呼函數：實作 CMARL 約束
# =====================================================================

MY_CONST_PARAM = CONST_PARAM
T_MAX = MY_CONST_PARAM.t_max
LAMBDA_W = 1.0
TARGET_K = MY_CONST_PARAM.target_k

print(f"[參數確認]")
print(f"- 衛星 const: {MY_CONST_NAME}")
print(f"- 最大步數 (T_max): {T_MAX}")
print(f"- target K: {TARGET_K}")
print("IS_MYOTIC:", IS_MYOTIC)
print("-" * 30)


class CMARL_LagrangianCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.lambda_weight = LAMBDA_W  
        self.target_e = 0.2       # 超時率必須 <= 20%
        self.lr_lambda = 1e-4      
        self.T_max = T_MAX
        self.max_lambda = 2.0

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        last_info = episode.last_info_for(episode.get_agents()[0]) 
        
        is_vio = last_info.get("is_violation", 0.0) if last_info else 0.0
        cost = last_info.get("cost", 0.0) if last_info else 0.0
        print("cost:", cost)
        comp_time = last_info.get("time", 0.0) if last_info else 0.0
        print("time:", comp_time)
        tx_cost = last_info.get("tx_cost", 0.0) if last_info else 0.0
        
        episode.custom_metrics["is_vio"] = is_vio
        episode.custom_metrics["episode_cost"] = cost
        episode.custom_metrics["completion_time"] = comp_time
        episode.custom_metrics["transmission_cost"] = tx_cost

    def on_train_result(self, *, algorithm, result, **kwargs):
        env_metrics = result.get("env_runners", {})
        custom_metrics = env_metrics.get("custom_metrics", {})
        
        avg_cost = custom_metrics["episode_cost_mean"]
        is_violated = custom_metrics["is_vio_mean"]
        print("avg_cost:", avg_cost)
        print("is_violated:", is_violated)

        diff = max(avg_cost - self.target_e, 0.0)
        step = self.lr_lambda * diff
            
        new_lambda = self.lambda_weight + step
        self.lambda_weight = min(self.max_lambda, max(0.0, new_lambda))

        result["custom_metrics"]["lambda_weight"] = self.lambda_weight

        def broadcast_lambda(env):
            actual_env = env.par_env if hasattr(env, "par_env") else env.unwrapped
            actual_env.current_lambda = self.lambda_weight

        worker_group = getattr(algorithm, "env_runner_group", None) or algorithm.workers
        worker_group.foreach_env(broadcast_lambda)

# =====================================================================
# 3. 主程式：設定與啟動訓練
# =====================================================================

def env_creator(args):
    env = SatelliteDataDisseminationEnv(
        const_param=MY_CONST_PARAM, T_max=T_MAX, lambda_w=LAMBDA_W, is_myotic=IS_MYOTIC, test_mode=IS_TEST_MODE, num_users=N_USER,
        erasure=ERASURE
    )
    return ParallelPettingZooEnv(env)

def main():
    ray.init()

    print("\n" + "="*40)
    print("硬體與 GPU 狀態檢查")
    print("="*40)
    
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"1. PyTorch CUDA 是否可用: {cuda_available}")
    if cuda_available:
        print(f"   -> 抓到的 GPU 型號: {torch.cuda.get_device_name(0)}")
    else:
        print("   -> ⚠️ 警告: PyTorch 抓不到 GPU！你可能安裝到了 CPU 版本的 PyTorch。")

    resources = ray.cluster_resources()
    gpu_count = resources.get("GPU", 0.0)
    print(f"2. Ray 叢集可用 GPU 數量: {gpu_count}")
    if gpu_count == 0.0:
        print("   -> ⚠️ 警告: Ray 沒有偵測到任何 GPU！")
    print("="*40 + "\n")

    env_name = "satellite_nc_env"
    register_env(env_name, env_creator)
    if not IS_MYOTIC:
        ModelCatalog.register_custom_model("my_ctde_model", MAPPO_LSTM_Model)
    else:
        ModelCatalog.register_custom_model("my_ctde_model", MAPPO_CTDE_Model)

    dummy_env = SatelliteDataDisseminationEnv(
        const_param=MY_CONST_PARAM, T_max=T_MAX, lambda_w=LAMBDA_W, is_myotic=IS_MYOTIC, test_mode=IS_TEST_MODE, num_users=N_USER
    )
    sample_agent = dummy_env.possible_agents[0]
    obs_space = dummy_env.observation_space(sample_agent)
    act_space = dummy_env.action_space(sample_agent)

    n_runner = 2
    train_batch_size = dummy_env.constellation.t * T_MAX * n_runner

    policies = {"shared_policy": (None, obs_space, act_space, {})}
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .environment(env=env_name)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=n_runner, 
            num_envs_per_env_runner=1,         
            rollout_fragment_length=30,  
            sample_timeout_s=600.0
        ) 
        .resources(
            num_gpus=1,                        
            num_cpus_per_worker=1              
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            count_steps_by="agent_steps"
        )
        .callbacks(CMARL_LagrangianCallback) 
        .training(
            gamma=0.99,            
            lr_schedule=[
                [0, 1e-4],          
                [10 * train_batch_size, 5e-5],     
                [30 * train_batch_size, 1e-5]     
            ],               
            train_batch_size=train_batch_size, 
            clip_param=0.2,       
            entropy_coeff=0.01,   
            model={
                "custom_model": "my_ctde_model",
            }
        )
        .debugging(log_level="WARN")
    )

    # 1. 建立設定好的全新訓練演算法 (包含了新的 lr_schedule 等)
    algo = config.build_algo()
    print("神經網路 (CTDE) 結構建構完成！")

    # =================================================================
    # 【新增】：Finetune Checkpoint 載入功能 (使用 .restore())
    # =================================================================
    if RESTORE_CHECKPOINT_PATH and os.path.exists(RESTORE_CHECKPOINT_PATH):
        print(f"🔄 發現 Checkpoint 設定，正在從 {RESTORE_CHECKPOINT_PATH} 載入模型權重進行 Finetune...")
        algo.restore(RESTORE_CHECKPOINT_PATH)
        print("✅ 大腦權重載入成功！開始進行接續訓練 / 微調！")
    else:
        print("▶️ 未指定 Checkpoint 或路徑不存在，將從第 0 代「從頭開始」訓練。")
    # =================================================================

    checkpoint_dir = f"./satellite_{MY_CONST_NAME}_checkpoints" if not IS_MYOTIC else f"./satellite_{MY_CONST_NAME}_myotic_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_file_path = os.path.join(checkpoint_dir, f"training_log_{MY_CONST_NAME}.csv")
    csv_file = open(log_file_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Iteration", "Reward", "Cost_Rate", "Lambda", "Tx_Cost", "Comp_Time"])

    for i in range(N_TRAIN_ITER):
        result = algo.train()
        reward_mean = result["env_runners"]["episode_reward_mean"]
        
        env_metrics = result.get("env_runners", {})
        custom_metrics = env_metrics.get("custom_metrics", {})
        cost_mean = custom_metrics.get("episode_cost_mean", 0.0)
        comp_time_mean = custom_metrics.get("completion_time_mean", 0.0)
        tx_cost_mean = custom_metrics.get("transmission_cost_mean", 0.0)
        lam = result["custom_metrics"].get("lambda_weight", 0.0)
        
        print(f"Iter {i:03d} | 全局 Reward: {reward_mean:.2f} | 超時率(Cost): {cost_mean*100:.1f}% | 懲罰權重(Lambda): {lam:.3f}")
        print(f"完成步數: {comp_time_mean:.1f} | 總流量: {tx_cost_mean:.1f}")

        csv_writer.writerow([i, reward_mean, cost_mean, lam, tx_cost_mean, comp_time_mean])
        csv_file.flush() 

        if i % 10 == 0:
            algo.save(checkpoint_dir)

    print("訓練結束！")
    csv_file.close()
    ray.shutdown()

if __name__ == "__main__":
    main()