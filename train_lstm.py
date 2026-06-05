import os
import numpy as np
import sys
import csv
from param import *
import ray
from ray import tune
# from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch

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
        # 1. 取得原始的線性預測值
        # raw_logits = self.actor_mlp(combined_features)
        
        # =========================================================
        # 【腦部手術：資源分配的數學約束】
        # RLlib 預設會把 logits 切半，前半部當 Mean，後半部當 Log_Std
        # =========================================================
        # mean_len = raw_logits.shape[1] // 2
        # means = raw_logits[:, :mean_len]
        # log_stds = raw_logits[:, mean_len:]
        
        # # 建立一個隱形的「留在肚子裡 (Hold Buffer)」選項，基準分數設為 0
        # virtual_hold_logit = torch.zeros(means.shape[0], 1, device=means.device)
        
        # # 將原本的通道與隱形通道合併，然後做 Softmax 算比例
        # concat_logits = torch.cat([means, virtual_hold_logit], dim=1)
        # probs = torch.softmax(concat_logits, dim=1)
        
        # # 把隱形通道的比例拿掉，剩下的實體通道比例加總【絕對會 <= 1.0】
        # bounded_means = probs[:, :-1]
        
        # # 把修飾過、符合物理極限的 Mean 跟原來的 Std 重新組合還給 RLlib
        # action_logits = torch.cat([bounded_means, log_stds], dim=1)
        # =========================================================
        
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

        # 1. 取得原始輸出
        raw_logits = self.actor(local_features)

        # =========================================================
        # 【腦部手術：資源分配的數學約束】
        # =========================================================
        mean_len = raw_logits.shape[1] // 2
        means = raw_logits[:, :mean_len]
        log_stds = raw_logits[:, mean_len:]
        
        virtual_hold_logit = torch.zeros(means.shape[0], 1, device=means.device)
        concat_logits = torch.cat([means, virtual_hold_logit], dim=1)
        probs = torch.softmax(concat_logits, dim=1)
        bounded_means = probs[:, :-1]
        
        action_logits = torch.cat([bounded_means, log_stds], dim=1)
        # =========================================================

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
        self.target_e = 0.1       # 超時率必須 <= 20%
        self.lr_lambda = 0.001 #1e-4      
        self.T_max = T_MAX
        self.max_lambda = 2.0

    # 【新增】：回合開始時，準備一個空陣列來裝 action
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["action_isl_history"] = []
        episode.user_data["action_dl_history"] = []

    # 【新增】：每個 Step 直接抓出神經網路輸出的 action
    def on_postprocess_trajectory(
        self, *, worker, episode, agent_id, policy_id, policies,
        postprocessed_batch, original_batches, **kwargs
    ):
        if SampleBatch.ACTIONS in postprocessed_batch:
            actions = postprocessed_batch[SampleBatch.ACTIONS]
            # actions 的形狀是 (Batch_Size, Action_Dim)
            if len(actions.shape) == 2 and actions.shape[1] > 1:
                # 前面 M 個維度是 ISL (取水平加總)，最後 1 個維度是 Downlink
                isl_sums = np.sum(actions[:, :-1], axis=1)
                dl_vals = actions[:, -1]
                
                # 存入 episode 的紀錄中
                episode.user_data["action_isl_history"].extend(isl_sums.tolist())
                episode.user_data["action_dl_history"].extend(dl_vals.tolist())

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

        # 【新增】：直接在這裡算平均，寫入 metrics
        isl_hist = episode.user_data.get("action_isl_history", [])
        dl_hist = episode.user_data.get("action_dl_history", [])
        
        if len(isl_hist) > 0:
            episode.custom_metrics["avg_isl"] = np.mean(isl_hist)
            episode.custom_metrics["avg_dl"] = np.mean(dl_hist)
        else:
            episode.custom_metrics["avg_isl"] = 0.0
            episode.custom_metrics["avg_dl"] = 0.0
            

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

# def env_creator(args):
    # env = SatelliteDataDisseminationEnv(
    #     const_param=MY_CONST_PARAM, T_max=T_MAX, lambda_w=LAMBDA_W, is_myotic=IS_MYOTIC, test_mode=IS_TEST_MODE, num_users=N_USER,
    #     erasure=ERASURE, use_deficit=USE_DEFICIT
    # )
    # return ParallelPettingZooEnv(env)

def env_creator(args):
    # 從 kwargs 取出權重，如果沒有就給預設值 0.5
    omega_t = args.get("omega_t", 0.5)
    omega_c = args.get("omega_c", 0.5)
    
    env = SatelliteDataDisseminationEnv(
        const_param=MY_CONST_PARAM, lambda_w=LAMBDA_W, 
        is_myotic=IS_MYOTIC, test_mode=IS_TEST_MODE, num_users=N_USER,
        erasure=ERASURE, use_deficit=USE_DEFICIT, target_k=TARGET_K,
        omega_t=omega_t, omega_c=omega_c  # 傳給環境
    )
    return ParallelPettingZooEnv(env)

def main():
    ray.init()

    print("\n" + "="*40)
    print("硬體與 GPU 狀態檢查")
    print("="*40)
    
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

    # 建立總表來記錄所有模型最終的表現 (用於畫 Pareto 圖)
    pareto_results = []
    
    for idx, config in enumerate(PARETO_CONFIGS):
        w_t = config["omega_t"]
        w_c = config["omega_c"]
        run_name = f"WT{int(w_t * 10)}_WC{int(w_c * 10)}"
        
        print("\n" + "="*50)
        print(f"🚀 開始訓練模型: {run_name}")
        print("="*50)
        
        # 註冊帶有特定權重的環境 (每次迴圈使用不同的名字避免衝突)
        env_name = f"satellite_nc_env_{run_name}"
        register_env(env_name, lambda config_args: env_creator({**config_args, **config}))
        
        if not IS_MYOTIC:
            ModelCatalog.register_custom_model("my_ctde_model", MAPPO_LSTM_Model)
        else:
            ModelCatalog.register_custom_model("my_ctde_model", MAPPO_CTDE_Model)

        # ... (取得 dummy_env 等初始化程式碼照舊) ...
        dummy_env = env_creator(config)
        sample_agent = dummy_env.par_env.possible_agents[0]
        obs_space = dummy_env.observation_space[sample_agent]
        act_space = dummy_env.action_space[sample_agent]

        n_runner = 2
        train_batch_size = dummy_env.par_env.constellation.t * T_MAX * n_runner

        policies = {"shared_policy": (None, obs_space, act_space, {})}
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return "shared_policy"

        # 設定 PPOConfig (指定剛剛註冊的環境名稱)
        algo_config = (
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
                lr_schedule=[[0, 1e-4], [10 * train_batch_size, 5e-5], [30 * train_batch_size, 1e-5]],               
                train_batch_size=train_batch_size, 
                clip_param=0.2,       
                entropy_coeff=0.01,   
                model={"custom_model": "my_ctde_model"}
            )
            .debugging(log_level="WARN")
        )

        algo = algo_config.build_algo()

        # 2. 【新增】：將預訓練的權重載入這個新實體中
        # 假設 checkpoint_path 有在 param.py 裡面定義 (例如 checkpoint_path = "./my_base_checkpoint")
        checkpoint_path = RESTORE_CHECKPOINT_PATH
        # checkpoint_path = f"{MY_CONST_NAME}_checkpoints/{run_name}"
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"📥 成功從 {checkpoint_path} 載入模型權重！")
            algo.restore(os.path.abspath(checkpoint_path))
        else:
            print("⚠️  警告：找不到 checkpoint_path，模型將從隨機權重開始訓練！")
        
        # 針對這個權重組合，建立專屬的資料夾
        checkpoint_dir = f"{MY_CONST_NAME}_checkpoints/{run_name}"
        if (IS_MYOTIC): checkpoint_dir = f"{MY_CONST_NAME}_myotic_checkpoints/{run_name}"

        #  ========== tmp: skipping trained model
        model_dir = os.path.join(checkpoint_dir, "algorithm_state.pkl")
        if os.path.exists(model_dir):
            print(f"⚠️  detected existing pareto model at {model_dir}, skipping..")
            continue

        os.makedirs(checkpoint_dir, exist_ok=True)
        log_file_path = os.path.join(checkpoint_dir, f"training_log.csv")
        
        with open(log_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Iteration", "Reward", "Cost_Rate", "Lambda", "Tx_Cost", "Comp_Time"])
            
            # 用來計算最後幾代的平均值
            final_tx_costs = []
            final_comp_times = []
            final_fulfill = []

            for i in range(N_TRAIN_ITER):
                result = algo.train()
                # 取出數據
                reward_mean = result["env_runners"]["episode_reward_mean"]
                custom_metrics = result.get("env_runners", {}).get("custom_metrics", {})
                cost_mean = custom_metrics.get("episode_cost_mean", 0.0)
                comp_time_mean = custom_metrics.get("completion_time_mean", 0.0)
                tx_cost_mean = custom_metrics.get("transmission_cost_mean", 0.0)
                lam = result["custom_metrics"].get("lambda_weight", 0.0)
                
                print(f"Iter {i:03d} | 超時率: {cost_mean*100:.1f}% | 步數: {comp_time_mean:.1f} | 流量: {tx_cost_mean:.1f}")
                csv_writer.writerow([i, reward_mean, cost_mean, lam, tx_cost_mean, comp_time_mean])
                csv_file.flush()
                
                # 如果是最後 10 代，把數據存起來算平均
                if i >= N_TRAIN_ITER - 10:
                    final_tx_costs.append(tx_cost_mean)
                    final_comp_times.append(comp_time_mean)
                    final_fulfill.append(1 - cost_mean)

                if i % 10 == 0:
                    algo.save(checkpoint_dir)

        # 訓練結束，計算該模型的最終表現，並存入總表
        avg_comp_time = np.mean(final_comp_times)
        avg_tx_cost = np.mean(final_tx_costs)
        avg_fulfill = np.mean(final_fulfill)
        pareto_results.append((w_t, w_c, avg_comp_time, avg_tx_cost, avg_fulfill))
        
        # 儲存最後的模型，並清理記憶體以準備跑下一個權重
        algo.save(checkpoint_dir)
        algo.stop()
        
    print("\n" + "="*50)
    print(f"[{MY_CONST_NAME}] 🏆 Pareto 掃描結束！以下是所有模型的最終表現：")

    # write result csv
    log_file_path = f"{MY_CONST_NAME}_checkpoints/pareto_result.csv"
    if (IS_MYOTIC): log_file_path = f"{MY_CONST_NAME}_myotic_checkpoints/pareto_result.csv"

    csv_file = open(log_file_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["omega_t", "omega_c", "Comp_Time", "Tx_Cost"])
    for res in pareto_results:
        print(f"Omega_T: {res[0]}, Omega_C: {res[1]} -> 平均完成時間: {res[2]:.2f}, 傳輸流量: {res[3]:.2f}")
        csv_writer.writerow(list(res))
    
    csv_file.close()
    ray.shutdown()

if __name__ == "__main__":
    main()