import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Dict
from datetime import datetime, timedelta, timezone
from skyfield.api import load
from Constellation import *
from param import *

class SatelliteDataDisseminationEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "satellite_nc_v0"}

    def __init__(self, const_param: Const_Param, num_neighbors=1, num_grids=1, T_max=90, num_users=10, lambda_w=0, target_k=20,
                 is_ORNC=False, is_ERNC=False, is_myotic=False, step_seconds=10):
        super().__init__()

        # 1. 定義 param
        self.e = 0.2            # reliability constraint: Pr(T > T_max) <= e
        self.T_max = T_max         # max time step (truncation)
        
        self.M = num_neighbors  # 鄰居數量 (Intra-tier)
        self.G = num_grids      # 覆蓋網格數量 (Inter-tier)
        self.Tw = 2             # time window for contact volume

        self.target_k = target_k
        self.step_seconds = step_seconds
        
        if (is_myotic): self.Tw = 1

        self.constellation = Constellation(param=const_param, t_max=T_max, num_users=num_users, target_k=target_k, step_seconds=step_seconds)
        self.N = len(self.constellation.agents)
        self.current_lambda = lambda_w

        self.PROGRESS_SCALE = 10.0
        self.COST_SCALE = 1.0

        # 【新增這行】預設關閉，當設為 True 時變身為 B1 基準算法
        self.is_ORNC_baseline = is_ORNC
        self.is_ERNC_baseline = is_ERNC
        self.is_myotic_baseline = is_myotic

        # 加入這兩行 (PettingZoo 鐵規則)
        self.possible_agents = [agent.name for agent in self.constellation.agents] #self.constellation.agents[:]
        self.agents = self.possible_agents[:]

        self.tx_cost_avg = {}
        for agent_name in self.agents:
            self.tx_cost_avg[agent_name] = 0.0

        # 2. 定義動作空間 (Action Space) - 連續變數
        # 每個 LEO 輸出一個長度為 M+1 的陣列，範圍 [0, 1]，代表流量分配比例
        self.action_shape = (self.M + 1,)
        self.action_spaces = {
            agent.name: Box(low=0.0, high=1.0, shape=self.action_shape, dtype=np.float32)
            for agent in self.constellation.agents
        }
        
        # 3. 定義觀測空間 (Observation Space) - 局部視角 (給 Actor 用)
        # 包含：自身 Buffer(1), 鄰居 Contact Volume(M), 地面 Contact Volume(G) (我自己可以送多少)
        self.observation_spaces = {
            agent.name: Dict({
                "local_obs": Dict({
                    "action_mask": Box(low=0.0, high=1.0, shape=self.action_shape, dtype=bool),
                    # 1. 庫存純量 (5 維)：[自己, 鄰居1, 鄰居2, 鄰居3, 鄰居4]
                    "buffers": Box(low=0.0, high=1.0, shape=(1 + self.M,), dtype=np.float32),
                
                    # 2. 接觸圖容量矩陣 (5 x T 維)
                    # Row 0: 對地廣播的未來 T 步
                    # Row 1~4: 給四個鄰居 ISL 的未來 T 步
                    "contact_volumes": Box(low=0.0, high=1.0, shape=(1 + self.M, self.Tw), dtype=np.float32)
                }), 
                # 3. Global State (N) 
                "global_state": Dict({
                    # "action_mask": Box(low=0.0, high=1.0, shape=self.action_shape, dtype=np.float32),
                    "buffers": Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
                    "contact_volumes": Box(low=0.0, high=1.0, shape=(self.N, self.Tw), dtype=np.float32)
                }) 
            })
            for agent in self.constellation.agents
        }

        # 初始化 Skyfield 時間與環境參數
        self.ts = load.timescale()
        self.current_step = 0
        self.episode_tx_cost = 0.0
        self.start_dt = datetime(2026, 4, 1, 0, 0, 0)
        self.reward_factor = 1.0 # scale down reward
        self.reward_factor_time = 1e5

        # 通訊參數
        self.broadcast_rate_bps = 30e6 * 1.0 
        self.packet_size_bits = 80e6 # 10 MB = 80 Mbits

    def reset(self, seed=None, options=None):
        """回合開始: 重置時間、位置、Buffer 與 DoF 進度"""
        self.agents = self.possible_agents[:]

        self.current_step = 0
        self.start_dt = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # 這裡重置你的 LEO buffers 與地面的 received_dof
        self.constellation.reset()

        for agent_name in self.agents:
            self.tx_cost_avg[agent_name] = 0.0

        # 取得初始觀測值
        # 【效能優化】：在這裡統一算一次全局狀態
        current_global_state = self.state()
        current_time = self.ts.from_datetime(self.start_dt)
        observations = {
            agent_name: {
            # "action_mask" : np.zeros(self.M + 1, dtype=np.float32),
            "local_obs" : self._get_obs(self.constellation.get_id_by_name(agent_name), current_time),
            "global_state" : current_global_state 
            } for agent_name in self.agents
        }

        self.episode_tx_cost = 0.0
        infos = {
            agent_name: {
                "is_violation" : 0.0, 
                "cost" : 0,  # ratio of receiver that not decode yet
                "tx_cost": self.episode_tx_cost,
                "time": self.constellation.get_finish_time_cost()
            } for agent_name in self.agents
        }
        
        return observations, infos

    def step(self, actions):
        """每一回合的環境互動 (核心邏輯)"""
        # 1. 更新 Skyfield 時間
        current_dt = self.start_dt + timedelta(seconds=self.current_step * self.step_seconds)
        current_time = self.ts.utc(current_dt.year, current_dt.month, current_dt.day,
                                   current_dt.hour, current_dt.minute, current_dt.second)
        
        # 2. 執行流量分配邏輯 (Flow Allocation)
        # MEO source distribution
        self.constellation.meo_broadcast_to_leos(current_time)

        # 3. 計算 Reward (獎勵設計)
        rewards = {}
        ft = self.constellation.get_finish_time_cost()

        max_buf = self.constellation.get_leo_max_buffer()

        all_done = bool(self.check_all_grids_fulfilled())
        is_truncated = bool(self.current_step >= self.T_max - 1)
        is_done = all_done or is_truncated
        
        for agent_name in self.agents:
            # name_i = agent_i.name
            i = self.constellation.get_id_by_name(agent_name)
            rewards[agent_name] = 0 
            # 神經網路輸出的比例 (0~1)
            raw_action = actions[agent_name] # {"LEO_i": action}

            # 【總量守恆限制】確保分配總和不超過 1.0
            action_sum = np.sum(raw_action)
            if action_sum > 1.0:
                # 如果總和大於 1，就按比例壓縮 (例如 2.5 會被等比例壓縮成總和剛好 1.0)
                action_probs = raw_action / action_sum
            else:
                # 如果總和小於或等於 1，代表衛星想「保留」一部分封包在自己的 Buffer 裡不傳，這是合法的！
                action_probs = raw_action

            # --- 套用物理拘束 (Contact Volume) ---
            # Intra-tier (給鄰居)
            acc_cost = 0.0
            acc_max_cost = 0.0

            # 【補救核心】：現場重新計算 Action Mask，並強制攔截無效動作
            action_mask = np.zeros(self.M + 1, dtype=np.float32)
            
            for j, agent_j in enumerate([self.constellation.get_neighbors(i)[0]]):
                
                if self.constellation.get_ISL_capacity(i, agent_j, current_time) > 0:
                    teg_j = self.constellation.get_teg_downlink_volume(agent_j, self.Tw, current_time)
                    if np.sum(teg_j) > 0:  
                        action_mask[j] = 1.0
                
                contact_capacity = self.constellation.get_ISL_capacity(i, agent_j, current_time)
                buf_i = self.constellation.get_leo_buffer(i)
                actual_flow = min(buf_i, action_probs[j] * contact_capacity * action_mask[j])
                self.constellation.transfer_buffer(sat_id=i, neighbor=agent_j, amount=actual_flow)
                # count in "actual_flow"
                acc_cost += actual_flow
                acc_max_cost += max_buf
                self.episode_tx_cost += actual_flow

            # Inter-tier (給地面)
            contact_capacity = self.constellation.get_downlink_capacity()
            # print("ENV:", i, current_time)
            if len(self.constellation.get_visible_grids(i, current_time)) > 0:
                action_mask[self.M] = 1.0
                
            buf_i = self.constellation.get_leo_buffer(i)
            actual_flow = min(buf_i, action_probs[self.M] * contact_capacity * action_mask[self.M]) 

            acc_cost += actual_flow
            acc_max_cost += max_buf
            self.tx_cost_avg[agent_name] += acc_cost / acc_max_cost
            self.episode_tx_cost += actual_flow

            # 在 step() 裡，下載前先記錄舊進度
            # if (agent_name == TEST_ID):
            #     print(action_probs[self.M], contact_capacity, action_mask[self.M])
            old_fulfill = self.constellation.get_user_fulfill_percent()

            # for g in range(self.constellation.get_visible_grids(i)):
            self.constellation.download_to_grid(i, amount=actual_flow, current_time=current_time)

            # 計算進度增量 → 這才是真正的正向信號
            new_fulfill = self.constellation.get_user_fulfill_percent()
            delta_fulfill = new_fulfill - old_fulfill

            rewards[agent_name] += self.PROGRESS_SCALE * delta_fulfill
            rewards[agent_name] -= self.COST_SCALE * (acc_cost / acc_max_cost)

            # time cost
            rewards[agent_name] -= 1 / self.T_max #self.reward_factor_time
        
        # 4. 判斷是否結束 (所有目標網格的 DoF 都達到 K)
        # update finish time
        self.constellation.set_finish_time(self.current_step)
        cost = float(1.0 - self.constellation.get_user_fulfill_percent())
        terminations = {agent_name: is_done for agent_name in self.agents}
        truncations = {agent_name: is_truncated for agent_name in self.agents} # 是否超時
        is_violation = 1.0 if (is_truncated and not all_done) else 0.0

        # if all_done:
        #     for agent_name in self.agents:
        #         rewards[agent_name] += 50.0    

        if is_done:
            for agent_name in self.agents:
                rewards[agent_name] -= self.current_lambda * cost
                rewards[agent_name] -= self.COST_SCALE * (self.tx_cost_avg[agent_name] / self.T_max)
        
        # 5. 更新狀態
        self.current_step += 1
        next_dt = self.start_dt + timedelta(seconds=self.current_step * self.step_seconds)
        next_time = self.ts.utc(next_dt.year, next_dt.month, next_dt.day,
                                   next_dt.hour, next_dt.minute, next_dt.second)
        
        # 【效能優化】：在這裡統一算一次全局狀態
        current_global_state = self.state()
        
        observations = {
            agent_name: {
            "local_obs" : self._get_obs(self.constellation.get_id_by_name(agent_name), next_time),
            "global_state" : current_global_state 
            } for agent_name in self.agents
        }
        infos = {
            agent_name: {
                "is_violation" : is_violation, 
                "cost" : cost,  # ratio of receiver that not decode yet
                "tx_cost": self.episode_tx_cost,
                "time": ft,
                "lambda": self.current_lambda
            } for agent_name in self.agents
        }

        # 當所有任務完成，清空 agents 列表 (PettingZoo 規範)
        # if all_done or is_truncated:
        #     self.agents = []

        return observations, rewards, terminations, truncations, infos

    def state(self):
        """
        【全局狀態】
        這是給 Centralized Critic (上帝視角) 看的。
        Actor 只看 _get_obs (局部 K-DoF 缺口)，但 Critic 可以看全網所有的狀態矩陣。
        1. buffer state of global
        2. contact volume
        """
        current_dt = self.start_dt + timedelta(seconds=self.current_step * self.step_seconds)
        current_time = self.ts.utc(current_dt.year, current_dt.month, current_dt.day,
                                   current_dt.hour, current_dt.minute, current_dt.second)
        
        global_buf = []
        max_buf = self.constellation.get_leo_max_buffer()

        # 1. 收集全網所有衛星的 Buffer (slow)
        for agent_name in self.possible_agents:
            if agent_name in self.agents:
                agent_id = self.constellation.get_id_by_name(agent_name)
                buf = self.constellation.get_leo_buffer(agent_id)
                global_buf.append(np.clip(buf / max_buf, 0.0, 1.0))
            else:
                global_buf.append(0.0)

        # 2. 收集全網所有衛星的 TEG (永遠掃描 possible_agents)
        global_cv = []
        for agent_name in self.possible_agents:
            if agent_name in self.agents:
                agent_id = self.constellation.get_id_by_name(agent_name)
                # covered_grids = self.constellation.get_visible_grids(agent_id, current_time)
                my_teg = self.constellation.get_teg_downlink_volume(agent_id, self.Tw, current_time)
                global_cv.append(my_teg)
            else:
                global_cv.append([0.0] * self.Tw) # 死掉就補連續的 0

        # return np.array(global_state, dtype=np.float32)
        return {
            "buffers": np.array(global_buf, dtype=np.float32),
            "contact_volumes": np.array(global_cv, dtype=np.float32)
        }

    def _get_obs(self, agent_id, current_time):
        """計算局部觀測值給 Actor"""
        # ==========================================
        # 特徵 1: 自己的 Buffer (1 維) + 鄰居的 Buffer (4 維)
        # ==========================================
        buf = self.constellation.get_leo_buffer(agent_id)
        max_buf = self.constellation.get_leo_max_buffer()
        norm_buf = np.clip(buf / max_buf, 0.0, 1.0)
        bufs = [norm_buf]

        for j in [self.constellation.get_neighbors(agent_id)[0]]:
            buf_j = self.constellation.get_leo_buffer(j)
            norm_buf_j = np.clip(buf_j / max_buf, 0.0, 1.0)
            bufs.append(norm_buf_j)

        # ==========================================
        # 特徵 2:  Contact Volume (5 * T 維)
        # ==========================================
        cv_matrix = np.zeros((1 + self.M, self.Tw), dtype=np.float32)
        # covered_grids = self.constellation.get_visible_grids(agent_id, current_time)
        
        my_teg = self.constellation.get_teg_downlink_volume(agent_id, self.Tw, current_time)
        # 填入自己對地的 TEG
        cv_matrix[0, :] = my_teg
        
        # 填入鄰居的 TEG
        for idx, j in enumerate([self.constellation.get_neighbors(agent_id)[0]]):
            # grids_j = self.constellation.get_visible_grids(j, current_time)
            teg_j = self.constellation.get_teg_downlink_volume(j, self.Tw, current_time)
            cv_matrix[idx + 1, :] = teg_j

        # action mask
        action_mask = np.zeros(self.M + 1, dtype=np.float32)

        # if (np.any(cv_matrix)):
        #     print(cv_matrix)
        
        # 1. 檢查鄰居 (ISL) 是否活著
        for idx, j in enumerate([self.constellation.get_neighbors(agent_id)[0]]):
            if self.constellation.get_ISL_capacity(agent_id, j, current_time) > 0:
                action_mask[idx] = 1.0
                
        # # 2. 檢查對地 (Downlink) 是否活著
        if len(self.constellation.get_visible_grids(agent_id, current_time)) > 0:
            if self.constellation.get_downlink_capacity() > 0:
                action_mask[self.M] = 1.0

        return {
            "action_mask": action_mask,
            "buffers": np.array(bufs, dtype=np.float32),
            "contact_volumes": cv_matrix
        }

    def check_all_grids_fulfilled(self):
        total_recv_percent = self.constellation.get_user_fulfill_percent()
        target = float(1 - self.e) # constraint
        return (total_recv_percent >= target)

