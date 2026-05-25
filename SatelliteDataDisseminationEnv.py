import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Dict
from datetime import datetime, timedelta, timezone
from skyfield.api import load
from Constellation import *
from param import *

class SatelliteDataDisseminationEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "satellite_nc_v0"}

    def __init__(self, const_param: Const_Param, num_grids=1, num_users=10, lambda_w=0, target_k=20, erasure=0.1,
                 is_unicast=False, is_ORNC=False, is_ERNC=False, is_myotic=False, step_seconds=10, test_mode=False, use_deficit=False,
                 omega_t=0.5, omega_c=0.5, seed=1234):
        super().__init__()

        # 1. 定義 param
        self.e = 0.2         # reliability constraint: Pr(T > T_max) <= e
        self.T_max = const_param.t_max         # max time step (truncation)
        
        self.M = const_param.n_neighbor  # 鄰居數量 (Intra-tier)
        self.G = num_grids      # 覆蓋網格數量 (Inter-tier)
        self.Tw = 2             # time window for contact volume

        self.target_k = target_k
        self.step_seconds = step_seconds
        self.is_unicast = is_unicast

        # pareto frontier param
        self.omega_t = omega_t
        self.omega_c = omega_c

        # # ablation param
        self.enable_RLNC = const_param.enable_RLNC
        
        if (is_myotic): self.Tw = 1

        self.grid_scale = const_param.grid_scale
        self.seed = seed
        self.num_users = num_users

        self.constellation = Constellation(
            param=const_param, 
            t_max=self.T_max, 
            num_users=num_users, 
            target_k=target_k, 
            step_seconds=step_seconds,
            test_mode=test_mode,
            grid_scale=self.grid_scale,
            erasure=erasure,
            seed=self.seed
        )
        self.N = len(self.constellation.agents)
        self.current_lambda = lambda_w

        self.PROGRESS_SCALE = omega_t #1.0
        self.COST_SCALE = omega_c

        # 加入這兩行 (PettingZoo 鐵規則)
        self.possible_agents = [agent.name for agent in self.constellation.agents] #self.constellation.agents[:]
        self.agents = self.possible_agents[:]

        # 初始化 N x N 的虛擬帳本 (row: 資料原創者, col: 資料目前持有者)
        # self.ledger = np.zeros((self.N, self.N), dtype=np.float32)
        # self.agent_progress = np.zeros(self.N, dtype=np.float32)

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
        self.ISL_cost_factor = 0.9

        # 通訊參數
        self.broadcast_rate_bps = 30e6 * 1.0 
        self.packet_size_bits = 80e6 # 10 MB = 80 Mbits

        # ARQ queue
        self.global_arq_queue = {}
        if not self.enable_RLNC:
            total_users = sum([len(g.users) for g in self.constellation.user_grids])
            all_user_ids = set(range(total_users))
            # 建立 Queue：封包 ID 0 ~ target_k-1，一開始每個封包都缺所有人的 ACK
            self.global_arq_queue = {pkt_id: set(all_user_ids) for pkt_id in range(self.target_k)}

    def reset(self, seed=None, options=None):
        """回合開始: 重置時間、位置、Buffer 與 DoF 進度"""
        self.agents = self.possible_agents[:]

        self.current_step = 0
        self.start_dt = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # 這裡重置你的 LEO buffers 與地面的 received_dof
        self.constellation.reset()

        if not getattr(self, 'enable_RLNC', True):
            # 算出全網格總共有多少個 users
            n_users = self.constellation.users_per_grid * self.constellation.n_grids #sum([len(g.users) for g in self.constellation.user_grids])
            all_user_ids = set(range(n_users))
            # 建立 Queue：封包 ID 0 ~ target_k-1，一開始每個封包都缺所有人的 ACK
            self.global_arq_queue = {pkt_id: set(all_user_ids) for pkt_id in range(self.target_k)}

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
        
        # 2. 執行流量分配邏輯 (MEO Flow Allocation)
        self.constellation.meo_broadcast_to_leos(current_time)

        # 3. 計算 Reward (獎勵設計)
        rewards = {a: 0.0 for a in self.agents}
        sent_user_count = 0

        # if (self.current_step == 1):
        #     print(actions)

        ft = self.constellation.get_finish_time_cost()
        max_buf = self.constellation.get_leo_max_buffer()
        # old_ful = self.constellation.get_user_received_percent()

        all_done = bool(self.check_all_grids_fulfilled())
        is_truncated = bool(self.current_step >= self.T_max - 1)
        is_done = all_done or is_truncated

        for agent_name in self.agents:
            i = self.constellation.get_id_by_name(agent_name)
            raw_action = actions[agent_name]

            # ==================================================
            # 【關鍵修正】：1. 先看清環境，計算出哪幾條路徑活著 (Action Mask)
            # ==================================================
            action_mask = np.zeros(self.M + 1, dtype=np.float32)
            for j, agent_j in enumerate(self.constellation.get_neighbors(i)[:self.M]):
                if self.constellation.get_ISL_capacity(i, agent_j, current_time) > 0:
                    teg_j = self.constellation.get_teg_downlink_volume(agent_j, self.Tw, current_time)
                    if np.sum(teg_j) > 0:  
                        action_mask[j] = 1.0
                        
            if len(self.constellation.get_visible_grids(i, current_time)) > 0:
                if self.constellation.get_downlink_capacity() > 0:
                    action_mask[self.M] = 1.0

            # ==================================================
            # 【關鍵修正】：2. 把想射向死路的意圖「歸零」，保留活路的意圖
            # ==================================================
            masked_action = raw_action * action_mask
            
            # ==================================================
            # 【關鍵修正】：3. 針對「活著的路徑」進行正規化，確保火力集中！
            # ==================================================
            action_sum = np.sum(masked_action)
            # 只有當 AI 的總火力超過 1.0 (超出物理極限) 時才等比例壓縮
            # 如果 AI 想省電 (sum <= 1.0)，就尊重它的決定
            if action_sum > 1.0:    
                action_probs = masked_action / action_sum
            else:                   
                action_probs = masked_action

            # --- 套用物理拘束 (Contact Volume) ---
            acc_cost = 0.0
            acc_max_cost = 0.0
            max_buf = self.constellation.get_leo_max_buffer()

            # Intra-tier (給鄰居)
            for j, agent_j in enumerate(self.constellation.get_neighbors(i)[:self.M]):
                contact_capacity = self.constellation.get_ISL_capacity(i, agent_j, current_time)
                buf_i = self.constellation.get_leo_buffer(i)
                # 這裡不需要再乘 mask 了，因為前面的 action_probs 已經處理過
                actual_flow = min(buf_i, action_probs[j] * contact_capacity)
                self.constellation.transfer_buffer(sat_id=i, neighbor=agent_j, amount=actual_flow)
                
                acc_cost += actual_flow * self.ISL_cost_factor
                acc_max_cost += max_buf
                self.episode_tx_cost += actual_flow * self.ISL_cost_factor

            # Inter-tier (給地面)
            visible_grids = self.constellation.get_visible_grids(i, current_time)
            if len(visible_grids) > 0:
                action_mask[self.M] = 1.0
                
            contact_capacity = self.constellation.get_downlink_capacity()
            buf_i = self.constellation.get_leo_buffer(i)

            if self.enable_RLNC:
                actual_flow = min(buf_i, action_probs[self.M] * contact_capacity * action_mask[self.M]) 
                acc_cost += actual_flow
                acc_max_cost += max_buf
                self.tx_cost_avg[agent_name] += acc_cost / acc_max_cost

                # record progress as reward
                old_ful = self.constellation.get_user_received_percent()
                sent_user_count = self.constellation.download_to_grid(i, amount=actual_flow, current_time=current_time)
                new_ful = self.constellation.get_user_received_percent()

            else:
                # ==================================================
                # 【真正的 No-RLNC 選擇性重傳 (Selective Repeat ARQ)】
                # ==================================================
                capacity = min(action_probs[self.M] * contact_capacity * action_mask[self.M], buf_i)
                actual_flow = 0.0
                sent_user_count = 0

                old_ful = self.constellation.get_user_received_percent()

                if capacity > 0 and len(self.global_arq_queue) > 0:
                    # 1. 找出目前這個衛星能看到的 User IDs
                    visible_users = set()
                    for g_idx in visible_grids:
                        for u in self.constellation.user_grids[g_idx].users:
                            visible_users.add(u.user_id)

                    # 2. 從 Queue 中挑出「對可見用戶有意義」的封包準備發射 (Head-of-Line 排程)
                    packets_to_send = []
                    for pkt_id in list(self.global_arq_queue.keys()):
                        pending_users = self.global_arq_queue[pkt_id]
                        # 只要這個封包的「欠款名單」跟「目前可見用戶」有交集，就值得發射
                        if len(pending_users.intersection(visible_users)) > 0:
                            packets_to_send.append(pkt_id)
                        # 受限於當前頻寬 capacity
                        if len(packets_to_send) >= capacity:
                            break
                            
                    actual_flow = len(packets_to_send)

                    # 3. 呼叫物理層傳輸，並回收 ACK
                    if actual_flow > 0:
                        success_acks = self.constellation.download_arq_to_grid(i, packets_to_send, current_time)
                        sent_user_count = len(set([u_id for u_id, p_id in success_acks]))

                        # 4. 根據 ACK 從 Queue 中劃掉名單
                        for u_id, pkt_id in success_acks:
                            if pkt_id in self.global_arq_queue and u_id in self.global_arq_queue[pkt_id]:
                                self.global_arq_queue[pkt_id].remove(u_id)

                        # 5. 清理 Queue：如果某個封包所有人都收到了，正式將其剔除！
                        for pkt_id in list(self.global_arq_queue.keys()):
                            if len(self.global_arq_queue[pkt_id]) == 0:
                                del self.global_arq_queue[pkt_id]
                    
                new_ful = self.constellation.get_user_received_percent()

            # count rewards
            delta_fulfill = new_ful - old_ful
            rewards[agent_name] += self.PROGRESS_SCALE * delta_fulfill
            for neighbor_id in self.constellation.get_rev_neighbors(i):
                _nei_name = self.constellation.get_name_by_id(neighbor_id)
                rewards[_nei_name] += self.PROGRESS_SCALE * delta_fulfill

            if (self.is_unicast): self.episode_tx_cost += actual_flow * max(sent_user_count, 1.0)
            else: self.episode_tx_cost += actual_flow

            # time cost
            rewards[agent_name] -= self.omega_t * 1  #self.reward_factor_time
            rewards[agent_name] -= self.omega_c * (acc_cost / acc_max_cost)

        # caclulate progress
        # new_ful = self.constellation.get_user_received_percent()
        # delta_fulfill = new_ful - old_ful
        # # for agent_name in self.agents:
        #     rewards[agent_name] += self.PROGRESS_SCALE * delta_fulfill
        
        # 4. 判斷是否結束 (所有目標網格的 DoF 都達到 K)
        # update finish time
        self.constellation.set_finish_time(self.current_step)
        cost = float(1.0 - self.constellation.get_user_fulfill_percent())
        terminations = {agent_name: is_done for agent_name in self.agents}
        truncations = {agent_name: is_truncated for agent_name in self.agents} # 是否超時
        is_violation = 1.0 if (is_truncated and not all_done) else 0.0

        if is_done:
            for agent_name in self.agents:
                rewards[agent_name] -= self.current_lambda * cost
        #         rewards[agent_name] -= self.COST_SCALE * (self.tx_cost_avg[agent_name])
        
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
                "lambda": self.current_lambda,
                "sent_user_count": sent_user_count,
                "current_progress": self.constellation.get_user_fulfill_percent()
            } for agent_name in self.agents
        }

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

        for j in self.constellation.get_neighbors(agent_id)[:self.M]:
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
        for idx, j in enumerate(self.constellation.get_neighbors(agent_id)[:self.M]):
            # grids_j = self.constellation.get_visible_grids(j, current_time)
            teg_j = self.constellation.get_teg_downlink_volume(j, self.Tw, current_time)
            cv_matrix[idx + 1, :] = teg_j

        # action mask
        action_mask = np.zeros(self.M + 1, dtype=np.float32)

        # if (np.any(cv_matrix)):
        #     print(cv_matrix)
        
        # 1. 檢查鄰居 (ISL) 是否活著
        for j, agent_j in enumerate(self.constellation.get_neighbors(agent_id)[:self.M]):
            if self.constellation.get_ISL_capacity(agent_id, agent_j, current_time) > 0:
                teg_j = self.constellation.get_teg_downlink_volume(agent_j, self.Tw, current_time)
                if np.sum(teg_j) > 0:  
                    action_mask[j] = 1.0
                
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
