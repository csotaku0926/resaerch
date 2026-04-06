import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Dict
from datetime import datetime, timedelta, timezone
from skyfield.api import load
from Constellation import Constellation

class SatelliteDataDisseminationEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "satellite_nc_v0"}

    def __init__(self, num_neighbors=4, num_grids=1):
        super().__init__()

        # 1. 定義 param
        self.e = 0.2            # reliability constraint: Pr(T > T_max) <= e
        self.T_max = 90         # max time step (truncation)
        
        self.M = num_neighbors  # 鄰居數量 (Intra-tier)
        self.G = num_grids      # 覆蓋網格數量 (Inter-tier)
        self.Tw = 2             # time window for contact volume
        
        self.constellation = Constellation()
        self.N = len(self.constellation.agents)

        # 加入這兩行 (PettingZoo 鐵規則)
        self.possible_agents = [agent.name for agent in self.constellation.agents] #self.constellation.agents[:]
        self.agents = self.possible_agents[:]

        # 2. 定義動作空間 (Action Space) - 連續變數
        # 每個 LEO 輸出一個長度為 M+1 的陣列，範圍 [0, 1]，代表流量分配比例
        self.action_spaces = {
            agent.name: Box(low=0.0, high=1.0, shape=(self.M + 1,), dtype=np.float32)
            for agent in self.constellation.agents
        }
        
        # 3. 定義觀測空間 (Observation Space) - 局部視角 (給 Actor 用)
        # 包含：自身 Buffer(1), 鄰居 Contact Volume(M), 地面 Contact Volume(G) (我自己可以送多少)
        self.observation_spaces = {
            agent.name: Dict({
                # 1. 庫存純量 (5 維)：[自己, 鄰居1, 鄰居2, 鄰居3, 鄰居4]
                "buffers": Box(low=0.0, high=1.0, shape=(1 + self.M,), dtype=np.float32),
                
                # 2. 接觸圖容量矩陣 (5 x T 維)
                # Row 0: 對地廣播的未來 T 步
                # Row 1~4: 給四個鄰居 ISL 的未來 T 步
                "contact_volumes": Box(low=0.0, high=1.0, shape=(1 + self.M, self.Tw), dtype=np.float32)
            })
            for agent in self.constellation.agents
        }

        # 初始化 Skyfield 時間與環境參數
        self.ts = load.timescale()
        self.step_seconds = 10
        self.current_step = 0
        self.start_dt = datetime(2026, 4, 1, 0, 0, 0)

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
        
        # 取得初始觀測值
        current_time = self.ts.from_datetime(self.start_dt)
        observations = {agent_name: self._get_obs(self.constellation.get_id_by_name(agent_name), 
                                                  current_time) for agent_name in self.agents}
        infos = {agent_name: {} for agent_name in self.agents}
        
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
        # rewards = {i:-self.current_step for i in range(self.N)}
        rewards = {}

        for agent_name in self.agents:
            # name_i = agent_i.name
            i = self.constellation.get_id_by_name(agent_name)
            rewards[agent_name] = -self.current_step
            # 神經網路輸出的比例 (0~1)
            action_probs = actions[agent_name] # {"LEO_i": action}
            
            # 將比例轉換為實際想傳的封包數 (乘以自身 Buffer 總量)
            desired_flows = action_probs * self.constellation.get_leo_buffer(i)
            
            # --- 套用物理拘束 (Contact Volume) ---
            # Intra-tier (給鄰居)
            for j, agent_j in enumerate(self.constellation.get_neighbors(i)):
                contact_capacity = self.constellation.get_ISL_capacity(i, j, current_time)
                actual_flow = min(desired_flows[j], contact_capacity)
                self.constellation.transfer_buffer(sat_id=i, neighbor=j, amount=actual_flow)
                rewards[agent_name] -= actual_flow
                
            # Inter-tier (給地面)
            contact_capacity = self.constellation.get_downlink_capacity()
            actual_flow = min(desired_flows[self.M], contact_capacity)
            rewards[agent_name] -= actual_flow
            # for g in range(self.constellation.get_visible_grids(i)):
            self.constellation.download_to_grid(i, amount=actual_flow, current_time=current_time)

        
        # 4. 判斷是否結束 (所有目標網格的 DoF 都達到 K)
        all_done = bool(self.check_all_grids_fulfilled())
        terminations = {agent_name: all_done for agent_name in self.agents}
        is_truncated = bool(self.current_step >= self.T_max)
        truncations = {agent_name: is_truncated for agent_name in self.agents} # 是否超時
        
        # 5. 更新狀態
        self.current_step += 1
        next_dt = self.start_dt + timedelta(seconds=self.current_step * self.step_seconds)
        next_time = self.ts.utc(next_dt.year, next_dt.month, next_dt.day,
                                   next_dt.hour, next_dt.minute, next_dt.second)
        
        observations = {agent_name: self._get_obs(i, next_time) for i, agent_name in enumerate(self.agents)}
        infos = {agent_name: {} for agent_name in self.agents}

        # 當所有任務完成，清空 agents 列表 (PettingZoo 規範)
        if all_done or is_truncated:
            self.agents = []

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
        
        global_state = []

        # 1. 收集全網所有衛星的 Buffer
        max_buf = self.constellation.get_leo_max_buffer()
        for agent_id in self.agents:
            buf = self.constellation.get_leo_buffer(agent_id)
            global_state.append(np.clip(buf / max_buf, 0.0, 1.0))

        # 2. 收集全網所有衛星對地的未來 T 步接觸容量 (TEG)
        for agent_id in self.agents:
            covered_grids = self.constellation.get_visible_grids(agent_id, current_time)
            # 這裡回傳的 my_teg 應該是一個長度為 self.Tw 的 list 或 array
            my_teg = self.constellation.get_teg_downlink_volume(agent_id, covered_grids, self.Tw, current_time)
            global_state.extend(my_teg)

        return np.array(global_state, dtype=np.float32)

    def _get_obs(self, agent_id, current_time):
        """計算局部觀測值給 Actor"""
        # ==========================================
        # 特徵 1: 自己的 Buffer (1 維) + 鄰居的 Buffer (4 維)
        # ==========================================
        buf = self.constellation.get_leo_buffer(agent_id)
        max_buf = self.constellation.get_leo_max_buffer()
        norm_buf = np.clip(buf / max_buf, 0.0, 1.0)
        bufs = [norm_buf]

        for j in self.constellation.get_neighbors(agent_id):
            buf_j = self.constellation.get_leo_buffer(j)
            norm_buf_j = np.clip(buf_j / max_buf, 0.0, 1.0)
            bufs.append(norm_buf_j)

        # ==========================================
        # 特徵 2:  Contact Volume (5 * T 維)
        # ==========================================
        cv_matrix = np.zeros((1 + self.M, self.Tw), dtype=np.float32)
        covered_grids = self.constellation.get_visible_grids(agent_id, current_time)
        
        my_teg = self.constellation.get_teg_downlink_volume(agent_id, covered_grids, self.Tw, current_time)
        # 填入自己對地的 TEG
        cv_matrix[0, :] = my_teg
        
        # 填入鄰居的 TEG
        for idx, j in enumerate(self.constellation.get_neighbors(agent_id)):
            teg_j = self.constellation.get_teg_downlink_volume(j, covered_grids, self.Tw, current_time)
            cv_matrix[idx + 1, :] = teg_j

        return {
            "buffers": np.array(bufs, dtype=np.float32),
            "contact_volumes": cv_matrix
        }

    def check_all_grids_fulfilled(self):
        total_recv_percent = self.constellation.get_user_fulfill_percent()
        target = (1 - self.e) # constraint
        return (total_recv_percent >= target)

