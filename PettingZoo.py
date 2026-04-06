import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box
from datetime import datetime, timedelta
from skyfield.api import load
from Constellation import Constellation

class SatelliteDataDisseminationEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "satellite_nc_v0"}

    def __init__(self, num_neighbors=4, num_grids=1):
        super().__init__()
        self.constellation = Constellation()

        # 1. 定義智能體 ID
        # self.N = len(self.constellation.constellation) # num of agent
        # self.constellation.agents = [i for i in range(self.N)]  #[f"leo_{i}" for i in range(num_leos)]
        # self.constellation.agents = self.possible_agents[:]
        
        self.e = 0.2            # reliability constraint: Pr(T > T_max) <= e
        
        self.M = num_neighbors  # 鄰居數量 (Intra-tier)
        self.G = num_grids      # 覆蓋網格數量 (Inter-tier)
        self.N = len(self.constellation.agents)
        self.Tw = 2             # time window for contact volume
        
        # 2. 定義動作空間 (Action Space) - 連續變數
        # 每個 LEO 輸出一個長度為 M+1 的陣列，範圍 [0, 2]，代表流量分配比例
        self.action_spaces = {
            agent: Box(low=0.0, high=2.0, shape=(self.M + 1), dtype=np.float32)
            for agent in self.constellation.agents
        }
        
        # 3. 定義觀測空間 (Observation Space) - 局部視角 (給 Actor 用)
        # 包含：自身 Buffer(1), 鄰居 Contact Volume(M), 地面 Contact Volume(G) (我自己可以送多少)
        obs_dim = 1 + self.M + 1
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.constellation.agents
        }

        # 初始化 Skyfield 時間與環境參數
        self.ts = load.timescale()
        self.step_seconds = 10
        self.max_dof = 100 # 目標 K 值
        self.current_step = 0
        self.start_dt = datetime(2026, 4, 1, 0, 0, 0)

        # 通訊參數
        self.broadcast_rate_bps = 30e6 * 1.0 
        self.packet_size_bits = 80e6 # 10 MB = 80 Mbits

    def reset(self, seed=None, options=None):
        """回合開始: 重置時間、位置、Buffer 與 DoF 進度"""
        self.constellation.agents = self.constellation.agents[:]
        self.current_step = 0
        self.start_dt = datetime(2026, 4, 1, 0, 0, 0)
        
        # 這裡重置你的 LEO buffers 與地面的 received_dof
        # ...
        
        # 取得初始觀測值
        observations = {agent: self._get_obs(agent) for agent in self.constellation.agents}
        infos = {agent: {} for agent in self.constellation.agents}
        
        return observations, infos

    def step(self, actions):
        """每一回合的環境互動 (核心邏輯)"""
        # 1. 更新 Skyfield 時間
        current_dt = self.start_dt + timedelta(seconds=self.current_step * self.step_seconds)
        current_time = self.ts.utc(current_dt.year, current_dt.month, current_dt.day,
                                   current_dt.hour, current_dt.minute, current_dt.second)
        
        # 2. 執行流量分配邏輯 (Flow Allocation)
        # MEO source distribution

        # 3. 計算 Reward (獎勵設計)
        rewards = {i:-self.current_step for i in range(self.N)}

        for i, agent_i in enumerate(self.constellation.agents):
            # 神經網路輸出的比例 (0~1)
            action_probs = actions[i] # [X_ISL_1, X_ISL_2, X_ISL_3, X_ISL_4, X_DL]
            
            # 將比例轉換為實際想傳的封包數 (乘以自身 Buffer 總量)
            desired_flows = action_probs * self.constellation.get_leo_buffer(i)
            
            # --- 套用物理拘束 (Contact Volume) ---
            # Intra-tier (給鄰居)
            for j, agent_j in enumerate(self.constellation.get_neighbors(i)):
                contact_capacity = self.constellation.get_ISL_capacity(i, j, current_time)
                actual_flow = min(desired_flows[j], contact_capacity)
                self.constellation.transfer_buffer(i, neighbor=j, amount=actual_flow)
                rewards[i] -= actual_flow
                
            # Inter-tier (給地面)
            actual_flow = min(desired_flows[self.M], contact_capacity)
            rewards[i] -= actual_flow
            # for g in range(self.constellation.get_visible_grids(i)):
            contact_capacity = self.constellation.get_downlink_capacity()
            self.constellation.download_to_grid(i, amount=actual_flow, current_time=current_time)

        
        # 4. 判斷是否結束 (所有目標網格的 DoF 都達到 K)
        all_done = self.check_all_grids_fulfilled()
        terminations = {agent: all_done for agent in self.constellation.agents}
        truncations = {agent: False for agent in self.constellation.agents} # 是否超時
        
        # 5. 更新狀態
        self.current_step += 1
        observations = {agent: self._get_obs(agent) for agent in self.constellation.agents}
        infos = {agent: {} for agent in self.constellation.agents}

        # 當所有任務完成，清空 agents 列表 (PettingZoo 規範)
        if all_done:
            self.constellation.agents = []

        return observations, rewards, terminations, truncations, infos

    def state(self):
        """
        【CTDE 的靈魂：全局狀態】
        這是給 Centralized Critic (上帝視角) 看的。
        Actor 只看 _get_obs (局部 K-DoF 缺口)，但 Critic 可以看全網所有的狀態矩陣。
        """
        global_buffer_state = ... # 所有 LEO 的庫存
        global_teg_state = ...    # 全局接觸圖的容量
        return np.concatenate([global_buffer_state, global_teg_state])

    def _get_obs(self, agent_id, current_time):
        """計算局部觀測值給 Actor"""
        # (buf state of self, contact volume: 4)
        # ==========================================
        # 特徵 1: 自己的 Buffer (1 維)
        # ==========================================
        buf = self.constellation.get_leo_buffer(agent_id)
        max_buf = self.constellation.get_leo_max_buffer()
        norm_buf = np.clip(buf / max_buf, 0.0, 1.0)

        # ==========================================
        # 特徵 2: 鄰居的 Buffer 與 ISL Volume (2 * M 維)
        # ==========================================
        neighbor_bufs = []

        for j in self.constellation.get_neighbors(agent_id):
            buf_j = self.constellation.get_leo_buffer(j)
            norm_buf_j = np.clip(buf_j / max_buf, 0.0, 1.0)
            neighbor_bufs.append()


    def check_all_grids_fulfilled(self):
        total_recv = self.constellation.get_total_received()
        total_k = self.constellation.get_total_target()
        target = int( self.e * total_k ) # constraint
        return (total_recv >= target)

