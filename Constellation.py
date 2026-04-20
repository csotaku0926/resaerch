import math
from skyfield.api import load, EarthSatellite
from Satellite import *
from GroundGrid import *
from sgp4.api import Satrec, WGS72
from datetime import timezone, timedelta

# 物理常數
MU = 398600.4418         # 地球標準重力參數 (km^3/s^2)
R = 6371.2    # SGP4 使用的地球半徑標準

"""
Starlink 2: s=20, p=36, h=570
Telesat:
OneWeb:
"""
class Const_Param:
    def __init__(self, alt=540.0, inc=53.2, p=10, s=10, f=17, t_max=40, target_k=10):
        self.alt = alt         # 高度 (km)
        self.inc = inc       # 傾角 (度)
        self.p = p                   # 軌道面數 (Planes)
        self.s = s                   # 每面衛星數 (Sats per plane)
        self.f = f                   # 相位因子
        self.t_max = t_max
        self.target_k = target_k

class Constellation:
    def __init__(self, param: Const_Param, # alt=540.0, inc=53.2, p=10, s=10, f=17, 
                 meo_alt=10000, meo_inc=45.0,
                 n_grids=10, num_users=10,
                 packet_size_bits=80e6, broadcast_rate_bps=10e6, meo_tx_rate_bps=50e6,
                 step_seconds=10, t_max=90, target_k=20, test_mode=False):
        # --- 1. Starlink Shell 2 官方參數 ---
        self.alt = param.alt         # 高度 (km)
        self.inc = param.inc       # 傾角 (度)
        self.p = param.p                   # 軌道面數 (Planes)
        self.s = param.s                   # 每面衛星數 (Sats per plane)
        self.f = param.f                   # 相位因子
        self.t = self.p * self.s                # 總共 1584 顆
        
        self.sat_id = 0
        self.meo_id = 9999

        self.n_grids = n_grids
        self.grid_scale = 10.0

        self.agents = []
        self.user_grids = []
        self.name_to_idx = {}
        self.meo_sat = None

        # --- MEO 參數 ---
        self.meo_alt = meo_alt
        self.meo_inc = meo_inc

        A = R + self.alt # 半長軸
        MEO_A = R + self.meo_alt

        # communication param
        self.meo_tx_rate_bps = meo_tx_rate_bps # 100 Mbps
        self.packet_size_bits = packet_size_bits
        self.broadcast_rate_bps = broadcast_rate_bps
        self.step_seconds = step_seconds
        self.target_k = param.target_k
        self.users_per_grid = num_users
        self.max_covered_grid = 4 # assume at most cover 4 grids per time
        self.t_max = param.t_max

        # 這裡使用簡化的通訊參數
        self.s_freq_hz = 2e9    # S-band 2GHz (for user erasure)
        self.ka_freq_hz = 23e9  # Ka-band 23GHz (for ISL)
        # 假設發射功率與天線增益等常數 (單位: dBm / dB)
        self.ptx_dbm = 30.0  # 發射功率
        self.gtx_db = 20.0   # 發射天線增益
        self.grx_db = 20.0   # 接收天線增益
        self.losses_db = 3.0 # 其他系統與指向損失
        self.noise_dbm = -100.0

        # time scale
        ts = load.timescale()
        self.t_init = ts.utc(2026, 4, 1, 0, 0, 0) # 你的預設起始時間

        # 計算 Mean Motion (rad/min)
        n_rad_per_sec = math.sqrt(MU / (A**3))
        self.NO_KOZAI = n_rad_per_sec * 60.0  # 轉換為每分鐘的弧度

        # 2. 計算 MEO 的 Mean Motion (rad/min)
        meo_n_rad_per_sec = math.sqrt(MU / (MEO_A**3))
        self.MEO_NO_KOZAI = meo_n_rad_per_sec * 60.0

        # random seed
        np.random.seed(1234)

        # ------------ build constellation ------------------
        self.test_mode = test_mode
        if not self.test_mode:
            self.initialize_roi(
                grid_size=self.grid_scale,
                users_per_grid=self.users_per_grid, 
                target_k=self.target_k
            )
            raan_offset = self.get_raan_offset()
            self.build_constellation(raan_offset=raan_offset)
        else:
            self.build_constellation()
            self.initialize_users_along_tracks(self.target_k)

    def reset(self):
        """reset whole constellation (buffer state, inital position)"""
        for i in range(len(self.agents)):
            self.agents[i].reset()

        for i in range(len(self.user_grids)):
            self.user_grids[i].reset()

        if not self.test_mode:
            self.initialize_roi(
                grid_size=self.grid_scale,
                users_per_grid=self.users_per_grid, 
                target_k=self.target_k
            )

        else:
            self.initialize_users_along_tracks(self.target_k)

    def build_constellation(self, raan_offset=0.0, rewind_angle_deg=45.0, do_log=False):
        # # 載入時間系統
        ts = load.timescale()
        # # 設定一個基準紀元時間 (Epoch)，所有衛星從這點開始跑
        # epoch = ts.utc(2026, 4, 1, 0, 0, 0)
        # sgp4_init 需要的是以 1949 年 12 月 31 日為基準的天數 (Skyfield ts.tt可以直接給出，這裡簡單使用 jd - 2433281.5)
        epoch_days = self.t_init.tt - 2433281.5 

        # 【關鍵修正】：算出倒帶的角度
        rewind_rad = math.radians(rewind_angle_deg)

        # --- 2. 生成 Walker-Delta 並直接建立 Skyfield 物件 ---
        for p in range(self.p):
            # 算出該軌道面的升交點赤經 (RAAN)，並轉成弧度
            raan_deg = p * (360.0 / self.p)
            if (self.test_mode): raan_deg = p * 5.0

            raan_rad = math.radians(raan_deg) + raan_offset
            
            for s in range(self.s):
                # 算出該衛星的平近點角 (Mean Anomaly)，並轉成弧度
                mean_anomaly_deg = (s * (360.0 / self.s)) + (p * self.f * (360.0 / self.t))
                mean_anomaly_rad = math.radians(mean_anomaly_deg % 360.0) - rewind_rad
                
                # [核心步驟] 建立 SGP4 核心物件 (Satrec)
                satrec = Satrec()
                satrec.sgp4init(
                    WGS72,               # 使用 WGS72 重力模型
                    'i',                 # 'i' 代表這是一個完美的理想軌道初始化
                    self.sat_id,              # 衛星編號
                    epoch_days,          # 軌道基準時間
                    0.0,                 # BSTAR (空氣阻力係數，完美模擬設為 0)
                    0.0,                 # NDOT (平均運動一階導數，設為 0)
                    0.0,                 # NDDOT (平均運動二階導數，設為 0)
                    0.0,                 # ECCO (離心率 Eccentricity，圓軌道為 0)
                    0.0,                 # ARGPO (近地點幅角 Argument of Perigee，設為 0)
                    math.radians(self.inc), # INCLO (軌道傾角，弧度)
                    mean_anomaly_rad,    # MO (平近點角，弧度)
                    self.NO_KOZAI,            # NO_KOZAI (平均運動 Mean Motion, rad/min)
                    raan_rad             # NODEO (升交點赤經 RAAN, 弧度)
                )
                
                # [核心步驟] 把 Satrec 包裝成 Skyfield 可以操作的 EarthSatellite
                sat_name = f"Starlink_Shell2_{p}_{s}"
                skyfield_sat = EarthSatellite.from_satrec(satrec, ts)
                skyfield_sat.name = sat_name
                sat_i = RelaySatellite(skyfield_sat)
                self.name_to_idx[ sat_name ] = self.sat_id
                
                self.agents.append(sat_i)
                self.sat_id += 1

        ## -- generate MEO object
        satrec_meo = Satrec()
        satrec_meo.sgp4init(
            WGS72,               
            'i',                 
            self.meo_id,                # 給它一個特別的 ID (例如 9999)
            epoch_days,          # 跟 LEO 用同一個基準時間
            0.0, 0.0, 0.0, 0.0, 0.0,
            math.radians(self.meo_inc), 
            0.0,                 # 初始位置 (平近點角設 0)
            self.MEO_NO_KOZAI,        
            0.0                  # 升交點赤經
        )

        meo_name = "MEO_Data_Source"
        meo_sat = EarthSatellite.from_satrec(satrec_meo, ts)
        self.meo_sat = MEOSatellite(meo_sat)
        self.meo_sat.name = meo_name

        # self.agents.append(meo_i)

        if (do_log): print(f"成功將 {len(self.agents)} 顆完美 Walker 衛星實體化為 Skyfield 物件！")

        # --- 3. 測試：取得第一顆衛星在 10 分鐘後的 3D 座標 ---
        t_test = ts.utc(2026, 4, 1, 0, 10, 0)
        pos = self.agents[0].get_pos(t_test)
        if (do_log): print(f"衛星 {self.agents[0].name} 在 {t_test.utc_datetime()} 的 3D 座標 (X, Y, Z) km: \n{pos}")
 
    def locate_sat_init(self, agent_id):
        subpoint = self.agents[agent_id].skyfield_sat.at(self.t_init).subpoint()
        sat_lat = subpoint.latitude.degrees
        sat_lon = subpoint.longitude.degrees
        return sat_lat, sat_lon

    def get_raan_offset(self, target_lat=10.0, target_lon=25.0) -> float:
        """
        將 Walker-Delta 星系的 0 號衛星，在 t_init 時精準對齊到 target_lat 與 target_lon。
        (適用於 0~20N, 0~50E 區域，中心點為 10N, 25E)
        """
        ts = load.timescale()
        epoch_days = self.t_init.tt - 2433281.5 
        
        # --- 步驟 A: 逆向推算需要的 Mean Anomaly (對齊緯度) ---
        # 簡化公式： sin(Lat) = sin(Inc) * sin(Argument of Latitude)
        # 由於圓軌道 Arg of Perigee = 0，Argument of Latitude 就等於 Mean Anomaly
        inc_rad = math.radians(self.inc)
        target_lat_rad = math.radians(target_lat)
        
        # 確保目標緯度沒有超過軌道傾角極限
        if abs(target_lat) > self.inc:
            raise ValueError("目標緯度大於軌道傾角，衛星永遠飛不到那裡！")
            
        base_ma_rad = math.asin(math.sin(target_lat_rad) / math.sin(inc_rad))
        
        # --- 步驟 B: 逆向推算需要的 RAAN (對齊經度) ---
        # 地球自轉也會影響經度，最暴力的做法是先建一顆測試衛星，看誤差多少再修正
        satrec_test = Satrec()
        satrec_test.sgp4init(WGS72, 'i', 999, epoch_days, 0.0, 0.0, 0.0, 0.0, 0.0,
                             inc_rad, base_ma_rad, self.NO_KOZAI, 0.0)
        test_sat = EarthSatellite.from_satrec(satrec_test, ts)
        test_lon = test_sat.at(self.t_init).subpoint().longitude.degrees
        
        # 計算經度偏差值
        lon_offset_deg = target_lon - test_lon
        raan_offset_rad = math.radians(lon_offset_deg)

        return raan_offset_rad

    def initialize_roi(self, lat_min=0.0, lat_max=90.0, lon_min=0.0, lon_max=90.0, grid_size=20.0, users_per_grid=10, target_k=100, do_log=False):
        """
        根據經緯度範圍，自動生成 GroundGrid 陣列與散佈在其中的 Users
        """
        grid_id_counter = 0
        user_id_counter = 0

        if (do_log): print(f"Initialize grids at lat {lat_min}~{lat_max}, lon {lon_min}~{lon_max}")

        # 產生經緯度的 grid
        lats = np.arange(lat_min, lat_max, grid_size)
        lons = np.arange(lon_min, lon_max, grid_size)

        # reset if exist
        self.user_grids.clear()
        if (do_log): print(f"generate {len(lats) * len(lons)} grids")

        for lat in lats:
            for lon in lons:
                # 計算網格中心點
                center_lat = lat + grid_size / 2.0
                center_lon = lon + grid_size / 2.0
                
                # 使用你的類別基準實例化 GroundGrid
                grid = GroundGrid(center_lat, center_lon, grid_id_counter, grid_size=grid_size, target_k=target_k)
                
                # 在這個網格範圍內，隨機生成用戶
                for _ in range(users_per_grid):
                    u_lat = np.random.uniform(lat, lat + grid_size)
                    u_lon = np.random.uniform(lon, lon + grid_size)
                    
                    user = User(user_id_counter, u_lat, u_lon, target_k=target_k)
                    grid.users.append(user)
                    grid.user_finish_time.append(-1)
                    user_id_counter += 1
                    
                self.user_grids.append(grid)
                grid_id_counter += 1

    def initialize_users_along_tracks(self, target_k=20, do_log=False):
        ts = load.timescale()
        
        # --- 第 1 階段：純掃描 ---
        # 只記錄衛星在這段時間內會經過的「不重複網格座標」
        target_grid_coords = []
        grid_size = self.grid_scale

        if do_log: print(f"正在掃描 {len(self.agents)} 顆衛星的模擬軌跡涵蓋範圍...")

        for agent in self.agents:
            for step in range(self.t_max):
                future_dt = self.t_init.utc_datetime() + timedelta(seconds=step * self.step_seconds)
                future_t = ts.from_datetime(future_dt.replace(tzinfo=timezone.utc))
                
                subpoint = agent.skyfield_sat.at(future_t).subpoint()
                lat = subpoint.latitude.degrees
                lon = subpoint.longitude.degrees
                
                grid_lat = math.floor(lat / grid_size) * grid_size
                grid_lon = math.floor(lon / grid_size) * grid_size
                
                # 把經過的座標丟進 set 裡 (自動去重)
                target_grid_coords.append([grid_lat, grid_lon])

        # --- 第 2 階段：真正初始化用戶 ---
        # 確定了涵蓋範圍後，才開始建立物件 (絕對不會重複生成！)
        self.user_grids = []
        grid_id_counter = 0
        user_id_counter = 0

        # random pick K grids
        target_grid_coords = list(target_grid_coords)
        np.random.shuffle(target_grid_coords)
        target_grid_coords = target_grid_coords[:self.n_grids]

        if do_log: print(f"掃描完畢，共有 {len(target_grid_coords)} 個網格落入涵蓋範圍。開始生成用戶...")

        for (grid_lat, grid_lon) in target_grid_coords:
            grid = GroundGrid(grid_lat + grid_size // 2, grid_lon + grid_size // 2, grid_id_counter, 
                             grid_size=grid_size, target_k=target_k)
            
            for _ in range(self.users_per_grid):
                u_lat = grid_lat + grid_size // 2 + np.random.uniform(-grid_size // 2, grid_size // 2)
                u_lon = grid_lon + grid_size // 2 + np.random.uniform(-grid_size // 2, grid_size // 2)
                user = User(user_id_counter, u_lat, u_lon, target_k=target_k)
                grid.users.append(user)
                grid.user_finish_time.append(-1)
                user_id_counter += 1
            
            self.user_grids.append(grid)
            grid_id_counter += 1

        if do_log: print(f"初始化完成！共建立 {grid_id_counter} 個網格，{user_id_counter} 個用戶。")
    
    def get_id_by_name(self, name: str):
        return self.name_to_idx[name]

    def get_neighbors(self, sat_id: int):
        # forward: (s+1) % N
        p = sat_id // self.s # plane id
        s = sat_id % self.s # sat id in plane
        
        forward = p * self.s + ((s + 1) % self.s)
        backward = p * self.s + ((s - 1) % self.s)
        left = ((p + 1) % self.p) * self.s + s
        right = ((p - 1) % self.p) * self.s + s
        
        return [forward, backward, left, right]

    def is_leo_visible_to_meo(self, t, ISL_max_range=10000):
        """get visible LEO to MEO source"""
        meo_pos = self.meo_sat.at(t).position.km
        
        candidates = []
        for i in range(self.t):
            leo_i = self.agents[i]
            leo_pos = leo_i.get_pos(t)
            dist = ((meo_pos - leo_pos)**2).sum()**0.5
            
            # 如果距離在 ISL 範圍內 (例如 10000km)，這顆 LEO 就能接收 MEO 的資料
            valid = (dist < ISL_max_range)
            candidates.append(valid)

        return candidates
    
    def get_ISL_capacity(self, agent1_id:int, agent2_id:int, current_time, MAX_ISL_DISTANCE=5000.0):
        
        if (agent1_id == agent2_id): return 0
        
        sat1 = self.agents[agent1_id]
        sat2 = self.agents[agent2_id]

        # relative distance and pos at current time
        diff = sat1.skyfield_sat.at(current_time) - sat2.skyfield_sat.at(current_time)
        dist = diff.distance().km

        # determine capacity
        # 若超出最大通訊距離，直接斷線
        
        # if dist > MAX_ISL_DISTANCE:
        #     return 0
            
        # --- 2. 實體層鏈路預算 (Link Budget) 模擬 ---
        
        # 計算自由空間路徑損失 FSPL (dB)
        # 公式: 20 * log10(d) + 20 * log10(f) + 20 * log10(4pi/c)
        fspl_db = 20 * np.log10(dist * 1000) + 20 * np.log10(self.ka_freq_hz) - 147.55
        
        # 接收功率 Prx (dBm)
        prx_dbm = self.ptx_dbm + self.gtx_db + self.grx_db - fspl_db - self.losses_db
        
        # 計算訊噪比 SNR (dB) 並轉為線性值
        snr_db = prx_dbm - self.noise_dbm
        snr_linear = 10 ** (snr_db / 10.0)

        # --- 3. 動態容量計算 (Shannon Capacity) ---
        bandwidth_hz = 500e6 # 物理頻寬 100 MHz
        
        # 理論最大傳輸速率 (bps)
        dynamic_rate_bps = bandwidth_hz * np.log2(1 + snr_linear)
        actual_rate_bps = dynamic_rate_bps 
        
        # --- 4. 轉換為這 10 秒內能傳遞的 NC 封包數 ---
        # packet_size_bits = 80e6 # 10 MB = 80 Mbits
        
        total_bits_in_step = actual_rate_bps * self.step_seconds
        max_packets = total_bits_in_step / self.packet_size_bits
        
        return int(10000 / dist) # --> 4 #int(max_packets) --> 0 

    def get_leo_buffer(self, agent_id):
        return self.agents[agent_id].get_buffer()   
    
    def get_leo_max_buffer(self):
        return self.agents[0].get_max_buffer()

    def transfer_buffer(self, neighbor, amount, sat_id=None):
        """sat_id=None --> MEO transfer"""
        if (sat_id is not None): self.agents[sat_id].send(amount)
        self.agents[neighbor].recv(amount) 

    def meo_broadcast_to_leos(self, current_time, max_dist=15000):
        """
        MEO 作為 Source, 將封包廣播給視距內的 LEOs
        """
        # 1. 定義 MEO 的發射能力 (假設 MEO 的頻寬比較大)
        # meo_tx_rate_bps = 100e6  # 100 Mbps
        # packet_size_bits = 80e6  # 10 MB = 80 Mbits (跟你 LEO 的設定一樣)
        
        # 算出 MEO 在這 10 秒內，總共噴了多少個封包 (DoF)
        meo_total_packets = 0.1 * self.step_seconds #(self.meo_tx_rate_bps * self.step_seconds) / self.packet_size_bits

        # 2. 掃描所有的 LEO (Agents)
        for i, agent in enumerate(self.agents):
            # leo_sat = agent.skyfield_sat # 取得 Skyfield 物件
            # meo_sat = self.meo_sat.skyfield_sat # 你的 MEO Skyfield 物件
            
            # 3. 檢查視距 (Line of Sight) 與仰角
            # 計算 MEO 看 LEO 的相對位置
            # difference = leo_sat - meo_sat
            # distance_km = difference.at(current_time).distance().km
            # if distance_km < max_dist:

            # 6. 將收到的封包加入 LEO 的 Buffer 裡 (使用我們上一篇討論的 add_buffer)
            self.transfer_buffer(neighbor=i, amount=meo_total_packets)

    def get_visible_grids(self, agent_id, current_time) -> list[int]:
        sat = self.agents[agent_id].skyfield_sat
        visible_grid_idx = []
        
        # 掃描所有的 GroundGrid
        for i, grid in enumerate(self.user_grids):
            difference = sat - grid.center_position # wgs84.latlon...
            alt, _, _ = difference.at(current_time).altaz()
            
            # 如果仰角 > 15 度，代表波束涵蓋到了這個網格
            if alt.degrees >= 15.0:
                visible_grid_idx.append(i)
                
        return visible_grid_idx
    
    def get_downlink_capacity(self):
        
        # 6. 轉換為封包數 (假設 1 個 NC 封包是 10 MB = 80 Mbits)
        # packet_size_bits = 80e6
        max_packets = (self.broadcast_rate_bps * self.step_seconds) / self.packet_size_bits

        # user amount counts..
        # user_num = self.users_per_grid
        # user_factor = max(1.0, user_num / 10.0)

        return 5 #int(max_packets)

    def calculate_erasure_rate(self, agent_id: int, user: User, current_time):
        """
        基於 LEO 衛星幾何與路徑損失的連續異質抹除率計算 (link quality)
        """
        sat = self.agents[agent_id].skyfield_sat
        difference = sat - user.pos
        alt, _, distance = difference.at(current_time).altaz()
        
        elevation_deg = alt.degrees
        slant_range_km = distance.km
        
        # 1. 物理視距極限保護
        if elevation_deg < 20.0:
            return 1.0 # 仰角過低，被地球曲率或建築物完全遮蔽，物理上100%掉包
            
        # 2. 嚴謹的連續物理計算：自由空間路徑損失 (FSPL)
        # 距離越遠 (仰角越低)，fspl_db 越大
        freq_hz = self.s_freq_hz # S-band
        fspl_db = 20 * np.log10(slant_range_km * 1000) + 20 * np.log10(freq_hz) - 147.55
        
        # 3. 大氣層吸收模型 (Atmospheric Absorption)
        # 仰角越低，穿過的大氣層越厚。這是一個標準的 csc(elevation) 關係
        # 假設天頂 (90度) 的大氣損耗為 0.5 dB
        zenith_attenuation_db = 0.5
        # 將仰角轉為弧度，計算餘割 (cosecant) 以反映穿透大氣層的相對厚度
        elevation_rad = np.radians(elevation_deg)
        atmospheric_loss_db = zenith_attenuation_db / np.sin(elevation_rad)
        
        # 4. 計算最終的接收 SNR (假設發射功率與天線增益固定)
        ptx_dbm = self.ptx_dbm
        gtx_db = self.gtx_db
        grx_db = self.grx_db
        noise_dbm = self.noise_dbm
        
        # SNR = 接收功率 - 雜訊
        snr_db = (ptx_dbm + gtx_db + grx_db - fspl_db - atmospheric_loss_db) - noise_dbm
        
        # =============================================================
        # 5. SNR 到 Erasure Rate 的平滑映射 (Sigmoid / 瀑布曲線)
        # 學術界常使用 Logistic 函數來逼近 FEC 晶片的解碼成功率曲線
        # 這裡沒有武斷的 step，只有平滑的機率轉換
        # =============================================================
        # 假設 SNR_THRESHOLD 是 5.0 dB (低於這個值，掉包率開始急劇上升)
        SNR_THRESHOLD = 8.0
        
        # 將 SNR 的差距轉換為 Erasure Rate (範圍 0~1)
        # 使用 sigmoid 函數： 1 / (1 + exp(k * (x - x0)))
        # k 是曲線的陡峭程度，這模擬了 FEC 的瀑布效應
        steepness = 1.5 
        erasure_rate = 1.0 / (1.0 + np.exp(steepness * (snr_db - SNR_THRESHOLD)))
        final_erasure = np.clip(erasure_rate, 0.0, 1.0)

        # ==========================================
        # 🌪️ [關鍵新增] 突發狀況：週期性極端氣候遮蔽 (Deterministic Blockage)
        # ==========================================
        dt = current_time.utc_datetime()
        minute = dt.minute
        second = dt.second
        
        # 陷阱設計：每分鐘的 第 20 秒 到 50 秒，會有一場嚴重的通訊遮蔽
        in_weather_event = (20 <= second <= 50)
        
        # 為了強迫 AI 使用 ISL 協作，我們讓這場風暴「只襲擊偶數號衛星」，奇數號天氣晴朗
        if in_weather_event and (agent_id % 2 == 0):
            return 0.99  # 突發 99% 掉包率 (通道幾乎全毀)
        
        return float(final_erasure)

    def download_to_grid(self, agent_id:int, amount, current_time):
        grid_is = self.get_visible_grids(agent_id, current_time)

        # sat = self.agents[agent_id]
        # sent buffer
        self.agents[agent_id].send(amount)
        # --- 3. 異質化接收發生在這裡！ ---
        for g_idx in grid_is:
            for ui, user in enumerate(self.user_grids[g_idx].users):
                # 算出自己的漏水率
                user_erasure_rate = self.calculate_erasure_rate(agent_id, user, current_time)
                
                # 【關鍵】大家都面對同樣的 37 滴水，但各自憑實力接水
                # User A: np.random.binomial(37, 0.95) -> 可能收到 35 滴
                # User B: np.random.binomial(37, 0.60) -> 可能只收到 22 滴
                received = np.random.binomial(int(amount), 1.0 - user_erasure_rate)
                
                self.user_grids[g_idx].users[ui].recv(received)

    def get_user_fulfill_percent(self) -> float:
        # return the percentage
        ful_cnt = 0
        user_cnt = 0

        for grid in self.user_grids:
            ful_cnt += grid.get_user_fulfill()
            user_cnt += grid.get_user_count()
        return ful_cnt / user_cnt
   
    def get_user_received_percent(self) -> float:
        # return the percentage
        ful_cnt = 0
        user_cnt = 0

        for grid in self.user_grids:
            ful_cnt += sum(grid.get_user_total_recv())
            user_cnt += grid.get_user_count()
        return ful_cnt / user_cnt

    def set_finish_time(self, current_step:int):
        k = self.target_k
        for gi, grid in enumerate(self.user_grids):
            total_recv = grid.get_user_total_recv()
            for ui, recv in enumerate(total_recv):
                # update finish time if done and not set yet
                if (recv >= k and self.user_grids[gi].user_finish_time[ui] == -1):
                    self.user_grids[gi].user_finish_time[ui] = current_step

    def get_finish_time_cost(self) -> float:
        """get avg time cost so far"""
        avg_time = 0.0
        t_max = self.t_max
        for grid in self.user_grids:
            finish_times = grid.get_finish_time()
            for ft in finish_times:
                if ft > -1: avg_time += ft
                else: avg_time += t_max

        avg_time /= self.get_user_count()

        return avg_time

    def get_user_count(self):
        return sum([ g.get_user_count() for g in self.user_grids ])
    
    def get_teg_downlink_volume(self, agent_id: int, n_time_window: int, current_time) -> list[int]:
        """
        真正的 TEG Contact Volume (時效性總量)
        往未來推演，計算這顆衛星離開這個網格前，"總共"還能砸下多少有效封包。
        """
        teg_vector = []
    
        # sat = self.agents[agent_id]
        # covered_grids = self.get_visible_grids(agent_id, current_time)
        # if (len(covered_grids) == 0):
        #     return [0] * n_time_window
        
        for future_step in range(n_time_window):
            # 1. 時間往未來推進
            ts = load.timescale()
            future_dt = current_time.utc_datetime() + timedelta(seconds=future_step * self.step_seconds)
            future_t = ts.utc(future_dt.year, future_dt.month, future_dt.day, 
                                future_dt.hour, future_dt.minute, future_dt.second)
            
            covered_grids = self.get_visible_grids(agent_id, future_t)

            # 2. 取得未來的仰角
            avg_ratio = 0.0
            for gi in covered_grids: 
                grid = self.user_grids[gi]
                # (這裡借用你寫好的物理公式，算平均掉包率)
                for user in grid.users:
                    e_rate = self.calculate_erasure_rate(agent_id, user, future_t)
                    avg_ratio += (1.0 - e_rate)
            
            avg_ratio /= (self.max_covered_grid * self.users_per_grid)
                    
            teg_vector.append(avg_ratio)
            
        return teg_vector

def main():
    C = Constellation()
    
if __name__ == '__main__':
    main()