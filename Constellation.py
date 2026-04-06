import math
from skyfield.api import load, EarthSatellite
from Satellite import *
from GroundGrid import *
from sgp4.api import Satrec, WGS72

# 物理常數
MU = 398600.4418         # 地球標準重力參數 (km^3/s^2)
R = 6371.2    # SGP4 使用的地球半徑標準

class Constellation:
    def __init__(self, alt=540.0, inc=53.2, p=72, s=22, f=17, 
                 meo_alt=10000, meo_inc=45.0,
                 n_grids=10,
                 packet_size_bits=80e6, broadcast_rate_bps=30e6,
                 step_seconds=10):
        # --- 1. Starlink Shell 2 官方參數 ---
        self.alt = alt         # 高度 (km)
        self.inc = inc       # 傾角 (度)
        self.p = p                   # 軌道面數 (Planes)
        self.s = s                   # 每面衛星數 (Sats per plane)
        self.f = f                   # 相位因子
        self.t = self.p * self.s                # 總共 1584 顆
        
        self.sat_id = 1
        self.meo_id = 9999

        self.n_grids = n_grids

        self.agents = []
        self.user_grids = []

        # --- MEO 參數 ---
        self.meo_alt = meo_alt
        self.meo_inc = meo_inc

        A = R + self.alt # 半長軸
        MEO_A = R + self.meo_alt

        # communication param
        self.packet_size_bits = packet_size_bits
        self.broadcast_rate_bps = broadcast_rate_bps
        self.step_seconds = step_seconds

        # 這裡使用簡化的通訊參數
        self.s_freq_hz = 2e9    # S-band 2GHz (for user erasure)
        self.ka_freq_hz = 23e9  # Ka-band 23GHz (for ISL)
        # 假設發射功率與天線增益等常數 (單位: dBm / dB)
        self.ptx_dbm = 30.0  # 發射功率
        self.gtx_db = 20.0   # 發射天線增益
        self.grx_db = 20.0   # 接收天線增益
        self.losses_db = 3.0 # 其他系統與指向損失
        self.noise_dbm = -100.0

        # 計算 Mean Motion (rad/min)
        n_rad_per_sec = math.sqrt(MU / (A**3))
        self.NO_KOZAI = n_rad_per_sec * 60.0  # 轉換為每分鐘的弧度

        # 2. 計算 MEO 的 Mean Motion (rad/min)
        meo_n_rad_per_sec = math.sqrt(MU / (MEO_A**3))
        self.MEO_NO_KOZAI = meo_n_rad_per_sec * 60.0

        # random seed
        np.random.seed(1234)

        # ------------ build constellation ------------------
        self.build_constellation()
        self.initialize_roi()

    def build_constellation(self):
        # 載入時間系統
        ts = load.timescale()
        # 設定一個基準紀元時間 (Epoch)，所有衛星從這點開始跑
        epoch = ts.utc(2026, 4, 1, 0, 0, 0)
        # sgp4_init 需要的是以 1949 年 12 月 31 日為基準的天數 (Skyfield ts.tt可以直接給出，這裡簡單使用 jd - 2433281.5)
        epoch_days = epoch.tt - 2433281.5 

        # --- 2. 生成 Walker-Delta 並直接建立 Skyfield 物件 ---
        for p in range(self.p):
            # 算出該軌道面的升交點赤經 (RAAN)，並轉成弧度
            raan_deg = p * (360.0 / self.p)
            raan_rad = math.radians(raan_deg)
            
            for s in range(self.s):
                # 算出該衛星的平近點角 (Mean Anomaly)，並轉成弧度
                mean_anomaly_deg = (s * (360.0 / self.s)) + (p * self.f * (360.0 / self.t))
                mean_anomaly_rad = math.radians(mean_anomaly_deg % 360.0)
                
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
        self.meo_sat = EarthSatellite.from_satrec(satrec_meo, ts)
        self.meo_sat.name = meo_name
        meo_i = MEOSatellite(self.meo_sat)

        self.agents.append(meo_i)

        print(f"成功將 {len(self.agents)} 顆完美 Walker 衛星實體化為 Skyfield 物件！")

        # --- 3. 測試：取得第一顆衛星在 10 分鐘後的 3D 座標 ---
        t_test = ts.utc(2026, 4, 1, 0, 10, 0)
        pos = self.agents[0].get_pos(t_test)
        print(f"衛星 {self.agents[0].name} 在 {t_test.utc_datetime()} 的 3D 座標 (X, Y, Z) km: \n{pos}")

    def initialize_roi(self, lat_min=21.0, lat_max=25.0, lon_min=119.0, lon_max=123.0, grid_size=2.0, users_per_grid=10, target_k=100):
        """
        根據經緯度範圍，自動生成 GroundGrid 陣列與散佈在其中的 Users
        """
        grid_id_counter = 0
        user_id_counter = 0

        # 產生經緯度的 grid
        lats = np.arange(lat_min, lat_max, grid_size)
        lons = np.arange(lon_min, lon_max, grid_size)

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
                    user_id_counter += 1
                    
                self.user_grids.append(grid)
                grid_id_counter += 1
            
    def get_neighbors(self, sat_id: int):
        # forward: (s+1) % N
        p = sat_id / self.s # plane id
        s = sat_id % self.s # sat id in plane
        
        forward = p * self.s + ((s + 1) % self.s)
        backward = p * self.s + ((s - 1) % self.s)
        left = ((p + 1) % self.p) * self.s + s
        right = ((p - 1) % self.p) * self.s + s
        
        return [forward, backward, left, right]

    def is_leo_visible_to_meo(self, t, ISL_max_range=10000):
        # 取得 MEO 座標
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
        sat1 = self.agents[agent1_id]
        sat2 = self.agents[agent2_id]

        # relative distance and pos at current time
        diff = sat1.skyfield_sat.at(current_time) - sat2.skyfield_sat.at(current_time)
        dist = diff.distance().km

        # determine capacity
        # 若超出最大通訊距離，直接斷線
        
        if dist > MAX_ISL_DISTANCE:
            return 0
            
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
        bandwidth_hz = 100e6 # 物理頻寬 100 MHz
        
        # 理論最大傳輸速率 (bps)
        dynamic_rate_bps = bandwidth_hz * np.log2(1 + snr_linear)
        
        # 打個折扣 (例如 0.5)，代表實際調變編碼機制 (AMC) 的極限
        actual_rate_bps = dynamic_rate_bps * 0.5 
        
        # --- 4. 轉換為這 10 秒內能傳遞的 NC 封包數 ---
        # packet_size_bits = 80e6 # 10 MB = 80 Mbits
        
        total_bits_in_step = actual_rate_bps * self.step_seconds
        max_packets = total_bits_in_step / self.packet_size_bits
        
        return int(max_packets)

    def get_leo_buffer(self, agent_id):
        return self.agents[agent_id].get_buffer()   

    def transfer_buffer(self, sat_id, neighbor, amount):
        self.agents[sat_id].buffer -= amount
        self.agents[neighbor].buffer += amount 

    def get_visible_grids(self, agent_id, current_time) -> list[int]:
            sat = self.agents[agent_id]
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
        
        return int(max_packets)

    def calculate_erasure_rate(self, agent_id: int, user: User, current_time):
        """
        基於 LEO 衛星幾何與路徑損失的連續異質抹除率計算 (無武斷的 if-else)
        """
        sat = self.agents[agent_id]
        difference = sat - user.pos
        alt, _, distance = difference.at(current_time).altaz()
        
        elevation_deg = alt.degrees
        slant_range_km = distance.km
        
        # 1. 物理視距極限保護
        if elevation_deg < 10.0:
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
        noise_dbm = self.noise_dbm
        
        # SNR = 接收功率 - 雜訊
        snr_db = (ptx_dbm + gtx_db - fspl_db - atmospheric_loss_db) - noise_dbm
        
        # =============================================================
        # 5. SNR 到 Erasure Rate 的平滑映射 (Sigmoid / 瀑布曲線)
        # 學術界常使用 Logistic 函數來逼近 FEC 晶片的解碼成功率曲線
        # 這裡沒有武斷的 step，只有平滑的機率轉換
        # =============================================================
        # 假設 SNR_THRESHOLD 是 5.0 dB (低於這個值，掉包率開始急劇上升)
        SNR_THRESHOLD = 5.0
        
        # 將 SNR 的差距轉換為 Erasure Rate (範圍 0~1)
        # 使用 sigmoid 函數： 1 / (1 + exp(k * (x - x0)))
        # k 是曲線的陡峭程度，這模擬了 FEC 的瀑布效應
        steepness = 1.5 
        erasure_rate = 1.0 / (1.0 + np.exp(steepness * (snr_db - SNR_THRESHOLD)))
        final_erasure = np.clip(erasure_rate, 0.0, 1.0)
        
        return float(final_erasure)

    def download_to_grid(self, agent_id:int, amount, current_time):
        grid_is = self.get_visible_grids()

        # sat = self.agents[agent_id]
        # --- 3. 異質化接收發生在這裡！ ---
        for g_idx in grid_is:
            for ui, user in enumerate(self.user_grids[ui]):
                # 算出自己的漏水率
                # User A 在中心，erasure_rate = 0.05
                # User B 在邊緣，erasure_rate = 0.40
                user_erasure_rate = self.calculate_erasure_rate(agent_id, user, current_time)
                
                # 【關鍵】大家都面對同樣的 37 滴水，但各自憑實力接水
                # User A: np.random.binomial(37, 0.95) -> 可能收到 35 滴
                # User B: np.random.binomial(37, 0.60) -> 可能只收到 22 滴
                received = np.random.binomial(int(amount), 1.0 - user_erasure_rate)
                
                self.user_grids[ui].recv(received)

    def get_total_received(self):
        # GroundGrid.get_user_total_recv
        total = sum([ m.get_user_total_recv() for m in self.user_grids])
        return total
    
    def get_total_target(self):
        # GroundGrid.get_user_total_recv
        total = sum([ m.get_user_total_target() for m in self.user_grids])
        return total
   
    def get_user_count(self):
        return sum([ g.get_user_count() for g in self.user_grids ])

def main():
    C = Constellation()
    print(type(C.constellation[0]))
    
if __name__ == '__main__':
    main()