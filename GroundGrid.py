from skyfield.api import wgs84
from skyfield.api import EarthSatellite

# User Class
class User:
    def __init__(self, user_id, lat, lon, target_k=100):
        self.user_id = user_id
        self.lat = lat
        self.lon = lon
        self.pos = wgs84.latlon(lat, lon)
        self.received_count = 0
        self.target_k = target_k

    def get_dist_from_sat(self, sat: EarthSatellite, current_time):
        diff = sat - self.pos

        # 3. 代入當下時間，取得觀測座標系
        topocentric = diff.at(current_time)

        # 4. 取得仰角與距離
        alt, az, distance = topocentric.altaz()

        return distance
    
    def recv(self, amount:int):
        self.received_count = min(self.received_count + amount, self.target_k)

    def reset(self):
        self.received_count = 0


# 實作邏輯：定義目標區域
class GroundGrid:
    def __init__(self, lat_center, lon_center, grid_id, grid_size=5, target_k=100):
        self.grid_id = grid_id
        self.lat_center = lat_center
        self.lon_center = lon_center
        self.center_position = wgs84.latlon(lat_center, lon_center)

        self.grid_size = grid_size
        self.target_k = target_k  # 該網格解碼所需的 DoF 數量
        # self.received_count = 0 # 累計收到的 NC 封包 (用於 Reward 計算)

        self.users = []

    def get_user_total_recv(self) -> list[int]:
        miss_lst = [user.received_count for user in self.users]
        return miss_lst
    
    def get_user_fulfill(self) -> int:
        miss_lst = [(user.received_count >= self.target_k) for user in self.users]
        return sum(miss_lst)
    
    def get_user_count(self) -> int:
        return len(self.users)
    
    def reset(self):
        for i in range(len(self.users)):
            self.users[i].reset()


def main():
    # 範例：建立一個位於台灣上空的目標網格
    taiwan_grid = GroundGrid(lat_center=23.0, lon_center=121, grid_id=0)
    target_pos = wgs84.latlon(taiwan_grid.lat_center, taiwan_grid.lon_center)
    G = GroundGrid(20, 30)
