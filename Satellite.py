import numpy as np
from skyfield.api import EarthSatellite

class MEOSatellite:
    def __init__(self, skyfield_sat: EarthSatellite):
        self.skyfield_sat = skyfield_sat  # 底層的物理衛星物件
        self.name = skyfield_sat.name

    def get_pos(self, t):
        return self.sat.at(t).position.km

class RelaySatellite:
    def __init__(self, skyfield_sat: EarthSatellite):
        self.skyfield_sat = skyfield_sat  # 底層的物理衛星物件
        self.name = skyfield_sat.name
        self.buffer = 0          # 肚子裡有多少個 NC 封包 (Degrees of Freedom)
        self.max_buffer = 1000   # 緩存上限
        
    def get_pos(self, t):
        return self.skyfield_sat.at(t).position.km
    
    def get_buffer(self):
        return self.buffer
    
    def leo_to_leo(self, t, leo_b):
        if (self.buffer <= 0): return

        # 判斷 LEO b 是否可連線
        dist = np.linalg.norm(self.get_pos(t) - leo_b.get_pos(t))
