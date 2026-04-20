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
        self.buffer = 0.0          # 肚子裡有多少個 NC 封包 (Degrees of Freedom)
        self.max_buffer = 30.0   # 緩存上限
        
    def recv(self, amount:int):
        self.buffer = min(self.buffer + amount, self.max_buffer)

    def send(self, amount:int):
        real_amount = min(amount, self.buffer)
        self.buffer -= real_amount
        return real_amount
        
    def get_pos(self, t):
        return self.skyfield_sat.at(t).position.km
    
    def get_buffer(self):
        return self.buffer
    
    def get_max_buffer(self):
        return self.max_buffer
    
    def reset(self):
        self.buffer = 0
