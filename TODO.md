# TODO list

## Topology Module
- 軌道生成 : 幾條 LEO 軌道面，衛星數量，高度傾斜角
- hetero receivers
- time slot : SNR, link capacity of each receiver and sat, with angle and distance
- existing framework
    - [LEOPath](https://github.com/Fundacio-i2CAT/LEOPath)
    - [Skyfield](https://rhodesmill.org/skyfield/)
    - StarryNet
- given parameters to generate constellation 

## TEG construction (?)
- local actor partial observation
    - 未來的下行接觸容積 (?) --> 如果鄰居服務比我還好，讓他來做
        - V_DL * T_contact
    - 當前這區域已經送了多少 --> 已經送了多少 DoF，達標 N 就好 (把 地面用戶群切成網格) 
    - 確保送滿 --> 看這區域最爛的用戶是誰，就往這邊總共丟這麼多

## data dissemination module
- compute effective received amount
- x' = x * (1 - p); c = max(0, N - x')
- MEO 原點事先決定要送多少冗餘量，LEO 根據狀態決定哪邊要送多少

## MARL interface
- step, reset

## Process
1. MEO : transmit to visible LEO
2. LEO --> LEO relay (action) (predefined routing?)
3. NC multipath, multicast
