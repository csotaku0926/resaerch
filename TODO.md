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

## 04/15 update
正確性: 觀察不用真實星系
一個人: 退縮回 unicast (other: multiple unicast)
simulation 告訴甚麼事情, 我想觀察甚麼，跟我的機制的關係，什麼樣的狀況下
看不出來有甚麼意義, 想要表達的 insight

理想環境
- 重疊覆蓋區域高, 用戶多 : 阻止多顆衛星同時盲目發射, 在衛星同時覆蓋的情形發生
- 低仰角超長 : 阻止衛星看到就急著發射 (?)
- MEO 補給的效率要低於 LEO ?

觀察
- 不要多個衛星同時對同一群用戶發送
    - 分配每個衛星的 quota，而不是每個都傳 max packet for 邊緣用戶
- 想讓接觸比較多用戶的衛星去做，不用分開傳送 (multi-unicast) --> 充分利用 NC 的 multicast advantage
    - 選擇一個 contact volume 最好的連結傳輸 (multicast), 不是只對某個人好 (multi-unicast)
    - 把所有人的需求分批在最適合的仰角 (對 "多人平均" 而言是最好的角度)，而不是只有看到一個人就硬傳
    - 根據 TEG ，發現接觸時間短，馬上傳完; 接觸長，等未來
- 在狀態不好的狀況發射 quota ，成效低而且太快把庫存射完了 (考慮 LEO 只作為純粹的搬運工，完全不產生新的資料)
- scalability : Tx cost 增長趨勢低於他人
- 衛星密度 ?

機制
- Critic 統籌全局，它會告訴衛星 A 傳，告訴衛星 B 和 C 停止，合作性退讓 ? 消除重疊覆蓋區的浪費

metrics
- 完成任務要花多少成本 (Tx cost v.s. User Number) --> Multi-Unicast 的 cost 隨用戶數量上升 ，只有一個用戶時轉變成 unicast
- CDF of fulfill rate over time --> 減少最糟用戶的等待時間, 別人都是馬上傳完導致後期 buf 不夠
- tail latency: 最糟用戶等多久 (?) --> 如何定義 "最糟的人"
- 實際接收 / 衛星發射 (找好時機) (?)
- 超時率 ? (Number of Completed Task) --> (每個衛星 buf 稀少的情形下，可以準確協調彼此衛星資源，不會過早射完 buf)

=====================================
 Praying for you 🕯️ O Great Mita 💝 
=====================================