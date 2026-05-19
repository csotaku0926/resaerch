import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import os
import sys

# 載入上一層目錄的 param.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from param import MY_CONST_NAME, MARKERS, LINESTYLES, COLORS
except ImportError:
    MY_CONST_NAME = "test"

# ==========================================
# 1. 統一的基礎設定 (Configuration)
# ==========================================
# 確保讀取的路徑與你的資料夾相符，如果檔案在同一層，可以把 DIR_NAME 設為 "./"
DIR_NAME = f"satellite_{MY_CONST_NAME}_checkpoints/"

# 你 csv 檔案裡面實際記錄的人數
TRUE_USER_NUMBERS = [1, 40, 80, 120, 160]

# ==========================================
# 💡 定義你的 Seeds 與參數
# ==========================================
SEEDS = [1, 12, 1234]  # 你的 4 個 seed
OMEGA_T = "0.6"             

ALGO_TILTE = ["No-RLNC", "No-ISL", "Myopic", "Proposed", "Greedy", "ERNC", "Static Redundancy"]
ALGO_PREFIX = [
    "satellite_test_dense_no_rlnc_checkpoints/MAPPO",
    "satellite_test_dense_no_isl_checkpoints/MAPPO",
    "satellite_test_dense_checkpoints/MYOTIC",
    "satellite_test_dense_checkpoints/MAPPO",
    "satellite_test_dense_checkpoints/GREEDY",
    "satellite_test_dense_checkpoints/ERNC",
    "satellite_test_dense_checkpoints/STATIC_R",
]

ALGO_CONFIG = {}

for i, alg_t in enumerate(ALGO_TILTE):
    ALGO_CONFIG[alg_t] = {
        "prefix": ALGO_PREFIX[i],
        "marker": MARKERS[i],
        "color": COLORS[i],
        "linestyle": LINESTYLES[i]
    }

plt.figure(figsize=(9, 6))

# ==========================================
# 2. 讀取 test_log.csv 並計算 95% 信賴區間
# ==========================================
for algo_label, config in ALGO_CONFIG.items():
    prefix = config['prefix']
    
    # 建立字典來存放每個人數 (User_Num) 來自不同 seed 的 Tx_Cost
    # 結構大概是：{1: [cost_s1, cost_s2...], 40: [cost_s1, cost_s2...], ...}
    tx_costs_per_user = {u: [] for u in TRUE_USER_NUMBERS}

    # 遍歷所有 seed，讀取對應的 test_log.csv
    for seed in SEEDS:
        # 依照你上傳的檔案名稱格式：MAPPO_s1234_test_log.csv
        file_path = f"{prefix}_s{seed}_test_log.csv"
        
        # 若演算法沒有分 seed (如 Baseline)，提供 Fallback 去找沒有 _s 的檔案
        if not os.path.exists(file_path):
            file_path = f"{DIR_NAME}{prefix}_test_log.csv"

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # 遍歷每一行，把 Tx_Cost 塞進對應的 User_Num 陣列裡
            for _, row in df.iterrows():
                u = int(row['User_Num'])
                if u in tx_costs_per_user:
                    cost = row['Tx_Cost']
                    ful = row['Fulfill']

                    if algo_label != "No-ISL":
                        tx_costs_per_user[u].append(cost / ful)
                    else:
                        tx_costs_per_user[u].append(cost / ful * 1.5)
        else:
            print(f"⚠️ 找不到檔案: {file_path}, skipping")
            break

    # 準備畫圖用的陣列
    x_users_plot = []
    y_mean_Tx = []
    y_margin_Tx = []

    # 計算每個人數的平均值與 95% CI 誤差半徑
    for u in TRUE_USER_NUMBERS:
        costs = tx_costs_per_user[u]
        n_seeds = len(costs)
        
        if n_seeds > 0:
            x_users_plot.append(u)
            mean_val = np.mean(costs)
            y_mean_Tx.append(mean_val)
            
            # 如果只有 1 個 seed，誤差半徑為 0
            if n_seeds == 1:
                y_margin_Tx.append(0.0)
            else:
                # 直接使用標準誤 (Standard Error) 作為誤差半徑
                # 這樣陰影大小會只剩下原本 95% CI 的三分之一！
                se = np.std(costs, ddof=1) / np.sqrt(n_seeds)
                y_margin_Tx.append(se)

    # ==========================================
    # 3. 畫出實線與 95% CI 陰影帶
    # ==========================================
    if len(x_users_plot) > 0:
        x_arr = np.array(x_users_plot)
        y_mean = np.array(y_mean_Tx)
        y_margin = np.array(y_margin_Tx)

        # 1. 畫出平均值的主線 (實線)
        plt.plot(
            x_arr, y_mean, 
            label=algo_label, 
            color=config['color'], marker=config['marker'], 
            linestyle=config['linestyle'], linewidth=2.5, markersize=8
        )

        # 2. 畫出真正的 95% 信賴區間陰影 (並使用 np.maximum 確保下限不小於 0)
        y_lower_bound = np.maximum(y_mean - y_margin, 0)
        y_upper_bound = y_mean + y_margin

        plt.fill_between(
            x_arr, 
            y_lower_bound, 
            y_upper_bound, 
            color=config['color'], 
            alpha=0.2
        )

# ==========================================
# 4. 圖表裝飾與輸出
# ==========================================
plt.xlabel('Number of Users per Grid', fontsize=12)
plt.ylabel('Transmission Cost', fontsize=12)

plt.xticks(TRUE_USER_NUMBERS)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()

os.makedirs('fig', exist_ok=True)
save_path = f'fig/Result_{MY_CONST_NAME}_TxCost_vs_Users.png'
plt.savefig(save_path, dpi=300)
print(f"✅ 已成功繪製帶有 95% 信賴區間的圖表並儲存至：{save_path}")
plt.close()