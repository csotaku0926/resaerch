import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 載入上一層目錄的 param.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from param import *
except ImportError:
    MY_CONST_NAME = "test"

# ==========================================
# 1. 統一的基礎設定 (Configuration)
# ==========================================
DATA_SRCS = [
    {"prefix": "satellite_test_dense_no_rlnc_checkpoints/MAPPO", "label": "No-RLNC"},
    {"prefix": "satellite_test_dense_no_isl_checkpoints/MAPPO", "label": "No-ISL"},
    {"prefix": "satellite_test_dense_checkpoints/MYOTIC", "label": "Myopic"},
    {"prefix": "satellite_test_dense_checkpoints/MAPPO", "label": "Proposed (Tw=2)"},
    {"prefix": "satellite_test_dense_checkpoints/GREEDY", "label": "Greedy"},
    {"prefix": "satellite_test_dense_checkpoints/ERNC", "label": "ERNC"},
    {"prefix": "satellite_test_dense_checkpoints/STATIC_R", "label": "Static Redundancy"},
    {"prefix": "satellite_test_checkpoints/OFFLINE", "label": "Offline"},
]

ERASURE_RATES = [0.1, 0.2, 0.3, 0.4] 

X_COLUMN = 'erasure'


# ==========================================
# 2. 繪製來自 test_log.csv 的指標 (Tx Cost, Comp Time)
# ==========================================
def plot_test_log_metrics():
    # 定義要從 test_log.csv 裡面抓哪些欄位出來畫圖
    METRICS_TO_PLOT = {
        'Tx_Cost': {'ylabel': 'Transmission Cost'},
    }

    for metric, labels in METRICS_TO_PLOT.items():
        plt.figure(figsize=(8, 6))
        
        for i, data in enumerate(DATA_SRCS):
            prefix = data["prefix"]
            label = data["label"]

            # 建立字典來存放每個人數 (User_Num) 來自不同 seed 的 Tx_Cost
            # {1: [cost_s1, cost_s2...], 40: [cost_s1, cost_s2...], ...}
            tx_costs_per_erasure = {u: [] for u in ERASURE_RATES} 

            # 遍歷所有 seed，讀取對應的 test_log.csv
            for seed in SEED_LIST:

                file_path = f"{prefix}_s{seed}_test_log_erasure.csv"

                if os.path.exists(file_path):

                    try:
                        df = pd.read_csv(file_path)
                        # 遍歷每一行，把 Tx_Cost 塞進對應的 User_Num 陣列裡
                        for _, row in df.iterrows():
                            u = row['erasure']
                            if u in tx_costs_per_erasure:
                                cost = row['Tx_Cost']
                                ful = row['Fulfill']
                                
                                if label == "Offline":
                                    tx_costs_per_erasure[u].append(cost / ful * 7)
                                    print("erasure:", u, cost / ful * 7)
                                else:    
                                    tx_costs_per_erasure[u].append(cost / ful)
                                    if (label == "Proposed (Tw=2)"):
                                        print('proposed', u, cost / ful)

                                

                    except pd.errors.EmptyDataError:
                        print(f"⚠️ empty file: {file_path}, skipping")
                        continue

                else:
                    print(f"⚠️ 找不到檔案: {file_path}, skipping")
                    break

            # 準備畫圖用的陣列
            x_users_plot = []
            y_mean_Tx = []
            y_margin_Tx = []

            # 計算每個人數的平均值與 95% CI 誤差半徑
            for u in ERASURE_RATES:
                costs = tx_costs_per_erasure[u]
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
                    label=label, 
                    color=COLORS[i], marker=MARKERS[i], 
                    linestyle=LINESTYLES[i], linewidth=2.5, markersize=8
                )

                # 2. 畫出真正的 95% 信賴區間陰影 (並使用 np.maximum 確保下限不小於 0)
                y_lower_bound = np.maximum(y_mean - y_margin, 0)
                y_upper_bound = y_mean + y_margin

                plt.fill_between(
                    x_arr, 
                    y_lower_bound, 
                    y_upper_bound, 
                    color=COLORS[i], 
                    alpha=0.2
                )

        # 設定圖表細節
        plt.xlabel('Average Erasure Probability', fontsize=12)
        plt.ylabel(labels['ylabel'], fontsize=12)
        
        # 強制 X 軸對齊我們設定的 USER_NUMBERS
        plt.xticks(ERASURE_RATES)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11, loc='best')
        plt.tight_layout()
        
        # 存檔
        os.makedirs("fig", exist_ok=True)
        save_filename = f"fig/Result_{MY_CONST_NAME}_{metric}_erasure.png"
        plt.savefig(save_filename, dpi=300)
        print(f"已儲存圖表：{save_filename}")
        plt.show()


# ==========================================
# 4. 主程式執行區
# ==========================================
if __name__ == "__main__":
    print("=== 開始繪製測試結果指標 (erasure) ===")
    plot_test_log_metrics()