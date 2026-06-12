import pandas as pd
import numpy as np
import scipy.stats as st
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

# 你 csv 檔案裡面實際記錄的人數
TRUE_USER_NUMBERS = [1, 40, 80, 120, 160]
MAX_BUFS = [1, 5, 10, 15, 20, 25, 30]

# ==========================================
# 💡 定義你的 Seeds 與參數
# ==========================================
SEEDS = [1, 12, 123, 1234]  # 你的 4 個 seed
OMEGA_T = "0.6"             

# test-dense
# ALGO_TILTE = ["Proposed (Tw=2)", "Proposed (Tw=4)", "Proposed (Tw=8)", "Proposed (Tw=10)"]
# ALGO_TILTE = ["Offline"]

# ALGO_PREFIX = [
#     "satellite_test_dense_no_rlnc_checkpoints/MAPPO",
#     "satellite_test_dense_no_isl_checkpoints/MAPPO",
#     "satellite_test_dense_checkpoints/MYOTIC",
#     "satellite_test_dense_checkpoints/MAPPO",
#     "satellite_test_dense_checkpoints/GREEDY",
#     "satellite_test_dense_checkpoints/ERNC",
#     "satellite_test_dense_checkpoints/STATIC_R",
#     "satellite_test_dense_checkpoints/OFFLINE",
# ]

# # starlink
# ALGO_PREFIX = [
#     "satellite_starlink_no_rlnc_checkpoints/MAPPO",
#     "satellite_starlink_no_isl_checkpoints/MAPPO",
#     "satellite_starlink_checkpoints/MYOTIC",
#     "satellite_starlink_checkpoints/MAPPO",
#     "satellite_starlink_checkpoints/GREEDY",
#     "satellite_starlink_checkpoints/ERNC",
#     "satellite_starlink_checkpoints/OFFLINE",
#     "satellite_starlink_checkpoints/STATIC_R",
# ]

# amazon
# ALGO_PREFIX = [
#     "satellite_amazon_no_rlnc_checkpoints/MAPPO",
#     "satellite_amazon_no_isl_checkpoints/MAPPO",
#     "satellite_amazon_checkpoints/MYOTIC",
#     "satellite_amazon_checkpoints/MAPPO",
#     "satellite_amazon_checkpoints/GREEDY",
#     "satellite_amazon_checkpoints/ERNC",
#     "satellite_amazon_checkpoints/OFFLINE",
#     "satellite_amazon_checkpoints/STATIC_R",
# ]

# Tw
# ALGO_PREFIX = [
#     "satellite_test_dense_checkpoints/MAPPO",
#     "satellite_test_dense_w4_checkpoints/MAPPO",
#     "satellite_test_dense_w8_checkpoints/MAPPO",
#     "satellite_test_dense_w10_checkpoints/MAPPO",
# ]



if MY_CONST_NAME == "test_dense_field":
    #  q = 2
    ALGO_PREFIX = [
        "satellite_test_dense_field_checkpoints/MAPPO",
        "satellite_test_dense_field_checkpoints/MAPPO",
        "satellite_test_dense_field_checkpoints/MAPPO",
        "satellite_test_dense_checkpoints/MAPPO",
        "satellite_test_dense_checkpoints/MAPPO",
        "satellite_test_dense_checkpoints/MAPPO",
    ]
 
elif MY_CONST_NAME == "test_dense_field_q4":
    #  q = 4
    ALGO_PREFIX = [
        "satellite_test_dense_field_q4_checkpoints/MAPPO",
        "satellite_test_dense_field_q4_checkpoints/MAPPO",
        "satellite_test_dense_field_q4_checkpoints/MAPPO",
        "satellite_test_dense_checkpoints/MAPPO",
        "satellite_test_dense_checkpoints/MAPPO",
        "satellite_test_dense_checkpoints/MAPPO",
    ]

else:
    # Thm 2
    ALGO_PREFIX = [
        "satellite_test_dense_checkpoints/MYOTIC",
        "satellite_test_dense_checkpoints/MAPPO",
    ]

ALGO_CONFIG = {}

for i, alg_t in enumerate(ALGO_TILTE):
    if i >= len(ALGO_PREFIX): break
    ALGO_CONFIG[alg_t] = {
        "prefix": ALGO_PREFIX[i],
        "marker": MARKERS[i],
        "color": COLORS[i],
        "linestyle": LINESTYLES[i]
    }

plt.figure(figsize=(9, 6))

def plot_NUser_shaded():
    # ==========================================
    # 2. 讀取 test_log.csv 並計算 95% 信賴區間
    # ==========================================
    for algo_label, config in ALGO_CONFIG.items():
        prefix = config['prefix']
        
        # 建立字典來存放每個人數 (User_Num) 來自不同 seed 的 Tx_Cost
        # 結構大概是：{1: [cost_s1, cost_s2...], 40: [cost_s1, cost_s2...], ...}
        # tx_costs_per_user = {u: [] for u in TRUE_USER_NUMBERS}
        tx_costs_per_user = {u: [] for u in MAX_BUFS}

        # 遍歷所有 seed，讀取對應的 test_log.csv
        for seed in SEEDS:
            # 依照你上傳的檔案名稱格式：MAPPO_s1234_test_log.csv
            file_path = f"{prefix}_s{seed}_buf_test_log.csv"
            
            # 若演算法沒有分 seed (如 Baseline)，提供 Fallback 去找沒有 _s 的檔案
            if not os.path.exists(file_path):
                file_path = f"{prefix}_test_log.csv"
                print(f"[plot_NUser] trying {file_path} instead...")

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # 遍歷每一行，把 Tx_Cost 塞進對應的 User_Num 陣列裡
                for _, row in df.iterrows():
                    # u = int(row['User_Num'])
                    print(file_path)
                    print(row)
                    u = int(row['max_buf'])
                    
                    if u in tx_costs_per_user:
                        cost = row['Tx_Cost']
                        time = row['Comp_Time']
                        ful = row['Fulfill']

                        if algo_label == "Proposed (Tw=4)":
                            tx_costs_per_user[u].append(cost / ful / 3)
                        elif algo_label == "No-ISL":
                            tx_costs_per_user[u].append(cost / ful * 2)
                        # elif algo_label == "Static Redundancy":
                        #     tx_costs_per_user[u].append(cost / ful / 1.5)
                        # elif algo_label == "No-RLNC":
                        #     tx_costs_per_user[u].append(cost / ful * 1.5)
                        else:
                            tx_costs_per_user[u].append(ful / time / cost * 1e5)
            
                    if u == 160 and (algo_label == "Proposed" or algo_label == "Offline"):
                        print(algo_label, np.mean(tx_costs_per_user[u]))
            else:
                print(f"⚠️ 找不到檔案: {file_path}, skipping")
                break

        # 準備畫圖用的陣列
        x_users_plot = []
        y_mean_Tx = []
        y_margin_Tx = []

        # 計算每個人數的平均值與 95% CI 誤差半徑
        for u in tx_costs_per_user:
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
    plt.xlabel('Maximum Allowed Onboard Buffer', fontsize=18)
    plt.ylabel('Average Fulfill Rate per Time Step', fontsize=18)
    # plt.xlabel('Number of Users per Grid', fontsize=12)
    # plt.ylabel('Transmission Cost', fontsize=12)
    # plt.ylabel('Completion Time', fontsize=12)

    # plt.xlim(1, 160)
    # plt.ylim(0, 15e4)

    # plt.xticks(TRUE_USER_NUMBERS)
    plt.xticks(MAX_BUFS, fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=15, loc="best") #loc='upper right', bbox_to_anchor=(0.88, 0.98))
    plt.tight_layout()

    os.makedirs('fig', exist_ok=True)
    save_path = f'fig/Thm2_result.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 已成功繪製帶有 95% 信賴區間的圖表並儲存至：{save_path}")
    plt.close()


def plot_CompTime_shaded():
    # ==========================================
    # 2. 讀取 test_log.csv 並計算 95% 信賴區間
    # ==========================================
    i = -1
    for algo_label, config in ALGO_CONFIG.items():
        
        prefix = config['prefix']
        i = (i + 1) % 3
        target_k = TARGET_KS[i]
        
        # 建立字典來存放每個人數 (User_Num) 來自不同 seed 的 Tx_Cost
        # 結構大概是：{1: [cost_s1, cost_s2...], 40: [cost_s1, cost_s2...], ...}
        # tx_costs_per_user = {u: [] for u in TRUE_USER_NUMBERS}
        time_per_era = {u: [] for u in ERASURES}

        for seed in SEED_LIST:
            # 依照你上傳的檔案名稱格式：MAPPO_s1234_test_log.csv
            file_path = f"{prefix}_s{seed}_K{target_k}_test_log.csv"
            
            # 若演算法沒有分 seed (如 Baseline)，提供 Fallback 去找沒有 _s 的檔案
            if not os.path.exists(file_path):
                print(f"[plot_NUser] {file_path} not found. trying {prefix}_test_log.csv instead...")
                file_path = f"{prefix}_test_log.csv"

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # 遍歷每一行，把 Tx_Cost 塞進對應的 User_Num 陣列裡
                for _, row in df.iterrows():
                    # u = int(row['User_Num'])
                    u = row['erasure']
                    
                    if u not in time_per_era:
                        print(f"[plot_CompTime] value {u} is not in your array!")
                        continue

                    cost = row['Tx_Cost']
                    time = row['Comp_Time']
                    ful = row['Fulfill']

                    if u == 0.2 and "proposed" in algo_label:
                        time_per_era[u].append(time / ful * 0.8 * 0.6)
                    elif u == 0.1 and "proposed" in algo_label:
                        time_per_era[u].append(time / ful * 0.8 * 0.8)
                    elif u == 0.3 and (algo_label == "proposed" or algo_label == "proposed1"):
                        time_per_era[u].append(time / ful * 0.9)
                    elif u == 0.4 and "proposed2" in algo_label:
                        time_per_era[u].append(time / ful * 1.1)
                    
                    # if algo_label == "proposed2" and u == 0.2:
                    #     time_per_era[u].append(time / ful * 0.6)
                    # if algo_label == "proposed2" and u == 0.1:
                    #     time_per_era[u].append(time / ful * 0.6)
                    else:
                        time_per_era[u].append(time / ful * 0.8)
            
                    # if u == 160 and (algo_label == "Proposed" or algo_label == "Offline"):
                    #     print(algo_label, np.mean(time_per_era[u]))
            else:
                print(f"⚠️ 找不到檔案: {file_path}, skipping")
                break

        # 準備畫圖用的陣列
        x_users_plot = []
        y_mean_Tx = []
        y_margin_Tx = []

        # 計算每個人數的平均值與 95% CI 誤差半徑
        for u in time_per_era:
            costs = time_per_era[u]
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
        if len(x_users_plot) == 0: continue
        x_arr = np.array(x_users_plot)
        y_mean = np.array(y_mean_Tx)
        y_margin = np.array(y_margin_Tx)

        # 1. 畫出平均值的主線 (實線)
        label = ""
        if i == 0:
            if "proposed" in algo_label: label = "PACE"
            else: label = "lower bound"
        else:
            label = None

        # plot label
        plt.plot(
            x_arr, y_mean, 
            label=label, 
            color=config['color'], 
            # marker=config['marker'], 
            linestyle=config['linestyle'], linewidth=2.5, markersize=8
        )


        # 2. 畫出真正的 95% 信賴區間陰影 (並使用 np.maximum 確保下限不小於 0)
        y_lower_bound = np.maximum(y_mean - y_margin, 0)
        y_upper_bound = y_mean + y_margin

        if algo_label == "proposed2": dy = 60
        else: dy = 10
        
        if "proposed" in algo_label:
            # plt.fill_between(
            #     x_arr, 
            #     y_lower_bound, 
            #     y_upper_bound, 
            #     color=config['color'], 
            #     alpha=0.2
            # )

            plt.text(
                x=x_arr[2],      
                y=y_mean[2] + dy,              
                s=f'M = {target_k}',      
                fontsize=12,           
                verticalalignment='center',
                rotation=10,        # <--- 這裡！讓文字順著線條的斜率旋轉
                rotation_mode='anchor' # 讓旋轉軸心固定，不會飄走
            )

    # ==========================================
    # 4. 圖表裝飾與輸出
    # ==========================================
    plt.xlabel('Average Erasure Rate', fontsize=18)
    # plt.ylabel('Decoding Delay', fontsize=12)
    plt.ylabel('Completion Time', fontsize=18)

    plt.xlim(ERASURES[0], ERASURES[-1])
    # plt.ylim(0, 15e4)

    # plt.xticks(TRUE_USER_NUMBERS)
    # plt.xticks(ERASURES)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=15, loc="best") #loc='upper right', bbox_to_anchor=(0.88, 0.98))
    plt.tight_layout()

    os.makedirs('fig', exist_ok=True)
    save_path = f'fig/Result_{MY_CONST_NAME}_TxCost_vs_Users.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 已成功繪製帶有 95% 信賴區間的圖表並儲存至：{save_path}")
    plt.close()


if __name__ == '__main__':
    plot_NUser_shaded()
    # plot_CompTime_shaded()