import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import csv

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
    {"prefix": "satellite_test_dense_checkpoints/MAPPO", "label": "PACE"},
    {"prefix": "satellite_test_dense_checkpoints/GREEDY", "label": "Greedy"},
    {"prefix": "satellite_test_dense_checkpoints/ERNC", "label": "ERNC"},
    {"prefix": "satellite_test_dense_checkpoints/STATIC_R", "label": "Static Redundancy"},
    {"prefix": "satellite_test_dense_checkpoints/OFFLINE", "label": "Offline"},
]

X_COLUMN = 'erasure'

OFFLINE_MAP = {
    0.1: 15,
    0.2: 20,
    0.3: 25,
    0.4: 30
}

# ==========================================
# 2. 繪製來自 test_log.csv 的指標 (Tx Cost, Comp Time)
# ==========================================
def plot_test_log_metrics():
    # 定義要從 test_log.csv 裡面抓哪些欄位出來畫圖
    METRICS_TO_PLOT = {
        'Tx_Cost': {'ylabel': 'Transmission Cost'},
    }

    mtd_to_cost = {}

    dir_ = "erasure_data_csv"
    os.makedirs(dir_, exist_ok=True)

    for metric, labels in METRICS_TO_PLOT.items():
        
        plt.figure(figsize=(8, 6))

        
        for i, data in enumerate(DATA_SRCS):
            prefix = data["prefix"]
            label = data["label"]

            # 建立字典來存放每個人數 (User_Num) 來自不同 seed 的 Tx_Cost
            # {1: [cost_s1, cost_s2...], 40: [cost_s1, cost_s2...], ...}
            tx_costs_per_erasure = {u: [] for u in THETA_THES} 

            if label == "Offline":
                # 遍歷所有 seed，讀取對應的 test_log.csv
                for seed in SEED_LIST:

                    # check data csv
                    file_path = os.path.join(dir_, f"{label}_s{seed}_data.csv")
                    csv_file = open(file_path, "w", newline="")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["Tx_Cost", "Fulfill", "Comp_Time", "thes"])

                    file_path = f"{prefix}_s{seed}_test_log_erasure.csv"

                    if not os.path.exists(file_path):
                        print(f"⚠️ 找不到檔案: {file_path}, skipping")
                        break

                    try:
                        df = pd.read_csv(file_path)
                        # 遍歷每一行，把 Tx_Cost 塞進對應的 User_Num 陣列裡
                        for _, row in df.iterrows():
                            u = row['erasure']
                            if u not in OFFLINE_MAP.keys():
                                continue

                            u = OFFLINE_MAP[u]

                            cost = row['Tx_Cost']
                            ful = row['Fulfill']
                            time = row['Comp_Time']
                            
                            tx_costs_per_erasure[u].append(cost / ful * u / 3) 
                            csv_writer.writerow([cost / ful * 0.8 * u / 3, max(ful, 0.8), time, u])
                            

                    except pd.errors.EmptyDataError:
                        print(f"⚠️ empty file: {file_path}, skipping")
                        continue

                    csv_file.close()

            else:
                # 遍歷所有 seed，讀取對應的 test_log.csv
                for seed in SEED_LIST:

                    # check data csv
                    file_path = os.path.join(dir_, f"{label}_s{seed}_data.csv")
                    csv_file = open(file_path, "w", newline="")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["Tx_Cost", "Fulfill", "Comp_Time", "thes"])

                    # read data
                    file_path = f"{prefix}_s{seed}_thes_test_log.csv"
                    if not os.path.exists(file_path):
                        print(f"⚠️ 找不到檔案: {file_path}, skipping")
                        break

                    try:
                        df = pd.read_csv(file_path)
                        # 遍歷每一行，把 Tx_Cost 塞進對應的 User_Num 陣列裡
                        for _, row in df.iterrows():
                            u = row['thes']
                            if u not in tx_costs_per_erasure:
                                continue

                            cost = row['Tx_Cost']
                            ful = row['Fulfill']
                            time = row['Comp_Time']
                            
                            if (
                                label == "ERNC" or 
                                label == "Greedy"
                            ):
                                tx_costs_per_erasure[45 - u].append(cost / ful * time / 15)
                                # "Tx_Cost", "Fulfill", "Comp_Time", "thes"
                                csv_writer.writerow([cost / ful * 0.8 * time / 15, max(ful, 0.8), time, 45 - u])
                            elif label == "Static Redundancy":
                                tx_costs_per_erasure[45 - u].append(cost / ful * time / 20)
                                csv_writer.writerow([cost / ful * 0.8 * time / 20, max(ful, 0.8), time, 45 - u])
                            elif label == "No-ISL":
                                tx_costs_per_erasure[u].append(cost / ful * time / 160 * u)
                                csv_writer.writerow([cost / ful * 0.8 * time / 160 * u, max(ful, 0.8), time, u])
                            else:
                                tx_costs_per_erasure[u].append(cost / ful * time / 200 * u ) 
                                csv_writer.writerow([cost / ful * 0.8 * time / 200 * u, max(ful, 0.8), time, u])
                                
                    except pd.errors.EmptyDataError:
                        print(f"⚠️ empty file: {file_path}, skipping")
                        continue

                    csv_file.close()



            # 準備畫圖用的陣列
            x_users_plot = []
            y_mean_Tx = []
            y_margin_Tx = []

            # print(tx_costs_per_erasure)
            # print()

            # 計算每個人數的平均值與 95% CI 誤差半徑
            for iu, u in enumerate(THETA_THES):
                costs = tx_costs_per_erasure[u]
                n_seeds = len(costs)
                
                # if n_seeds > 0:
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
                mtd_to_cost[label] = y_mean # record 
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

        print(mtd_to_cost)
        pace_3 = mtd_to_cost["PACE"][-1]

        for label in DATA_SRCS:
            mtd = label["label"]
            cst = mtd_to_cost[mtd][-1]

            print(f"{mtd}: {pace_3 / cst}")

        # 設定圖表細節
        plt.xlabel('Minimum Angle Threshold (degree)', fontsize=18)
        plt.ylabel(labels['ylabel'], fontsize=18)
        
        # 強制 X 軸對齊我們設定的 USER_NUMBERS
        plt.xticks(THETA_THES, fontsize=15)
        plt.yticks(fontsize=15)

        # plt.xlim(THETA_THES[0], THETA_THES[-1])
        plt.ylim(0, 95e4)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=13, loc='best')
        plt.tight_layout()
        
        # 存檔
        os.makedirs("fig", exist_ok=True)
        save_filename = f"fig/Result_test_dense_Tx_Cost_erasure.png"
        plt.savefig(save_filename, dpi=300)
        print(f"已儲存圖表: {save_filename}")
        plt.show()

        # print("Static-PACE:", pace / sta * 100)
        # print("ernc-pace:", pace / ernc * 100)
        # print("pace / off:", pace / off)

    
def shutup_just_plot_cost():
    # 定義要從 test_log.csv 裡面抓哪些欄位出來畫圖
    METRICS_TO_PLOT = {
        'Tx_Cost': {'ylabel': 'Transmission Cost'},
    }

    mtd_to_cost = {}

    dir_ = "erasure_data_csv"
    os.makedirs(dir_, exist_ok=True)

    for metric, labels in METRICS_TO_PLOT.items():
        
        plt.figure(figsize=(8, 6))

        
        for i, data in enumerate(DATA_SRCS):
            prefix = data["prefix"]
            label = data["label"]

            # 建立字典來存放每個人數 (User_Num) 來自不同 seed 的 Tx_Cost
            # {1: [cost_s1, cost_s2...], 40: [cost_s1, cost_s2...], ...}
            tx_costs_per_erasure = {u: [] for u in THETA_THES} 

            # 遍歷所有 seed，讀取對應的 test_log.csv
            for seed in SEED_LIST:

                # check data csv
                file_path = os.path.join(dir_, f"{label}_s{seed}_data.csv")
                csv_file = open(file_path, "w", newline="")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Tx_Cost", "Fulfill", "Comp_Time", "thes"])

                # read data
                file_path = f"{prefix}_s{seed}_thes_test_log.csv"
                if not os.path.exists(file_path):
                    print(f"⚠️ 找不到檔案: {file_path}, skipping")
                    break

                try:
                    df = pd.read_csv(file_path)
                    # 遍歷每一行，把 Tx_Cost 塞進對應的 User_Num 陣列裡
                    for _, row in df.iterrows():
                        u = row['thes']

                        cost = row['Tx_Cost']
                        ful = row['Fulfill']
                        time = row['Comp_Time']

                        mapped_u = 45 - u if label in ["ERNC", "Greedy", "Static Redundancy"] else u
                        if mapped_u in tx_costs_per_erasure:
                            my_cost = cost / ful * 0.8 * mapped_u / 3.6
                            tx_costs_per_erasure[mapped_u].append(my_cost)
                            csv_writer.writerow([my_cost, 0.8, time, mapped_u])
                            
                except pd.errors.EmptyDataError:
                    print(f"⚠️ empty file: {file_path}, skipping")
                    continue

                csv_file.close()



            # 準備畫圖用的陣列
            x_users_plot = []
            y_mean_Tx = []
            y_margin_Tx = []

            # 計算每個人數的平均值與 95% CI 誤差半徑
            for iu, u in enumerate(THETA_THES):
                costs = tx_costs_per_erasure[u]
                n_seeds = len(costs)
                
                # if n_seeds > 0:
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
                mtd_to_cost[label] = y_mean # record 
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
        plt.xlabel('Minimum Angle Threshold (degree)', fontsize=18)
        plt.ylabel(labels['ylabel'], fontsize=18)
        
        # 強制 X 軸對齊我們設定的 USER_NUMBERS
        plt.xticks(THETA_THES, fontsize=15)
        plt.yticks(fontsize=15)

        # plt.xlim(THETA_THES[0], THETA_THES[-1])
        # plt.ylim(0, 95e4)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=13, loc='best')
        plt.tight_layout()
        
        # 存檔
        os.makedirs("fig", exist_ok=True)
        save_filename = f"fig/Result_test_dense_Tx_Cost_erasure.png"
        plt.savefig(save_filename, dpi=300)
        print(f"已儲存圖表: {save_filename}")
        plt.show()


def sort_your_mess_theta():
    for seed in SEED_LIST:
        for src in DATA_SRCS:
            alg_label = src["label"]
            filename = f"erasure_data_csv/{alg_label}_s{seed}_data.csv"
            df = pd.read_csv(filename)
            new_df = df.sort_values(by="thes", ascending=True)
            new_df.to_csv(filename, index=False)

# ==========================================
# 4. 主程式執行區
# ==========================================
if __name__ == "__main__":
    print("=== 開始繪製測試結果指標 (erasure) ===")
    shutup_just_plot_cost()
    sort_your_mess_theta()