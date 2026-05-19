import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from param import *

# ==========================================
# 💡 嚴格遵守原始檔案的 Seeds 與參數
# ==========================================
DIR_NAME = f"satellite_{MY_CONST_NAME}_checkpoints/"

SEEDS = [12, 123, 1234]  
OMEGA_T = "0.6"         
N_USER = 160    

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

def plot_cost_efficiency():
    plt.figure(figsize=(10, 6))
    max_tx_all = 0

    for algo, info in ALGO_CONFIG.items():
        all_x = []
        all_y = []
        local_max_x = 0
        
        for seed in SEEDS:
            # 【關鍵修復 1】：完全遵守 No-RLNC 讀取 t0.5 的邏輯
            if algo != "No-RLNC":
                file_path = f"{info['prefix']}_0.1_{N_USER}_t{OMEGA_T}_s{seed}_curve.csv"
            else:
                file_path = f"{info['prefix']}_0.1_{N_USER}_t0.5_s{seed}_curve.csv"
            
            if not os.path.exists(file_path):
                print(f"⚠️ 找不到檔案: {file_path}，直接跳過。")
                continue

            df = pd.read_csv(file_path)
            if df.empty: 
                continue
                
            # 【關鍵修復 2】：完全遵守原本的 X 軸邏輯
            if algo not in ["Greedy", "ERNC", "Static Redundancy"]:
                if algo == "Proposed":
                    x_vals = df["tx_cost"].values * 3.0
                else:
                    x_vals = df["tx_cost"].values
            else:
                x_vals = df["tx_cost"].values / 1.5
                    
            # 【關鍵修復 3】：完全遵守原本的 Y 軸動態正規化邏輯 (除以 max_f)
            max_f = df["fulfill"].max()
            if max_f > 0:
                y_vals = (df["fulfill"].values / max_f) * 100
            else:
                y_vals = df["fulfill"].values * 0

            # 過濾掉 X 軸重複的值 (插值法要求 X 必須嚴格遞增)
            x_unique, indices = np.unique(x_vals, return_index=True)
            y_unique = y_vals[indices]

            all_x.append(x_unique)
            all_y.append(y_unique)
            local_max_x = max(local_max_x, x_unique[-1])
            max_tx_all = max(max_tx_all, local_max_x)

        # 進行插值 (對齊 X 軸) 與計算平均/陰影
        if len(all_x) > 0:
            common_x = np.linspace(0, local_max_x, 300)
            interp_y_list = []

            for x_arr, y_arr in zip(all_x, all_y):
                f = interp1d(x_arr, y_arr, kind='linear', bounds_error=False, fill_value=(y_arr[0], y_arr[-1]))
                interp_y_list.append(f(common_x))

            interp_y_matrix = np.array(interp_y_list)
            n_seeds = len(interp_y_list)

            mean_y = np.mean(interp_y_matrix, axis=0)
            
            plt.plot(common_x, mean_y, color=info["color"], label=f"{algo}", linewidth=2.5, linestyle=info["linestyle"])
            plt.scatter(common_x[-1], mean_y[-1], color=info["color"], marker=info.get("marker", "o"), s=60, zorder=5)

            if n_seeds > 1:
                se_y = np.std(interp_y_matrix, ddof=1, axis=0) / np.sqrt(n_seeds)
                
                y_lower = np.clip(mean_y - se_y, 0, 100)
                y_upper = np.clip(mean_y + se_y, 0, 100)

                plt.fill_between(common_x, y_lower, y_upper, color=info["color"], alpha=0.15)

    # 圖表裝飾與輸出
    plt.xlabel('Accumulated Transmission Cost', fontsize=12)
    plt.ylabel('Task Completion Rate (%)', fontsize=12)

    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()

    os.makedirs('fig', exist_ok=True)
    save_path = f'fig/{MY_CONST_NAME}_N{N_USER}_cost_efficiency_curve_shaded.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 已成功繪製帶有陰影的曲線圖並儲存至：{save_path}")
    plt.close()


if __name__ == "__main__":
    # plot_step_ful_curves() 
    plot_cost_efficiency()