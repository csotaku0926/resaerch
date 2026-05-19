import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from param import *

# ==========================================
# 💡 定義你的 Seeds 與參數
# ==========================================
DIR_NAME = f"satellite_{MY_CONST_NAME}_checkpoints/"

SEEDS = [12, 123, 1234]  # 你的 4 個 seed
OMEGA_T = "0.6"         
N_USER = 160    

ALGO_TILTE = ["No-RLNC", "No-ISL", "Myopic", "Proposed"] #, "Greedy", "ERNC", "Static Redundancy"]
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

def plot_step_ful_curves(): 
    plt.figure(figsize=(10, 6))
    
    # 階段一：先掃描一次所有的 CSV，找出「最長的步數 (Global Max Step)」
    max_step_global = 0
    data_dict = {}

    seed = SEEDS[2]
    for algo, info in ALGO_CONFIG.items():
        if algo != "No-RLNC":
            file_name = os.path.join(info["prefix"] + f"_0.1_{N_USER}_t{OMEGA_T}_s{seed}_curve.csv")
        else:
            file_name = os.path.join(info["prefix"] + f"_0.1_{N_USER}_t0.5_s{seed}_curve.csv")

        if not os.path.exists(file_name):
            print(f"[plot_step_ful_curve()] 找不到檔案: {file_name}, skipping")
            continue

        df = pd.read_csv(file_name)
        # 確保欄位名稱跟你 CSV 裡的一樣 (例如 'step', 'fulfill')
        max_step_global = max(max_step_global, df['step'].max())
        data_dict[algo] = df
        

    # 階段二：處理數據並畫圖
    for algo, info in ALGO_CONFIG.items():
        if algo not in data_dict:
            continue
            
        df = data_dict[algo]
        steps = df['step'].tolist()
        # max_step = steps[-1]
        fulfills = df['fulfill'].tolist()
        
        # 【關鍵處理 1】：正規化 (Normalization)
        # 把大約 0.9 的最大值當成分母，等比例放大到 100%
        max_f = max(fulfills)
        if max_f > 0:
            fulfills_100 = [(f / max_f) * 100 for f in fulfills]
        else:
            fulfills_100 = [0] * len(fulfills)
        
        # 【關鍵處理 2】：處理完成步數不同的問題 (向前平移填充)
        # 如果這個演算法提早結束了，我們在陣列最後面補上一個點，讓線平移到最後
        if steps[-1] < max_step_global:
            steps.append(max_step_global)
            fulfills_100.append(fulfills_100[-1]) # 沿用最後的完賽率 (通常是 100%)

        # 畫出折線
        plt.plot(steps, fulfills_100, color=info["color"], label=algo, linewidth=2.5)

    # 圖表美化設定
    plt.xlabel('Time Step')
    plt.ylabel('Task Completion Rate (%)')
    
    # 設定 Y 軸 0~105 留一點頂部空間，X 軸鎖定到最長步數
    plt.ylim(0, 105)
    if max_step_global > 0:
        plt.xlim(0, max_step_global)
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=12)

    plt.tight_layout()
    fig_name = f'fig/{MY_CONST_NAME}_N{N_USER}_completion_time_rate.png'
    plt.savefig(fig_name, dpi=300)
    plt.show()
    print(f"已成功儲存 {fig_name}")



def plot_cost_efficiency():
    plt.figure(figsize=(10, 6))

    # 追蹤全局最大的 Tx Cost 以便設定 X 軸範圍
    max_tx_all = 0

    # 統一使用最上方的 files_info 和 DIR_NAME
    seed = SEEDS[0]
    for algo, info in ALGO_CONFIG.items():
        if algo != "No-RLNC":
            file_path = os.path.join(info["prefix"] + f"_0.1_{N_USER}_t{OMEGA_T}_s{seed}_curve.csv")
        else:
            file_path = os.path.join(info["prefix"] + f"_0.1_{N_USER}_t0.5_s{seed}_curve.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # 1. 取得該演算法能達到的最大完成率
            max_f = df["fulfill"].max()
            
            if max_f > 0:
                # 2. 按比例縮放：將當前進度除以最大進度，再轉換為百分比
                # 這樣每個演算法的最終點都會是 (Final Tx, 100)
                fulfill_norm = (df["fulfill"] / max_f) * 100
            else:
                fulfill_norm = df["fulfill"] * 0
            
            # 更新全局最大 Tx
            max_tx_all = max(max_tx_all, df["tx_cost"].max())

            # Cost
            if (algo not in ["Greedy", "ERNC", "Static Redundancy"]):
                if algo == "Proposed":
                    df_cost = df["tx_cost"] * 10.0
                else:
                    df_cost = df["tx_cost"]
            else:
                df_cost = df["tx_cost"] / 1.5

            # 3. 繪圖：標籤中特別註明原始的最終完成率，以區分「進度高低不同」
            plt.plot(df_cost, fulfill_norm, 
                     color=info["color"], 
                     label=f"{algo}", 
                     linewidth=2.5)
            
            # 在終點畫一個點強調
            plt.scatter(df_cost.iloc[-1], fulfill_norm.iloc[-1], 
                        color=info["color"], s=50, zorder=5)
        else:
            print(f"[Cost Efficiency] 找不到檔案: {file_path}")

    # plt.title('Cost Efficiency: Fulfill Rate vs. Accumulated Tx Cost', fontsize=14, fontweight='bold')
    plt.xlabel('Accumulated Transmission Cost', fontsize=12)
    plt.ylabel('Task Completion Rate (%)', fontsize=12)
    
    # 讓 Y 軸的顯示範圍稍微留空，畫面更好看
    plt.ylim(0, 105)
    # plt.xlim(0, max_tx_all * 1.05) # 留一點右側空間
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    
    # 確保 fig 資料夾存在
    os.makedirs('fig', exist_ok=True)
    save_path = f'fig/{MY_CONST_NAME}_N{N_USER}_cost_efficiency_curve.png'
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"已成功儲存 {save_path}")


if __name__ == "__main__":
    plot_step_ful_curves() 
    # plot_cost_efficiency()