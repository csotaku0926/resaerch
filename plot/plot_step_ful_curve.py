import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from param import *

# 1. 定義你要比較的算法與對應的 CSV 檔名
# 你可以自行把其他 baseline 的 csv 檔名加進來
DIR_NAME = f"satellite_{MY_CONST_NAME}_checkpoints/"
USER_NUM = 400

files_info = {
    "MAPPO": {"file": f"MAPPO_{USER_NUM}_curve.csv", "color": "blue", "label": "MAPPO-CTDE (Proposed)"},
    "MYOTIC": {"file": f"MYOTIC_{USER_NUM}_curve.csv", "color": "red", "label": "MYOPIC-CTDE"},
    "GREEDY": {"file": f"GREEDY_{USER_NUM}_curve.csv", "color": "gray", "label": "Greedy"},
    "ERNC": {"file": f"ERNC_{USER_NUM}_curve.csv", "color": "orange", "label": "ER-NC"},
    "STATIC": {"file": f"STATIC_R_{USER_NUM}_curve.csv", "color": "green", "label": "Static Redundancy"},
}


def plot_step_ful_curves(): 
    plt.figure(figsize=(10, 6))
    
    # 階段一：先掃描一次所有的 CSV，找出「最長的步數 (Global Max Step)」
    max_step_global = 0
    data_dict = {}

    for algo, info in files_info.items():
        file_name = os.path.join(DIR_NAME, info["file"])
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            # 確保欄位名稱跟你 CSV 裡的一樣 (例如 'step', 'fulfill')
            max_step_global = max(max_step_global, df['step'].max())
            data_dict[algo] = df
        else:
            print(f"找不到檔案: {file_name}")

    # 階段二：處理數據並畫圖
    for algo, info in files_info.items():
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
        plt.plot(steps, fulfills_100, color=info["color"], label=info["label"], linewidth=2.5)

    # 圖表美化設定
    # plt.title('Fulfill Rate CDF over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Task Completion Rate (%)')
    
    # 設定 Y 軸 0~105 留一點頂部空間，X 軸鎖定到最長步數
    plt.ylim(0, 105)
    if max_step_global > 0:
        plt.xlim(0, max_step_global)
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=12)

    plt.tight_layout()
    fig_name = f'fig/{MY_CONST_NAME}_N{USER_NUM}_all_curves_comparison.png'
    plt.savefig(fig_name, dpi=300)
    plt.show()
    print(f"已成功儲存 {fig_name}")

def plot_cost_efficiency():
    plt.figure(figsize=(10, 6))

    # 統一使用最上方的 files_info 和 DIR_NAME
    for algo, info in files_info.items():
        file_path = os.path.join(DIR_NAME, info["file"])
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # X軸是累積流量 (tx_cost)，Y軸是任務完成率 (fulfill)
            # Matplotlib 會自動把 tx_cost 當作連續刻度，完美對齊
            plt.plot(df["tx_cost"], df["fulfill"], 
                     color=info["color"], label=info["label"], linewidth=2.5)
        else:
            print(f"[Cost Efficiency] 找不到檔案: {file_path}")

    plt.title('Cost Efficiency: Fulfill Rate vs. Accumulated Tx Cost', fontsize=14, fontweight='bold')
    plt.xlabel('Accumulated Transmission Cost (packets)', fontsize=12)
    plt.ylabel('User Fulfill Rate', fontsize=12)
    
    # 讓 Y 軸的顯示範圍稍微留空，畫面更好看
    plt.ylim(0, 1.05)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    
    # 確保 fig 資料夾存在
    os.makedirs('fig', exist_ok=True)
    save_path = f'fig/{MY_CONST_NAME}_N{USER_NUM}_cost_efficiency_curve.png'
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"已成功儲存 {save_path}")


if __name__ == "__main__":
    plot_step_ful_curves() 
    plot_cost_efficiency()