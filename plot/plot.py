import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from param import *

# ==========================================
# 1. 基礎設定與檔案讀取
# ==========================================
# 定義你的演算法標籤與對應的 CSV 檔案名稱
# 假設你的檔案分別命名為以下名稱，請依照實際檔名修改
algorithm_files = {
    'Greedy': 'GREEDY_test_log.csv',
    'Static': 'STATIC_R_test_log.csv',
    'ERNC': 'ERNC_test_log.csv',
    'Myopic MARL': 'MYOTIC_test_log.csv',
    'Proposed MARL (Ours)': 'MAPPO_test_log.csv'
}

# 讀取所有 CSV 檔案並存入字典中
dir_name = f"satellite_{MY_CONST_NAME}_checkpoints/"
data_dict = {}
for algo_name, file_ in algorithm_files.items():
    file_name = os.path.join(dir_name, file_)
    if os.path.exists(file_name):
        data_dict[algo_name] = pd.read_csv(file_name)
    else:
        print(f"警告：找不到檔案 {file_name}，將跳過此演算法的繪製。")

# ==========================================
# 2. 定義要畫的三張圖的參數
# ==========================================
# 定義 Y 軸要畫的欄位名稱，以及對應的圖表標題與 Y 軸標籤
metrics_to_plot = {
    'Tx_Cost': {
        'title': 'Tx Cost vs User Number',
        'ylabel': 'Tx Cost'
    }
    # ,
    # 'Fulfill': {
    #     'title': 'Fulfillment vs User Number',
    #     'ylabel': 'Fulfill %'
    # },
    # 'Comp_Time': {
    #     'title': 'Completion Time vs User Number',
    #     'ylabel': 'Completion Time (Step)'
    # }
}

# X 軸的欄位名稱 (假設你的 CSV 裡叫做 User_Num)
x_column = 'User_Num'

# 定義四個演算法的線條顏色與標記符號，讓圖表更具學術專業感
styles = {
    'Greedy': {'color': 'gray', 'marker': 'x', 'linestyle': '--'},
    'Static': {'color': 'green', 'marker': 's', 'linestyle': '-.'},
    'ERNC': {'color': 'orange', 'marker': '^', 'linestyle': ':'},
    'Myopic MARL': {'color': 'red', 'marker': 'o', 'linestyle': '-'},
    'Proposed MARL (Ours)': {'color': 'blue', 'marker': 'o', 'linestyle': '-'}
}

# ==========================================
# 3. 執行繪圖迴圈
# ==========================================
for metric, labels in metrics_to_plot.items():
    plt.figure(figsize=(8, 6)) # 設定畫布大小
    
    # 畫出每個演算法在該指標下的折線
    for algo_name, df in data_dict.items():
        if metric in df.columns and x_column in df.columns:
            plt.plot(
                df[x_column], 
                df[metric], 
                label=algo_name, 
                color=styles[algo_name]['color'], 
                marker=styles[algo_name]['marker'], 
                linestyle=styles[algo_name]['linestyle'],
                linewidth=2,   # 線條粗細
                markersize=8   # 標記大小
            )
        else:
            print(f"警告：{algo_name} 的資料中找不到欄位 {metric} 或 {x_column}")

    # 設定圖表細節
    # plt.title(labels['title'], fontsize=14, fontweight='bold')
    plt.xlabel('Number of Users per Grid', fontsize=12)
    plt.ylabel(labels['ylabel'], fontsize=12)
    
    # 設定 X 軸刻度只顯示整數 (例如 10, 20, 30...)
    if data_dict:
        sample_df = list(data_dict.values())[0]
        plt.xticks(sample_df[x_column].unique())

    # 加入網格線與圖例
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='best') # 自動將圖例放在不遮擋線條的好位置
    plt.tight_layout()
    
    # 儲存圖片 (高解析度 300 dpi 適合放進論文)
    save_filename = f"fig/Result_{MY_CONST_NAME}_{metric}.png"
    os.makedirs("fig", exist_ok=True)
    plt.savefig(save_filename, dpi=300)
    print(f"已儲存圖表：{save_filename}")
    
    # 顯示圖表
    plt.show()