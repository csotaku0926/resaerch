import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

# 載入上一層目錄的 param.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from param import MY_CONST_NAME
except ImportError:
    MY_CONST_NAME = "test"

# 1. 定義檔案路徑與你想在圖例(Legend)上顯示的名稱
# (你可以把 label 改成你在論文裡命名的演算法名稱，例如 "Proposed CMARL (w=3)")
data_sources = [
    {"path": "test_dense_no_rlnc_checkpoints/pareto_result.csv", "label": "No-RLNC"},
    {"path": "test_dense_no_isl_checkpoints/pareto_result.csv", "label": "No-ISL"},
    {"path": "test_dense_myotic_checkpoints/pareto_result.csv", "label": "Myopic"},
    {"path": "test_dense_checkpoints/pareto_result.csv", "label": "Proposed"},
    # {"path": "test_w3_checkpoints/pareto_result.csv", "label": "Proposed (Tw=3)"},
    # {"path": "test_w4_checkpoints/pareto_result.csv", "label": "Proposed (Tw=4)"}
]

# 設定圖片大小
plt.figure(figsize=(9, 6))

# 定義每條線的點樣式(Marker)與顏色
markers = ['x', 'x', 'x', 'o', 's', '^']
colors = ["#7a7a7a", "#b81dff", "#b41f21", '#1f77b4', '#ff7f0e', '#2ca02c'] # 經典的藍、橘、綠
linestyles = ['dotted', 'dotted', 'dashed', 'solid', 'dashdot', 'dashed']

# 2. 依序讀取檔案並繪圖
for i, data in enumerate(data_sources):
    path = data["path"]
    label = data["label"]
    
    if os.path.exists(path):
        # 讀取 CSV
        df = pd.read_csv(path)

        if "Fulfill" not in df.columns:
            print(f"⚠️ Fulfill col not found in {path}. Skipping")
        else:
            # 【重要】根據 X 軸 (Tx_Cost) 進行排序
            # 這是為了確保折線是「從左畫到右」，避免點的順序錯亂導致線條來回折返
            df_sorted = df.sort_values(by="Tx_Cost")

            
            # 繪製曲線
            plt.plot(
                df_sorted["Tx_Cost"] / df_sorted["Fulfill"] * 0.8, 
                df_sorted["Comp_Time"] / df_sorted["Fulfill"] * 0.8, 
                # marker=markers[i], 
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=2.5, 
                markersize=8, 
                label=label
        )
    else:
        print(f"⚠️ 警告: 找不到檔案 {path}，將跳過繪製此線。")

# ==========================================
# 2. 加入 Greedy Baseline 的單一座標點 (點)
# ==========================================
# greedy_path = f"satellite_test_checkpoints/GREEDY_test_log.csv"
# if os.path.exists(greedy_path):
#     greedy_df = pd.read_csv(greedy_path)
    
    # 提取 Greedy 測試結果的單一數值
    # 使用 .iloc[0] 確保即使檔案有多行也只取第一筆 (或者你也可以用 .mean() 取平均)
    # greedy_tx_cost = greedy_df['Tx_Cost'].iloc[0] 
    # greedy_comp_time = greedy_df['Comp_Time'].iloc[0]
    # ful_factor = 0.8 / greedy_df['Fulfill'].iloc[0]
    
    # # 使用 scatter 畫出一個醒目的單點
    # plt.scatter(
    #     greedy_tx_cost * ful_factor, 
    #     greedy_comp_time * ful_factor, 
    #     color='red',           # 紅色
    #     marker='*',            # 星形
    #     s=300,                 # s 控制點的大小，300 很大很醒目
    #     edgecolors='black',    # 加個黑邊框更有質感
    #     label='Greedy Baseline', 
    #     zorder=5               # zorder=5 確保這個點絕對壓在所有線條的最上層
    # )
    # print(f"✅ 成功讀取 Greedy 資料點: (Cost: {greedy_tx_cost:.1f}, Time: {greedy_comp_time:.1f})")
# else:
#     print(f"⚠️ 警告: 找不到檔案 {greedy_path}，將跳過繪製 Greedy 點。")

# 3. 美化圖表外觀
# plt.title('Energy-Latency Pareto Frontier', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Transmission Cost', fontsize=14)
plt.ylabel('Completion Time', fontsize=14)

# 加上網格線，讓讀者更容易對齊數值
plt.grid(True, linestyle='--', alpha=0.6)

# 顯示圖例 (Legend)
plt.legend(fontsize=12, loc='best')

# 自動調整邊界
plt.tight_layout()

# 4. 儲存圖片並顯示
output_filename = f'fig/pareto_frontier.png'
plt.savefig(output_filename, dpi=300)
print(f"✅ 繪圖完成！圖片已儲存為: {output_filename}")

plt.show()