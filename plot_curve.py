import pandas as pd
import matplotlib.pyplot as plt
import os
from param import *

# ==========================================
# 1. File Paths Configuration
# ==========================================
# 請將這裡的檔名替換成你實際對應兩個演算法的 CSV 檔名
file_mine = f'satellite_{MY_CONST_NAME}_checkpoints/training_log_{MY_CONST_NAME}.csv'             # 對應 'mine' 演算法的檔案
file_myotic = f'satellite_{MY_CONST_NAME}_myotic_checkpoints/training_log_{MY_CONST_NAME}.csv'    # 對應 'myotic' 演算法的檔案
file_ERNC = f'satellite_{MY_CONST_NAME}_checkpoints/ERNC_test_log.csv'
file_GREEDY = f'satellite_{MY_CONST_NAME}_checkpoints/GREEDY_test_log.csv'
file_STATIC = f'satellite_{MY_CONST_NAME}_checkpoints/STATIC_R_test_log.csv'

# ==========================================
# 2. Plotting Setup
# ==========================================
METRICS = ["Tx_Cost", "Comp_Time"] # Iteration,Reward,Cost_Rate,Lambda,Tx_Cost,Comp_Time

for m in METRICS:
    ylab = ""
    if m == "Reward": ylab = "Reward"
    elif m == "Tx_Cost" : ylab = "Transmission Cost"
    elif m == "Comp_Time" : ylab = "Completion Time (Step)"
    else : ylab = "?" 

    plt.figure(figsize=(10, 6))

    # Read and plot 'Mine' algorithm
    if os.path.exists(file_mine):
        df_mine = pd.read_csv(file_mine)
        plt.plot(df_mine['Iteration'], df_mine[m], 
                label='Mine', color='blue', linewidth=2, alpha=0.8)
    else:
        print(f"Warning: Cannot find {file_mine}")

    # Read and plot 'Myotic' algorithm
    if os.path.exists(file_myotic):
        df_myotic = pd.read_csv(file_myotic)
        plt.plot(df_myotic['Iteration'], df_myotic[m], 
                label='Myotic', color='red', linewidth=2, alpha=0.8)
    else:
        print(f"Warning: Cannot find {file_myotic}")

    # --- Plot 'Baseline' (Horizontal Line) ---

    BASELINE_FILES = [file_ERNC, file_GREEDY, file_STATIC]
    COLORS = ["orange", "gray", "green"]
    BL_NAMES = ['Opportunistic ERNC', 'Greedy Baseline', 'Static ERNC']
    # comp_time is problematic...
    TMP_TIME = [11.5, 12.0, 12.2]

    for i, bl in enumerate(BASELINE_FILES):
        if (m == "Reward"):
            print("Reward is chosen. skipping baselines")
            break

        if os.path.exists(bl):
            df_baseline = pd.read_csv(bl)
            
            # 萃取你要的單一數值
            # 假設你的 CSV 裡面那個欄位叫做 'Reward'，並且你取第一筆資料：
            # (如果是別的欄位名稱，請把 'Reward' 換掉)
            baseline_value = df_baseline[m].iloc[0] 

            # if m == "Comp_Time": baseline_value = TMP_TIME[i]
            
            # 如果裡面有多行數據，你也可以選擇取平均值：
            # baseline_value = df_baseline['Reward'].mean()

            # 使用 axhline 畫出一條貫穿整張圖的水平直線
            plt.axhline(y=baseline_value, 
                        color=COLORS[i],          # 設定顏色
                        linestyle='--',         # 設定為虛線 (dashed line)
                        linewidth=2,            # 線條粗細
                        label=BL_NAMES[i])       # 圖例名稱
        else:
            print(f"Warning: Cannot find {bl}")

    # ==========================================
    # 3. Chart Formatting (English Only, Default Font)
    # ==========================================
    plt.xlabel('Iteration')
    plt.ylabel(ylab)

    # Add legend to distinguish algorithms
    plt.legend(title='Algorithm', loc='best')

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # ==========================================
    # 4. Save and Show
    # ==========================================
    os.makedirs("fig", exist_ok=True)
    save_filename = f'fig/{MY_CONST_NAME}_{m}_trend_comparison.png'
    plt.savefig(save_filename, dpi=300)
    print(f"Chart successfully saved as '{save_filename}'")

    plt.show()