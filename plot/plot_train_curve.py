import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from param import *

# ==========================================
# 1. File Paths Configuration
# ==========================================
# 請將這裡的檔名替換成你實際對應兩個演算法的 CSV 檔名
file_mine = f'satellite_{MY_CONST_NAME}_checkpoints/training_log_{MY_CONST_NAME}.csv'             
file_myotic = f'satellite_{MY_CONST_NAME}_myotic_checkpoints/training_log_{MY_CONST_NAME}.csv'    
file_ERNC = f'satellite_{MY_CONST_NAME}_checkpoints/ERNC_test_log.csv'
file_GREEDY = f'satellite_{MY_CONST_NAME}_checkpoints/GREEDY_test_log.csv'
file_STATIC = f'satellite_{MY_CONST_NAME}_checkpoints/STATIC_R_test_log.csv'

# ==========================================
# 1.5 【核心修改】：預先讀取所有資料，找出「最短的」終點並裁切
# ==========================================
df_dict = {}
max_iters = []

# 讀取 Mine
if os.path.exists(file_mine):
    df_mine = pd.read_csv(file_mine)
    df_dict['mine'] = df_mine
    max_iters.append(df_mine['Iteration'].max())
else:
    print(f"Warning: Cannot find {file_mine}")

# 讀取 Myotic
if os.path.exists(file_myotic):
    df_myotic = pd.read_csv(file_myotic)
    df_dict['myotic'] = df_myotic
    max_iters.append(df_myotic['Iteration'].max())
else:
    print(f"Warning: Cannot find {file_myotic}")

# 自動找出大家共同的「最短極限」
if max_iters:
    global_min_max_iter = min(max_iters)
    print(f"自動偵測到最短的訓練終點為 Iteration: {global_min_max_iter}，將以此對齊所有曲線。")
else:
    global_min_max_iter = 0

# 將所有的 DataFrame 裁切，強迫它們都只保留到 global_min_max_iter
for key in df_dict:
    df_dict[key] = df_dict[key][df_dict[key]['Iteration'] <= global_min_max_iter]

# ==========================================
# 2. Plotting Setup
# ==========================================
METRICS = ["Tx_Cost", "Comp_Time"] 

for m in METRICS:
    ylab = ""
    if m == "Reward": ylab = "Reward"
    elif m == "Tx_Cost" : ylab = "Transmission Cost"
    elif m == "Comp_Time" : ylab = "Completion Time (Step)"
    else : ylab = "?" 

    plt.figure(figsize=(10, 6))

    # 畫出裁切後的 'Mine'
    if 'mine' in df_dict:
        df = df_dict['mine']
        plt.plot(df['Iteration'], df[m], 
                label='Proposed MARL', color='blue', linewidth=2, alpha=0.8)

    # 畫出裁切後的 'Myotic'
    if 'myotic' in df_dict:
        df = df_dict['myotic']
        plt.plot(df['Iteration'], df[m], 
                label='Myopic MARL', color='red', linewidth=2, alpha=0.8)

    # --- Plot 'Baseline' (Horizontal Line) ---
    # (原本你註解掉畫 Baseline 虛線的程式碼我省略了，若有需要可以直接貼回這個位置)

    # ==========================================
    # 3. Chart Formatting (English Only, Default Font)
    # ==========================================
    plt.xlabel('Iteration')
    plt.ylabel(ylab)

    # Add legend to distinguish algorithms
    plt.legend(title='Algorithm', loc='best')

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 【畫面美化】：強制鎖定 X 軸的右側邊界，讓線條完美頂到右邊，完全不會留白
    if global_min_max_iter > 0:
        plt.xlim(0, global_min_max_iter)

    plt.tight_layout()

    # ==========================================
    # 4. Save and Show
    # ==========================================
    os.makedirs("fig", exist_ok=True)
    save_filename = f'fig/{MY_CONST_NAME}_{m}_trend_comparison.png'
    plt.savefig(save_filename, dpi=300)
    print(f"Chart successfully saved as '{save_filename}'")

    plt.show()