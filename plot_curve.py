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
METRICS = "Tx_Cost" # Iteration,Reward,Cost_Rate,Lambda,Tx_Cost,Comp_Time

YLAB = ""
if METRICS == "Reward": YLAB = "Reward"
elif METRICS == "Tx_Cost" : YLAB = "Transmission Cost"
elif METRICS == "Comp_Time" : YLAB = "Completion Time (Step)"
else : YLAB = "?" 

plt.figure(figsize=(10, 6))

# Read and plot 'Mine' algorithm
if os.path.exists(file_mine):
    df_mine = pd.read_csv(file_mine)
    plt.plot(df_mine['Iteration'], df_mine[METRICS], 
             label='Mine', color='blue', linewidth=2, alpha=0.8)
else:
    print(f"Warning: Cannot find {file_mine}")

# Read and plot 'Myotic' algorithm
if os.path.exists(file_myotic):
    df_myotic = pd.read_csv(file_myotic)
    plt.plot(df_myotic['Iteration'], df_myotic[METRICS], 
             label='Myotic', color='red', linewidth=2, alpha=0.8)
else:
    print(f"Warning: Cannot find {file_myotic}")

# ==========================================
# 3. Chart Formatting (English Only, Default Font)
# ==========================================
plt.xlabel('Iteration')
plt.ylabel(YLAB)

# Add legend to distinguish algorithms
plt.legend(title='Algorithm', loc='lower right')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# ==========================================
# 4. Save and Show
# ==========================================
os.makedirs("fig", exist_ok=True)
save_filename = f'fig/{MY_CONST_NAME}_{METRICS}_trend_comparison.png'
plt.savefig(save_filename, dpi=300)
print(f"Chart successfully saved as '{save_filename}'")

plt.show()