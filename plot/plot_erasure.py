import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 載入上一層目錄的 param.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from param import MY_CONST_NAME
except ImportError:
    MY_CONST_NAME = "test"

# ==========================================
# 1. 統一的基礎設定 (Configuration)
# ==========================================
DIR_NAME = f"satellite_{MY_CONST_NAME}_checkpoints/"

# 請確認這裡的 USER_NUMBERS 跟你跑 test.py 時的設定一致
ERASURE_RATES = [0.1, 0.2, 0.3, 0.4] 

X_COLUMN = 'erasure'

# 【統一管理區】在這裡設定演算法名稱、檔案前綴與繪圖樣式
# 之後所有的圖都會統一使用這裡的設定，保證風格完全一致
ALGO_CONFIG = {
    'Proposed MARL (Ours)': {
        'prefix': 'MAPPO', 'color': 'blue', 'marker': 'o', 'linestyle': '-'
    },
    'Myopic MARL': {
        'prefix': 'MYOTIC', 'color': 'red', 'marker': 'o', 'linestyle': '-'
    },
    'ERNC': {
        'prefix': 'ERNC', 'color': 'orange', 'marker': '^', 'linestyle': ':'
    },
    'Greedy': {
        'prefix': 'GREEDY', 'color': 'gray', 'marker': 'x', 'linestyle': '--'
    },
    'Static': {
        'prefix': 'STATIC_R', 'color': 'green', 'marker': 's', 'linestyle': '-.'
    }
}

# ==========================================
# 2. 繪製來自 test_log.csv 的指標 (Tx Cost, Comp Time)
# ==========================================
def plot_test_log_metrics():
    # 定義要從 test_log.csv 裡面抓哪些欄位出來畫圖
    METRICS_TO_PLOT = {
        'Tx_Cost': {'ylabel': 'Transmission Cost'},
    }

    for metric, labels in METRICS_TO_PLOT.items():
        plt.figure(figsize=(8, 6))
        
        for algo_label, config in ALGO_CONFIG.items():
            # 讀取例如 MAPPO_test_log.csv
            file_path = os.path.join(DIR_NAME, f"{config['prefix']}_test_log.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if metric in df.columns and X_COLUMN in df.columns:
                    plt.plot(
                        df[X_COLUMN], df[metric] / df["Fulfill"] * 0.8, 
                        label=algo_label, 
                        color=config['color'], marker=config['marker'], 
                        linestyle=config['linestyle'], linewidth=2.5, markersize=8
                    )
            else:
                pass # 可以選擇印出警告: print(f"找不到 {file_path}")

        # 設定圖表細節
        plt.xlabel('Average Erasure Probability', fontsize=12)
        plt.ylabel(labels['ylabel'], fontsize=12)
        
        # 強制 X 軸對齊我們設定的 USER_NUMBERS
        plt.xticks(ERASURE_RATES)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11, loc='best')
        plt.tight_layout()
        
        # 存檔
        os.makedirs("fig", exist_ok=True)
        save_filename = f"fig/Result_{MY_CONST_NAME}_{metric}_vs_Fulfill.png"
        plt.savefig(save_filename, dpi=300)
        print(f"已儲存圖表：{save_filename}")
        plt.show()


# ==========================================
# 4. 主程式執行區
# ==========================================
if __name__ == "__main__":
    print("=== 開始繪製測試結果指標 (erasure) ===")
    plot_test_log_metrics()