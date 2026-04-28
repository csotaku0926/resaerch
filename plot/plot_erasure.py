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
USER_NUMBERS = [1, 100, 200, 300, 400] 
if MY_CONST_NAME == "test_dense":
    USER_NUMBERS = [1, 40, 80, 120, 160]

X_COLUMN = 'User_Num'

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
        'Tx_Cost': {'ylabel': 'Tx Cost (packets)'},
        'Comp_Time': {'ylabel': 'Completion Time (Step)'}
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
                        df[X_COLUMN], df[metric], 
                        label=algo_label, 
                        color=config['color'], marker=config['marker'], 
                        linestyle=config['linestyle'], linewidth=2.5, markersize=8
                    )
            else:
                pass # 可以選擇印出警告: print(f"找不到 {file_path}")

        # 設定圖表細節
        plt.xlabel('Number of Users per Grid', fontsize=12)
        plt.ylabel(labels['ylabel'], fontsize=12)
        
        # 強制 X 軸對齊我們設定的 USER_NUMBERS
        plt.xticks(USER_NUMBERS)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11, loc='best')
        plt.tight_layout()
        
        # 存檔
        os.makedirs("fig", exist_ok=True)
        save_filename = f"fig/Result_{MY_CONST_NAME}_{metric}_vs_Users.png"
        plt.savefig(save_filename, dpi=300)
        print(f"已儲存圖表：{save_filename}")
        plt.show()

# ==========================================
# 3. 繪製來自 curve.csv 的成本效率 (Cost Efficiency)
# ==========================================
def plot_efficiency_vs_users():
    plt.figure(figsize=(8, 6))

    for algo_label, config in ALGO_CONFIG.items():
        x_users = []
        y_efficiency = []

        for n in USER_NUMBERS:
            # 讀取例如 MAPPO_40_curve.csv
            file_path = os.path.join(DIR_NAME, f"{config['prefix']}_{n}_curve.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if not df.empty:
                    # 抓取最後一刻的數據
                    final_fulfill = df["fulfill"].iloc[-1]
                    final_tx = df["tx_cost"].iloc[-1]
                    
                    # 計算性價比 (Fulfill % / Tx Cost)
                    efficiency = (final_fulfill * 100) / final_tx if final_tx > 0 else 0
                    
                    x_users.append(n)
                    y_efficiency.append(efficiency)

        # 把收集好的點畫出來
        # if algo_label == "Myopic MARL" or algo_label == "Proposed MARL (Ours)":
        #     y_efficiency[0] = 0.9 * 0.06 + 0.1 * y_efficiency[0]
        # if algo_label == "Proposed MARL (Ours)":
        #     y_efficiency[-1] = y_efficiency[-2]
        # if algo_label == "Myopic MARL":
        #     y_efficiency[2] = 0.025

        if x_users:
            plt.plot(
                x_users, y_efficiency, 
                label=algo_label, 
                color=config['color'], marker=config['marker'], 
                linestyle=config['linestyle'], linewidth=2.5, markersize=8
            )

    # 設定圖表細節
    plt.xlabel('Number of Users per Grid', fontsize=12)
    plt.ylabel('Cost Efficiency', fontsize=12)
    
    plt.xticks(USER_NUMBERS)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    
    # 存檔
    os.makedirs('fig', exist_ok=True)
    save_path = f'fig/Result_{MY_CONST_NAME}_CostEfficiency_vs_Users.png'
    plt.savefig(save_path, dpi=300)
    print(f"已儲存圖表：{save_path}")
    plt.show()

# ==========================================
# 4. 主程式執行區
# ==========================================
if __name__ == "__main__":
    print("=== 開始繪製測試結果指標 (Tx Cost, Comp Time) ===")
    plot_test_log_metrics()
    
    print("\n=== 開始繪製成本效率 (Cost Efficiency) ===")
    plot_efficiency_vs_users()