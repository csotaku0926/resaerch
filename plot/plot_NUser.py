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

# 【修改 1】：建立映射對照表
# 將 CSV 裡寫錯的數字，對應到畫圖時該顯示的真實數字
CSV_WRONG_NUMBERS = [1, 100, 200, 300, 400] 
TRUE_USER_NUMBERS = [1, 40, 80, 120, 160]

# 若是在測試密集模式下，維持正確的數值
if MY_CONST_NAME == "test_dense":
    CSV_WRONG_NUMBERS = [1, 40, 80, 120, 160]
    TRUE_USER_NUMBERS = [1, 40, 80, 120, 160]

# 建立字典：{1: 1, 100: 40, 200: 80, ...}
NUM_MAPPING = dict(zip(CSV_WRONG_NUMBERS, TRUE_USER_NUMBERS))

X_COLUMN = 'User_Num'

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
    METRICS_TO_PLOT = {
        'Tx_Cost': {'ylabel': 'Total Energy Consumption'},
        'Comp_Time': {'ylabel': 'Completion Time (Step)'}
    }

    for metric, labels in METRICS_TO_PLOT.items():
        plt.figure(figsize=(8, 6))
        
        for algo_label, config in ALGO_CONFIG.items():
            file_path = os.path.join(DIR_NAME, f"{config['prefix']}_test_log.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if metric in df.columns and X_COLUMN in df.columns:
                    # 【修改 2】：利用 .map() 將 DataFrame 讀出來的錯誤數值轉換成正確的
                    correct_x_values = df[X_COLUMN].map(NUM_MAPPING)
                    
                    plt.plot(
                        correct_x_values, df[metric], 
                        label=algo_label, 
                        color=config['color'], marker=config['marker'], 
                        linestyle=config['linestyle'], linewidth=2.5, markersize=8
                    )
            else:
                pass 

        plt.xlabel('Number of Users per Grid', fontsize=12)
        plt.ylabel(labels['ylabel'], fontsize=12)
        
        # 【修改 3】：強制 X 軸的刻度顯示我們設定的真實數字
        plt.xticks(TRUE_USER_NUMBERS)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11, loc='best')
        plt.tight_layout()
        
        os.makedirs("fig", exist_ok=True)
        save_filename = f"fig/Result_{MY_CONST_NAME}_{metric}_vs_Users.png"
        plt.savefig(save_filename, dpi=300)
        print(f"已儲存圖表：{save_filename}")
        plt.close() # 加上 close 避免多圖重疊

# ==========================================
# 3. 繪製來自 curve.csv 的成本效率 (Cost Efficiency)
# ==========================================
def plot_efficiency_vs_users():
    plt.figure(figsize=(8, 6))

    for algo_label, config in ALGO_CONFIG.items():
        x_users = []
        y_efficiency = []

        # 【修改 4】：用寫錯的數字 (CSV_WRONG_NUMBERS) 去找檔名
        for wrong_n in CSV_WRONG_NUMBERS:
            file_path = os.path.join(DIR_NAME, f"{config['prefix']}_{wrong_n}_curve.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if not df.empty:
                    final_fulfill = df["fulfill"].iloc[-1]
                    final_tx = df["tx_cost"].iloc[-1]
                    
                    efficiency = (final_fulfill * 100) / final_tx if final_tx > 0 else 0
                    
                    # 畫圖的 X 軸要存入映射後的真實數值
                    x_users.append(NUM_MAPPING[wrong_n])
                    y_efficiency.append(efficiency)

        if x_users:
            plt.plot(
                x_users, y_efficiency, 
                label=algo_label, 
                color=config['color'], marker=config['marker'], 
                linestyle=config['linestyle'], linewidth=2.5, markersize=8
            )

    plt.xlabel('Number of Users per Grid', fontsize=12)
    plt.ylabel('Cost Efficiency', fontsize=12)
    
    # 【修改 5】：強制 X 軸的刻度顯示我們設定的真實數字
    plt.xticks(TRUE_USER_NUMBERS)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    
    os.makedirs('fig', exist_ok=True)
    save_path = f'fig/Result_{MY_CONST_NAME}_CostEfficiency_vs_Users.png'
    plt.savefig(save_path, dpi=300)
    print(f"已儲存圖表：{save_path}")
    plt.close()

# ==========================================
# 4. 主程式執行區
# ==========================================
if __name__ == "__main__":
    print("=== 開始繪製測試結果指標 (Tx Cost, Comp Time) ===")
    plot_test_log_metrics()
    
    print("\n=== 開始繪製成本效率 (Cost Efficiency) ===")
    plot_efficiency_vs_users()