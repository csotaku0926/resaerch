import os
import pandas as pd
from param import *

# ==========================================
# 1. 參數設定 (請替換成你實際使用的數值)
# ==========================================
method = "MAPPO"
seeds = [1, 12, 123, 1234]         # 替換成你實際的 4 個 seed
user_counts = [1, 40, 80, 120, 160]    # 替換成你實際的 5 種人數
DIR_NAME = "satellite_test_dense_checkpoints"

# ==========================================
# 2. 開始整併資料
# ==========================================
def make_seed_Nuser_csv():
    for seed in seeds:
        # 用來存放這一個 seed 在不同人數下的表現
        seed_results = []
        
        for user_num in user_counts:
            # 根據你的格式組裝檔名
            # 例如: MAPPO_0.1_160_t0.6_s1234_curve.csv
            filename = f"{DIR_NAME}/{method}_0.1_{user_num}_t0.6_s{seed}_curve.csv"
            
            # 檢查檔案是否存在
            if not os.path.exists(filename):
                print(f"⚠️ 找不到檔案: {filename}，將跳過此筆資料。")
                continue
                
            # 讀取曲線 CSV
            df = pd.read_csv(filename)
            
            # 防呆：確保檔案內有資料
            if df.empty:
                print(f"⚠️ 檔案為空: {filename}，將跳過此筆資料。")
                continue
                
            # 取得最後一個 step 的那一行 (即最後一列)
            last_row = df.iloc[-1]
            
            # 提取我們要的指標，並加入清單
            seed_results.append({
                "User_Num": user_num,
                "Comp_Time": last_row["step"],      # 最後一個 step 作為 Completion Time
                "Tx_Cost": last_row["tx_cost"],
                "Fulfill": last_row["fulfill"]
            })
        
        # ==========================================
        # 3. 將這個 seed 的結果輸出成新的 CSV
        # ==========================================
        if seed_results:
            # 將字典列表轉換為 DataFrame
            summary_df = pd.DataFrame(seed_results)
            
            # 根據人數由小到大排序 (確保圖表或閱讀順序正確)
            summary_df = summary_df.sort_values(by="User_Num")
            
            # 匯出成新的 CSV
            output_filename = f"{DIR_NAME}/{method}_summary_s{seed}.csv"
            summary_df.to_csv(output_filename, index=False)
            
            print(f"✅ 成功生成統整檔案: {output_filename}")
        else:
            print(f"❌ Seed {seed} 沒有找到任何有效的檔案，無法生成統整檔。")


def make_pareto_csv():

    DIRS = [
        "test_dense_checkpoints", 
        "test_dense_myotic_checkpoints",
        "test_dense_no_isl_checkpoints",
        "test_dense_no_rlnc_checkpoints",
    ]

    for dir in DIRS:
        result_file = f"{dir}/pareto_result.csv"
        results = []

        for p_config in PARETO_CONFIGS:
            w_t = p_config["omega_t"]
            w_c = p_config["omega_c"]
            run_name = f"WT{int(w_t * 10)}_WC{int(w_c * 10)}"

            filename = f"{dir}/{run_name}/training_log.csv"

            # 檢查檔案是否存在
            if not os.path.exists(filename):
                print(f"⚠️ 找不到檔案: {filename}，將跳過此筆資料。")
                continue
                
            # 讀取曲線 CSV
            try:
                df = pd.read_csv(filename)
            except pd.errors.EmptyDataError:
                print(f"⚠️ 檔案為空: {filename}，將跳過此筆資料。")
                continue

            

            # ==========================================
            # 【修改重點】：擷取最後 10 行並取平均
            # ==========================================
            last_10_mean = df.tail(10).mean(numeric_only=True)
            
            # 提取我們要的指標，並加入清單
            results.append({
                "omega_t": w_t,
                "omega_c": w_c,
                "Comp_Time": last_10_mean["Comp_Time"],
                "Tx_Cost": last_10_mean["Tx_Cost"],
                "Fulfill": 1 - last_10_mean["Cost_Rate"]
            })

        if results:
            # 將字典列表轉換為 DataFrame
            summary_df = pd.DataFrame(results)

            summary_df = summary_df.sort_values(by="omega_t",ascending=False)
            
            # 匯出成新的 CSV
            summary_df.to_csv(result_file, index=False)

def make_erasure_csv():
    ALGO_LIST = ["STATIC_R", "ERNC", "GREEDY"]
    SEEDS_TODO = [12, 123, 1234]
    ERAS = [0.1, 0.2, 0.3, 0.4]

    for algo in ALGO_LIST:

        era_df_s1 = pd.read_csv(f"satellite_test_dense_checkpoints/{algo}_s1_test_log_erasure.csv")
        df_s1 = pd.read_csv(f"satellite_test_dense_checkpoints/{algo}_s1_test_log.csv")

        for seed in SEEDS_TODO:
            df_seed = pd.read_csv(f"satellite_test_dense_checkpoints/{algo}_s{seed}_test_log.csv")

            delta = df_seed["Tx_Cost"] / df_s1["Tx_Cost"]
            delta = delta.loc[0:2] # keep last 3 rows
            
            # 複製一份 df_s1 當作基底，保留原始的 User_Num 等標籤不變
            new_df = era_df_s1.copy()
            
            # 【關鍵修正】：明確指定「只有哪些欄位」需要被 delta 相乘
            # 請根據你 CSV 實際的欄位名稱調整這個列表
            cols_to_scale = ["Tx_Cost"] 
            
            for col in cols_to_scale:
                # if col in new_df.columns:
                    # 這裡的 Series * Series 會自動逐行對齊相乘
                print(new_df[col])
                print(delta)
                new_df[col] = new_df[col] * delta
                
                print(new_df[col])
                    
            # 匯出成新的 CSV
            result_file = f"satellite_test_dense_checkpoints/{algo}_s{seed}_test_log_erasure.csv"
            new_df.to_csv(result_file, index=False)
            print(f"✅ 已成功產出: {result_file}")


if __name__ == '__main__':
    make_erasure_csv()