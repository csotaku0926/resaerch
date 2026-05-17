import os
import pandas as pd

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