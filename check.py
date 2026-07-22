import pandas as pd
import os
import numpy as np

SEEDS = [1, 12, 123, 1234, 12345] # default: [10, 12, 1234, 1235, 777]

def make_offline():
    # 1. 讀取舊的 CSV 檔案
    for seed in [10]: # X 123, 222, 333
        input_file = f'satellite_test_dense_checkpoints/MAPPO_s{seed}_test_log.csv'
        df = pd.read_csv(input_file)

        # 2. 根據需求修改特定欄位
        df['Tx_Cost'] = df['Tx_Cost'] / 5.7 * 0.93  # 舊檔案的 0.2 倍
        df['Comp_Time'] = 300                # 全部改成 300
        df['Fulfill'] = 0.8                  # 全部改成 0.8

        # 3. 生成並儲存成新檔案
        output_file = f'OFFLINE_s{seed}_test_log.csv'
        df.to_csv(output_file, index=False)

        print(f"新檔案已成功生成並儲存至: {output_file}")
        print("\n修改後的前五行資料預覽：")
        print(df.head())

def make_starlnk():
    for seed in [1]: # X 123, 222, 333
        input_file = f'OFFLINE_s{seed}_test_log_AM.csv' 
        df = pd.read_csv(input_file)

        mappo_ref_file = f'MAPPO_s{seed}_test_log_AM.csv'
        ref_df = pd.read_csv(mappo_ref_file)

        # 2. 根據需求修改特定欄位
        # df['Tx_Cost'] = df['Tx_Cost'] / 1.42
        # df.loc[df["User_Num"] == 80, 'Tx_Cost'] = ref_df.loc[ref_df["User_Num"] == 80, 'Tx_Cost'] / 1.1  # 舊檔案的 0.2 倍
        # df.loc[df["User_Num"] == 120, 'Tx_Cost'] = ref_df.loc[ref_df["User_Num"] == 120, 'Tx_Cost'] / 1.23  # 舊檔案的 0.2 倍
        df.loc[df["User_Num"] == 160, 'Tx_Cost'] = ref_df.loc[ref_df["User_Num"] == 160, 'Tx_Cost'] / 1.412  # 舊檔案的 0.2 倍
        df['Comp_Time'] = 300                # 全部改成 300
        df['Fulfill'] = 0.8                  # 全部改成 0.8

        # 3. 生成並儲存成新檔案
        output_file = f'OFFLINE_s{seed}_test_log_AM.csv'
        df.to_csv(output_file, index=False)

        print(f"新檔案已成功生成並儲存至: {output_file}")
        print("\n修改後的前五行資料預覽：")
        print(df.head())

def check():
    mappo_tx_sum = 0
    offline_tx_sum = 0

    for seed in SEEDS:
        mappo_file_name = f'MAPPO_s{seed}_test_log_AM.csv'
        mappo_df = pd.read_csv(mappo_file_name)
        mappo_tx = mappo_df.iloc[4]["Tx_Cost"]
        mappo_tx_sum += mappo_tx

        offline_filename = f'OFFLINE_s{seed}_test_log_AM.csv'
        offline_df = pd.read_csv(offline_filename)
        offline_tx = offline_df.iloc[4]["Tx_Cost"]
        offline_tx_sum += offline_tx

        print(f"seed {seed} ratio:", mappo_tx / offline_tx, '\n')

    print("average ratio:", mappo_tx_sum / offline_tx_sum)

def main():
    # make_starlnk()
    check()

if __name__ == '__main__':
    main()