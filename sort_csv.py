import pandas as pd
import os
from param import SEED_LIST

def sort_csv_with_pandas(filename:str):

    # 檢查檔案是否存在
    if not os.path.exists(filename):
        print(f"錯誤：找不到檔案 '{filename}'")
        return

    try:
        # 讀取 CSV 檔案
        df = pd.read_csv(filename)

        # 檢查表格中是否包含 'max_buf' 欄位
        if 'max_buf' not in df.columns:
            print("錯誤：CSV 檔案中找不到 'max_buf' 欄位。")
            return

        # 依照 'max_buf' 欄位由小到大排序 (若要由大到小，請加上 ascending=False)
        df_sorted = df.sort_values(by='max_buf', ascending=True)

        # 覆寫原本的檔案 (index=False 代表不寫入最左側的索引值)
        df_sorted.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"成功！檔案 '{filename}' 已根據 'max_buf' 排序並覆寫完成。")

    except Exception as e:
        print(f"處理檔案時發生錯誤: {e}")


def main():
    for seed in SEED_LIST:
        filename = f"satellite_test_dense_checkpoints/MYOTIC_s{seed}_buf_test_log.csv"
        sort_csv_with_pandas(filename)

if __name__ == "__main__":
    main()