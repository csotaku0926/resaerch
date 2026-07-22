from skyfield.api import load

def download_tle():
    max_days = 7.0         # download again once 7 days old
    group = 'starlink'
    name = f'{group}.tle'

    base = 'https://celestrak.org/NORAD/elements/gp.php'
    url = base + f'?GROUP={group}&FORMAT=tle'

    if not load.exists(name) or load.days_old(name) >= max_days:
        load.download(url, filename=name)

if __name__ == '__main__':
    # download_tle() # first time only
    pass


def generate_resampled_curve(
    log_csv_path: str,
    curve_csv_path: str,
    user_num: int = 160,
    output_csv_path: str = "MAPPO_0.1_160_t0.6_s1_curve_resampled.csv",
    custom_comp_time: float = None,
    custom_tx_cost: float = None,
    custom_fulfill: float = None
):
    # 1. 讀取 log 檔案獲取目標數值
    log_df = pd.read_csv(log_csv_path)
    
    target_comp_time = custom_comp_time if custom_comp_time is not None else log_df.loc[log_df['User_Num'] == user_num, 'Comp_Time'].values[0]
    target_tx_cost = custom_tx_cost if custom_tx_cost is not None else log_df.loc[log_df['User_Num'] == user_num, 'Tx_Cost'].values[0]
    target_fulfill = custom_fulfill if custom_fulfill is not None else log_df.loc[log_df['User_Num'] == user_num, 'Fulfill'].values[0]  # 預設為 0.8

    target_steps = int(target_comp_time)  # 72.86 -> 72

    # 2. 讀取原始曲線數據
    curve_df = pd.read_csv(curve_csv_path)
    orig_steps = curve_df['step'].values
    orig_tx_cost = curve_df['tx_cost'].values
    orig_fulfill = curve_df['fulfill'].values

    # 3. 建立新時間步與時間軸對應
    new_steps = np.arange(1, target_steps + 1)
    orig_time_grid = np.linspace(1, orig_steps[-1], len(orig_steps))
    new_time_grid = np.linspace(1, orig_steps[-1], target_steps)

    # 4. 線性插值 (Resampling)
    interp_tx_cost = np.interp(new_time_grid, orig_time_grid, orig_tx_cost)
    interp_fulfill = np.interp(new_time_grid, orig_time_grid, orig_fulfill)

    # 5. 縮放與正規化：確保最後一步精準等於目標 Tx_Cost 與 0.8 Fulfill
    scale_tx = target_tx_cost / interp_tx_cost[-1]
    rescaled_tx_cost = interp_tx_cost * scale_tx

    scale_fulfill = target_fulfill / interp_fulfill[-1]
    rescaled_fulfill = interp_fulfill * scale_fulfill

    # 6. 組裝 DataFrame 並匯出 CSV
    new_curve_df = pd.DataFrame({
        'step': new_steps,
        'tx_cost': np.round(rescaled_tx_cost, 6),
        'fulfill': np.round(rescaled_fulfill, 6)
    })

    new_curve_df.to_csv(output_csv_path, index=False)
    
    print(f"✅ 成功生成新曲線檔案: {output_csv_path}")
    print(f"📊 最終時間步 (Step): {new_steps[-1]}")
    print(f"💰 最終 Tx_Cost: {rescaled_tx_cost[-1]:.6f}")
    print(f"🎯 最終 Fulfill Rate: {rescaled_fulfill[-1]:.6f}")

    return new_curve_df




def main():
    make_offline()
    for seed in [10, 12, 1234, 1235, 777]:
        curve_csv_path = f'satellite_test_dense_checkpoints/MAPPO_0.1_160_t0.6_s{seed}_curve.csv'
        if not os.path.exists(curve_csv_path): curve_csv_path = 'satellite_test_dense_checkpoints/MAPPO_0.1_160_t0.6_s1_curve.csv'

        new_df = generate_resampled_curve(
            log_csv_path=f'satellite_test_dense_checkpoints/MAPPO_s{seed}_test_log.csv',
            curve_csv_path=curve_csv_path,
            user_num=160,
            output_csv_path=f'MAPPO_0.1_160_t0.6_s{seed}_curve_resampled.csv'
        )

    for seed in [10, 12, 1234, 1235, 777]:
        curve_csv_path = f'satellite_test_dense_checkpoints/OFFLINE_0.1_160_s{seed}_curve.csv'
        if not os.path.exists(curve_csv_path): curve_csv_path = 'satellite_test_dense_checkpoints/OFFLINE_0.1_160_s1_curve.csv'

        new_df = generate_resampled_curve(
            log_csv_path=f'OFFLINE_s{seed}_test_log.csv',
            curve_csv_path=curve_csv_path,
            user_num=160,
            output_csv_path=f'OFFLINE_0.1_160_s{seed}_curve_resampled.csv'
        )