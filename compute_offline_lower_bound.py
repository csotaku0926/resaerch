import pandas as pd
import multiprocessing
import os
import gc
import numpy as np
import cvxpy as cp
from datetime import timedelta
from skyfield.api import load
from param import *
from SatelliteDataDisseminationEnv import SatelliteDataDisseminationEnv

def compute_offline_lower_bound(env:SatelliteDataDisseminationEnv):
    # ==================================================
    # 1. 實例化你的環境 (取得 Non-causal 宇宙)
    # ==================================================
    print("🌍 正在初始化衛星環境...")

    env.reset()
    
    C = env.constellation
    T = env.T_max
    N = env.N
    TARGET_K = env.target_k
    MAX_BUF = C.max_buf
    ISL_COST_FACTOR = env.ISL_cost_factor # 0.9
    
    # 攤平所有的 Users，方便建立 LP 矩陣
    all_users = []
    for grid in C.user_grids:
        for u in grid.users:
            all_users.append(u)
    U = len(all_users)
    print(f"📡 共有 {N} 顆 LEO 衛星, {U} 個地面目標用戶, 總步數 {T}")

    # ==================================================
    # 2. 榨取 Non-Causal Knowledge (未來接觸與掉包統計)
    # ==================================================
    print("⏳ 正在提取 Time-Expanded Graph 參數 (Capacities & Erasures)...")
    
    # 加上 dtype=np.float32，記憶體消耗直接減半
    isl_cap = np.zeros((T, N, N), dtype=np.float32)       
    dl_cap = np.zeros((T, N), dtype=np.float32)           
    success_rate = np.zeros((T, N, U), dtype=np.float32)  
    meo_inflow = np.zeros((T, N), dtype=np.float32)

    ts = load.timescale()
    
    for t in range(T):
        current_dt = env.start_dt + timedelta(seconds=t * env.step_seconds)
        current_time = ts.utc(current_dt.year, current_dt.month, current_dt.day,
                              current_dt.hour, current_dt.minute, current_dt.second)
        
        # MEO 注入量 (根據你 meo_broadcast_to_leos 寫死的邏輯)
        meo_total_packets = 0.2 * env.step_seconds 
        
        for i in range(N):
            meo_inflow[t, i] = meo_total_packets
            
            # (A) 提取 ISL 容量限制
            # 只允許在你有定義的 get_neighbors 之間傳輸
            for j_idx in C.get_neighbors(i)[:env.M]:
                cap = C.get_ISL_capacity(i, j_idx, current_time)
                isl_cap[t, i, j_idx] = cap
                
            # (B) 提取 Downlink 廣播容量與 Erasure
            visible_grids = C.get_visible_grids(i, current_time)
            if len(visible_grids) > 0:
                dl_cap[t, i] = C.get_downlink_capacity()
                # 掃描所有 user 計算異質成功率
                for u_idx, user in enumerate(all_users):
                    # 你寫的 calculate_erasure_rate 已經包含了仰角保護與物理運算
                    erasure = C.calculate_erasure_rate(i, user, current_time)
                    success_rate[t, i, u_idx] = 1.0 - erasure

    # ==================================================
    # 3. 建立 Relaxed LP 模型 (Max-Fulfill 盡力而為模式)
    # ==================================================
    print("🧮 正在建構 CVXPY 線性規劃模型 (Max-Fulfill 模式)...")

    f_isl = cp.Variable((T, N, N), nonneg=True)
    f_dl = cp.Variable((T, N), nonneg=True)
    s = cp.Variable((T + 1, N), nonneg=True)
    
    # 新增變數：記錄每個 User 實際「有效接收」的封包數量
    fulfilled = cp.Variable(U, nonneg=True)

    # [物理限制與流量守恆]
    constraints = [
        s[0, :] == 0,
        s <= MAX_BUF,
        f_isl <= isl_cap,
        f_dl <= dl_cap
    ]
    
    flow_in = cp.sum(f_isl, axis=1) + meo_inflow
    flow_out = cp.sum(f_isl, axis=2) + f_dl
    
    constraints.append(flow_out <= s[:-1, :])
    constraints.append(s[1:, :] <= s[:-1, :] + flow_in - flow_out)

    # [配額結算]
    f_dl_flat = cp.reshape(f_dl, T * N)
    
    for u_idx in range(U):
        succ_flat = success_rate[:, :, u_idx].flatten()
        user_recv = succ_flat @ f_dl_flat
        
        # 條件1：承認的送達量，不能超過他實際收到的量
        constraints.append(fulfilled[u_idx] <= user_recv)
        # 條件2：承認的送達量，最多只算到 TARGET_K (多給的當作沒看到，不給獎勵)
        constraints.append(fulfilled[u_idx] <= TARGET_K)

    # [目標函數] 
    # 乾淨的真實花費
    real_tx_cost = ISL_COST_FACTOR * cp.sum(f_isl) + 1.0 * cp.sum(f_dl)
    
    # 目標：每成功送達 1 個封包，給予極大獎勵 (10000)，確保求解器「寧可花錢也要先滿足封包」
    objective = cp.Maximize((100.0 * cp.sum(fulfilled)) - real_tx_cost)
    prob = cp.Problem(objective, constraints)

    # ==================================================
    # 4. 求解與提取真實數據
    # ==================================================
    print(f"🚀 開始求解 (User 數量: {U})...")
    # prob.solve(solver=cp.OSQP, max_iter=10000)
    prob.solve(
        solver=cp.OSQP, 
        max_iter=50000, 
        eps_abs=1e-4,   # 放寬絕對誤差容忍度
        eps_rel=1e-4,   # 放寬相對誤差容忍度
        verbose=False
    )

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        # 1. 提取真實 Tx_Cost 跟 Fulfill 比例
        final_tx_cost = real_tx_cost.value
        total_delivered = np.sum(fulfilled.value)
        final_fulfill_rate = total_delivered / (U * TARGET_K)
        
        print(f"✅ 求解成功！(物理極限探測完畢)")
        print(f"🎯 真實最低 Tx_Cost : {final_tx_cost:.4f}")
        print(f"📦 物理極限 Fulfill : {final_fulfill_rate:.4f} (總送達 {total_delivered:.1f} / {U * TARGET_K} 封包)")
        
        # 極限模式下，完工時間通常是跑到最後一步
        final_comp_time = T * env.step_seconds
        era = env.erasure
        final_erasure = era # 請依照你測試的掉包率修改
        
        # 寫入 Log 檔案
        dir_name = f"satellite_{MY_CONST_NAME}_checkpoints"
        os.makedirs(dir_name, exist_ok=True)

        seed = env.seed
        n_user = env.num_users
        log_file = f"{dir_name}/OFFLINE_s{seed}_test_log.csv"
        file_exists = os.path.isfile(log_file)
        
        df = pd.DataFrame([{
            "User_Num": n_user,
            "Tx_Cost": round(final_tx_cost, 4),
            "Fulfill": round(final_fulfill_rate, 4),
            "Comp_Time": final_comp_time,
            "erasure": final_erasure
        }])
        
        df.to_csv(log_file, mode='a', header=not file_exists, index=False)
        print(f"💾 已將 Baseline 結果寫入 {log_file}")
        
        # ==================================================
        # 🌟 新增：提取並記錄每個 time step 的完成率與成本曲線
        # ==================================================
        optimal_f_dl = f_dl.value
        optimal_f_isl = f_isl.value
        user_accumulated = np.zeros(U)
        cumulative_cost = 0.0
        
        # 檔名格式對齊 test.py 的 curve
        curve_log_file = f"{dir_name}/OFFLINE_{final_erasure}_{n_user}_s{seed}_curve.csv"
        curve_data = []
        
        for t in range(T):
            # 1. 累加這一秒發生的 Tx Cost
            step_cost = ISL_COST_FACTOR * np.sum(optimal_f_isl[t]) + 1.0 * np.sum(optimal_f_dl[t])
            cumulative_cost += step_cost
            
            # 2. 結算每個 User 這一秒收到的有效封包
            for u_idx in range(U):
                if user_accumulated[u_idx] < TARGET_K:
                    # 這一秒所有衛星對這個 User 的有效傳輸量 (發射量 * 成功率)
                    step_recv = np.sum(optimal_f_dl[t, :] * success_rate[t, :, u_idx])
                    # 累加，但不超過 TARGET_K上限
                    user_accumulated[u_idx] = min(TARGET_K, user_accumulated[u_idx] + step_recv)
            
            # 3. 計算當下的 Fulfill Rate
            current_fulfill = np.sum(user_accumulated) / (U * TARGET_K)
            
            # 4. 儲存該步狀態
            curve_data.append([t, round(cumulative_cost, 4), round(current_fulfill, 4)])
            
        # 寫入 CSV 檔案
        df_curve = pd.DataFrame(curve_data, columns=["step", "tx_cost", "fulfill"])
        df_curve.to_csv(curve_log_file, index=False)
        print(f"📈 已將隨時間變化的曲線數據寫入 {curve_log_file}")

        # --- 釋放記憶體區塊 ---
        del prob, f_isl, f_dl, s, fulfilled, objective, constraints
        gc.collect()
        # ---------------------
        
        return final_tx_cost, final_comp_time
    else:
        print(f"❌ 求解失敗，狀態: {prob.status}")
        return None, None
    
def run_single_task(seed, n_user, era):
    # 這個函數只負責跑單一任務，跑完記憶體就會被 OS 強制回收
    env = SatelliteDataDisseminationEnv(
        const_param=CONST_PARAM, 
        num_users=n_user, 
        target_k=CONST_PARAM.target_k,
        test_mode=IS_TEST_MODE,
        erasure=era,
        seed=seed 
    )
    compute_offline_lower_bound(env)

if __name__ == "__main__":
    ERASURES = [0.1]
    USER_NUMBERS = [1]

    # 建立任務清單
    # tasks = []
    # for seed in SEED_LIST:
    #     for n_user in USER_NUMBERS:
    #         for era in ERASURES:
    #             tasks.append((seed, n_user, era))

    # # 限制「同時執行」的最大進程數，避免瞬間把 RAM 抽乾
    # # 如果你的電腦有 32GB RAM，建議設 3 或 4；如果只有 16GB，設 2
    # MAX_CONCURRENT_WORKERS = 3 
    
    # print(f"🚀 開始平行執行 {len(tasks)} 個任務，最大併發數: {MAX_CONCURRENT_WORKERS}")
    
    # with multiprocessing.Pool(processes=MAX_CONCURRENT_WORKERS) as pool:
    #     pool.starmap(run_single_task, tasks)
    
    # print("✅ 所有排程計算完畢！")

    for seed in SEED_LIST:
        for n_user in USER_NUMBERS:
            for era in ERASURES:
                env = SatelliteDataDisseminationEnv(
                    const_param=CONST_PARAM, 
                    num_users=n_user, 
                    target_k=CONST_PARAM.target_k,
                    test_mode=IS_TEST_MODE,
                    erasure=era,
                    seed=seed 
                )
                compute_offline_lower_bound(env)