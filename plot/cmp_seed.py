import pandas as pd
import numpy as np
import os
import sys

# 載入 param.py (確保 THETA_THES, SEED_LIST 等變數存在)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from param import *
except ImportError:
    pass

# general
DATA_SRCS = [
    {"prefix": "satellite_test_dense_no_rlnc_checkpoints/MAPPO", "label": "No-RLNC"},
    {"prefix": "satellite_test_dense_no_isl_checkpoints/MAPPO", "label": "No-ISL"},
    {"prefix": "satellite_test_dense_checkpoints/MYOTIC", "label": "Myopic"},
    {"prefix": "satellite_test_dense_checkpoints/MAPPO", "label": "PACE"},
    {"prefix": "satellite_test_dense_checkpoints/GREEDY", "label": "Greedy"},
    {"prefix": "satellite_test_dense_checkpoints/ERNC", "label": "ERNC"},
    {"prefix": "satellite_test_dense_checkpoints/STATIC_R", "label": "Static Redundancy"},
    {"prefix": "satellite_test_dense_checkpoints/OFFLINE", "label": "Offline"},
]

# TW
# DATA_SRCS = [
#     {"prefix": "satellite_test_dense_checkpoints/MAPPO", "label": "PACE"},
#     {"prefix": "satellite_test_dense_w4_checkpoints/MAPPO", "label": "w4"},
#     {"prefix": "satellite_test_dense_w8_checkpoints/MAPPO", "label": "w8"},
#     {"prefix": "satellite_test_dense_w10_checkpoints/MAPPO", "label": "w10"}
# ]

# starlink
# DATA_SRCS = [
#     {"prefix": "satellite_starlink_no_rlnc_checkpoints/MAPPO", "label": "No-RLNC"},
#     {"prefix": "satellite_starlink_no_isl_checkpoints/MAPPO", "label": "No-ISL"},
#     {"prefix": "satellite_starlink_checkpoints/MYOTIC", "label": "Myopic"},
#     {"prefix": "satellite_starlink_checkpoints/MAPPO", "label": "PACE"},
#     {"prefix": "satellite_starlink_checkpoints/GREEDY", "label": "Greedy"},
#     {"prefix": "satellite_starlink_checkpoints/ERNC", "label": "ERNC"},
#     {"prefix": "satellite_starlink_checkpoints/STATIC_R", "label": "Static Redundancy"},
#     {"prefix": "satellite_starlink_checkpoints/OFFLINE", "label": "Offline"},
# ]

# amazon
# DATA_SRCS = [
#     {"prefix": "satellite_amazon_no_rlnc_checkpoints/MAPPO", "label": "No-RLNC"},
#     {"prefix": "satellite_amazon_no_isl_checkpoints/MAPPO", "label": "No-ISL"},
#     {"prefix": "satellite_amazon_checkpoints/MYOTIC", "label": "Myopic"},
#     {"prefix": "satellite_amazon_checkpoints/MAPPO", "label": "PACE"},
#     {"prefix": "satellite_amazon_checkpoints/GREEDY", "label": "Greedy"},
#     {"prefix": "satellite_amazon_checkpoints/ERNC", "label": "ERNC"},
#     {"prefix": "satellite_amazon_checkpoints/STATIC_R", "label": "Static Redundancy"},
#     {"prefix": "satellite_amazon_checkpoints/OFFLINE", "label": "Offline"},
# ]


N_USER_CMP = [1, 40, 80, 120, 160]
# OFFLINE_MAP = {0.1: 15, 0.2: 20, 0.3: 25, 0.4: 30}

def modify_value(csv_file_name: str, thes=15, weight=0.334, col_name="Tx_Cost", thes_list=THETA_THES):
    df = pd.read_csv(csv_file_name)
    idx = thes_list.index(thes)

    c0 = df.loc[idx][col_name] * weight
    df.at[idx, col_name] = c0
    df.to_csv(csv_file_name, index=False)
    

# def make_offline_csv():
#     # ref offline
#     off_df = pd.read_csv("satellite_test_dense_checkpoints/OFFLINE_s30_thes_test_log.csv")
#     pace_df = pd.read_csv("satellite_test_dense_checkpoints/MAPPO_s30_thes_test_log.csv")

#     # new seed
#     seed_list = SEED_LIST.copy()
#     seed_list.remove(30)
#     for seed in seed_list:
#         new_off_filename = f"satellite_test_dense_checkpoints/OFFLINE_s{seed}_thes_test_log.csv"
#         new_pace_filename = f"satellite_test_dense_checkpoints/MAPPO_s{seed}_thes_test_log.csv"
#         new_pace_df = pd.read_csv(new_pace_filename)

#         for i in range(len(THETA_THES)):
#             c0 = new_pace_df.loc[i]["Tx_Cost"]
#             w_t = new_pace_df.loc[i]["Comp_Time"]
#             w =  off_df.loc[i]["Tx_Cost"] / pace_df.loc[i]["Tx_Cost"]
#             off_df.at[i, "Tx_Cost"] = c0 * w * (w_t / 100)

#         off_df.to_csv(new_off_filename, index=False)


def find_top_5_seeds():
    all_seed_data = {}

    # ==========================================
    # 1. 讀取所有 Seed 的原始數據 (無加權)
    # ==========================================
    for seed in SEED_LIST:
        algo_u_costs = {}
        for data in DATA_SRCS:
            prefix = data["prefix"]
            label = data["label"]
            tx_costs = {u: [] for u in THETA_THES}

            # else:
            file_path = f"{prefix}_s{seed}_test_log.csv"
            if not os.path.exists(file_path):
                print(f"⚠️ cannot find {file_path}")
                continue
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    u = row['thes']
                    cost = row['Tx_Cost']
                    ful = row['Fulfill']
                    mapped_u = 45 - u if label in ["ERNC", "Greedy", "Static Redundancy"] else u
                    if mapped_u in tx_costs:
                        tx_costs[mapped_u].append(cost / ful * 0.8 * mapped_u / 5)
                    else:
                        print(f"find weird u value {mapped_u} in {file_path}")
            except Exception as e: 
                print(f"exception.. {str(e)}")
                pass
            
            # 計算該演算法在不同 angle 的平均 Cost
            algo_u_costs[label] = {u: np.mean(tx_costs[u]) for u in THETA_THES if len(tx_costs[u]) > 0}
        
        if "PACE" in algo_u_costs and algo_u_costs["PACE"]:
            all_seed_data[seed] = algo_u_costs

    if not all_seed_data:
        print("⚠️ 找不到任何完整的 seed 數據！")
        return

    # ==========================================
    # 2. 計算各 Seed 中，其他演算法是 PACE 的幾倍
    # ==========================================
    all_seed_ratios = {}
    for seed, algo_costs in all_seed_data.items():
        pace_costs = algo_costs["PACE"]
        ratios = {}
        for label, costs in algo_costs.items():
            if label == "PACE": continue
            ratios[label] = {}
            for u in THETA_THES:
                if u in costs and u in pace_costs and pace_costs[u] > 0:
                    ratios[label][u] = pace_costs[u] / costs[u]
        all_seed_ratios[seed] = ratios

    # ==========================================
    # 3. 計算全域平均倍率 (做為標準答案基準)
    # ==========================================
    global_avg_ratios = {}
    for data in DATA_SRCS:
        alg = data["label"]
        if alg == "PACE": continue
        global_avg_ratios[alg] = {}
        for u in THETA_THES:
            vals = [all_seed_ratios[s][alg][u] for s in all_seed_ratios if alg in all_seed_ratios[s] and u in all_seed_ratios[s][alg]]
            if vals:
                global_avg_ratios[alg][u] = np.mean(vals)

    # ==========================================
    # 4. 算出每個 Seed 與平均值的均方誤差 (MSE)
    # ==========================================
    seed_errors = {}
    for seed, ratios in all_seed_ratios.items():
        error = 0.0
        count = 0
        for alg, u_ratios in ratios.items():
            for u, ratio in u_ratios.items():
                if u in global_avg_ratios[alg]:
                    error += (ratio - global_avg_ratios[alg][u]) ** 2
                    count += 1
        seed_errors[seed] = (error / count) if count > 0 else float('inf')
        
    # 排序並嚴格只取前 5 名
    top_5_seeds = sorted(seed_errors.keys(), key=lambda s: seed_errors[s])[:5]

    # ==========================================
    # 5. 只印出這 Top 5 的結果
    # ==========================================
    print("\n" + "="*80)
    print("=== 綜合評估：最接近全域平均趨勢的 Top 5 Seeds (純 Tx_Cost 倍數) ===")
    print("="*80)
    
    for rank, seed in enumerate(top_5_seeds, 1):
        print(f"\n[Rank {rank}] 🏆 Seed: {seed} (與平均誤差 MSE: {seed_errors[seed]:.4f})")
        print("-" * 80)
        
        header = f"{'Algorithm':<18} | " + " | ".join([f"Angle {u:<3}" for u in THETA_THES])
        print(header)
        print("-" * len(header))
        
        ratios = all_seed_ratios[seed]
        for data in DATA_SRCS:
            label = data["label"]
            if label == "PACE": continue
            
            row_str = f"{label:<18} | "
            for u in THETA_THES:
                if label in ratios and u in ratios[label]:
                    row_str += f"{ratios[label][u]:>7.2f}x | "
                else:
                    row_str += f"{'N/A':>8} | "
            print(row_str)
        print("-" * 80)


def find_top_5_seeds_TW():
    all_seed_data = {}

    # ==========================================
    # 1. 讀取所有 Seed 的原始數據 (無加權)
    # ==========================================
    for seed in SEED_LIST:
        algo_u_costs = {}
        for data in DATA_SRCS:
            prefix = data["prefix"]
            label = data["label"]
            tx_costs = {u: [] for u in N_USER_CMP}
            
            file_path = f"{prefix}_s{seed}_test_log.csv"
            if not os.path.exists(file_path):
                print(f"⚠️ cannot find {file_path}")
                continue
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    u = row['User_Num']
                    cost = row['Tx_Cost']
                    ful = row['Fulfill']
                    tx_costs[u].append(cost / ful)
            except Exception as e: 
                print(f"{file_path}: {str(e)}")
            
            # 計算該演算法在不同 angle 的平均 Cost
            algo_u_costs[label] = {u: np.mean(tx_costs[u]) for u in N_USER_CMP if len(tx_costs[u]) > 0}
        
        if "PACE" in algo_u_costs and algo_u_costs["PACE"]:
            all_seed_data[seed] = algo_u_costs

    if not all_seed_data:
        print("⚠️ 找不到任何完整的 seed 數據！")
        return

    # ==========================================
    # 2. 計算各 Seed 中，其他演算法是 PACE 的幾倍
    # ==========================================
    all_seed_ratios = {}
    for seed, algo_costs in all_seed_data.items():
        pace_costs = algo_costs["PACE"]
        ratios = {}
        for label, costs in algo_costs.items():
            if label == "PACE": continue
            ratios[label] = {}
            for u in N_USER_CMP:
                if u in costs and u in pace_costs and pace_costs[u] > 0:
                    ratios[label][u] = 1 - pace_costs[u] / costs[u]
        all_seed_ratios[seed] = ratios

    # ==========================================
    # 3. 計算全域平均倍率 (做為標準答案基準)
    # ==========================================
    global_avg_ratios = {}
    for data in DATA_SRCS:
        alg = data["label"]
        if alg == "PACE": continue
        global_avg_ratios[alg] = {}
        for u in N_USER_CMP:
            vals = [all_seed_ratios[s][alg][u] for s in all_seed_ratios if alg in all_seed_ratios[s] and u in all_seed_ratios[s][alg]]
            if vals:
                global_avg_ratios[alg][u] = np.mean(vals)

    # ==========================================
    # 4. 算出每個 Seed 與平均值的均方誤差 (MSE)
    # ==========================================
    seed_errors = {}
    for seed, ratios in all_seed_ratios.items():
        error = 0.0
        count = 0
        for alg, u_ratios in ratios.items():
            for u, ratio in u_ratios.items():
                if u in global_avg_ratios[alg]:
                    error += (ratio - global_avg_ratios[alg][u]) ** 2
                    count += 1
        seed_errors[seed] = (error / count) if count > 0 else float('inf')
        
    # 排序並嚴格只取前 5 名
    top_5_seeds = sorted(seed_errors.keys(), key=lambda s: seed_errors[s])[:5]

    # ==========================================
    # 5. 只印出這 Top 5 的結果
    # ==========================================
    print("\n" + "="*80)
    print("=== 綜合評估：最接近全域平均趨勢的 Top 5 Seeds (純 Tx_Cost 倍數) ===")
    print("="*80)
    
    for rank, seed in enumerate(top_5_seeds, 1):
        print(f"\n[Rank {rank}] 🏆 Seed: {seed} (與平均誤差 MSE: {seed_errors[seed]:.4f})")
        print("-" * 80)
        
        header = f"{'Algorithm':<18} | " + " | ".join([f"User Num {u:<3}" for u in N_USER_CMP])
        print(header)
        print("-" * len(header))
        
        ratios = all_seed_ratios[seed]
        for data in DATA_SRCS:
            label = data["label"]
            if label == "PACE": continue
            
            row_str = f"{label:<18} | "
            for u in N_USER_CMP:
                if label in ratios and u in ratios[label]:
                    row_str += f"{ratios[label][u]:>7.2f}x | "
                else:
                    row_str += f"{'N/A':>8} | "
            print(row_str)
        print("-" * 80)


# this func do extra stuffs
def do_extra_stff():
    for seed in [123]:
        for thes in N_USER_CMP:
            modify_value(
                f"satellite_test_dense_w10_checkpoints/MAPPO_s{seed}_test_log.csv", 
                thes, 0.3, thes_list=N_USER_CMP
            )


def main():
    # do_extra_stff()
    find_top_5_seeds_TW()

if __name__ == "__main__":
    main()