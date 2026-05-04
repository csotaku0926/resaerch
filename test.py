"""
三種 Baseline 的正確實作
========================

B1: Greedy-RLNC  —— 按 TEG contact volume 比例分配（有 ISL + downlink）
B2: ER-NC        —— 有 downlink contact 就全打，不用 ISL，不做 planning
B3: Static-R     —— 離線算 N*，每次固定送 N* 封包，不用 ISL，不做即時決策
"""

import numpy as np
import ray
import matplotlib.pyplot as plt
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from datetime import timedelta
import csv
import os
import sys

from SatelliteDataDisseminationEnv import SatelliteDataDisseminationEnv
from train_lstm import *
from param import *

# ── 執行設定 ──────────────────────────────────────
# USER_NUMBERS = [1, 40, 80, 120, 160] # [1, 40, 80, 120, 160]
# ERASURES = [0.2]
USER_NUMBERS = [1, 40, 80, 120, 160]
ERASURES = [0.1] #[0.1, 0.2, 0.3, 0.4]
NUM_EPISODES = 3
T_MAX = 90 #CONST_PARAM.t_max
print(f"[參數確認]")
print(f"- 衛星 const: {MY_CONST_NAME}")
print(f"- 最大步數 (T_max): {T_MAX}")
print(f"- target K: {TARGET_K}")
print(f"test mode list: {TEST_MODES}")
print("-" * 30)
# ─────────────────────────────────────────────────


def current_skyfield_time(actual_env):
    dt = actual_env.start_dt + timedelta(
        seconds=actual_env.current_step * actual_env.step_seconds)
    return actual_env.ts.utc(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second)

# ERNC should swap with Greedy..
# ╔══════════════════════════════════════════════════════╗
# ║  B1: Greedy-RLNC                                     ║
# ║                                                      ║
# ║  邏輯：                                              ║
# ║    對每條 link（ISL ×M + downlink ×1），              ║
# ║    用 TEG contact volume 當作「連結品質」指標，       ║
# ║    按比例分配流量：action[j] ∝ TEG_j                 ║
# ║                                                      ║
# ║  ─ 有 ISL（傳給鄰居讓他們之後再送地面）              ║
# ║  ─ 有 downlink（直接送地面）                         ║
# ║  ─ 不看 deficit，純粹依連結品質貪婪分配              ║
# ╚══════════════════════════════════════════════════════╝
def action_greedy_rlnc(real_id, actual_env, current_time):
    M             = actual_env.M
    Tw            = 1
    constellation = actual_env.constellation
    action        = np.zeros(M + 1, dtype=np.float32)

    buf = constellation.get_leo_buffer(real_id)
    if buf <= 0:
        return action

    # ── 收集每條 link 的 TEG contact volume ──────────────
    # TEG 是未來 Tw 步的容量向量，加總得到總 contact volume
    cvs = np.zeros(M + 1, dtype=np.float32)

    # ISL links：用鄰居的 downlink TEG 代表「透過這條 ISL 最終能到地面的量」
    for idx, neighbor_id in enumerate(constellation.get_neighbors(real_id)[:M]):
        if constellation.get_ISL_capacity(real_id, neighbor_id, current_time) > 0:
            teg = constellation.get_teg_downlink_volume(neighbor_id, Tw, current_time)
            cvs[idx] = float(np.sum(teg))

    # 自己的 downlink TEG
    if len(constellation.get_visible_grids(real_id, current_time)) > 0:
        teg_self = constellation.get_teg_downlink_volume(real_id, Tw, current_time)
        cvs[M] = float(np.sum(teg_self))

    total_cv = float(np.sum(cvs))
    if total_cv <= 0:
        return action

    # ── 按 TEG 比例分配：比例和為 1 → 用掉整個 buffer ──
    action = (cvs / total_cv).astype(np.float32)
    return action


# ╔══════════════════════════════════════════════════════╗
# ║  B2: ER-NC (Erasure-Recovery NC)                     ║
# ║                                                      ║
# ║  邏輯：                                              ║
# ║    有 downlink contact → 整個 buffer 全打出去        ║
# ║    沒有 downlink contact → 什麼都不傳                ║
# ║    完全不用 ISL，不做任何 planning 或 buffer 管理    ║
# ╚══════════════════════════════════════════════════════╝
def action_ernc(real_id, actual_env, current_time):
    M             = actual_env.M
    constellation = actual_env.constellation
    action        = np.zeros(M + 1, dtype=np.float32)

    # 有可見 grid → 全力打 downlink（比例 1.0 = 整個 buffer）
    # 環境 step() 內部會自動把實際流量 cap 在 downlink capacity
    if len(constellation.get_visible_grids(real_id, current_time)) > 0:
        action[M] = 1.0

    # ISL 全部為 0（不用 ISL）
    return action


# ╔══════════════════════════════════════════════════════════════════╗
# ║  B3: Static Redundancy (Contact-Plan Based Fixed Forwarding)     ║
# ║                                                                  ║
# ║  概念來源：                                                      ║
# ║    Fraire et al., "Design Challenges in Contact Plans for        ║
# ║    Disruption-Tolerant Satellite Networks",                      ║
# ║    IEEE Communications Magazine, 2015                            ║
# ║                                                                  ║
# ║  與 ER-NC 的關鍵差異：                                          ║
# ║    ER-NC  → 完全不用 ISL，每顆衛星各自打地面                    ║
# ║    Static-R → 使用 ISL，但 ISL 比例由離線接觸統計決定，         ║
# ║               執行時固定不變（不看即時狀態）                    ║
# ║                                                                  ║
# ║  邏輯：                                                          ║
# ║    【離線】掃描整個 episode 的接觸視窗，                         ║
# ║           統計每顆衛星平均的 downlink / ISL 接觸時間比例，      ║
# ║           算出固定分配向量 static_plan[sat_id] = [r0..rM]       ║
# ║    【執行】直接套用固定比例：action = static_plan[sat_id]        ║
# ║           不管當前 buffer 多少、deficit 多少、channel 狀態如何  ║
# ╚══════════════════════════════════════════════════════════════════╝
def compute_static_plan(actual_env):
    """
    離線階段：統計整個 episode 中每顆衛星的平均接觸容量，
    算出固定的流量分配比例 static_plan。

    static_plan[sat_id] = np.array([r_isl0, r_isl1, ..., r_isl_{M-1}, r_downlink])
    所有比例加總 <= 1.0，代表「把 buffer 按此比例分配」。

    文獻精神：離線的 contact plan 決定轉發策略，
    等同於 CGR 的最簡化版本（有 ISL relay，但無即時 volume tracking）。
    """
    constellation = actual_env.constellation
    M             = actual_env.M
    n_sats        = len(constellation.agents)
    ts            = actual_env.ts
    start_dt      = actual_env.start_dt
    step_sec      = actual_env.step_seconds

    # 累積每顆衛星在各 link 上的「有效接觸步數」
    # contact_counts[sat_id][j] = 這條 link 在整個 episode 有接觸的步數
    contact_counts = np.zeros((n_sats, M + 1), dtype=np.float32)

    sample_steps = np.linspace(0, actual_env.T_max - 1, 20, dtype=int)

    for s in sample_steps:
        dt = start_dt + timedelta(seconds=int(s) * step_sec)
        t  = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        for sat_id in range(n_sats):
            # ISL：鄰居在這個時刻是否有 capacity
            for idx, neighbor_id in enumerate(constellation.get_neighbors(sat_id)[:M]):
                cap = constellation.get_ISL_capacity(sat_id, neighbor_id, t)
                if cap > 0:
                    contact_counts[sat_id][idx] += cap   # 用 capacity 加權，而非只計次數

            # Downlink：這個時刻是否有可見 grid
            if len(constellation.get_visible_grids(sat_id, t)) > 0:
                cap_dl = constellation.get_downlink_capacity()
                contact_counts[sat_id][M] += cap_dl

    # 把累積接觸量轉成比例：按接觸量等比例分配，和為 1
    static_plan = np.zeros((n_sats, M + 1), dtype=np.float32)
    for sat_id in range(n_sats):
        total = np.sum(contact_counts[sat_id])
        if total > 0:
            static_plan[sat_id] = contact_counts[sat_id] / total

    print(f"  [Static-R] 離線計畫計算完成，範例 sat_0: {np.round(static_plan[0], 3)}")
    return static_plan


def action_static_r(real_id, actual_env, current_time, static_plan):
    """
    執行階段：直接套用離線計算好的固定比例。
    不看即時狀態，不做任何調整。
    """
    buf = actual_env.constellation.get_leo_buffer(real_id)
    if buf <= 0:
        return np.zeros(actual_env.M + 1, dtype=np.float32)

    # 直接回傳固定比例，環境 step() 會把它乘以 buf 當作 desired_flow
    return static_plan[real_id].copy()


# ╔══════════════════════════════════════════════════════╗
# ║  測試主迴圈                                           ║
# ╚══════════════════════════════════════════════════════╝
def run_mode(mode, user_numbers, num_episodes, algo=None, write_log=True, write_curve=True):
    avg_tx_costs      = []
    avg_fulfill_rates = []
    avg_comp_times    = []

    checkpoint_dir = f"./satellite_{MY_CONST_NAME}_checkpoints"
    if write_log:
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_file_path = os.path.join(checkpoint_dir, f"{mode}_test_log.csv")
        csv_file = open(log_file_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["User_Num", "Tx_Cost", "Fulfill", "Comp_Time", "erasure"])


    for era in ERASURES:

        for n_users in user_numbers:
            print(f"\n[{mode}] ══ erasure={era} ══ n_users={n_users} ══")

            if write_curve:
                curve_file_path = os.path.join(checkpoint_dir, f"{mode}_{era}_{n_users}_curve.csv")
                curve_csv_file = open(curve_file_path, "w", newline="")
                curve_csv_writer = csv.writer(curve_csv_file)
                curve_csv_writer.writerow(["step", "tx_cost", "fulfill"])

            raw_env = SatelliteDataDisseminationEnv(
                const_param=CONST_PARAM, T_max=T_MAX, num_users=n_users, is_myotic=(mode == "MYOTIC"), test_mode=IS_TEST_MODE,
                erasure=era,
                is_unicast=(not (mode == "MAPPO" or mode == "MYOTIC"))
            )
            env = ParallelPettingZooEnv(raw_env)

            # B3：離線計算固定分配計畫，整個 n_users 設定共用一個值
            static_plan = None
            if mode == "STATIC_R":
                # env.reset()
                actual_env = env.par_env if hasattr(env, "par_env") else env.unwrapped
                print("  計算 Static Plan（離線步驟）...")
                static_plan = compute_static_plan(actual_env)

            tx_costs      = []
            comp_times    = []
            fulfill_rates = []
            final_curve = []

            for ep in range(num_episodes):
                obs, _ = env.reset()
                actual_env = env.par_env if hasattr(env, "par_env") else env.unwrapped
                done          = False
                final_tx_cost = 0.0
                current_ep_curve = []

                # print([u.pos for u in actual_env.constellation.user_grids[0].users])

                while not done:
                    current_time = current_skyfield_time(actual_env)
                    actions = {}

                    for agent_id, agent_obs in obs.items():
                        real_id = actual_env.constellation.get_id_by_name(agent_id)

                        if mode == "MAPPO" or mode == "MYOTIC":
                            actions[agent_id] = algo.compute_single_action(
                                observation=agent_obs,
                                policy_id="shared_policy",
                                explore=False)

                            # print(actions[agent_id])

                        elif mode == "GREEDY":
                            actions[agent_id] = action_greedy_rlnc(
                                real_id, actual_env, current_time)
                            # print(actions[agent_id])

                        elif mode == "ERNC":
                            actions[agent_id] = action_ernc(
                                real_id, actual_env, current_time)

                        elif mode == "STATIC_R":
                            actions[agent_id] = action_static_r(
                                real_id, actual_env, current_time, static_plan)

                    obs, _, terminations, truncations, infos = env.step(actions)

                    # 【新增 3】：記錄當下 Step 的完賽率
                    step_val = actual_env.current_step
                    current_fulfill = actual_env.constellation.get_user_fulfill_percent()
                    current_tx_cost = actual_env.episode_tx_cost
                    current_ep_curve.append((step_val, current_tx_cost, current_fulfill))

                    if infos:
                        first = list(infos.keys())[0]
                        final_tx_cost = infos[first].get("tx_cost", 0.0)
                        final_comp_time = infos[first].get("time", 0.0)

                    done = (terminations.get("__all__", False) or
                            truncations.get("__all__", False))
                    
                # 【新增 4】：如果這是最困難的一局 (例如 400 user)，就把曲線存起來
                if write_curve and (len(final_curve) == 0 or (
                    ((mode == "MAPPO") or (mode == "MYOTIC")) and len(current_ep_curve) < len(final_curve)
                ) or (
                    not ((mode == "MAPPO") or (mode == "MYOTIC")) and len(current_ep_curve) > len(final_curve)
                )): 
                    final_curve = current_ep_curve

                fulfill = actual_env.constellation.get_user_fulfill_percent()
                tx_costs.append(final_tx_cost)
                comp_times.append(final_comp_time)
                fulfill_rates.append(fulfill)
                print(f"  ep {ep+1:02d}: tx={final_tx_cost:.1f}, time={final_comp_time}"
                    f" fulfill={fulfill*100:.1f}%")

            avg_tx  = float(np.mean(tx_costs))
            avg_ful = float(np.mean(fulfill_rates))
            avg_time = float(np.mean(comp_times))
            avg_tx_costs.append(avg_tx)
            avg_fulfill_rates.append(avg_ful)
            avg_comp_times.append(avg_time)
            print(f"  → avg tx_cost={avg_tx:.2f}, fulfill={avg_ful*100:.1f}%")
            if write_log:
                csv_writer.writerow([n_users, avg_tx, avg_ful, avg_time, era])
                csv_file.flush() # 強制寫入硬碟，這樣就算跑到一半強制中斷，前面的紀錄也都會在！


            if write_curve: 
                for step, tx, ful in final_curve: 
                    curve_csv_writer.writerow([step, tx, ful])
                    curve_csv_file.flush()
                curve_csv_file.close()

    if write_log: csv_file.close()

    return avg_tx_costs, avg_fulfill_rates, avg_comp_times


def main():
    ray.init(ignore_reinit_error=True)

    algo = None

    for mode in TEST_MODES: # "MAPPO" , "MYOTIC", "GREEDY" , "ERNC" , "STATIC_R"

        if mode == "MAPPO":
            ModelCatalog.register_custom_model("my_ctde_model", MAPPO_LSTM_Model)
            def env_creator(cfg):
                return ParallelPettingZooEnv(
                    SatelliteDataDisseminationEnv(
                        const_param=CONST_PARAM, 
                        T_max=T_MAX, 
                        num_users=cfg.get("num_users", 80),
                        erasure=cfg.get("erasure", 0.1),    
                        test_mode=IS_TEST_MODE
                ))
            register_env("satellite_nc_env", env_creator)
            algo = Algorithm.from_checkpoint(os.path.abspath(f"./satellite_{MY_CONST_NAME}_checkpoints"))
            print("MAPPO 載入完成")

        elif mode == "MYOTIC":
            ModelCatalog.register_custom_model("my_ctde_model", MAPPO_CTDE_Model)
            def env_creator(cfg):
                return ParallelPettingZooEnv(
                    SatelliteDataDisseminationEnv(
                        const_param=CONST_PARAM, T_max=T_MAX, 
                        num_users=cfg.get("num_users", 80),
                        erasure=cfg.get("erasure", 0.1),    
                        is_myotic=True, 
                        test_mode=IS_TEST_MODE
                ))
            register_env("satellite_nc_env", env_creator)
            algo = Algorithm.from_checkpoint(os.path.abspath(f"./satellite_{MY_CONST_NAME}_myotic_checkpoints"))
            print("MYOTIC 載入完成")

        tx_costs, fulfill_rates, times = run_mode(
            mode, USER_NUMBERS, NUM_EPISODES, algo=algo)

    ray.shutdown()


if __name__ == "__main__":
    main()