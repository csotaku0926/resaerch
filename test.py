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
USER_NUMBERS = [1, 50, 100, 200, 400]
NUM_EPISODES = 3
T_MAX = CONST_PARAM.t_max
print(f"[參數確認]")
print(f"- 衛星 const: {MY_CONST_NAME}")
print(f"- 最大步數 (T_max): {T_MAX}")
print(f"- target K: {TARGET_K}")
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
    Tw            = actual_env.Tw
    constellation = actual_env.constellation
    action        = np.zeros(M + 1, dtype=np.float32)

    buf = constellation.get_leo_buffer(real_id)
    if buf <= 0:
        return action

    # ── 收集每條 link 的 TEG contact volume ──────────────
    # TEG 是未來 Tw 步的容量向量，加總得到總 contact volume
    cvs = np.zeros(M + 1, dtype=np.float32)

    # ISL links：用鄰居的 downlink TEG 代表「透過這條 ISL 最終能到地面的量」
    for idx, neighbor_id in enumerate([constellation.get_neighbors(real_id)[0]]):
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


# ╔══════════════════════════════════════════════════════╗
# ║  B3: Static Redundancy                               ║
# ║                                                      ║
# ║  邏輯：                                              ║
# ║    【離線】根據長期平均 erasure rate p̄ 算出 N*        ║
# ║       N* = K / (1 - p̄) × (1 + safety_margin)       ║
# ║    【執行】每次 downlink contact 固定送 N* 封包       ║
# ║    不看當前 buffer 大小、不看 deficit、不用 ISL      ║
# ║    N* 在整個測試過程中不變                           ║
# ╚══════════════════════════════════════════════════════╝
def compute_n_star(actual_env, safety_margin=0.3):
    """
    離線估算 N*：對軌道多個時刻取樣，
    算出所有衛星對所有用戶的平均 erasure rate p̄，
    回傳 N* = K / (1 - p̄) × (1 + safety_margin)

    參考: Courtade & Wesel, IEEE Trans. Commun., 2011
    """
    constellation = actual_env.constellation
    TARGET_K      = constellation.target_k
    ts            = actual_env.ts
    start_dt      = actual_env.start_dt
    step_sec      = actual_env.step_seconds

    erasure_samples = []
    sample_steps = np.linspace(0, actual_env.T_max - 1, 10, dtype=int)

    for s in sample_steps:
        dt = start_dt + timedelta(seconds=int(s) * step_sec)
        t  = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        for sat_id in range(len(constellation.agents)):
            visible = constellation.get_visible_grids(sat_id, t)
            for gi in visible:
                for user in constellation.user_grids[gi].users:
                    p = constellation.calculate_erasure_rate(sat_id, user, t)
                    if 0.0 <= p < 1.0:
                        erasure_samples.append(p)

    p_avg  = float(np.mean(erasure_samples)) if erasure_samples else 0.3
    n_star = (TARGET_K / max(1e-9, 1.0 - p_avg)) * (1.0 + safety_margin)
    print(f"  [Static-R] p̄={p_avg:.4f}, K={TARGET_K}, N*={n_star:.2f}")
    return n_star


def action_static_r(real_id, actual_env, current_time, n_star):
    """
    執行階段：固定送 N* 封包到 downlink，不用 ISL。
    N* 是離線算好的常數，不隨即時狀況改變。
    """
    M             = actual_env.M
    constellation = actual_env.constellation
    action        = np.zeros(M + 1, dtype=np.float32)

    visible = constellation.get_visible_grids(real_id, current_time)
    buf     = constellation.get_leo_buffer(real_id)
    cap_dl  = constellation.get_downlink_capacity()

    if len(visible) == 0 or buf == 0: return action

    target_flow = min(n_star, cap_dl, buf)
    action[M]   = float(np.clip(target_flow / cap_dl, 0.0, 1.0))

    return action


# ╔══════════════════════════════════════════════════════╗
# ║  測試主迴圈                                          ║
# ╚══════════════════════════════════════════════════════╝
def run_mode(mode, user_numbers, num_episodes, algo=None, write_log=True):
    avg_tx_costs      = []
    avg_fulfill_rates = []
    avg_comp_times    = []

    checkpoint_dir = f"./satellite_{MY_CONST_NAME}_checkpoints"
    if write_log:
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_file_path = os.path.join(checkpoint_dir, f"{mode}_test_log.csv")
        csv_file = open(log_file_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        # 寫入標題列 (Headers)
        csv_writer.writerow(["User_Num", "Tx_Cost", "Fulfill", "Comp_Time"])

    for n_users in user_numbers:
        print(f"\n[{mode}] ══ n_users={n_users} ══")

        raw_env = SatelliteDataDisseminationEnv(
            const_param=CONST_PARAM, T_max=T_MAX, num_users=n_users, is_myotic=(mode == "MYOTIC"), test_mode=IS_TEST_MODE
        )
        env = ParallelPettingZooEnv(raw_env)

        # B3：離線預算 N*，整個 n_users 設定共用一個值
        n_star = None
        if mode == "STATIC_R":
            env.reset()
            actual_env = env.par_env if hasattr(env, "par_env") else env.unwrapped
            print("  計算 N*（離線步驟）...")
            n_star = compute_n_star(actual_env)

        tx_costs      = []
        comp_times    = []
        fulfill_rates = []

        for ep in range(num_episodes):
            obs, _ = env.reset()
            actual_env = env.par_env if hasattr(env, "par_env") else env.unwrapped
            done          = False
            final_tx_cost = 0.0

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
                            real_id, actual_env, current_time, n_star)

                obs, _, terminations, truncations, infos = env.step(actions)

                if infos:
                    first = list(infos.keys())[0]
                    final_tx_cost = infos[first].get("tx_cost", 0.0)
                    final_comp_time = infos[first].get("time", 0.0)

                done = (terminations.get("__all__", False) or
                        truncations.get("__all__", False))

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
        csv_writer.writerow([n_users, avg_tx, avg_ful, avg_time])
        csv_file.flush() # 強制寫入硬碟，這樣就算跑到一半強制中斷，前面的紀錄也都會在！

    csv_file.close()

    return avg_tx_costs, avg_fulfill_rates, avg_comp_times


def plot_results(user_numbers, results):
    colors  = {"MAPPO": "blue", "GREEDY": "green",
               "ERNC": "orange", "STATIC_R": "red"}
    markers = {"MAPPO": "o", "GREEDY": "s",
               "ERNC": "^", "STATIC_R": "D"}
    labels  = {
        "MAPPO":    "MAPPO-CTDE (proposed)",
        "GREEDY":   "B1: Greedy-RLNC",
        "ERNC":     "B2: ER-NC",
        "STATIC_R": "B3: Static Redundancy",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mode, (tx_costs, fulfill_rates) in results.items():
        kw = dict(marker=markers[mode], linestyle='-',
                  color=colors[mode], label=labels[mode])
        ax1.plot(user_numbers, tx_costs, **kw)
        ax2.plot(user_numbers, fulfill_rates, **kw)

    ax1.set_title('Transmission Cost vs Users')
    ax1.set_xlabel('Number of Users')
    ax1.set_ylabel('Avg Tx Cost (packets)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.set_title('Fulfill Rate vs Users')
    ax2.set_xlabel('Number of Users')
    ax2.set_ylabel('Avg Fulfill Rate')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300)
    plt.show()
    print("已儲存 baseline_comparison.png")


def main():
    ray.init(ignore_reinit_error=True)

    algo = None

    for mode in TEST_MODES: # "MAPPO" , "MYOTIC", "GREEDY" , "ERNC" , "STATIC_R"

        if mode == "MAPPO":
            ModelCatalog.register_custom_model("my_ctde_model", MAPPO_LSTM_Model)
            def env_creator(cfg):
                return ParallelPettingZooEnv(
                    SatelliteDataDisseminationEnv(
                        const_param=CONST_PARAM, T_max=T_MAX, num_users=cfg.get("num_users", 10), test_mode=IS_TEST_MODE
                ))
            register_env("satellite_nc_env", env_creator)
            algo = Algorithm.from_checkpoint(os.path.abspath(f"./satellite_{MY_CONST_NAME}_checkpoints"))
            print("MAPPO 載入完成")

        elif mode == "MYOTIC":
            ModelCatalog.register_custom_model("my_ctde_model", MAPPO_CTDE_Model)
            def env_creator(cfg):
                return ParallelPettingZooEnv(
                    SatelliteDataDisseminationEnv(
                        const_param=CONST_PARAM, T_max=T_MAX, num_users=cfg.get("num_users", 10), is_myotic=True, test_mode=IS_TEST_MODE
                ))
            register_env("satellite_nc_env", env_creator)
            algo = Algorithm.from_checkpoint(os.path.abspath(f"./satellite_{MY_CONST_NAME}_myotic_checkpoints"))
            print("MYOTIC 載入完成")

        tx_costs, fulfill_rates, times = run_mode(
            mode, USER_NUMBERS, NUM_EPISODES, algo=algo)

    # plot_results(USER_NUMBERS, {MODE: (tx_costs, fulfill_rates)})
    ray.shutdown()


if __name__ == "__main__":
    main()