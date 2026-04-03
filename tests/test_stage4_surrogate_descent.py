from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal
from tests.stage4_diagnostics import (
    apply_single_row_update_with_info,
    compute_eta_matrix,
    compute_surrogate_E1,
    save_csv_rows,
    save_txt_log,
)


def run(quick: bool = False) -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    tol = 1e-12

    d_list = [4] if quick else [4, 8, 16]
    nd_list = [4] if quick else [12, 20]
    omega_list = [0.5, 1.0] if quick else [1.0]
    seeds = [0, 1] if quick else list(range(6))
    rows_per_seed = 3 if quick else 20

    lines: List[str] = []
    csv_rows: List[List[float | int]] = []
    scatter_x_eta: List[float] = []
    scatter_y_delta: List[float] = []
    scatter_x_rl: List[float] = []
    scatter_y_drop: List[float] = []
    scatter_rows: List[List[float | int]] = []
    omega_descent_stats: Dict[float, List[float]] = {w: [] for w in omega_list}
    total_configs = len(d_list) * len(nd_list) * len(omega_list)
    total_cases_plan = total_configs * len(seeds) * rows_per_seed
    config_idx = 0
    global_case_done = 0
    t0 = time.perf_counter()
    print(
        "[Stage4-D Progress] "
        f"planned_configs={total_configs}, planned_cases={total_cases_plan}, "
        f"seeds_per_config={len(seeds)}, rows_per_seed={rows_per_seed}"
    )

    for d in d_list:
        for nd in nd_list:
            n = nd * d
            for omega in omega_list:
                config_idx += 1
                cfg_t0 = time.perf_counter()
                total_cases = 0
                valid_cases = 0
                skipped_cases = 0
                theorem_violations = 0
                descent_condition_cases = 0
                descent_success_cases = 0
                empirical_descent_count = 0
                max_ineq_gap = -np.inf

                for seed in seeds:
                    torch.manual_seed(seed * 3000 + n + d + 13)
                    np.random.seed(seed * 3000 + n + d + 13)
                    A = torch.randn(n, d, 8, device=device, dtype=dtype)
                    x_true = normalize_oct_signal(torch.randn(d, 8, device=device, dtype=dtype))
                    y = intensity_measurements_explicit(A, x_true)
                    x = normalize_oct_signal(torch.randn(d, 8, device=device, dtype=dtype))
                    eta, _beta = compute_eta_matrix(A)
                    rows = np.random.choice(n, size=rows_per_seed, replace=True)
                    for l in rows:
                        total_cases += 1
                        global_case_done += 1
                        e1_before = compute_surrogate_E1(A, x, y)
                        x_plus, info = apply_single_row_update_with_info(x, A[l], y[l], omega=omega)
                        valid = bool(info["valid"]) and (not bool(info["skipped"]))
                        if not valid:
                            skipped_cases += 1
                            continue
                        valid_cases += 1
                        abs_r_l = float(info["abs_r_amp"])
                        eta_col_sum_excl = float(eta[:, int(l)].sum().item() - eta[int(l), int(l)].item())
                        e1_after = compute_surrogate_E1(A, x_plus, y)
                        predicted_upper = e1_before - float(omega) * (1.0 - eta_col_sum_excl) * abs_r_l
                        ineq_gap = e1_after - predicted_upper
                        max_ineq_gap = max(max_ineq_gap, ineq_gap)
                        if ineq_gap > tol:
                            theorem_violations += 1
                            lines.append(
                                f"[D-violation] seed={seed}, d={d}, n={n}, l={int(l)}, omega={omega}, "
                                f"E1_before={e1_before:.6e}, E1_after={e1_after:.6e}, "
                                f"eta_col_sum_excl={eta_col_sum_excl:.6e}, abs_r_l={abs_r_l:.6e}, "
                                f"predicted_upper={predicted_upper:.6e}, ineq_gap={ineq_gap:.6e}"
                            )
                        if eta_col_sum_excl < 1.0 and abs_r_l > 1e-14:
                            descent_condition_cases += 1
                            if e1_after < e1_before + tol:
                                descent_success_cases += 1
                        if e1_after < e1_before + tol:
                            empirical_descent_count += 1

                        scatter_x_eta.append(eta_col_sum_excl)
                        scatter_y_delta.append(e1_after - e1_before)
                        scatter_x_rl.append(abs_r_l)
                        scatter_y_drop.append(e1_before - e1_after)
                        scatter_rows.append(
                            [
                                d,
                                n,
                                omega,
                                seed,
                                int(l),
                                eta_col_sum_excl,
                                e1_after - e1_before,  # delta_E1
                                abs_r_l,
                                e1_before - e1_after,  # drop_E1
                            ]
                        )
                        if global_case_done % max(1, total_cases_plan // 100) == 0:
                            elapsed = time.perf_counter() - t0
                            pct = 100.0 * global_case_done / max(1, total_cases_plan)
                            print(
                                "[Stage4-D Progress] "
                                f"global_cases={global_case_done}/{total_cases_plan} ({pct:.1f}%), "
                                f"elapsed={elapsed:.1f}s, current_cfg=(d={d}, n={n}, omega={omega}), seed={seed}"
                            )

                empirical_descent_rate = float(empirical_descent_count / max(1, valid_cases))
                omega_descent_stats[omega].append(empirical_descent_rate)
                cond_rate = float(descent_success_cases / max(1, descent_condition_cases))
                freq_condition = float(descent_condition_cases / max(1, valid_cases))
                lines.extend(
                    [
                        "[Stage4-D Summary]",
                        f"d={d}, n={n}, omega={omega}",
                        f"cases={total_cases}, valid={valid_cases}, skip={skipped_cases}",
                        f"theorem_violations={theorem_violations}, max_ineq_gap={max_ineq_gap:.6e}",
                        f"descent_condition_cases={descent_condition_cases}, "
                        f"descent_success_cases={descent_success_cases}, descent_success_rate={cond_rate:.6e}",
                        f"empirical_descent_rate_all={empirical_descent_rate:.6e}, "
                        f"freq_eta_col_sum_excl_lt_1={freq_condition:.6e}",
                    ]
                )
                csv_rows.append(
                    [
                        d,
                        n,
                        omega,
                        total_cases,
                        valid_cases,
                        skipped_cases,
                        theorem_violations,
                        max_ineq_gap,
                        descent_condition_cases,
                        descent_success_cases,
                        cond_rate,
                        empirical_descent_rate,
                        freq_condition,
                    ]
                )
                cfg_elapsed = time.perf_counter() - cfg_t0
                total_elapsed = time.perf_counter() - t0
                print(
                    "[Stage4-D Progress] "
                    f"config={config_idx}/{total_configs} done "
                    f"(d={d}, n={n}, omega={omega}), "
                    f"cfg_cases={total_cases}, valid={valid_cases}, skip={skipped_cases}, "
                    f"cfg_elapsed={cfg_elapsed:.1f}s, total_elapsed={total_elapsed:.1f}s"
                )

    out_dir = PROJECT_ROOT / "output"
    out_txt = out_dir / "stage4_D_surrogate_descent.txt"
    out_csv = out_dir / "stage4_D_surrogate_descent.csv"
    out_scatter_csv = out_dir / "stage4_D_scatter.csv"
    save_txt_log(out_txt, lines)
    save_csv_rows(
        out_csv,
        [
            "d",
            "n",
            "omega",
            "cases",
            "valid_cases",
            "skipped_cases",
            "theorem_violations",
            "max_ineq_gap",
            "descent_condition_cases",
            "descent_success_cases",
            "descent_success_rate",
            "empirical_descent_rate_all",
            "freq_eta_col_sum_excl_lt_1",
        ],
        csv_rows,
    )
    save_csv_rows(
        out_scatter_csv,
        [
            "d",
            "n",
            "omega",
            "seed",
            "row_index",
            "eta_col_sum_excl",
            "delta_E1",
            "abs_r_l",
            "drop_E1",
        ],
        scatter_rows,
    )

    fig1 = out_dir / "stage4_D_eta_vs_deltaE.png"
    fig2 = out_dir / "stage4_D_rl_vs_drop.png"
    fig3 = out_dir / "stage4_D_omega_descent_rate.png"

    if len(scatter_x_eta) > 0:
        plt.figure()
        plt.scatter(scatter_x_eta, scatter_y_delta, s=8, alpha=0.5)
        plt.xlabel("eta_col_sum_excl")
        plt.ylabel("E1_after - E1_before")
        plt.title("Surrogate Change vs Dominance Proxy")
        plt.tight_layout()
        plt.savefig(fig1)
        plt.close()

        plt.figure()
        plt.scatter(scatter_x_rl, scatter_y_drop, s=8, alpha=0.5)
        plt.xlabel("|r_l|")
        plt.ylabel("E1_before - E1_after")
        plt.title("Surrogate Drop vs Selected Row Residual")
        plt.tight_layout()
        plt.savefig(fig2)
        plt.close()

        plt.figure()
        omegas = sorted(omega_descent_stats.keys())
        vals = [float(np.mean(omega_descent_stats[w])) if len(omega_descent_stats[w]) else np.nan for w in omegas]
        plt.bar([str(w) for w in omegas], vals)
        plt.xlabel("omega")
        plt.ylabel("empirical descent rate")
        plt.title("Empirical Surrogate Descent by Step Size")
        plt.tight_layout()
        plt.savefig(fig3)
        plt.close()

    for ln in lines:
        print(ln)
    print(f"[Stage4-D] txt={out_txt}")
    print(f"[Stage4-D] csv={out_csv}")
    print(f"[Stage4-D] scatter_csv={out_scatter_csv}")
    print(f"[Stage4-D] fig1={fig1}")
    print(f"[Stage4-D] fig2={fig2}")
    print(f"[Stage4-D] fig3={fig3}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(quick=args.quick)
