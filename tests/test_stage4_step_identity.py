from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal, oct_array_norm
from tests.stage4_diagnostics import (
    apply_single_row_update_with_info,
    save_csv_rows,
    save_txt_log,
    summarize_errors,
)


def run(quick: bool = False) -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    phase_eps = 1e-18
    beta_eps = 1e-18
    tol = 1e-12

    d_list = [4] if quick else [4, 8, 16]
    nd_list = [4] if quick else [12, 20]
    omega_list = [0.5, 1.0] if quick else [1.0]
    seeds = [0, 1] if quick else list(range(6))
    rows_per_seed = 4 if quick else 20

    lines: List[str] = []
    csv_rows: List[List[float | int]] = []
    scatter_rows: List[List[float | int]] = []
    total_configs = len(d_list) * len(nd_list) * len(omega_list)
    total_cases_plan = total_configs * len(seeds) * rows_per_seed
    config_idx = 0
    global_case_done = 0
    t0 = time.perf_counter()
    print(
        "[Stage4-A Progress] "
        f"planned_configs={total_configs}, planned_cases={total_cases_plan}, "
        f"seeds_per_config={len(seeds)}, rows_per_seed={rows_per_seed}"
    )

    for d in d_list:
        for nd in nd_list:
            n = nd * d
            for omega in omega_list:
                config_idx += 1
                cfg_t0 = time.perf_counter()
                err_step_all: List[float] = []
                err_er_all: List[float] = []
                total_cases = 0
                valid_cases = 0
                skipped_cases = 0
                for seed in seeds:
                    torch.manual_seed(seed * 1000 + n + d)
                    np.random.seed(seed * 1000 + n + d)
                    A = torch.randn(n, d, 8, device=device, dtype=dtype)
                    x_true = normalize_oct_signal(torch.randn(d, 8, device=device, dtype=dtype))
                    y = intensity_measurements_explicit(A, x_true)
                    x = normalize_oct_signal(torch.randn(d, 8, device=device, dtype=dtype))
                    rows = np.random.choice(n, size=rows_per_seed, replace=True)
                    for l in rows:
                        total_cases += 1
                        global_case_done += 1
                        x_plus, info = apply_single_row_update_with_info(
                            x, A[l], y[l], omega=omega, phase_eps=phase_eps, beta_eps=beta_eps
                        )
                        valid = bool(info["valid"]) and (not bool(info["skipped"])) and (
                                float(info["beta_l"]) > beta_eps)
                        if not valid:
                            skipped_cases += 1
                            continue
                        valid_cases += 1
                        lhs1 = float(oct_array_norm(x_plus - x).item())
                        rhs1 = float(abs(omega) / np.sqrt(float(info["beta_l"])) * float(info["abs_r_amp"]))
                        err1 = abs(lhs1 - rhs1)
                        lhs2 = float(info["abs_e"])
                        rhs2 = float(info["abs_r_amp"])
                        err2 = abs(lhs2 - rhs2)

                        scatter_rows.append(
                            [
                                d,
                                n,
                                omega,
                                seed,
                                int(l),
                                rhs1,  # predicted_step_size
                                lhs1,  # observed_step_size
                                float(info["abs_r_amp"]),  # abs_r_l
                                float(info["beta_l"]),
                                err1,  # err_step
                            ]
                        )

                        err_step_all.append(err1)
                        err_er_all.append(err2)
                        if err1 > tol or err2 > tol:
                            lines.append(
                                f"[A-violation] seed={seed}, d={d}, n={n}, l={int(l)}, omega={omega}, "
                                f"lhs1={lhs1:.6e}, rhs1={rhs1:.6e}, err1={err1:.6e}, "
                                f"lhs2={lhs2:.6e}, rhs2={rhs2:.6e}, err2={err2:.6e}"
                            )
                        if global_case_done % max(1, total_cases_plan // 100) == 0:
                            elapsed = time.perf_counter() - t0
                            pct = 100.0 * global_case_done / max(1, total_cases_plan)
                            print(
                                "[Stage4-A Progress] "
                                f"global_cases={global_case_done}/{total_cases_plan} ({pct:.1f}%), "
                                f"elapsed={elapsed:.1f}s, current_cfg=(d={d}, n={n}, omega={omega}), seed={seed}"
                            )

                s1 = summarize_errors(err_step_all, tol)
                s2 = summarize_errors(err_er_all, tol)
                lines.extend(
                    [
                        "[Stage4-A Summary]",
                        f"d={d}, n={n}, omega={omega}",
                        f"cases={total_cases}, valid={valid_cases}, skip={skipped_cases}",
                        f"max_err_step={s1['max']:.6e}, mean_err_step={s1['mean']:.6e}, "
                        f"median_err_step={s1['median']:.6e}, pass_rate_step={s1['pass_rate']:.6e}",
                        f"max_err_e_r={s2['max']:.6e}, mean_err_e_r={s2['mean']:.6e}, "
                        f"median_err_e_r={s2['median']:.6e}, pass_rate_e_r={s2['pass_rate']:.6e}",
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
                        s1["max"],
                        s1["mean"],
                        s1["median"],
                        s1["pass_rate"],
                        s2["max"],
                        s2["mean"],
                        s2["median"],
                        s2["pass_rate"],
                    ]
                )
                cfg_elapsed = time.perf_counter() - cfg_t0
                total_elapsed = time.perf_counter() - t0
                print(
                    "[Stage4-A Progress] "
                    f"config={config_idx}/{total_configs} done "
                    f"(d={d}, n={n}, omega={omega}), "
                    f"cfg_cases={total_cases}, valid={valid_cases}, skip={skipped_cases}, "
                    f"cfg_elapsed={cfg_elapsed:.1f}s, total_elapsed={total_elapsed:.1f}s"
                )

    out_txt = PROJECT_ROOT / "output" / "stage4_A_step_identity.txt"
    out_csv = PROJECT_ROOT / "output" / "stage4_A_step_identity.csv"
    out_scatter_csv = PROJECT_ROOT / "output" / "stage4_A_scatter.csv"
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
            "max_err_step",
            "mean_err_step",
            "median_err_step",
            "pass_rate_step",
            "max_err_e_r",
            "mean_err_e_r",
            "median_err_e_r",
            "pass_rate_e_r",
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
            "predicted_step_size",
            "observed_step_size",
            "abs_r_l",
            "beta_l",
            "err_step",
        ],
        scatter_rows,
    )
    for ln in lines:
        print(ln)
    print(f"[Stage4-A] txt={out_txt}")
    print(f"[Stage4-A] csv={out_csv}")
    print(f"[Stage4-A] scatter_csv={out_scatter_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(quick=args.quick)
