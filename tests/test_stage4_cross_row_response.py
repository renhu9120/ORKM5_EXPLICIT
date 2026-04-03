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

from core.octonion_inner import intensity_measurements_explicit, row_inner_batch_fast
from core.octonion_metric import normalize_oct_signal
from core.octonion_ops import oct_abs
from tests.stage4_diagnostics import (
    apply_single_row_update_with_info,
    compute_transfer_operator,
    compute_transfer_operator_wrong,
    save_csv_rows,
    save_txt_log,
    summarize_errors,
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
    total_configs = len(d_list) * len(nd_list) * len(omega_list)
    total_cases_plan = total_configs * len(seeds) * rows_per_seed
    config_idx = 0
    global_case_done = 0
    global_start = time.perf_counter()
    print(
        "[Stage4-B Progress] "
        f"planned_configs={total_configs}, planned_cases={total_cases_plan}, "
        f"seeds_per_config={len(seeds)}, rows_per_seed={rows_per_seed}"
    )

    for d in d_list:
        for nd in nd_list:
            n = nd * d
            for omega in omega_list:
                config_idx += 1
                cfg_start = time.perf_counter()
                case_max_err: List[float] = []
                case_self_err: List[float] = []
                case_max_wrong_err: List[float] = []
                total_cases = 0
                valid_cases = 0
                skipped_cases = 0
                for seed in seeds:
                    torch.manual_seed(seed * 1000 + n + d + 31)
                    np.random.seed(seed * 1000 + n + d + 31)
                    A = torch.randn(n, d, 8, device=device, dtype=dtype)
                    x_true = normalize_oct_signal(torch.randn(d, 8, device=device, dtype=dtype))
                    y = intensity_measurements_explicit(A, x_true)
                    x = normalize_oct_signal(torch.randn(d, 8, device=device, dtype=dtype))
                    rows = np.random.choice(n, size=rows_per_seed, replace=True)
                    for l in rows:
                        total_cases += 1
                        global_case_done += 1
                        x_plus, info = apply_single_row_update_with_info(x, A[l], y[l], omega=omega)
                        valid = bool(info["valid"]) and (not bool(info["skipped"]))
                        if not valid:
                            skipped_cases += 1
                            continue
                        valid_cases += 1
                        s_before = row_inner_batch_fast(A, x)
                        s_after = row_inner_batch_fast(A, x_plus)
                        e_l = info["e_l"]
                        beta_l = info["beta_l"]
                        errs = []
                        errs_wrong = []
                        for m in range(n):
                            t_ml = compute_transfer_operator(A, m, int(l), e_l, beta_l)
                            lhs = s_after[m]
                            rhs = s_before[m] + float(omega) * t_ml
                            err = float(oct_abs(lhs - rhs).item())
                            errs.append(err)

                            t_wrong = compute_transfer_operator_wrong(A, m, int(l), e_l, beta_l)
                            rhs_wrong = s_before[m] + float(omega) * t_wrong
                            err_wrong = float(oct_abs(lhs - rhs_wrong).item())
                            errs_wrong.append(err_wrong)
                        self_t = compute_transfer_operator(A, int(l), int(l), e_l, beta_l)
                        self_err = float(oct_abs(self_t - e_l).item())
                        case_max = float(np.max(np.asarray(errs, dtype=np.float64)))
                        case_max_wrong = float(np.max(np.asarray(errs_wrong, dtype=np.float64)))
                        case_max_err.append(case_max)
                        case_max_wrong_err.append(case_max_wrong)
                        case_self_err.append(self_err)
                        if case_max > tol or self_err > tol:
                            lines.append(
                                f"[B-violation] seed={seed}, d={d}, n={n}, l={int(l)}, omega={omega}, "
                                f"max_err_response={case_max:.6e}, self_err={self_err:.6e}, "
                                f"max_err_wrong={case_max_wrong:.6e}"
                            )
                        if global_case_done % max(1, total_cases_plan // 100) == 0:
                            elapsed = time.perf_counter() - global_start
                            pct = 100.0 * global_case_done / max(1, total_cases_plan)
                            print(
                                "[Stage4-B Progress] "
                                f"global_cases={global_case_done}/{total_cases_plan} ({pct:.1f}%), "
                                f"elapsed={elapsed:.1f}s, "
                                f"current_cfg=(d={d}, n={n}, omega={omega}), seed={seed}"
                            )

                s_resp = summarize_errors(case_max_err, tol)
                s_self = summarize_errors(case_self_err, tol)
                s_wrong = summarize_errors(case_max_wrong_err, tol)
                lines.extend(
                    [
                        "[Stage4-B Summary]",
                        f"d={d}, n={n}, omega={omega}",
                        f"cases={total_cases}, valid={valid_cases}, skip={skipped_cases}",
                        f"max_err_cross_response={s_resp['max']:.6e}, mean_err_cross_response={s_resp['mean']:.6e}, "
                        f"median_err_cross_response={s_resp['median']:.6e}, pass_rate_cross_response={s_resp['pass_rate']:.6e}",
                        f"max_err_self_transfer={s_self['max']:.6e}, mean_err_self_transfer={s_self['mean']:.6e}, "
                        f"pass_rate_self_transfer={s_self['pass_rate']:.6e}",
                        f"max_err_wrong_bracketing={s_wrong['max']:.6e}, mean_err_wrong_bracketing={s_wrong['mean']:.6e}",
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
                        s_resp["max"],
                        s_resp["mean"],
                        s_resp["median"],
                        s_resp["pass_rate"],
                        s_self["max"],
                        s_self["mean"],
                        s_self["pass_rate"],
                        s_wrong["max"],
                        s_wrong["mean"],
                    ]
                )
                cfg_elapsed = time.perf_counter() - cfg_start
                all_elapsed = time.perf_counter() - global_start
                print(
                    "[Stage4-B Progress] "
                    f"config={config_idx}/{total_configs} done "
                    f"(d={d}, n={n}, omega={omega}), "
                    f"cfg_cases={total_cases}, valid={valid_cases}, skip={skipped_cases}, "
                    f"cfg_elapsed={cfg_elapsed:.1f}s, total_elapsed={all_elapsed:.1f}s"
                )

    out_txt = PROJECT_ROOT / "output" / "stage4_B_cross_row_response.txt"
    out_csv = PROJECT_ROOT / "output" / "stage4_B_cross_row_response.csv"
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
            "max_err_cross_response",
            "mean_err_cross_response",
            "median_err_cross_response",
            "pass_rate_cross_response",
            "max_err_self_transfer",
            "mean_err_self_transfer",
            "pass_rate_self_transfer",
            "max_err_wrong_bracketing",
            "mean_err_wrong_bracketing",
        ],
        csv_rows,
    )
    for ln in lines:
        print(ln)
    print(f"[Stage4-B] txt={out_txt}")
    print(f"[Stage4-B] csv={out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(quick=args.quick)
