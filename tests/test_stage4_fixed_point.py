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

from algorithms.gradients.grad_orkm import orkm_single_row_update_fast
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal, oct_array_norm
from tests.stage4_diagnostics import apply_single_row_update_with_info, save_csv_rows, save_txt_log


def run(quick: bool = False) -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    tol_e = 1e-12
    tol_update = 1e-12
    tol_res = 1e-10
    tol_full = 1e-12


    d_list = [4] if quick else [4, 8, 16]
    nd_list = [4] if quick else [12, 20]
    seeds = [0, 1] if quick else list(range(6))
    noise_levels = [1e-10] if quick else [1e-12, 1e-10, 1e-8]

    lines: List[str] = []
    csv_rows: List[List[float | int]] = []
    total_configs = len(d_list) * len(nd_list) * len(seeds)
    config_idx = 0
    t0 = time.perf_counter()
    print(
        "[Stage4-E Progress] "
        f"planned_configs={total_configs}, d_list={d_list}, nd_list={nd_list}, seeds={len(seeds)}"
    )

    cases = 0
    zero_update_pass = 0
    max_abs_e_consistent = 0.0
    max_update_norm_consistent = 0.0
    max_fullpass_change = 0.0
    fullpass_pass = 0
    zero_implies_consistent_checks = 0
    zero_implies_consistent_pass = 0

    for d in d_list:
        for nd in nd_list:
            n = nd * d
            for seed in seeds:
                config_idx += 1
                cfg_t0 = time.perf_counter()
                torch.manual_seed(seed * 4000 + n + d + 17)
                np.random.seed(seed * 4000 + n + d + 17)
                A = torch.randn(n, d, 8, device=device, dtype=dtype)
                x_true = normalize_oct_signal(torch.randn(d, 8, device=device, dtype=dtype))
                y = intensity_measurements_explicit(A, x_true)

                # E1: consistent row -> zero update
                for l in range(n):
                    x_plus, info = apply_single_row_update_with_info(x_true, A[l], y[l], omega=1.0)
                    abs_e = float(info["abs_e"])
                    upd = float(oct_array_norm(x_plus - x_true).item())
                    max_abs_e_consistent = max(max_abs_e_consistent, abs_e)
                    max_update_norm_consistent = max(max_update_norm_consistent, upd)
                    cases += 1
                    if abs_e <= tol_e and upd <= tol_update:
                        zero_update_pass += 1

                # E2: near consistent perturbation, zero update => small residual
                for nl in noise_levels:
                    x_pert = normalize_oct_signal(x_true + float(nl) * torch.randn_like(x_true))
                    for l in range(n):
                        x_plus, info = apply_single_row_update_with_info(x_pert, A[l], y[l], omega=1.0)
                        if bool(info["skipped"]):
                            continue
                        upd = float(oct_array_norm(x_plus - x_pert).item())
                        if upd <= tol_update:
                            zero_implies_consistent_checks += 1
                            abs_r = float(info["abs_r_amp"])
                            if abs_r <= tol_res:
                                zero_implies_consistent_pass += 1
                            else:
                                lines.append(
                                    f"[E2-warning] seed={seed}, d={d}, n={n}, l={l}, noise={nl}, "
                                    f"update_norm={upd:.6e}, abs_r={abs_r:.6e}"
                                )

                # E3: full-pass fixed point at consistent x_true
                perm = torch.randperm(n, device=device)
                x_pass = x_true.clone()
                for l in perm.tolist():
                    x_pass, _ = orkm_single_row_update_fast(
                        x_pass, A[l], y[l], omega=1.0, return_info=True
                    )
                full_change = float(oct_array_norm(x_pass - x_true).item())
                max_fullpass_change = max(max_fullpass_change, full_change)
                if full_change <= tol_full:
                    fullpass_pass += 1
                else:
                    lines.append(
                        f"[E3-violation] seed={seed}, d={d}, n={n}, fullpass_change={full_change:.6e}"
                    )

                csv_rows.append(
                    [
                        d,
                        n,
                        seed,
                        max_abs_e_consistent,
                        max_update_norm_consistent,
                        full_change,
                    ]
                )
                cfg_elapsed = time.perf_counter() - cfg_t0
                total_elapsed = time.perf_counter() - t0
                print(
                    "[Stage4-E Progress] "
                    f"config={config_idx}/{total_configs} done (d={d}, n={n}, seed={seed}), "
                    f"running_cases={cases}, running_zero_pass={zero_update_pass}, "
                    f"cfg_elapsed={cfg_elapsed:.1f}s, total_elapsed={total_elapsed:.1f}s"
                )

    pass_rate_zero_update = float(zero_update_pass / max(1, cases))
    pass_rate_fullpass_fixed = float(fullpass_pass / max(1, len(d_list) * len(nd_list) * len(seeds)))
    pass_rate_zero_impl = float(zero_implies_consistent_pass / max(1, zero_implies_consistent_checks))
    lines.extend(
        [
            "[Stage4-E Summary]",
            f"cases={cases}",
            f"max_abs_e_consistent={max_abs_e_consistent:.6e}",
            f"max_update_norm_consistent={max_update_norm_consistent:.6e}",
            f"pass_rate_zero_update={pass_rate_zero_update:.6e}",
            f"zero_implies_consistent_checks={zero_implies_consistent_checks}, "
            f"zero_implies_consistent_pass_rate={pass_rate_zero_impl:.6e}",
            f"max_fullpass_change={max_fullpass_change:.6e}",
            f"pass_rate_fullpass_fixed={pass_rate_fullpass_fixed:.6e}",
        ]
    )

    out_txt = PROJECT_ROOT / "output" / "stage4_E_fixed_point.txt"
    out_csv = PROJECT_ROOT / "output" / "stage4_E_fixed_point.csv"
    out_summary_csv = PROJECT_ROOT / "output" / "stage4_E_summary.csv"
    save_txt_log(out_txt, lines)
    save_csv_rows(
        out_csv,
        ["d", "n", "seed", "running_max_abs_e_consistent", "running_max_update_norm_consistent", "fullpass_change"],
        csv_rows,
    )
    save_csv_rows(
        out_summary_csv,
        [
            "cases",
            "max_abs_e_consistent",
            "max_update_norm_consistent",
            "pass_rate_zero_update",
            "zero_implies_consistent_checks",
            "zero_implies_consistent_pass_rate",
            "max_fullpass_change",
            "pass_rate_fullpass_fixed",
        ],
        [
            [
                cases,
                max_abs_e_consistent,
                max_update_norm_consistent,
                pass_rate_zero_update,
                zero_implies_consistent_checks,
                pass_rate_zero_impl,
                max_fullpass_change,
                pass_rate_fullpass_fixed,
            ]
        ],
    )
    for ln in lines:
        print(ln)
    print(f"[Stage4-E] txt={out_txt}")
    print(f"[Stage4-E] csv={out_csv}")
    print(f"[Stage4-E] summary_csv={out_summary_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(quick=args.quick)
