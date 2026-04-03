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
from core.octonion_metric import normalize_oct_signal
from core.octonion_ops import oct_abs
from tests.stage4_diagnostics import (
    apply_single_row_update_with_info,
    compute_eta_matrix,
    compute_residual_vector,
    compute_transfer_operator,
    save_csv_rows,
    save_txt_log,
)


def run(quick: bool = False) -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    tol_ratio = 1e-12
    eps_small = 1e-30


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
    t0 = time.perf_counter()
    print(
        "[Stage4-C Progress] "
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
                max_ratio_t = 0.0
                max_ratio_s = 0.0
                max_ratio_amp = 0.0
                max_ratio_res = 0.0
                max_ratio_cs = 0.0
                violations_t = 0
                violations_s = 0
                violations_amp = 0
                violations_res = 0
                violations_cs = 0
                max_eta_diag_abs = 0.0

                for seed in seeds:
                    torch.manual_seed(seed * 2000 + n + d + 7)
                    np.random.seed(seed * 2000 + n + d + 7)
                    A = torch.randn(n, d, 8, device=device, dtype=dtype)
                    x_true = normalize_oct_signal(torch.randn(d, 8, device=device, dtype=dtype))
                    y = intensity_measurements_explicit(A, x_true)
                    x = normalize_oct_signal(torch.randn(d, 8, device=device, dtype=dtype))
                    eta, beta = compute_eta_matrix(A)

                    for l in range(n):
                        eta_ll_err = abs(float(eta[l, l].item()) - 1.0)
                        max_eta_diag_abs = max(max_eta_diag_abs, eta_ll_err)
                        for m in range(n):
                            cs_bnd = np.sqrt(float(beta[m].item()) / float(beta[l].item()))
                            ratio_cs = float(eta[m, l].item()) / (cs_bnd + eps_small)
                            max_ratio_cs = max(max_ratio_cs, ratio_cs)
                            if ratio_cs > 1.0 + tol_ratio:
                                violations_cs += 1

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
                        e_l = info["e_l"]
                        abs_e = float(info["abs_e"])
                        abs_r_l = float(info["abs_r_amp"])
                        beta_l = info["beta_l"]

                        pre = compute_residual_vector(A, x, y)
                        post = compute_residual_vector(A, x_plus, y)
                        s_before = pre["s_all"]
                        s_after = post["s_all"]
                        r_before = pre["r_all"]
                        r_after = post["r_all"]
                        amp_before = pre["amp_all"]
                        amp_after = post["amp_all"]

                        for m in range(n):
                            eta_ml = float(eta[m, int(l)].item())
                            t_ml = compute_transfer_operator(A, m, int(l), e_l, beta_l)
                            val1 = float(oct_abs(t_ml).item())
                            bnd1 = eta_ml * abs_e
                            ratio1 = val1 / (bnd1 + eps_small)
                            max_ratio_t = max(max_ratio_t, ratio1)
                            if ratio1 > 1.0 + tol_ratio:
                                violations_t += 1

                            val2 = float(oct_abs(s_after[m] - s_before[m]).item())
                            bnd2 = abs(omega) * eta_ml * abs_r_l
                            ratio2 = val2 / (bnd2 + eps_small)
                            max_ratio_s = max(max_ratio_s, ratio2)
                            if ratio2 > 1.0 + tol_ratio:
                                violations_s += 1

                            val3 = abs(float(amp_after[m].item()) - float(amp_before[m].item()))
                            bnd3 = abs(omega) * eta_ml * abs_r_l
                            ratio3 = val3 / (bnd3 + eps_small)
                            max_ratio_amp = max(max_ratio_amp, ratio3)
                            if ratio3 > 1.0 + tol_ratio:
                                violations_amp += 1

                            val4 = abs(float(r_after[m].item()) - float(r_before[m].item()))
                            bnd4 = abs(omega) * eta_ml * abs_r_l
                            ratio4 = val4 / (bnd4 + eps_small)
                            max_ratio_res = max(max_ratio_res, ratio4)
                            if ratio4 > 1.0 + tol_ratio:
                                violations_res += 1
                        if global_case_done % max(1, total_cases_plan // 100) == 0:
                            elapsed = time.perf_counter() - t0
                            pct = 100.0 * global_case_done / max(1, total_cases_plan)
                            print(
                                "[Stage4-C Progress] "
                                f"global_cases={global_case_done}/{total_cases_plan} ({pct:.1f}%), "
                                f"elapsed={elapsed:.1f}s, current_cfg=(d={d}, n={n}, omega={omega}), seed={seed}"
                            )

                lines.extend(
                    [
                        "[Stage4-C Summary]",
                        f"d={d}, n={n}, omega={omega}",
                        f"cases={total_cases}, valid={valid_cases}, skip={skipped_cases}",
                        f"max_ratio_T={max_ratio_t:.6e}, max_ratio_s={max_ratio_s:.6e}, "
                        f"max_ratio_amp={max_ratio_amp:.6e}, max_ratio_res={max_ratio_res:.6e}",
                        f"violations_T={violations_t}, violations_s={violations_s}, "
                        f"violations_amp={violations_amp}, violations_res={violations_res}",
                        f"max_ratio_cs={max_ratio_cs:.6e}, violations_cs={violations_cs}, "
                        f"max_abs_eta_ll_minus_1={max_eta_diag_abs:.6e}",
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
                        max_ratio_t,
                        max_ratio_s,
                        max_ratio_amp,
                        max_ratio_res,
                        violations_t,
                        violations_s,
                        violations_amp,
                        violations_res,
                        max_ratio_cs,
                        violations_cs,
                        max_eta_diag_abs,
                    ]
                )
                cfg_elapsed = time.perf_counter() - cfg_t0
                total_elapsed = time.perf_counter() - t0
                print(
                    "[Stage4-C Progress] "
                    f"config={config_idx}/{total_configs} done "
                    f"(d={d}, n={n}, omega={omega}), "
                    f"cfg_cases={total_cases}, valid={valid_cases}, skip={skipped_cases}, "
                    f"cfg_elapsed={cfg_elapsed:.1f}s, total_elapsed={total_elapsed:.1f}s"
                )

    out_txt = PROJECT_ROOT / "output" / "stage4_C_cross_row_bound.txt"
    out_csv = PROJECT_ROOT / "output" / "stage4_C_cross_row_bound.csv"
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
            "max_ratio_T",
            "max_ratio_s",
            "max_ratio_amp",
            "max_ratio_res",
            "violations_T",
            "violations_s",
            "violations_amp",
            "violations_res",
            "max_ratio_cs",
            "violations_cs",
            "max_abs_eta_ll_minus_1",
        ],
        csv_rows,
    )
    for ln in lines:
        print(ln)
    print(f"[Stage4-C] txt={out_txt}")
    print(f"[Stage4-C] csv={out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(quick=args.quick)
