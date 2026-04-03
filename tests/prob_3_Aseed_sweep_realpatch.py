from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from algorithms.algs.alg_orkm import alg_orkm
from prob_realpatch_common import (
    PROJECT_ROOT,
    aligned_metrics,
    append_csv,
    default_balloons_folder,
    default_device,
    load_one_center_patch,
    log_final_line,
    make_measurement_matrix,
    print_orbit_progress,
    prob_milestone,
    wall_synchronized,
)
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal, oct_array_norm


def run(
        *,
        smoke: bool = False,
        no_csv: bool = False,
        out_dir: str = "",
        omega: float = 1.0,
        verbose: bool = False,
        show_progress: bool = True,
) -> None:
    device = default_device()
    dtype = torch.float64
    patch_size = 8
    d = patch_size * patch_size
    roi_side = 3
    patch_index_in_roi = 4
    n_over_d = 20
    n = n_over_d * d
    seed = 1
    power_iters = 1
    passes = 8 if smoke else 80
    progress_every = max(1, passes // 10) if not verbose else 8
    omega = float(omega)

    if smoke:
        a_seeds = [5, 11]
    else:
        a_seeds = [1, 5, 11, 21, 31, 52, 71, 101, 212, 314]

    if show_progress:
        prob_milestone(
            "PROB3",
            f"start A_seed sweep: {len(a_seeds)} seeds, passes={passes}, omega={omega}, device={device}",
        )

    folder = default_balloons_folder()
    if show_progress:
        prob_milestone("PROB3", "loading real patch ...")
    x_true_raw = load_one_center_patch(
        folder=folder,
        obj_name_prefix="balloons",
        patch_size=patch_size,
        roi_side=roi_side,
        patch_index_in_roi=patch_index_in_roi,
        device=device,
        dtype=dtype,
    )
    x_true = normalize_oct_signal(x_true_raw)
    print(
        f"[ASEED] device={device}, d={d}, n={n}, passes={passes}, omega={omega}, seed={seed}, "
        f"patch norm={float(oct_array_norm(x_true).item()):.6e}"
    )

    rows: List[Dict[str, Any]] = []
    n_s = len(a_seeds)
    for ia, A_seed in enumerate(a_seeds):
        if show_progress:
            prob_milestone("PROB3", f"A_seed {ia + 1}/{n_s} (A_seed={A_seed}) - building A, y ...")
        torch.manual_seed(A_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(A_seed)
        A = make_measurement_matrix(A_seed, n, d, device, dtype)
        y = intensity_measurements_explicit(A, x_true)

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        if show_progress:
            prob_milestone("PROB3", f"A_seed {ia + 1}/{n_s} - alg_orkm ...")
        t0 = time.time()
        x_est, info = alg_orkm(
            A=A,
            y=y,
            T=passes,
            seed=seed,
            power_iters=power_iters,
            omega=omega,
            record_meas_rel=verbose,
            x_true_proc=x_true,
            verbose=verbose,
            progress_every=progress_every,
        )
        wall = wall_synchronized(device, t0)
        m = aligned_metrics(x_true, x_est, A, y)
        if show_progress:
            prob_milestone("PROB3", f"A_seed {ia + 1}/{n_s} done (wall inner={wall:.1f}s)")
        print(f"[ASEED] A_seed={A_seed}")
        log_final_line("[ASEED]", m, wall)
        if verbose:
            print_orbit_progress(info, progress_stride=8, prefix="[ASEED-PROG]")
        rows.append({
            "test": "prob3_Aseed",
            "case": "real_patch",
            "A_seed": A_seed,
            "omega": omega,
            "seed": seed,
            "power_iters": power_iters,
            "passes": passes,
            "final_dist": m["final_dist"],
            "log_final_dist": m["log_final_dist"],
            "final_rel_l2": m["final_rel_l2"],
            "final_meas_rel": m["final_meas_rel"],
            "wall_sec": wall,
        })

    dists = [float(r["final_dist"]) for r in rows]
    print(
        f"\n[ASEED] stats final_dist: mean={statistics.mean(dists):.6e}, median={statistics.median(dists):.6e}, "
        f"min={min(dists):.6e}, max={max(dists):.6e}"
    )
    best = min(rows, key=lambda r: r["final_dist"])
    worst = max(rows, key=lambda r: r["final_dist"])
    print(f"[ASEED] best A_seed={best['A_seed']}, worst A_seed={worst['A_seed']}")

    od = out_dir or os.path.join(str(PROJECT_ROOT), "scripts", "../scripts/output", "prob")
    csv_path = "" if no_csv else os.path.join(od, "prob_3_Aseed_sweep.csv")
    fieldnames = [
        "test", "case", "A_seed", "omega", "seed", "power_iters", "passes",
        "final_dist", "log_final_dist", "final_rel_l2", "final_meas_rel", "wall_sec",
    ]
    if csv_path:
        for row in rows:
            append_csv(csv_path, fieldnames, row)

    print("\n[ASEED] summary table")
    print("A_seed | final_dist | log(final_dist) | final_rel | final_meas_rel")
    print("-------|------------|-----------------|-----------|---------------")
    for row in rows:
        print(
            f"{row['A_seed']} | {row['final_dist']:.6e} | {row['log_final_dist']:.6e} | "
            f"{row['final_rel_l2']:.6e} | {row['final_meas_rel']:.6e}"
        )
    if show_progress:
        prob_milestone("PROB3", "finished.")


def main() -> None:
    p = argparse.ArgumentParser(description="TEST 3: A_seed sweep on real patch")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--verbose", action="store_true", help="grad_orkm iteration logs + meas_rel orbit")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    run(
        smoke=args.smoke,
        no_csv=args.no_csv,
        out_dir=args.out_dir,
        omega=args.omega,
        verbose=args.verbose,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
