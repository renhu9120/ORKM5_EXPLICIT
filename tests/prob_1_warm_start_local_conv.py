from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from algorithms.gradients.grad_orkm import grad_orkm
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
    sample_normalized_gaussian_oct,
    wall_synchronized,
    warm_start_x0,
)
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal, oct_array_norm


def _run_one_warm(
        *,
        x_true: torch.Tensor,
        A: torch.Tensor,
        y: torch.Tensor,
        eps: float,
        repeat_idx: int,
        seed_noise: int,
        iter_seed: int,
        passes: int,
        power_iters: int,
        omega: float,
        device: torch.device,
        dtype: torch.dtype,
        verbose: bool,
        progress_every: int,
) -> Tuple[Dict[str, Any], Dict[str, float], float, Dict[str, Any]]:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed_noise))
    x0, init_dist, init_rel = warm_start_x0(x_true, eps, device=device, dtype=dtype, gen=gen)

    torch.manual_seed(iter_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(iter_seed)
    t0 = time.time()
    x_est, info = grad_orkm(
        A=A,
        y=y,
        x0=x0,
        max_iters=passes,
        omega=omega,
        x_true_proc=x_true,
        record_meas_rel=verbose,
        verbose=verbose,
        progress_every=progress_every,
        return_info=True,
    )
    wall = wall_synchronized(device, t0)
    m = aligned_metrics(x_true, x_est, A, y)
    row: Dict[str, Any] = {
        "test": "prob1_warm_start",
        "eps": eps,
        "repeat": repeat_idx,
        "seed_noise": seed_noise,
        "iter_seed": iter_seed,
        "init_dist": init_dist,
        "init_rel": init_rel,
        "A_seed": None,
        "omega": omega,
        "power_iters": power_iters,
        "passes": passes,
        "final_dist": m["final_dist"],
        "log_final_dist": m["log_final_dist"],
        "final_rel_l2": m["final_rel_l2"],
        "final_meas_rel": m["final_meas_rel"],
        "wall_sec": wall,
    }
    return info, m, wall, row


def _signal_block(
        name: str,
        x_true: torch.Tensor,
        A_seed: int,
        n: int,
        d: int,
        eps_list: List[float],
        n_rep: int,
        base_noise_seed: int,
        iter_seed: int,
        passes: int,
        power_iters: int,
        omega: float,
        device: torch.device,
        dtype: torch.dtype,
        verbose: bool,
        progress_every: int,
        csv_path: str,
        show_progress: bool,
) -> None:
    torch.manual_seed(A_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(A_seed)
    A = make_measurement_matrix(A_seed, n, d, device, dtype)
    y = intensity_measurements_explicit(A, x_true)

    rows_out: List[Dict[str, Any]] = []
    n_eps = len(eps_list)
    if show_progress:
        prob_milestone(
            "PROB1",
            f"signal={name}: {n_eps} eps values x {n_rep} repeats x {passes} passes "
            f"(total inner runs={n_eps * n_rep})",
        )
    for ie, eps in enumerate(eps_list):
        finals: List[float] = []
        if show_progress:
            prob_milestone("PROB1", f"signal={name} eps {ie + 1}/{n_eps} (eps={eps:g})")
        for r in range(n_rep):
            if show_progress:
                prob_milestone("PROB1", f"signal={name} eps={eps:g} repeat {r + 1}/{n_rep} - grad_orkm ...")
            seed_noise = base_noise_seed + 10007 * r
            info, m, wall, row = _run_one_warm(
                x_true=x_true,
                A=A,
                y=y,
                eps=eps,
                repeat_idx=r,
                seed_noise=seed_noise,
                iter_seed=iter_seed,
                passes=passes,
                power_iters=power_iters,
                omega=omega,
                device=device,
                dtype=dtype,
                verbose=verbose,
                progress_every=progress_every,
            )
            row["signal"] = name
            row["A_seed"] = A_seed
            rows_out.append(row)
            finals.append(m["final_dist"])
            init_dist = float(row["init_dist"])
            init_rel = float(row["init_rel"])
            print(f"[WARM] signal={name} eps={eps:g} repeat={r} init_dist={init_dist:.6e} init_rel={init_rel:.6e}")
            log_final_line("[WARM]", m, wall)
            if verbose:
                print_orbit_progress(info, progress_stride=8, prefix="[WARM-PROG]")
        mean_f = statistics.mean(finals)
        med_f = statistics.median(finals)
        best_f = min(finals)
        print(f"[WARM] signal={name} eps={eps:g} aggregate mean_final_dist={mean_f:.6e} median={med_f:.6e} best={best_f:.6e}")

    fieldnames = [
        "test", "signal", "eps", "repeat", "seed_noise", "iter_seed", "init_dist", "init_rel",
        "A_seed", "omega", "power_iters", "passes",
        "final_dist", "log_final_dist", "final_rel_l2", "final_meas_rel", "wall_sec",
    ]
    if csv_path and csv_path != os.devnull:
        for row in rows_out:
            append_csv(csv_path, fieldnames, row)

    print("\n[WARM] summary table")
    print("eps | repeat | init_dist | final_dist | final_rel | final_meas_rel")
    print("----|--------|-----------|------------|-----------|---------------")
    for row in rows_out:
        print(
            f"{row['eps']:g} | {row['repeat']} | {row['init_dist']:.6e} | {row['final_dist']:.6e} | "
            f"{row['final_rel_l2']:.6e} | {row['final_meas_rel']:.6e}"
        )

    by_eps: Dict[float, List[float]] = defaultdict(list)
    for row in rows_out:
        by_eps[float(row["eps"])].append(float(row["final_dist"]))
    print("\n[WARM] per-eps stats (mean / median / best final_dist)")
    for eps in sorted(by_eps.keys()):
        v = by_eps[eps]
        print(
            f"  eps={eps:g}: mean={statistics.mean(v):.6e}, median={statistics.median(v):.6e}, best={min(v):.6e}"
        )


def run(
        *,
        smoke: bool = False,
        no_csv: bool = False,
        out_dir: str = "",
        verbose: bool = False,
        gaussian_only: bool = False,
        real_only: bool = False,
        show_progress: bool = True,
) -> None:
    device = default_device()
    dtype = torch.float64
    d = 64
    patch_size = 8
    roi_side = 3
    patch_index_in_roi = 4
    n_over_d = 20
    n = n_over_d * d
    A_seed = 5
    iter_seed = 1
    base_noise_seed = 12345
    omega = 1.0
    power_iters = 1
    passes = 8 if smoke else 80
    progress_every = max(1, passes // 10) if not verbose else 8

    if smoke:
        eps_list = [1e-2]
        n_rep = 1
    else:
        eps_list = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4]
        n_rep = 3

    od = out_dir or os.path.join(str(PROJECT_ROOT), "scripts", "../scripts/output", "prob")
    csv_path = "" if no_csv else os.path.join(od, "prob_1_warm_start.csv")

    folder = default_balloons_folder()
    obj_name_prefix = "balloons"

    if show_progress:
        prob_milestone(
            "PROB1",
            f"start warm-start local convergence: passes={passes}, eps_list len={len(eps_list)}, n_rep={n_rep}, "
            f"device={device}",
        )

    if not gaussian_only:
        if show_progress:
            prob_milestone("PROB1", "loading real patch ...")
        x_true_real_raw = load_one_center_patch(
            folder=folder,
            obj_name_prefix=obj_name_prefix,
            patch_size=patch_size,
            roi_side=roi_side,
            patch_index_in_roi=patch_index_in_roi,
            device=device,
            dtype=dtype,
        )
        x_true_real = normalize_oct_signal(x_true_real_raw)
        print(
            f"[WARM] real patch raw norm={float(oct_array_norm(x_true_real_raw).item()):.6e}, "
            f"normed norm={float(oct_array_norm(x_true_real).item()):.6e}"
        )
        _signal_block(
            "real_patch",
            x_true_real,
            A_seed,
            n,
            d,
            eps_list,
            n_rep,
            base_noise_seed,
            iter_seed,
            passes,
            power_iters,
            omega,
            device,
            dtype,
            verbose,
            progress_every,
            csv_path,
            show_progress,
        )

    if not real_only:
        if show_progress:
            prob_milestone("PROB1", "Gaussian baseline: sampling x_true ...")
        gen_g = torch.Generator(device=device)
        gen_g.manual_seed(999)
        x_true_g = sample_normalized_gaussian_oct(d, device, dtype, gen_g)
        _signal_block(
            "gaussian",
            x_true_g,
            A_seed,
            n,
            d,
            eps_list,
            n_rep,
            base_noise_seed + 1,
            iter_seed,
            passes,
            power_iters,
            omega,
            device,
            dtype,
            verbose,
            progress_every,
            csv_path,
            show_progress,
        )

    if show_progress:
        prob_milestone("PROB1", "finished.")


def main() -> None:
    p = argparse.ArgumentParser(description="TEST 1: warm-start local convergence (dist_align focus)")
    p.add_argument("--smoke", action="store_true", help="minimal sweep for pass/fail smoke")
    p.add_argument("--out-dir", type=str, default="", help="CSV output directory (default: scripts/output/prob)")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--gaussian-only", action="store_true")
    p.add_argument("--real-only", action="store_true")
    p.add_argument("--no-progress", action="store_true", help="disable [PROB1] milestone lines")
    args = p.parse_args()
    run(
        smoke=args.smoke,
        no_csv=args.no_csv,
        out_dir=args.out_dir,
        verbose=args.verbose,
        gaussian_only=args.gaussian_only,
        real_only=args.real_only,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
