from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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
    sample_normalized_gaussian_oct,
    wall_synchronized,
)
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal


def _stats_line(x: torch.Tensor) -> str:
    return (
        f"min={float(x.min().item()):.6e}, max={float(x.max().item()):.6e}, "
        f"mean={float(x.mean().item()):.6e}, std={float(x.std().item()):.6e}"
    )


def case_gaussian(d: int, device: torch.device, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    return sample_normalized_gaussian_oct(d, device, dtype, gen)


def case_real_patch(
        folder: str,
        patch_size: int,
        roi_side: int,
        patch_index_in_roi: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    raw = load_one_center_patch(
        folder=folder,
        obj_name_prefix="balloons",
        patch_size=patch_size,
        roi_side=roi_side,
        patch_index_in_roi=patch_index_in_roi,
        device=device,
        dtype=dtype,
    )
    return normalize_oct_signal(raw)


def case_shuffled_patch(x_raw: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    flat = x_raw.reshape(-1).clone()
    perm = torch.randperm(flat.numel(), device=flat.device, generator=gen)
    shuf = flat[perm].view(x_raw.shape)
    return normalize_oct_signal(shuf)


def case_resampled_patch(x_raw: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    flat = x_raw.reshape(-1)
    idx = torch.randint(0, flat.numel(), (flat.numel(),), device=flat.device, generator=gen)
    samp = flat[idx].view(x_raw.shape)
    return normalize_oct_signal(samp)


def case_band_shuffled(x_raw: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    out = x_raw.clone()
    d = x_raw.shape[0]
    for j in range(8):
        perm = torch.randperm(d, device=x_raw.device, generator=gen)
        out[:, j] = x_raw[perm, j]
    return normalize_oct_signal(out)


def case_smooth_low_rank(d: int, device: torch.device, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    base = torch.full((d, 8), 0.5, dtype=dtype, device=device)
    noise = torch.randn(d, 8, dtype=dtype, device=device, generator=gen) * 0.02
    return normalize_oct_signal(base + noise)


def run(
        *,
        smoke: bool = False,
        no_csv: bool = False,
        out_dir: str = "",
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
    A_seed = 5
    seed = 1
    power_iters = 1
    passes = 8 if smoke else 80
    progress_every = max(1, passes // 10) if not verbose else 8
    omega = 1.0

    folder = default_balloons_folder()
    if show_progress:
        prob_milestone("PROB4", f"start matched control signals: passes={passes}, device={device}")
    if show_progress:
        prob_milestone("PROB4", "loading raw patch for S/R/B/U constructors ...")
    x_raw = load_one_center_patch(
        folder=folder,
        obj_name_prefix="balloons",
        patch_size=patch_size,
        roi_side=roi_side,
        patch_index_in_roi=patch_index_in_roi,
        device=device,
        dtype=dtype,
    )

    case_builders: List[Tuple[str, Callable[[], torch.Tensor]]] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    case_builders.append(("G", lambda: case_gaussian(d, device, dtype, gen)))

    gen_p = torch.Generator(device=device)
    gen_p.manual_seed(43)
    case_builders.append(("P", lambda: case_real_patch(folder, patch_size, roi_side, patch_index_in_roi, device, dtype)))

    gen_s = torch.Generator(device=device)
    gen_s.manual_seed(44)
    case_builders.append(("S", lambda: case_shuffled_patch(x_raw, gen_s)))

    gen_r = torch.Generator(device=device)
    gen_r.manual_seed(45)
    case_builders.append(("R", lambda: case_resampled_patch(x_raw, gen_r)))

    gen_b = torch.Generator(device=device)
    gen_b.manual_seed(46)
    case_builders.append(("B", lambda: case_band_shuffled(x_raw, gen_b)))

    gen_u = torch.Generator(device=device)
    gen_u.manual_seed(47)
    case_builders.append(("U", lambda: case_smooth_low_rank(d, device, dtype, gen_u)))

    if smoke:
        case_builders = [case_builders[0], case_builders[1]]

    labels = [c[0] for c in case_builders]
    if show_progress:
        prob_milestone("PROB4", f"cases order: {' '.join(labels)} ({len(case_builders)} runs)")

    torch.manual_seed(A_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(A_seed)
    A = make_measurement_matrix(A_seed, n, d, device, dtype)

    rows: List[Dict[str, Any]] = []
    n_c = len(case_builders)
    for ic, (case_name, builder) in enumerate(case_builders):
        if show_progress:
            prob_milestone("PROB4", f"case {ic + 1}/{n_c} ({case_name}) - build x_true, y ...")
        x_true = builder()
        y = intensity_measurements_explicit(A, x_true)
        print(f"[CONTROL] case={case_name} {_stats_line(x_true)}")
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        if show_progress:
            prob_milestone("PROB4", f"case {ic + 1}/{n_c} ({case_name}) - alg_orkm ...")
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
            prob_milestone("PROB4", f"case {ic + 1}/{n_c} ({case_name}) done (wall inner={wall:.1f}s)")
        log_final_line("[CONTROL]", m, wall)
        if verbose:
            print_orbit_progress(info, progress_stride=8, prefix="[CONTROL-PROG]")
        rows.append({
            "test": "prob4_control",
            "case": case_name,
            "seed": seed,
            "A_seed": A_seed,
            "omega": omega,
            "power_iters": power_iters,
            "passes": passes,
            "final_dist": m["final_dist"],
            "log_final_dist": m["log_final_dist"],
            "final_rel_l2": m["final_rel_l2"],
            "final_meas_rel": m["final_meas_rel"],
            "wall_sec": wall,
        })

    od = out_dir or os.path.join(str(PROJECT_ROOT), "scripts", "../scripts/output", "prob")
    csv_path = "" if no_csv else os.path.join(od, "prob_4_control_signals.csv")
    fieldnames = [
        "test", "case", "seed", "A_seed", "omega", "power_iters", "passes",
        "final_dist", "log_final_dist", "final_rel_l2", "final_meas_rel", "wall_sec",
    ]
    if csv_path:
        for row in rows:
            append_csv(csv_path, fieldnames, row)

    print("\n[CONTROL] summary")
    print("case | final_dist | log(final_dist) | final_rel | final_meas_rel")
    print("-----|------------|-----------------|-----------|---------------")
    for row in rows:
        print(
            f"{row['case']} | {row['final_dist']:.6e} | {row['log_final_dist']:.6e} | "
            f"{row['final_rel_l2']:.6e} | {row['final_meas_rel']:.6e}"
        )
    if show_progress:
        prob_milestone("PROB4", "finished.")


def main() -> None:
    p = argparse.ArgumentParser(description="TEST 4: matched control signals")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    run(
        smoke=args.smoke,
        no_csv=args.no_csv,
        out_dir=args.out_dir,
        verbose=args.verbose,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
