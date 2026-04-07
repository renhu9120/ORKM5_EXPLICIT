from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.constants import DEFAULT_ORKM_OMEGA
from algorithms.algs.alg_orkm import alg_orkm
from algorithms.algs.alg_orkm_cuda import alg_orkm_cuda_success_rate_batched
from core.octonion_align import right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit, intensity_measurements_independent_batches
from core.octonion_metric import normalize_oct_signal
from core.octonion_sign import sign_aligned_distance

TrialFn = Callable[..., Any]

SOLVER_REGISTRY: Dict[str, TrialFn] = {
    "alg_orkm_cuda_success_rate_batched": alg_orkm_cuda_success_rate_batched,
    "alg_orkm": alg_orkm,
}


def default_param_list() -> np.ndarray:
    range1 = np.arange(5, 7.9, 1)
    range2 = np.arange(8, 9.9, 0.2)
    range3 = np.arange(10, 12.01, 1)
    return np.unique(np.concatenate([range1, range2, range3]))


def nd_grid_descending(param_list: np.ndarray) -> List[float]:
    u = np.unique(np.asarray(param_list, dtype=float))
    return u[::-1].tolist()


def _right_aligned_distance_row(x_true: Tensor, x_est: Tensor) -> float:
    return float(right_aligned_distance(x_true, x_est).item())


def run_trial_batch_cuda(
        *,
        nd_rate: float,
        d: int,
        batch_size: int,
        passes: int,
        stop_err: float,
        device: torch.device,
        dtype: torch.dtype,
        seed_offset: int,
        power_iters: int,
) -> Dict[str, Any]:
    n = int(np.floor(float(nd_rate) * float(d)))
    if n < 1:
        raise ValueError(f"invalid n/d={nd_rate} with d={d} -> n={n}")
    B = int(batch_size)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed_offset))

    A = torch.randn((B, n, d, 8), dtype=dtype, device=device, generator=gen)
    x_true = torch.randn((B, d, 8), dtype=dtype, device=device, generator=gen)
    for b in range(B):
        x_true[b] = normalize_oct_signal(x_true[b])
    y = intensity_measurements_independent_batches(A, x_true)

    t0 = time.perf_counter()
    with torch.inference_mode():
        x_est, info = alg_orkm_cuda_success_rate_batched(
            A,
            y,
            int(passes),
            seed=None,
            power_iters=int(power_iters),
            omega=DEFAULT_ORKM_OMEGA,
            x_true=x_true,
            stop_err=float(stop_err),
            verbose=False,
            progress_every=1,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0

    d_plus = torch.linalg.norm((x_true - x_est).reshape(B, -1), ord=2, dim=1)
    d_minus = torch.linalg.norm((x_true + x_est).reshape(B, -1), ord=2, dim=1)
    d_sign_v = torch.minimum(d_plus, d_minus)
    d_sign_list = [float(d_sign_v[i].item()) for i in range(B)]
    d_align_list = [_right_aligned_distance_row(x_true[i], x_est[i]) for i in range(B)]

    thr = float(stop_err)
    succ_list = [1 if d_sign_list[i] <= thr else 0 for i in range(B)]
    steps_list = [int(x) for x in info["per_trial_epochs"]]  # type: ignore[index]
    time_each = wall / float(B)
    times_list = [time_each] * B

    return {
        "successes": int(sum(succ_list)),
        "d_sign_list": d_sign_list,
        "d_align_list": d_align_list,
        "steps_list": steps_list,
        "times_list": times_list,
        "wall_batch": wall,
    }


def run_trial_single_cpu(
        *,
        nd_rate: float,
        d: int,
        passes: int,
        stop_err: float,
        device: torch.device,
        dtype: torch.dtype,
        seed_val: Optional[int],
        power_iters: int,
) -> Dict[str, Any]:
    n = int(np.floor(float(nd_rate) * float(d)))
    if seed_val is not None:
        torch.manual_seed(int(seed_val))
    A = torch.randn((n, d, 8), dtype=dtype, device=device)
    x_true = normalize_oct_signal(torch.randn((d, 8), dtype=dtype, device=device))
    y = intensity_measurements_explicit(A, x_true)
    t0 = time.perf_counter()
    with torch.inference_mode():
        x_est, info = alg_orkm(
            A,
            y,
            int(passes),
            seed=None,
            power_iters=int(power_iters),
            omega=DEFAULT_ORKM_OMEGA,
            record_meas_rel=False,
            x_true_proc=x_true,
            stop_err=float(stop_err),
            verbose=False,
        )
    wall = time.perf_counter() - t0
    d_sign = float(sign_aligned_distance(x_true, x_est).item())
    d_align = _right_aligned_distance_row(x_true, x_est)
    thr = float(stop_err)
    succ = 1 if d_sign <= thr else 0
    ec = info.get("epoch_row_update_counts")
    steps = int(len(ec)) if isinstance(ec, list) else int(passes)
    return {
        "successes": succ,
        "d_sign_list": [d_sign],
        "d_align_list": [d_align],
        "steps_list": [steps],
        "times_list": [wall],
        "wall_batch": wall,
    }


def aggregate_trial_stats(
        *,
        d_sign_list: List[float],
        d_align_list: List[float],
        steps_list: List[int],
        times_list: List[float],
        successes: int,
        trials: int,
) -> Dict[str, float]:
    sr = float(successes) / float(trials) if trials > 0 else float("nan")
    ds = np.asarray(d_sign_list, dtype=float)
    da = np.asarray(d_align_list, dtype=float)
    st = np.asarray(steps_list, dtype=float)
    tm = np.asarray(times_list, dtype=float)
    out: Dict[str, float] = {
        "success_rate": float(sr),
        "mean_steps": float(np.mean(st)),
        "std_steps": float(np.std(st, ddof=1)) if st.size > 1 else 0.0,
        "mean_time_s": float(np.mean(tm)),
        "std_time_s": float(np.std(tm, ddof=1)) if tm.size > 1 else 0.0,
        "mean_dist_sign": float(np.mean(ds)),
        "median_dist_sign": float(np.median(ds)),
        "max_dist_sign": float(np.max(ds)),
        "mean_dist_align": float(np.mean(da)),
        "median_dist_align": float(np.median(da)),
        "max_dist_align": float(np.max(da)),
    }
    return out


def sweep_success_rate(
        *,
        solver_name: str,
        d: int,
        param_list: np.ndarray,
        trials_per_rate: int,
        trial_batch_size: int,
        passes: int,
        stop_err: float,
        power_iters: int,
        device: torch.device,
        dtype: torch.dtype,
        base_seed: Optional[int],
        out_csv: Path,
        out_fig: Optional[Path],
        no_stop_on_zero: bool,
        verbose: bool,
) -> List[Dict[str, Any]]:
    grid = nd_grid_descending(param_list)
    if solver_name not in SOLVER_REGISTRY:
        raise KeyError(f"unknown solver {solver_name!r}; known: {sorted(SOLVER_REGISTRY)}")

    use_cuda_batched = solver_name == "alg_orkm_cuda_success_rate_batched" and device.type == "cuda"
    rows: List[Dict[str, Any]] = []

    for r_idx, nd_rate in enumerate(grid, start=1):
        if verbose:
            print(f"[Main2-SR] [{r_idx}/{len(grid)}] n/d={nd_rate:.4f} (n={int(np.floor(nd_rate * d))})")

        d_sign_all: List[float] = []
        d_align_all: List[float] = []
        steps_all: List[int] = []
        times_all: List[float] = []
        successes = 0
        processed = 0
        n_batches = (trials_per_rate + trial_batch_size - 1) // trial_batch_size

        for b_idx in range(n_batches):
            bs = min(trial_batch_size, trials_per_rate - processed)
            seed_off = (hash((nd_rate, d, base_seed, b_idx)) % (2**31)) if base_seed is None else int(
                base_seed + r_idx * 100_003 + b_idx * 1_009
            )
            if use_cuda_batched:
                pack = run_trial_batch_cuda(
                    nd_rate=nd_rate,
                    d=d,
                    batch_size=bs,
                    passes=passes,
                    stop_err=stop_err,
                    device=device,
                    dtype=dtype,
                    seed_offset=seed_off,
                    power_iters=power_iters,
                )
                d_sign_all.extend(pack["d_sign_list"])
                d_align_all.extend(pack["d_align_list"])
                steps_all.extend(pack["steps_list"])
                times_all.extend(pack["times_list"])
                successes += int(pack["successes"])
                processed += bs
            else:
                wall_batch = 0.0
                for j in range(bs):
                    pack = run_trial_single_cpu(
                        nd_rate=nd_rate,
                        d=d,
                        passes=passes,
                        stop_err=stop_err,
                        device=device,
                        dtype=dtype,
                        seed_val=seed_off + j,
                        power_iters=power_iters,
                    )
                    d_sign_all.extend(pack["d_sign_list"])
                    d_align_all.extend(pack["d_align_list"])
                    steps_all.extend(pack["steps_list"])
                    times_all.extend(pack["times_list"])
                    successes += int(pack["successes"])
                    wall_batch += float(pack["wall_batch"])
                processed += bs
                pack = {"wall_batch": wall_batch}
            if verbose:
                print(
                    f"  batch {b_idx + 1}/{n_batches} wall={pack['wall_batch']:.3f}s "
                    f"processed={processed}/{trials_per_rate} successes_so_far={successes}"
                )

        stats = aggregate_trial_stats(
            d_sign_list=d_sign_all,
            d_align_list=d_align_all,
            steps_list=steps_all,
            times_list=times_all,
            successes=successes,
            trials=trials_per_rate,
        )
        row = {
            "solver_name": solver_name,
            "d": int(d),
            "nd_rate": float(nd_rate),
            "trials": int(trials_per_rate),
            "successes": int(successes),
            **{k: float(v) for k, v in stats.items()},
        }
        rows.append(row)

        if verbose:
            print(
                f"  -> success {successes}/{trials_per_rate} ({stats['success_rate'] * 100:.1f}%), "
                f"mean_steps={stats['mean_steps']:.2f}, mean_time_s={stats['mean_time_s']:.4f}"
            )

        if successes == 0 and not no_stop_on_zero:
            if verbose:
                print("[Main2-SR] successes==0: stopping grid sweep (remaining n/d skipped).")
            break

    save_success_stats_csv(rows, out_csv)
    if out_fig is not None and len(rows) > 0:
        plot_rows_to_fig(rows, out_fig)
        if verbose:
            print(f"[Main2-SR] figure saved: {out_fig}")
    return rows


def save_success_stats_csv(rows: List[Dict[str, Any]], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "solver_name",
        "d",
        "nd_rate",
        "trials",
        "successes",
        "success_rate",
        "mean_steps",
        "std_steps",
        "mean_time_s",
        "std_time_s",
        "mean_dist_sign",
        "median_dist_sign",
        "max_dist_sign",
        "mean_dist_align",
        "median_dist_align",
        "max_dist_align",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[Main2-SR] csv saved: {csv_path}")
    return csv_path


def plot_rows_to_fig(rows: List[Dict[str, Any]], out_path: Path) -> None:
    rs = sorted(rows, key=lambda x: float(x["nd_rate"]))
    nd = np.array([float(r["nd_rate"]) for r in rs], dtype=float)
    p = np.array([float(r["success_rate"]) for r in rs], dtype=float)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.0, 2.8), dpi=150)
    ax.plot(nd, p, marker="o", linewidth=1.2, markersize=3.5)
    ax.set_xlabel(r"$n/d$")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic success-rate experiment (main_2_sr).")
    p.add_argument("--solver-name", type=str, default="alg_orkm_cuda_success_rate_batched")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--trials-per-rate", type=int, default=16)
    p.add_argument("--trial-batch-size", type=int, default=8)
    p.add_argument("--passes", type=int, default=2000)
    p.add_argument("--stop-err", type=float, default=1e-5)
    p.add_argument("--power-iters", type=int, default=5)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "output" / "success_rate"),
    )
    p.add_argument("--no-save-fig", action="store_true")
    p.add_argument("--no-stop-on-zero", action="store_true", help="Run full grid even after first all-fail n/d.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny grid and few trials for pipeline test.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not bool(args.quiet)
    device = torch.device(str(args.device))
    dtype = torch.float64

    if args.solver_name == "alg_orkm_cuda_success_rate_batched" and device.type != "cuda":
        raise SystemExit("alg_orkm_cuda_success_rate_batched requires CUDA; pass --device cuda or use --solver-name alg_orkm")

    d_run = int(args.d)
    if bool(args.smoke):
        param_list = np.array([12.0, 10.0], dtype=float)
        trials_per_rate = 16
        trial_batch_size = 8
        passes = 80
        d_run = min(d_run, 16)
    else:
        param_list = default_param_list()
        trials_per_rate = int(args.trials_per_rate)
        trial_batch_size = int(args.trial_batch_size)
        passes = int(args.passes)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sr_{args.solver_name}_d{d_run}_{stamp}"
    out_csv = out_dir / f"{stem}.csv"
    out_fig = None if bool(args.no_save_fig) else out_dir / f"{stem}.pdf"

    grid = nd_grid_descending(param_list)
    if verbose:
        print("[Main2-SR] start")
        print(f"[Main2-SR] solver={args.solver_name}")
        print(f"[Main2-SR] d={d_run}")
        print(f"[Main2-SR] nd_grid(desc)={grid[: min(8, len(grid))]}{'...' if len(grid) > 8 else ''} (len={len(grid)})")
        print(f"[Main2-SR] trials_per_rate={trials_per_rate}, trial_batch_size={trial_batch_size}")
        print(f"[Main2-SR] passes={passes}, stop_err={float(args.stop_err)}")
        print(f"[Main2-SR] device={device}, dtype={dtype}")

    t0 = time.perf_counter()
    sweep_success_rate(
        solver_name=str(args.solver_name),
        d=d_run,
        param_list=param_list,
        trials_per_rate=trials_per_rate,
        trial_batch_size=trial_batch_size,
        passes=passes,
        stop_err=float(args.stop_err),
        power_iters=int(args.power_iters),
        device=device,
        dtype=dtype,
        base_seed=args.seed,
        out_csv=out_csv,
        out_fig=out_fig,
        no_stop_on_zero=bool(args.no_stop_on_zero),
        verbose=verbose,
    )
    if verbose:
        print(f"[Main2-SR] done wall={time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
