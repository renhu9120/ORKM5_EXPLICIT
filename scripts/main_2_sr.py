import os

import csv

from algorithms.algs.alg_orkm import alg_orkm
from core.octonion_align import right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal
from utils.img_utils import plot_sr_curvs
import time
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# ==========================================================
# Register all solver functions here
# ==========================================================

SOLVER_MAP = {
    "alg_orkm": alg_orkm,
}


def _standardize_solver_output(
        solver_result,
        *,
        d: int,
        device: torch.device,
        dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict]:
    """
    Normalize heterogeneous solver outputs into:
      - x_est: Tensor with shape (d, 8)
      - history: dict
    """
    history: Dict = {}
    x_est = None

    # Common patterns:
    #   1) solver(...) -> x_est
    #   2) solver(...) -> (x_est, history)
    #   3) solver(...) -> {"x_est": ..., "history": ...}
    if isinstance(solver_result, tuple):
        if len(solver_result) >= 1:
            x_est = solver_result[0]
        if len(solver_result) >= 2 and isinstance(solver_result[1], dict):
            history = solver_result[1]
    elif isinstance(solver_result, dict):
        for key in ("x_est", "x_final", "x", "zT", "z_final", "solution"):
            if key in solver_result:
                x_est = solver_result[key]
                break
        hist_obj = solver_result.get("history", {})
        if isinstance(hist_obj, dict):
            history = hist_obj
    else:
        x_est = solver_result

    if x_est is None:
        raise ValueError(
            "Cannot parse solver output: expected Tensor, (Tensor, history), "
            "or dict containing one of keys {'x_est','x_final','x','zT','z_final','solution'}."
        )

    x_est = torch.as_tensor(x_est, dtype=dtype, device=device)

    # Accept a few common shapes and convert to (d, 8).
    if x_est.ndim == 3 and x_est.shape[1:] == (1, 8):
        x_est = x_est[:, 0, :]
    elif x_est.ndim == 3 and x_est.shape[0:2] == (1, d) and x_est.shape[2] == 8:
        x_est = x_est[0, :, :]
    elif x_est.ndim == 1 and x_est.numel() == d * 8:
        x_est = x_est.reshape(d, 8)
    elif x_est.ndim == 2 and x_est.shape == (8, d):
        x_est = x_est.transpose(0, 1).contiguous()
    elif x_est.ndim == 2 and x_est.shape == (d, 8):
        pass
    else:
        raise ValueError(
            f"Unsupported solution shape {tuple(x_est.shape)}; expected one of "
            f"(d,8), (d,1,8), (1,d,8), (8,d), or flat length d*8 with d={d}."
        )

    return x_est, history


def _infer_effective_steps(history: Dict, fallback_steps: int) -> int:
    """
    Infer executed steps/iterations from history with robust fallbacks.
    """
    if not isinstance(history, dict) or len(history) == 0:
        return int(fallback_steps)

    # Prefer explicit per-iteration counters when available.
    if isinstance(history.get("epoch_row_update_counts", None), list):
        return int(len(history["epoch_row_update_counts"]))

    iters = history.get("iter", None)
    if isinstance(iters, list) and len(iters) > 0:
        # Many implementations log iter=[0,1,2,...,k].
        if iters[0] == 0:
            return int(max(0, len(iters) - 1))
        return int(len(iters))

    return int(fallback_steps)


def _effective_steps_from_passes(
        solver_fn: Callable,
        *,
        passes: float,
        n: int,
        block_size: int = 1,
) -> int:
    """
    Convert pass budget into algorithm steps.
    - OWF / ORKM: T = round(passes * n)
    - OBKM      : T = round(passes * n / block_size)
    """
    name = solver_fn.__name__
    if name == "alg_obkm":
        return int(round(float(passes) * float(n) / float(block_size)))
    return int(round(float(passes) * float(n)))


def _load_success_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a success-vs-nd CSV produced by `save_success_stats_csv`.

    Returns
    -------
    nd : np.ndarray shape (N,)
    p  : np.ndarray shape (N,)  # success_rate in [0,1]
    """
    nd_vals, p_vals, trials, successes = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nd = float(row["nd_rate"])
            # prefer recomputing from successes/trials if present and consistent
            if row.get("trials", "") != "" and row.get("successes", "") != "":
                t = float(row["trials"])
                s = float(row["successes"])
                p = (s / t) if t > 0 else 0.0
            else:
                p = float(row["success_rate"])
            nd_vals.append(nd)
            p_vals.append(p)
    # sort by nd
    idx = np.argsort(np.asarray(nd_vals, dtype=float))
    nd = np.asarray(nd_vals, dtype=float)[idx]
    p = np.asarray(p_vals, dtype=float)[idx]
    return nd, p


def plot_multi_success_vs_nd(
        csv_paths: List[str],
        *,
        labels: Optional[List[str]] = None,
        out_path: Optional[str] = None,
        title: Optional[str] = None,
        y_as_percentage: bool = True,
        figsize: Tuple[float, float] = (3.5, 2.6),
        dpi: int = 300,
        usetex: bool = False,
        marker: str = "o",
        linewidth: float = 1.6,
        markersize: float = 4.0,
        legend_loc: str = "best",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple success-rate vs n/d curves (no confidence bands).

    Parameters
    ----------
    csv_paths : list of str
        Paths to CSV files produced by `save_success_stats_csv`.
    labels : list of str, optional
        Legend labels for each curve; if None, uses the CSV base filename stem.
    out_path : str, optional
        If provided, save the figure to this path (e.g., 'figs/success_multi.pdf').
    y_as_percentage : bool
        Show y-axis as percentage (0–100%). If False, show raw rate (0–1).
    figsize, dpi, usetex, marker, linewidth, markersize, legend_loc
        Presentation options suitable for papers.

    Returns
    -------
    (fig, ax)
    """
    if labels is not None and len(labels) != len(csv_paths):
        raise ValueError("`labels` must have the same length as `csv_paths` (or be None).")

    # minimal paper-ish style (no seaborn)
    plt.rcParams.update({
        "text.usetex": usetex,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
    })

    fig, ax = plt.subplots(figsize=figsize)

    for i, csv_path in enumerate(csv_paths):
        nd, p = _load_success_csv(csv_path)
        label = labels[i] if labels is not None else Path(csv_path).stem
        ax.plot(nd, p, marker=marker, linewidth=linewidth, markersize=markersize, label=label)

    ax.set_xlabel(r"$n/d$")
    ax.set_ylabel("Success rate")

    ax.set_ylim(-0.05, 1.1)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    if title:
        ax.set_title(title)
    ax.legend(loc=legend_loc)
    fig.tight_layout()

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"[plot] saved to: {out}")
    plt.show()

    return fig, ax


def plot_success_vs_nd(rows: List[Dict[str, float]],
                       out_path: Optional[str] = None,
                       *,
                       title: Optional[str] = None,
                       figsize: Tuple[float, float] = (3.5, 2.6),
                       dpi: int = 300,
                       usetex: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot success rate vs n/d from `rows` returned by sweep_success_vs_nd_rate(...).

    Parameters
    ----------
    rows : list of dict
        Each dict must contain keys: 'nd_rate', 'trials', 'successes'.
        (If it also has 'success_rate', it's ignored and recomputed from successes/trials.)
    out_path : str or None
        If given (e.g., 'fig_success_nd.pdf' or '.png' / '.svg'), figure will be saved.
    title : str or None
        Optional plot title.
    show_ci : bool
        Whether to plot Wilson confidence interval bands.
    ci_level : float
        Confidence level for CI (e.g., 0.95).
    y_as_percentage : bool
        If True, y-axis shows % (0–100%); else shows rate (0–1).
    figsize : (w, h) in inches
        Figure size suitable for paper columns.
    dpi : int
        Dots per inch for raster formats; PDF/SVG are vector anyway.
    usetex : bool
        If True, enable LaTeX rendering (requires a TeX installation). Otherwise uses mathtext.

    Returns
    -------
    (fig, ax) : matplotlib figure and axes objects.
    """
    # sort by nd_rate
    rows_sorted = sorted(rows, key=lambda r: r["nd_rate"])
    nd = np.array([r["nd_rate"] for r in rows_sorted], dtype=float)
    trials = np.array([r["trials"] for r in rows_sorted], dtype=int)
    successes = np.array([r["successes"] for r in rows_sorted], dtype=int)
    p = np.divide(successes, trials, out=np.zeros_like(successes, dtype=float), where=trials > 0)

    # Minimal paper-ish style (no seaborn)
    plt.rcParams.update({
        "text.usetex": usetex,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # Main curve
    ax.plot(nd, p, marker="o", linewidth=1.5, markersize=4)

    # Axes labels
    ax.set_xlabel(r"$n/d$")
    ax.set_ylabel("Success rate")

    # Percent formatting if requested

    ax.set_ylim(-0.05, 1.1)

    # Ticks and grid for readability
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # Optional title
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        # Optionally also save a second vector/raster format
        # e.g., if out_path.endswidth(".pdf"), also save .png:
        # stem, ext = os.path.splitext(out_path)
        # fig.savefig(stem + ".png", bbox_inches="tight")

    return fig, ax


def sweep_success_vs_nd_rate(
        nd_grid: List[float],
        *,
        trials_per_rate: int = 50,
        T: int = 1500,
        d: int = 64,
        solver_fn: Optional[Callable[..., Tuple[List[torch.Tensor], List[torch.Tensor]]]] = alg_orkm,
        base_seed: Optional[int] = None,
        verbose: bool = True,
        stop_err: float = 1e-5,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float64,
) -> List[Dict[str, float]]:
    """
    Sweep n/d over [nd_start, nd_end] with step nd_step, and for each rate run
    `trials_per_rate` Monte Carlo trials using `run_single_trial`. For each rate
    report:
        - success_rate = (#successes / trials_per_rate)
        - mean_steps   = average #iterations among successful runs (NaN if none)
        - mean_time_s  = average wall time (seconds) among successful runs (NaN if none)

    Parameters
    ----------
    stop_err : float, optional
        Paper orbit-distance threshold: passed to solvers for early stopping and used
        as the success criterion in ``run_single_trial`` (default ``1e-5`` for SR).

    Returns
    -------
    rows : List[dict]
        One dict per n/d rate with keys:
        {'nd_rate','trials','successes','success_rate','mean_steps','std_steps',
         'mean_time_s','std_time_s'}
    """
    # build the grid of n/d rates, inclusive of nd_end (with a tiny epsilon for float)

    rows: List[Dict[str, float]] = []

    for r_idx, nd_rate in enumerate(nd_grid, start=1):
        succ_flags: List[int] = []
        steps_list: List[int] = []
        times_list: List[float] = []

        if verbose:
            print(f"\n=== [{r_idx}/{len(nd_grid)}] n/d = {nd_rate:.3f} | trials = {trials_per_rate} ===")

        for t_idx in range(trials_per_rate):
            # Optional deterministic seeding per (rate, trial)
            if base_seed is not None:
                seed_val = int(base_seed + r_idx * 10_000 + t_idx)
                torch.manual_seed(seed_val)
                np.random.seed(seed_val)

            succ, steps, tsec = run_single_trial(
                n_over_d=nd_rate,
                d=d,
                passes=T,
                stop_err=stop_err,
                solver_fn=solver_fn,
                device=device,
                dtype=dtype,
            )
            succ_flags.append(succ)
            if succ == 1:
                steps_list.append(steps)
                times_list.append(tsec)

            if verbose and ((t_idx + 1) % max(1, trials_per_rate // 10) == 0):
                print(f"{solver_fn.__name__} - {d} - trial {t_idx + 1:>3}/{trials_per_rate}: "
                      f"{'OK' if succ == 1 else 'FAIL'}"
                      f"{'' if succ == 0 else f', steps={steps}, time={tsec:.3f}s'}")

        successes = int(sum(succ_flags))
        trials = trials_per_rate
        success_rate = successes / trials if trials > 0 else float("nan")

        # stats on successful runs only
        if successes > 0:
            mean_steps = float(np.mean(steps_list))
            std_steps = float(np.std(steps_list, ddof=1)) if len(steps_list) > 1 else 0.0
            mean_time = float(np.mean(times_list))
            std_time = float(np.std(times_list, ddof=1)) if len(times_list) > 1 else 0.0
        else:
            mean_steps = float("nan")
            std_steps = float("nan")
            mean_time = float("nan")
            std_time = float("nan")

        row = {
            "nd_rate": nd_rate,
            "trials": trials,
            "successes": successes,
            "success_rate": success_rate,
            "mean_steps": mean_steps,
            "std_steps": std_steps,
            "mean_time_s": mean_time,
            "std_time_s": std_time,
        }
        rows.append(row)

        if verbose:
            print(f"--> n/d={nd_rate:.3f} | success={successes}/{trials} "
                  f"({success_rate * 100:.1f}%), "
                  f"mean_steps={mean_steps:.2f}, mean_time={mean_time:.3f}s")

        if successes == 0:
            if verbose:
                print("\n*** Success rate reached 0. Filling remaining nd_rate entries with zeros. ***")

            for nd_remain in nd_grid[r_idx:]:
                rows.append({
                    "nd_rate": nd_remain,
                    "trials": trials_per_rate,
                    "successes": 0,
                    "success_rate": 0.0,
                    "mean_steps": float("nan"),
                    "std_steps": float("nan"),
                    "mean_time_s": float("nan"),
                    "std_time_s": float("nan"),
                })
            break
    return rows


def run_single_trial(n_over_d: float = 7.5, d: int = 4, passes: int = 2000, stop_err: float = 1e-5,
                     solver_fn: Optional[Callable[..., Tuple[List[torch.Tensor], List[torch.Tensor]]]] = None,
                     device: Optional[str] = None,
                     dtype: torch.dtype = torch.float64,
                     ):



    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # generate data
    n = int(np.floor(n_over_d * d))
    A = torch.randn(n, d, 8, dtype=dtype, device=dev)
    x_true = normalize_oct_signal(torch.randn(d, 8, dtype=dtype, device=dev))
    y = intensity_measurements_explicit(A, x_true)

    T=passes
    solver_kwargs = {
        "A": A,
        "y": y,
        "T": T,
        "x_true_proc": x_true,
        "stop_err": stop_err,
    }

    # start trial
    start_time = time.perf_counter()
    with torch.inference_mode():
        solver_result = solver_fn(**solver_kwargs)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    x_final, history = _standardize_solver_output(
        solver_result,
        d=d,
        device=dev,
        dtype=dtype,
    )
    steps_done = _infer_effective_steps(history, fallback_steps=T)
    dist = right_aligned_distance(x_true, x_final).item()

    success = 1 if dist <= float(stop_err) else 0
    end_time = time.perf_counter()
    return success, steps_done, end_time - start_time


# ----------------------------
# Save stats to CSV
# ----------------------------
def save_success_stats_csv(
        rows: List[Dict[str, float]],
        csv_path: str,
) -> Path:
    """
    Save stats (rows from sweep) to CSV.
    Columns: nd_rate, trials, successes, success_rate, mean_steps, std_steps, mean_time_s, std_time_s
    """
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "nd_rate", "trials", "successes", "success_rate",
        "mean_steps", "std_steps", "mean_time_s", "std_time_s",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["nd_rate"]):
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[csv] saved to: {out}")
    return out


def run_sr_trial_call(
        d: int = 64,
        solver_fn=alg_orkm,
        *,
        trials_per_rate: int = 10,
        passes: int = 2000,
        stop_err: float = 1e-5,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float64,
):
    # range1 = np.arange(3, 3.9, 1)
    # range2 = np.arange(4, 8.9, 1)
    range3 = np.arange(12, 16.01, 0.2)
    # param_list = np.concatenate([range1, range2, range3])
    param_list = range3
    grid = param_list[::-1].tolist()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[config] device={device}, dtype={dtype}")

    stats = sweep_success_vs_nd_rate(
        nd_grid=grid,
        trials_per_rate=trials_per_rate,
        T=passes,
        d=d,
        solver_fn=solver_fn,  # ← 子进程传入的 solver_fn
        stop_err=stop_err,
        device=device,
        dtype=dtype,
    )

    filename = f"{solver_fn.__name__}"
    csv_path = "../output/data_sr_" + filename + "_d_" + str(d) + ".csv"

    save_success_stats_csv(
        stats,
        csv_path=csv_path
    )

    # # 假设你已经得到 rows = sweep_success_vs_nd_rate(...)
    out_path = "../output/fig_sr_" + filename + ".pdf"  # 改成 .svg / .png 皆可
    #
    fig, ax = plot_success_vs_nd(
        stats,
        out_path=out_path,
        title=None,  # 论文里一般不需要标题
        figsize=(3.5, 2.6),
        dpi=300,
        usetex=False,  # 若你的环境装了 TeX，可改 True
    )

    # send email to self
    # send_email_with_attachment(subject=csv_path, body='result', filename=csv_path, sender='mosheng_88@163.com',
    #                            password='GVfC8utuJiLPSc8G', recipient='renhu@gbu.edu.cn')


def draw_all_sr_data():
    d = 64
    alg_list = [
        {"file": "output/data_sr_alg_qwf_d_" + str(d) + ".csv", "label": "QWF", "linestyle": "-x", "color": "purple"},
        {"file": "output/data_sr_alg_qrwf_d_" + str(d) + ".csv", "label": "QRWF", "linestyle": "-s", "color": "pink"},
        {"file": "output/data_sr_alg_qraf_d_" + str(d) + ".csv", "label": "QRAF", "linestyle": "-o", "color": "black"},
        {"file": "output/data_sr_alg_qaraf_d_" + str(d) + ".csv", "label": "QARAF", "linestyle": "-*",
         "color": "orange"},
        {"file": "output/data_sr_alg_qn_d_" + str(d) + ".csv", "label": "QN", "linestyle": "-", "color": "red"},
        {"file": "output/data_sr_alg_qgn_d_" + str(d) + ".csv", "label": "QGN", "linestyle": "-.",
         "color": "blue"},
    ]

    plot_sr_curvs(
        alg_list,
        x_key="nd_rate",
        y_key="success_rate",
        xlabel="n/d of d = " + str(d),
        ylabel="Success rate",
        save_name="output/fig_success_rate_" + str(d) + ".pdf",
        figsize=(8, 4)
    )

    plot_sr_curvs(
        alg_list,
        x_key="nd_rate",
        y_key="mean_steps",
        xlabel="n/d of d = " + str(d),
        ylabel="Mean steps",
        save_name="output/fig_mean_steps_" + str(d) + ".pdf",
        figsize=(8, 4)
    )

    plot_sr_curvs(
        alg_list,
        x_key="nd_rate",
        y_key="mean_time_s",
        xlabel="n/d of d = " + str(d),
        ylabel="Mean time",
        save_name="output/fig_mean_time_s_" + str(d) + ".pdf",
        figsize=(8, 4)
    )


def run_sr_single_experiment(d: int, solver_name: str):
    """
    子进程执行一次完整的 sweep（多个 nd_rate + 多个 trial）。
    solver_name 为字符串，通过 SOLVER_MAP 查找真实函数。
    """

    solver_fn = SOLVER_MAP[solver_name]
    print(f"[START] d={d} | solver={solver_name}")

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    if auto_device != "cuda":
        print("[warn] CUDA unavailable; fallback to CPU.")

    run_sr_trial_call(
        d=d,
        solver_fn=solver_fn,
        trials_per_rate=10,
        passes=2000,
        stop_err=1e-5,
        device=auto_device,
        dtype=torch.float64,
    )

    print(f"[DONE ] d={d} | solver={solver_name}")


if __name__ == '__main__':
    d_list = [64]
    solver_list = ["alg_orkm"]

    # ====== 单程循环，每次运行一个 solver × d ======
    for d in d_list:
        for solver_fn in solver_list:
            run_sr_single_experiment(d, solver_fn)

    print("\n=== ALL EXPERIMENTS COMPLETED ===")

    # draw_all_sr_data()
