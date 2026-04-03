from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.algs.alg_orkm import alg_orkm
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal


def run_t38(
    *,
    seeds: list[int] | None = None,
    d: int = 30,
    n_over_d: int = 20,
    passes: int = 20,
    out_dir: str = "output/t38",
) -> None:
    if seeds is None:
        seeds = [10, 11]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = n_over_d * d

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    iters_csv = out / "summary_iters.csv"
    trials_csv = out / "summary_trials.csv"
    fig_path = out / f"curve_mean_d{d}.png"

    rows_iters: list[list[float | int]] = []
    rows_trials: list[list[float | int]] = []
    mean_curves: dict[str, list[float]] = {"raw": [], "align": [], "pm": [], "absip": []}
    ref_iters: list[int] | None = None

    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
        x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
        y = intensity_measurements_explicit(A, x_true)

        zt, history = alg_orkm(
            A=A,
            y=y,
            T=passes,
            passes=passes,
            power_iters=1,
            x_true_proc=x_true,
            verbose=False,
            progress_every=max(1, passes // 5),
        )
        _ = zt

        iters = [int(v) for v in history.get("orbit_log_iters", [])]
        raw = [float(v) for v in history.get("orbit_log_raw", [])]
        align = [float(v) for v in history.get("orbit_log_align", [])]
        pm = [float(v) for v in history.get("orbit_log_pm", [])]
        absip = [float(v) for v in history.get("orbit_log_absip", [])]

        if ref_iters is None:
            ref_iters = iters
            mean_curves["raw"] = [0.0] * len(iters)
            mean_curves["align"] = [0.0] * len(iters)
            mean_curves["pm"] = [0.0] * len(iters)
            mean_curves["absip"] = [0.0] * len(iters)

        for i, (k, r, a, p, s) in enumerate(zip(iters, raw, align, pm, absip)):
            rows_iters.append([seed, d, k, r, a, p, s, float("nan")])
            mean_curves["raw"][i] += r
            mean_curves["align"][i] += a
            mean_curves["pm"][i] += p
            mean_curves["absip"][i] += s

        final_align = align[-1]
        final_pm = pm[-1]
        rows_trials.append(
            [seed, d, passes, raw[-1], final_align, final_pm, absip[-1], int(final_align < 1e-5), int(final_pm < 1e-5)]
        )

    m = float(len(seeds))
    for key in mean_curves:
        mean_curves[key] = [v / m for v in mean_curves[key]]

    with iters_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "d", "iter", "dist_raw", "dist_align", "dist_pm", "absip", "residual"])
        w.writerows(rows_iters)

    with trials_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "d", "max_iter", "final_raw", "final_align", "final_pm", "final_absip", "success_align", "success_pm"])
        w.writerows(rows_trials)

    if ref_iters is None:
        raise RuntimeError("No iteration logs were collected.")

    plt.figure(figsize=(7, 4))
    plt.plot(ref_iters, mean_curves["raw"], label="raw")
    plt.plot(ref_iters, mean_curves["align"], label="align")
    plt.plot(ref_iters, mean_curves["pm"], label="pm")
    plt.yscale("log")
    plt.xlabel("iter")
    plt.ylabel("distance")
    plt.title(f"T38 mean curve (d={d}, seeds={len(seeds)})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    print("=== T38 compare align vs sign on ORKM ===")
    print(f"device={device}, d={d}, n/d={n_over_d}, passes={passes}, seeds={seeds}")
    print(f"saved: {iters_csv}")
    print(f"saved: {trials_csv}")
    print(f"saved: {fig_path}")


if __name__ == "__main__":
    run_t38()
