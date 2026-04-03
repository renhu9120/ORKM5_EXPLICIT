from __future__ import annotations

import csv
import os
import time
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.algs.alg_orkm import alg_orkm
from core.octonion_align import right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal, raw_distance
from core.octonion_sign import absolute_inner_product_similarity, sign_aligned_distance


def run_orkm_main_sign(metric_mode: str = "all") -> None:
    seed = 11
    d = 64
    n_over_d = 16
    n = n_over_d * d
    passes = 80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress_every = max(1, passes // 10)
    dist_tol = 1e-5

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print(
        f"[run_orkm_sign] config: device={device}, seed={seed}, d={d}, n={n}, "
        f"passes={passes}, metric_mode={metric_mode}"
    )
    print(
        "[run_orkm_sign] per-iter logs (verbose): dist_align / log(dist_align) "
        "vs dist_sign / log(dist_sign) — right_aligned_distance vs sign_aligned_distance."
    )

    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
    y = intensity_measurements_explicit(A, x_true)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    zt, history = alg_orkm(
        A=A,
        y=y,
        T=passes,
        passes=passes,
        power_iters=1,
        x_true_proc=x_true,
        verbose=True,
        progress_every=progress_every,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0

    final_raw = float(raw_distance(x_true, zt).item())
    final_align = float(right_aligned_distance(x_true, zt).item())
    final_pm = float(sign_aligned_distance(x_true, zt).item())
    final_absip = float(absolute_inner_product_similarity(x_true, zt).item())
    print(
        f"[alg_orkm] final_raw={final_raw:.6e}, final_align={final_align:.6e}, "
        f"final_pm={final_pm:.6e}, final_absip={final_absip:.6e}, "
        f"success_align={final_align <= dist_tol}, success_pm={final_pm <= dist_tol}, wall={wall:.3f}s"
    )

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output", "main_1_conv_sign")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "iter_metrics.csv")
    fig_path = os.path.join(out_dir, "conv_metrics.png")

    iters = [int(k) for k in history.get("orbit_log_iters", [])]
    raw_vals = [float(v) for v in history.get("orbit_log_raw", [])]
    align_vals = [float(v) for v in history.get("orbit_log_align", [])]
    pm_vals = [float(v) for v in history.get("orbit_log_pm", [])]
    absip_vals = [float(v) for v in history.get("orbit_log_absip", [])]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "d", "iter", "dist_raw", "dist_align", "dist_pm", "absip"])
        for i, dist_raw_i, dist_align_i, dist_pm_i, absip_i in zip(
            iters, raw_vals, align_vals, pm_vals, absip_vals
        ):
            writer.writerow([seed, d, i, dist_raw_i, dist_align_i, dist_pm_i, absip_i])

    if len(iters) > 0:
        plt.figure(figsize=(8, 4))
        if metric_mode in ("raw", "all"):
            plt.plot(iters, raw_vals, label="raw_distance", linewidth=1.5)
        if metric_mode in ("align", "all"):
            plt.plot(iters, align_vals, label="right_aligned_distance", linewidth=1.5)
        if metric_mode in ("sign", "all"):
            plt.plot(iters, pm_vals, label="sign_aligned_distance", linewidth=1.5)
        if metric_mode == "all":
            plt.plot(iters, [1.0 - a for a in absip_vals], label="1-absip", linewidth=1.2, linestyle="--")
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("metric value (log scale)")
        plt.title("ORKM convergence under multiple metrics")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

    print(f"[run_orkm_sign] saved: csv={csv_path}, fig={fig_path}")


if __name__ == "__main__":
    run_orkm_main_sign(metric_mode="all")
