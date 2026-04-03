import os
import time

import numpy as np
import torch

from algorithms.algs.alg_orkm import alg_orkm
from core.octonion_metric import normalize_oct_signal
from core.octonion_align import right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from utils.img_utils import plot_conv_curvs


def run_orkm_main() -> None:
    seed = 212
    d = 64
    n_over_d = 20
    n = n_over_d * d
    passes = 80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist_tol = 1e-5
    progress_every = max(1, passes // 10)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print(
        f"[run_orkm] config: device={device}, seed={seed}, d={d}, n={n}, "
        f"passes={passes}, dist_tol={dist_tol:.1e}"
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
        # stop_err=dist_tol,
        verbose=True,
        progress_every=progress_every,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    dist = right_aligned_distance(x_true, zt).item()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    plot_file = os.path.join(out_dir, "smoke_alg_orkm")
    iters = [int(k) for k in history.get("orbit_log_iters", [])]
    ln_vals = [float(v) for v in history.get("orbit_log_ln", [])]
    x_steps = np.asarray(iters, dtype=float)
    data_series = [
        {"y": np.asarray(ln_vals, dtype=float), "label": "ORKM", "style": "-", "color": "darkorange"},
    ]
    if len(iters) > 0 and len(ln_vals) > 0:
        plot_conv_curvs(
            x=x_steps,
            data_series=data_series,
            title="ORKM: ln(dist) vs passes",
            xlabel=f"passes for d={d} and n={n}",
            ylabel="ln(dist)",
            figsize=(8, 4),
            filename=plot_file,
        )
        plot_path = f"{plot_file}.pdf"
    else:
        plot_path = "N/A (no orbit history)"
    print(
        f"[alg_orkm] final_dist={dist:.6e}, success={bool(float(dist) <= dist_tol)}, "
        f"iters={len(history.get('iter', []))}, wall={wall:.3f}s, plot={plot_path}"
    )


if __name__ == "__main__":
    run_orkm_main()
