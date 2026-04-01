import os
import time

import numpy as np
import torch

from core.init_osi import init_osi, normalize_oct_signal
from core.octonion_align import right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import raw_distance
from core.ork_m import orkm_main
from utils.img_utils import plot_conv_curvs


def pick_device() -> torch.device:
    return torch.device("cpu")


def main() -> None:
    device = pick_device()
    torch.manual_seed(18)
    print("=== T18 smoke convergence check ===")
    print(f"selected device: {device}")

    d = 30
    n_over_d = 20
    n = int(n_over_d * d)
    total_iters = 600
    print(f"config: d={d}, n_over_d={n_over_d}, n={n}, total_iters={total_iters}")

    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
    y = intensity_measurements_explicit(A, x_true)
    x_est = init_osi(A, y, power_iters=5)

    hist_raw: list[float] = []
    hist_align: list[float] = []
    ln_vals: list[float] = []
    x_steps: list[int] = []
    t0 = time.perf_counter()
    t_last = t0

    for k in range(total_iters + 1):
        d_raw = raw_distance(x_true, x_est).item()
        d_align = right_aligned_distance(x_true, x_est).item()
        ln_dist = float(np.log(max(d_align, 1e-300)))
        t_now = time.perf_counter()
        dt_iter = t_now - t_last
        t_total = t_now - t0
        t_last = t_now

        hist_raw.append(d_raw)
        hist_align.append(d_align)
        ln_vals.append(ln_dist)
        x_steps.append(k)
        if k % 1 == 0:
            print(
                f"iter={k:03d}, d_raw={d_raw:.6e}, d_align={d_align:.6e}, "
                f"ln_dist={ln_dist:.6e}, dt={dt_iter:.3f}s, t={t_total:.3f}s"
            )
        if k < total_iters:
            x_est = orkm_main(A, y, x_est, max_iters=1)

    decrease_steps = sum(
        1 for i in range(1, len(hist_align)) if hist_align[i] <= hist_align[i - 1]
    )
    decrease_ratio = decrease_steps / total_iters
    overall_drop = hist_align[-1] < hist_align[0]

    print(f"align decrease ratio: {decrease_ratio:.2%}")
    print(
        f"align overall drop: {hist_align[0]:.6e} -> {hist_align[-1]:.6e} "
        f"({'YES' if overall_drop else 'NO'})"
    )

    ok = overall_drop
    print(f"T18 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T18 failed: no overall right-aligned distance drop")

    fig_dir = "results"
    os.makedirs(fig_dir, exist_ok=True)
    plot_file = os.path.join(fig_dir, "T18_owf_smoke_ln_orbit")
    data_series = [
        {
            "y": np.asarray(ln_vals, dtype=float),
            "label": "OWF",
            "style": "-",
            "color": "darkorange",
        },
    ]
    plot_conv_curvs(
        x=x_steps,
        data_series=data_series,
        title="OWF smoke: ln(orbit distance) vs iteration",
        xlabel="iteration",
        ylabel="ln(paper orbit distance)",
        figsize=(8, 4),
        filename=plot_file,
    )


if __name__ == "__main__":
    main()
