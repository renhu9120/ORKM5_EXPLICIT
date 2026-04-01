import statistics

import torch

from core.init_osi import init_osi, normalize_oct_signal
from core.octonion_align import right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from core.ork_m import orkm_main


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T22 smoke multi-seed consistency ===")
    print(f"device={device}, dtype=float64")

    d = 20
    n_over_d = 20
    n = int(n_over_d * d)
    total_iters = 40
    seed_list = list(range(10))
    print(
        f"config: d={d}, n_over_d={n_over_d}, n={n}, total_iters={total_iters}, "
        f"seeds={seed_list}"
    )

    final_vals: list[float] = []
    success_flags: list[bool] = []
    ratios: list[float] = []

    for seed in seed_list:
        torch.manual_seed(seed)
        A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
        x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
        y = intensity_measurements_explicit(A, x_true)
        x_est = init_osi(A, y, power_iters=5)

        initial_align = right_aligned_distance(x_true, x_est).item()
        for _ in range(total_iters):
            x_est = orkm_main(A, y, x_est, max_iters=1, omega=1.0)
        final_align = right_aligned_distance(x_true, x_est).item()

        drop_ratio = final_align / max(initial_align, 1e-300)
        success = (final_align < 1e-3) or (final_align < initial_align * 1e-2)
        final_vals.append(final_align)
        success_flags.append(success)
        ratios.append(drop_ratio)

        print(
            f"seed={seed:02d}, initial_d_align={initial_align:.6e}, "
            f"final_d_align={final_align:.6e}, drop_ratio={drop_ratio:.6e}, "
            f"success={'YES' if success else 'NO'}"
        )

    success_count = sum(int(v) for v in success_flags)
    success_ratio = success_count / len(seed_list)
    median_final = statistics.median(final_vals)
    median_ratio = statistics.median(ratios)

    print("---- overall summary ----")
    print(f"success_count={success_count}/{len(seed_list)}")
    print(f"success_ratio={success_ratio:.2%}")
    print(f"median_final_d_align={median_final:.6e}")
    print(f"median_drop_ratio={median_ratio:.6e}")
    print("T22 status: DONE (compare with target success ratio >= 80%)")


if __name__ == "__main__":
    main()
