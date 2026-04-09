import torch

from algorithms.initializations.init_osi import init_osi, normalize_oct_signal
from core.octonion_align import right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from discards.ork_m import orkm_main


def pick_device() -> torch.device:
    return torch.device("cpu")


def main() -> None:
    device = pick_device()
    torch.manual_seed(19)
    print("=== T19 small success rate check ===")
    print(f"selected device: {device}")

    trials = 20
    d = 3
    n_over_d = 10
    n = int(n_over_d * d)
    max_iters = 40
    success_threshold = 1e-2
    target_rate = 0.60
    print(
        f"config: trials={trials}, d={d}, n_over_d={n_over_d}, n={n}, "
        f"max_iters={max_iters}, init=init_osi"
    )

    success = 0
    for t in range(trials):
        x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
        A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
        y = intensity_measurements_explicit(A, x_true)
        x0 = init_osi(A, y, power_iters=5)

        x_est = orkm_main(A, y, x0, max_iters=max_iters)
        d_align = right_aligned_distance(x_true, x_est).item()
        is_success = d_align < success_threshold
        success += int(is_success)
        print(
            f"trial={t + 1:02d}, d_align={d_align:.6e}, "
            f"success={'YES' if is_success else 'NO'}"
        )

    rate = success / trials
    print(f"success rate: {rate * 100:.2f}%")

    ok = rate > target_rate
    print(f"T19 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(
            f"warning: success rate {rate * 100:.2f}% <= {target_rate * 100:.2f}% "
            "(kept for analysis)."
        )


if __name__ == "__main__":
    main()
