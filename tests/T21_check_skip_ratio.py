import torch

from algorithms.initializations.init_osi import init_osi, normalize_oct_signal
from core.octonion_align import right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import oct_array_norm, raw_distance
from core.ork_m import orkm_main


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    torch.manual_seed(21)
    print("=== T21 skip-ratio profile ===")
    print(f"device={device}, dtype=float64")

    d = 12
    n_over_d = 12
    n = int(n_over_d * d)
    max_iters = 8
    print(f"config: d={d}, n_over_d={n_over_d}, n={n}, max_iters={max_iters}")

    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
    y = intensity_measurements_explicit(A, x_true)

    cases: list[tuple[str, torch.Tensor]] = [
        ("random_init", torch.randn(d, 8, dtype=torch.float64, device=device)),
        ("init_osi", init_osi(A, y, power_iters=5)),
        ("near_zero_init", 1e-14 * torch.randn(d, 8, dtype=torch.float64, device=device)),
        ("small_norm_init", 1e-6 * torch.randn(d, 8, dtype=torch.float64, device=device)),
    ]

    for case_name, x0 in cases:
        x_est, info = orkm_main(A, y, x0, max_iters=max_iters, omega=1.0, return_info=True)
        init_norm = oct_array_norm(x0).item()
        final_raw = raw_distance(x_true, x_est).item()
        final_align = right_aligned_distance(x_true, x_est).item()

        print("----")
        print(f"case={case_name}")
        print(f"init_norm={init_norm:.6e}")
        print(f"total_updates={info['total_row_updates']}")
        print(f"total_skips={info['skip_count_total']}")
        print(f"skip_ratio={info['skip_ratio']:.6%}")
        print(f"epoch_skip_counts={info['epoch_skip_counts']}")
        print(f"epoch_row_update_counts={info['epoch_row_update_counts']}")
        print(f"final_d_raw={final_raw:.6e}")
        print(f"final_d_align={final_align:.6e}")

    print("T21 status: DONE (interpret skip ratios by initialization type)")


if __name__ == "__main__":
    main()
