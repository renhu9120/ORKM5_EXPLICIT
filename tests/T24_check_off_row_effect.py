import statistics

import torch

from algorithms.initializations.init_osi import normalize_oct_signal
from core.octonion_inner import intensity_measurements_explicit, row_inner_explicit
from core.octonion_ops import oct_abs
from discards.ork_m import orkm_single_row_update


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    torch.manual_seed(24)
    print("=== T24 off-row coupling profile ===")
    print(f"device={device}, dtype=float64")

    n, d = 24, 10
    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
    y = intensity_measurements_explicit(A, x_true)
    x0 = torch.randn(d, 8, dtype=torch.float64, device=device)
    selected_l = 3
    print(f"selected_row={selected_l}, omega=1.0")

    x1, info = orkm_single_row_update(x0, A[selected_l], y[selected_l], omega=1.0, return_info=True)
    print(f"selected_row_skipped={info['skipped']}, skip_reason={info['skip_reason']}")

    residual_before: list[float] = []
    residual_after: list[float] = []
    for m in range(n):
        b_m = torch.sqrt(torch.clamp(y[m], min=0.0))
        s_m_before = row_inner_explicit(A[m], x0)
        s_m_after = row_inner_explicit(A[m], x1)
        r_before = torch.abs(oct_abs(s_m_before) - b_m).item()
        r_after = torch.abs(oct_abs(s_m_after) - b_m).item()
        residual_before.append(r_before)
        residual_after.append(r_after)

    selected_before = residual_before[selected_l]
    selected_after = residual_after[selected_l]
    off_before = [residual_before[m] for m in range(n) if m != selected_l]
    off_after = [residual_after[m] for m in range(n) if m != selected_l]
    off_delta = [a - b for a, b in zip(off_after, off_before)]

    off_median_before = statistics.median(off_before)
    off_median_after = statistics.median(off_after)
    off_max_increase = max(off_delta)
    off_inc_count = sum(1 for v in off_delta if v > 0)
    off_dec_count = sum(1 for v in off_delta if v < 0)
    off_eq_count = len(off_delta) - off_inc_count - off_dec_count

    print(f"selected_row_before={selected_before:.6e}")
    print(f"selected_row_after={selected_after:.6e}")
    print(f"off_row_median_before={off_median_before:.6e}")
    print(f"off_row_median_after={off_median_after:.6e}")
    print(f"off_row_max_increase={off_max_increase:.6e}")
    print(
        f"off_row_count_decrease={off_dec_count}, "
        f"off_row_count_increase={off_inc_count}, off_row_count_equal={off_eq_count}"
    )
    print("T24 status: DONE (descriptive structural check)")


if __name__ == "__main__":
    main()
