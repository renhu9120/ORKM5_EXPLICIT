import torch

from algorithms.initializations.init_osi import normalize_oct_signal
from core.octonion_inner import intensity_measurements_explicit, row_inner_explicit
from core.octonion_ops import oct_abs
from discards.ork_m import orkm_single_row_update


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    torch.manual_seed(17)
    print("=== T17 single update direction check ===")
    print(f"selected device: {device}")

    n, d = 16, 10
    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
    y = intensity_measurements_explicit(A, x_true)
    x0 = torch.randn(d, 8, dtype=torch.float64, device=device)

    l = 0
    s_before = row_inner_explicit(A[l], x0)
    b_l = torch.sqrt(torch.clamp(y[l], min=0.0))
    before = torch.abs(oct_abs(s_before) - b_l)

    x1 = orkm_single_row_update(x0, A[l], y[l])
    s_after = row_inner_explicit(A[l], x1)
    after = torch.abs(oct_abs(s_after) - b_l)

    print(f"before: {before.item():.6e}")
    print(f"after:  {after.item():.6e}")

    ok = after.item() < before.item()
    print(f"T17 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T17 failed: single-row residual did not decrease")


if __name__ == "__main__":
    main()
