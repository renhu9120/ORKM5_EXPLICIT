import torch

from core.octonion_inner import (
    amplitude_measurements_explicit,
    amplitude_measurements_fast,
    intensity_measurements_explicit,
    intensity_measurements_fast,
    row_inner_batch,
    row_inner_batch_fast,
    row_inner_explicit,
    row_inner_fast,
)
from core.octonion_ops import oct_mul


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max(torch.abs(a - b)).item()


def check_oct_mul_broadcast(device: torch.device) -> None:
    d, n = 11, 7
    p1 = torch.randn(d, 8, dtype=torch.float64, device=device)
    q1 = torch.randn(8, dtype=torch.float64, device=device)
    out1 = oct_mul(p1, q1)
    assert out1.shape == (d, 8)

    p2 = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    q2 = torch.randn(1, d, 8, dtype=torch.float64, device=device)
    out2 = oct_mul(p2, q2)
    assert out2.shape == (n, d, 8)

    q3 = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    out3 = oct_mul(p2, q3)
    assert out3.shape == (n, d, 8)


def main() -> None:
    device = pick_device()
    torch.manual_seed(30)
    print("=== T30 fast inner equivalence check ===")
    print(f"device={device}, dtype=float64")
    check_oct_mul_broadcast(device)
    print("oct_mul broadcasting: PASS")

    groups = 8
    tol = 1e-12
    err_single = 0.0
    err_batch = 0.0
    err_y = 0.0
    err_b = 0.0

    for g in range(groups):
        n = 10 + g
        d = 9 + g
        A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
        x = torch.randn(d, 8, dtype=torch.float64, device=device)

        s_single_old = row_inner_explicit(A[0], x)
        s_single_new = row_inner_fast(A[0], x)
        s_batch_old = row_inner_batch(A, x)
        s_batch_new = row_inner_batch_fast(A, x)
        y_old = intensity_measurements_explicit(A, x)
        y_new = intensity_measurements_fast(A, x)
        b_old = amplitude_measurements_explicit(A, x)
        b_new = amplitude_measurements_fast(A, x)

        err_single = max(err_single, max_abs_diff(s_single_old, s_single_new))
        err_batch = max(err_batch, max_abs_diff(s_batch_old, s_batch_new))
        err_y = max(err_y, max_abs_diff(y_old, y_new))
        err_b = max(err_b, max_abs_diff(b_old, b_new))

    print(f"max |s_single_old - s_single_new| = {err_single:.6e}")
    print(f"max |s_batch_old  - s_batch_new|  = {err_batch:.6e}")
    print(f"max |y_old        - y_new|        = {err_y:.6e}")
    print(f"max |b_old        - b_new|        = {err_b:.6e}")

    ok = max(err_single, err_batch, err_y, err_b) < tol
    print(f"T30 status: {'PASS' if ok else 'FAIL'} (tol={tol:.1e})")
    if not ok:
        raise RuntimeError("T30 failed")


if __name__ == "__main__":
    main()
