import time
from collections.abc import Sequence

import torch

from core.init_osi import normalize_oct_signal
from core.octonion_inner import intensity_measurements_explicit
from core.ork_m import orkm_single_row_update
from core.orkm_fast import orkm_main_fast_fixed_perm


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def run_old_fixed_perm(
    A: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor,
    perm_list: Sequence[torch.Tensor],
    *,
    omega: float = 1.0,
) -> torch.Tensor:
    x_est = x0.clone()
    for perm in perm_list:
        for l in perm.tolist():
            x_est = orkm_single_row_update(x_est, A[l], y[l], omega=omega)
    return x_est


def run_fast_fixed_perm(
    A: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor,
    perm_list: Sequence[torch.Tensor],
    *,
    omega: float = 1.0,
) -> torch.Tensor:
    return orkm_main_fast_fixed_perm(A, y, x0, perm_list, omega=omega)


def timed_run(fn, device: torch.device) -> tuple[torch.Tensor, float]:
    maybe_sync(device)
    t0 = time.perf_counter()
    out = fn()
    maybe_sync(device)
    return out, time.perf_counter() - t0


def main() -> None:
    device = pick_device()
    torch.manual_seed(33)
    print("=== T33 fast speed profile ===")
    print(f"device={device}, dtype=float64")

    d = 20
    n = 240
    epochs = 3
    omega = 1.0
    rows = n * epochs
    print(f"config: d={d}, n={n}, epochs={epochs}, rows={rows}")

    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
    y = intensity_measurements_explicit(A, x_true)
    x0 = torch.randn(d, 8, dtype=torch.float64, device=device)
    perm_list = [torch.randperm(n, device=device) for _ in range(epochs)]

    _ = run_old_fixed_perm(A, y, x0, perm_list, omega=omega)
    _ = run_fast_fixed_perm(A, y, x0, perm_list, omega=omega)
    maybe_sync(device)

    _, old_t = timed_run(
        lambda: run_old_fixed_perm(A, y, x0, perm_list, omega=omega), device
    )
    _, new_t = timed_run(
        lambda: run_fast_fixed_perm(A, y, x0, perm_list, omega=omega), device
    )

    speedup = old_t / new_t if new_t > 0 else float("inf")
    old_ep = old_t / epochs
    new_ep = new_t / epochs
    old_row = old_t / rows
    new_row = new_t / rows

    print(f"old total time: {old_t:.6f} s")
    print(f"new total time: {new_t:.6f} s")
    print(f"speedup old/new: {speedup:.2f}x")
    print(f"old time/epoch: {old_ep:.6f} s")
    print(f"new time/epoch: {new_ep:.6f} s")
    print(f"old time/row: {old_row:.6e} s")
    print(f"new time/row: {new_row:.6e} s")

    if speedup >= 10.0:
        flag = "PASS (>=10x)"
    elif speedup >= 5.0:
        flag = "WARN (>=5x and <10x)"
    else:
        flag = "WARN (<5x)"
    print(f"T33 status: {flag}")


if __name__ == "__main__":
    main()
