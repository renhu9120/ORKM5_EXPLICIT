import time
from collections.abc import Sequence
import os

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


def timed(fn, device: torch.device) -> float:
    maybe_sync(device)
    t0 = time.perf_counter()
    _ = fn()
    maybe_sync(device)
    return time.perf_counter() - t0


def main() -> None:
    device = pick_device()
    print("=== T35 fast speed scaling ===")
    print(f"device={device}, dtype=float64")

    smoke = os.getenv("FAST_SPEED_SMOKE", "0") == "1"
    if smoke:
        configs = [(16, 64 * 8), (30, 20 * 10)]
    else:
        configs = [(16, 64 * 16), (30, 20 * 30)]
    epochs = 3
    omega = 1.0

    for idx, (d, n) in enumerate(configs):
        torch.manual_seed(350 + idx)
        A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
        x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
        y = intensity_measurements_explicit(A, x_true)
        x0 = torch.randn(d, 8, dtype=torch.float64, device=device)
        perm_list = [torch.randperm(n, device=device) for _ in range(epochs)]

        _ = run_old_fixed_perm(A, y, x0, perm_list, omega=omega)
        _ = orkm_main_fast_fixed_perm(A, y, x0, perm_list, omega=omega)
        maybe_sync(device)

        old_t = timed(lambda: run_old_fixed_perm(A, y, x0, perm_list, omega=omega), device)
        new_t = timed(
            lambda: orkm_main_fast_fixed_perm(A, y, x0, perm_list, omega=omega), device
        )
        speedup = old_t / new_t if new_t > 0 else float("inf")
        print(
            f"(d={d:3d}, n={n:5d}) -> old={old_t:.6f}s, new={new_t:.6f}s, speedup={speedup:.2f}x"
        )

    print("T35 status: DONE")


if __name__ == "__main__":
    main()
