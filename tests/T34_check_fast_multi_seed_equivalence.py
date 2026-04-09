import os
from collections.abc import Sequence

import torch

from algorithms.initializations.init_osi import normalize_oct_signal
from core.octonion_inner import intensity_measurements_explicit
from discards.ork_m import orkm_single_row_update
from discards.orkm_fast import orkm_main_fast_fixed_perm


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max(torch.abs(a - b)).item()


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


def main() -> None:
    device = pick_device()
    print("=== T34 multi-seed equivalence check ===")
    print(f"device={device}, dtype=float64")

    seed_count = int(os.getenv("FAST_EQ_SEEDS", "10"))
    seeds = list(range(seed_count))
    d = 16
    n = 160
    epochs = 2
    omega = 1.0
    tol = 1e-12
    print(f"config: d={d}, n={n}, epochs={epochs}, omega={omega}, seeds={len(seeds)}")

    pass_count = 0
    for seed in seeds:
        torch.manual_seed(seed)
        A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
        x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
        y = intensity_measurements_explicit(A, x_true)
        x0 = torch.randn(d, 8, dtype=torch.float64, device=device)
        perm_list = [torch.randperm(n, device=device) for _ in range(epochs)]

        x_old = run_old_fixed_perm(A, y, x0, perm_list, omega=omega)
        x_fast = orkm_main_fast_fixed_perm(A, y, x0, perm_list, omega=omega)
        err = max_abs_diff(x_old, x_fast)
        ok = err < tol
        pass_count += int(ok)
        print(
            f"seed={seed:02d}, max|x_old-x_fast|={err:.6e}, status={'PASS' if ok else 'FAIL'}"
        )

    ratio = pass_count / len(seeds)
    print(f"success ratio: {pass_count}/{len(seeds)} = {ratio:.2%}")
    ok_all = pass_count == len(seeds)
    print(f"T34 status: {'PASS' if ok_all else 'FAIL'}")
    if not ok_all:
        raise RuntimeError("T34 failed")


if __name__ == "__main__":
    main()
