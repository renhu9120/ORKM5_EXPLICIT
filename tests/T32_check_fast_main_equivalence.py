from collections.abc import Sequence

import torch

from core.init_osi import normalize_oct_signal
from core.octonion_align import right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import raw_distance
from core.ork_m import orkm_single_row_update
from core.orkm_fast import orkm_main_fast_fixed_perm


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
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    x_est = x0.clone()
    traj = [x_est.clone()]
    for perm in perm_list:
        for l in perm.tolist():
            x_est = orkm_single_row_update(x_est, A[l], y[l], omega=omega)
        traj.append(x_est.clone())
    return x_est, traj


def run_fast_fixed_perm(
    A: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor,
    perm_list: Sequence[torch.Tensor],
    *,
    omega: float = 1.0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    x_est = x0.clone()
    traj = [x_est.clone()]
    for perm in perm_list:
        x_est = orkm_main_fast_fixed_perm(A, y, x_est, [perm], omega=omega)
        traj.append(x_est.clone())
    return x_est, traj


def main() -> None:
    device = pick_device()
    torch.manual_seed(32)
    print("=== T32 fast main equivalence check ===")
    print(f"device={device}, dtype=float64")

    d = 20
    n = 240
    epochs = 3
    omega = 1.0
    tol = 1e-12
    print(f"config: d={d}, n={n}, epochs={epochs}, omega={omega}")

    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
    y = intensity_measurements_explicit(A, x_true)
    x0 = torch.randn(d, 8, dtype=torch.float64, device=device)
    perm_list = [torch.randperm(n, device=device) for _ in range(epochs)]

    x_old, traj_old = run_old_fixed_perm(A, y, x0, perm_list, omega=omega)
    x_fast, traj_fast = run_fast_fixed_perm(A, y, x0, perm_list, omega=omega)

    final_err = max_abs_diff(x_old, x_fast)
    final_raw = raw_distance(x_old, x_fast).item()
    final_align = right_aligned_distance(x_old, x_fast).item()

    print(f"final max |x_old - x_fast|      = {final_err:.6e}")
    print(f"final raw_distance(old, fast)   = {final_raw:.6e}")
    print(f"final aligned_distance(old,fast)= {final_align:.6e}")

    traj_max_err = 0.0
    for k, (xo, xf) in enumerate(zip(traj_old, traj_fast)):
        e = max_abs_diff(xo, xf)
        traj_max_err = max(traj_max_err, e)
        print(f"epoch={k:02d}, max|x_old-x_fast|={e:.6e}")

    print(f"trajectory max error = {traj_max_err:.6e}")
    ok = max(final_err, traj_max_err) < tol
    print(f"T32 status: {'PASS' if ok else 'FAIL'} (tol={tol:.1e})")
    if not ok:
        raise RuntimeError("T32 failed")


if __name__ == "__main__":
    main()
