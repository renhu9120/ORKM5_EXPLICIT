import math

import torch

from core.ork_m import orkm_single_row_update
from core.orkm_fast import orkm_single_row_update_fast


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max(torch.abs(a - b)).item()


def beta_equal(a: float, b: float, tol: float = 1e-15) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol


def main() -> None:
    device = pick_device()
    torch.manual_seed(31)
    print("=== T31 fast single update equivalence check ===")
    print(f"device={device}, dtype=float64")

    omegas = [0.25, 0.5, 0.75, 1.0]
    groups = 12
    tol = 1e-12
    max_err = 0.0
    info_ok = True

    for g in range(groups):
        d = 16 + (g % 5)
        a_row = torch.randn(d, 8, dtype=torch.float64, device=device)
        x = torch.randn(d, 8, dtype=torch.float64, device=device)
        y_l = torch.rand((), dtype=torch.float64, device=device) * 5.0

        for omega in omegas:
            x_old, info_old = orkm_single_row_update(
                x, a_row, y_l, omega=omega, return_info=True
            )
            x_new, info_new = orkm_single_row_update_fast(
                x, a_row, y_l, omega=omega, return_info=True
            )
            err = max_abs_diff(x_old, x_new)
            max_err = max(max_err, err)

            same_info = (
                bool(info_old["skipped"]) == bool(info_new["skipped"])
                and str(info_old["skip_reason"]) == str(info_new["skip_reason"])
                and bool(info_old["phase_valid"]) == bool(info_new["phase_valid"])
                and beta_equal(float(info_old["beta_value"]), float(info_new["beta_value"]))
            )
            info_ok = info_ok and same_info

            print(
                f"group={g:02d}, omega={omega:.2f}, "
                f"max|x_old-x_fast|={err:.6e}, info_match={'YES' if same_info else 'NO'}"
            )

    print("---- overall ----")
    print(f"max |x_old - x_fast| = {max_err:.6e}")
    print(f"info consistency      = {'PASS' if info_ok else 'FAIL'}")

    ok = (max_err < tol) and info_ok
    print(f"T31 status: {'PASS' if ok else 'FAIL'} (tol={tol:.1e})")
    if not ok:
        raise RuntimeError("T31 failed")


if __name__ == "__main__":
    main()
