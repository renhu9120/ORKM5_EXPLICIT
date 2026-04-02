import torch

from algorithms.initializations.init_osi import normalize_oct_signal
from core.octonion_inner import intensity_measurements_explicit, row_inner_explicit
from core.octonion_ops import oct_abs
from core.ork_m import orkm_single_row_update


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    torch.manual_seed(23)
    print("=== T23 relaxation scan consistency ===")
    print(f"device={device}, dtype=float64")

    n, d = 16, 8
    omega_list = [0.25, 0.5, 0.75, 1.0]
    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
    y = intensity_measurements_explicit(A, x_true)
    x0 = torch.randn(d, 8, dtype=torch.float64, device=device)

    selected_l = None
    for l in range(n):
        _, info = orkm_single_row_update(x0, A[l], y[l], omega=1.0, return_info=True)
        if not bool(info["skipped"]):
            selected_l = l
            break
    if selected_l is None:
        raise RuntimeError("all rows skipped; cannot run relaxation scan")

    print(f"selected_row={selected_l}")
    b_l = torch.sqrt(torch.clamp(y[selected_l], min=0.0)).item()
    s_before = row_inner_explicit(A[selected_l], x0)
    before = oct_abs(s_before).item() - b_l

    print(f"baseline: |s_before|={oct_abs(s_before).item():.6e}, b_l={b_l:.6e}, before={before:.6e}")
    print("omega scan details:")
    for omega in omega_list:
        x1, info = orkm_single_row_update(
            x0, A[selected_l], y[selected_l], omega=omega, return_info=True
        )
        s_after = row_inner_explicit(A[selected_l], x1)
        after = oct_abs(s_after).item() - b_l
        predicted = (1.0 - omega) * before
        abs_error = abs(after - predicted)
        if abs(predicted) < 1e-15:
            rel_error = float("nan")
        else:
            rel_error = abs_error / abs(predicted)
        print(
            f"omega={omega:.2f}, skipped={info['skipped']}, before={before:.6e}, "
            f"after={after:.6e}, predicted={predicted:.6e}, "
            f"abs_error={abs_error:.6e}, rel_error={rel_error:.6e}"
        )

    print("T23 status: DONE (inspect abs/rel errors near machine precision)")


if __name__ == "__main__":
    main()
