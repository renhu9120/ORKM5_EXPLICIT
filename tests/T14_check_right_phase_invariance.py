import torch

from core.octonion_inner import intensity_measurements_explicit
from core.octonion_ops import oct_normalize, oct_right_mul_global


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T14 right-phase invariance check ===")
    print(f"selected device: {device}")

    n, d = 20, 32
    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x = torch.randn(d, 8, dtype=torch.float64, device=device)

    q0 = torch.randn(8, dtype=torch.float64, device=device)
    q, valid = oct_normalize(q0)
    if not bool(valid.item()):
        raise RuntimeError("failed to sample valid unit octonion q")

    x_q = oct_right_mul_global(x, q)
    y1 = intensity_measurements_explicit(A, x)
    y2 = intensity_measurements_explicit(A, x_q)

    abs_err = torch.max(torch.abs(y1 - y2))
    rel_err = abs_err / torch.clamp(torch.max(torch.abs(y1)), min=1e-18)

    print(f"max |y1-y2|: {abs_err.item():.6e}")
    print(f"relative error: {rel_err.item():.6e}")

    ok = abs_err.item() < 1e-10
    print(f"T14 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T14 failed")


if __name__ == "__main__":
    main()
