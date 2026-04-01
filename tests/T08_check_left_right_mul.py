import torch

from core.octonion_ops import (
    oct_left_mul,
    oct_left_mul_global,
    oct_mul,
    oct_right_mul,
    oct_right_mul_global,
)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T08 left/right multiplication helpers check ===")
    print(f"selected device: {device}")

    a = torch.randn(8, dtype=torch.float64, device=device)
    x = torch.randn(16, 8, dtype=torch.float64, device=device)

    left_direct = oct_mul(a, x)
    left_api = oct_left_mul(a, x)
    right_direct = oct_mul(x, a)
    right_api = oct_right_mul(x, a)

    e_left = torch.max(torch.abs(left_direct - left_api)).item()
    e_right = torch.max(torch.abs(right_direct - right_api)).item()
    print(f"left mul error: {e_left:.6e}")
    print(f"right mul error: {e_right:.6e}")

    left_global = oct_left_mul_global(a, x)
    right_global = oct_right_mul_global(x, a)

    left_loop = torch.zeros_like(x)
    right_loop = torch.zeros_like(x)
    for i in range(x.shape[0]):
        left_loop[i] = oct_mul(a, x[i])
        right_loop[i] = oct_mul(x[i], a)

    e_left_global = torch.max(torch.abs(left_global - left_loop)).item()
    e_right_global = torch.max(torch.abs(right_global - right_loop)).item()
    print(f"global left error: {e_left_global:.6e}")
    print(f"global right error: {e_right_global:.6e}")

    ok = (
        e_left < 1e-12
        and e_right < 1e-12
        and e_left_global < 1e-12
        and e_right_global < 1e-12
    )
    print(f"T08 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T08 failed")


if __name__ == "__main__":
    main()
