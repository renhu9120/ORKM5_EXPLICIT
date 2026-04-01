import torch

from core.octonion_base import (
    ensure_octonion_tensor,
    oct_basis,
    oct_one,
    oct_stack_basis,
    oct_zero,
)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T01 base layout check ===")
    print(f"torch version: {torch.__version__}")
    print(f"selected device: {device}")

    raw = torch.randn(3, 4, 8, device=device, dtype=torch.float32)
    x = ensure_octonion_tensor(raw, name="raw")
    z0 = oct_zero(device=device)
    z5 = oct_zero(5, device=device)
    zhw = oct_zero(4, 6, device=device)
    one = oct_one(device=device)
    b0 = oct_basis(0, device=device)
    b1 = oct_basis(1, device=device)
    b7 = oct_basis(7, device=device)
    basis = oct_stack_basis(device=device)

    print(f"x shape: {tuple(x.shape)}, dtype: {x.dtype}, device: {x.device}")
    print(f"z0 shape: {tuple(z0.shape)}, dtype: {z0.dtype}")
    print(f"z5 shape: {tuple(z5.shape)}, dtype: {z5.dtype}")
    print(f"zhw shape: {tuple(zhw.shape)}, dtype: {zhw.dtype}")
    print(f"one: {one}")
    print(f"basis[0]: {basis[0]}")
    print(f"basis[1]: {basis[1]}")
    print(f"basis[7]: {basis[7]}")
    print(f"b0 equals one: {bool(torch.allclose(b0, one))}")
    print(f"b1: {b1}")
    print(f"b7: {b7}")

    ok = True
    ok = ok and (x.dtype == torch.float64)
    ok = ok and (x.shape[-1] == 8)
    ok = ok and (z0.shape == (8,))
    ok = ok and (z5.shape == (5, 8))
    ok = ok and (zhw.shape == (4, 6, 8))
    ok = ok and bool(torch.allclose(one, torch.tensor([1.0] + [0.0] * 7, dtype=torch.float64, device=device)))
    ok = ok and (basis.shape == (8, 8))
    ok = ok and all(basis[i, i].item() == 1.0 for i in range(8))

    print(f"T01 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T01 failed")


if __name__ == "__main__":
    main()
