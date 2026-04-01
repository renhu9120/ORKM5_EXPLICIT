import torch

from core.octonion_base import oct_basis, oct_one
from core.octonion_ops import oct_mul


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T03 multiplication identity and basis check ===")
    print(f"selected device: {device}")

    q = torch.randn(32, 8, dtype=torch.float64, device=device)
    one = oct_one(device=device)
    left = oct_mul(one, q)
    right = oct_mul(q, one)
    err_left = torch.max(torch.abs(left - q)).item()
    err_right = torch.max(torch.abs(right - q)).item()

    print(f"max ||1*q - q||: {err_left:.3e}")
    print(f"max ||q*1 - q||: {err_right:.3e}")

    neg_one = torch.tensor([-1.0] + [0.0] * 7, dtype=torch.float64, device=device)
    basis_sq_err = []
    for i in range(1, 8):
        ei = oct_basis(i, device=device)
        sq = oct_mul(ei, ei)
        err = torch.max(torch.abs(sq - neg_one)).item()
        basis_sq_err.append(err)
        print(f"e{i} * e{i} = {sq}, error to -1: {err:.3e}")

    selected_pairs = [(1, 2), (2, 3), (3, 1), (1, 4), (4, 5), (6, 7)]
    for i, j in selected_pairs:
        ei = oct_basis(i, device=device)
        ej = oct_basis(j, device=device)
        prod = oct_mul(ei, ej)
        print(f"e{i} * e{j} = {prod}")

    ok = err_left < 1e-12 and err_right < 1e-12 and max(basis_sq_err) < 1e-12
    print(f"T03 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T03 failed")


if __name__ == "__main__":
    main()
