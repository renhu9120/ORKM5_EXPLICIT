import torch

from core.octonion_ops import oct_conj, oct_mul


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T05 conjugate multiplication relation check ===")
    print(f"selected device: {device}")

    count = 20
    errs = []
    for k in range(count):
        p = torch.randn(8, dtype=torch.float64, device=device)
        q = torch.randn(8, dtype=torch.float64, device=device)
        lhs = oct_conj(oct_mul(p, q))
        rhs = oct_mul(oct_conj(q), oct_conj(p))
        err = torch.norm(lhs - rhs).item()
        errs.append(err)
        print(f"group {k:02d}: relation error={err:.6e}")

    max_err = max(errs)
    print(f"max error: {max_err:.6e}")
    ok = max_err < 1e-12
    print(f"T05 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T05 failed")


if __name__ == "__main__":
    main()
