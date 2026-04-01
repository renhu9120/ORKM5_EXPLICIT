import torch

from core.octonion_ops import oct_abs, oct_abs_sq, oct_mul


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T06 norm properties check ===")
    print(f"selected device: {device}")

    z = torch.randn(128, 8, dtype=torch.float64, device=device)
    abs_sq_z = oct_abs_sq(z)
    min_abs_sq = torch.min(abs_sq_z).item()
    print(f"min abs_sq(z): {min_abs_sq:.6e}")

    count = 40
    errs_abs_sq = []
    errs_abs = []
    for _ in range(count):
        p = torch.randn(8, dtype=torch.float64, device=device)
        q = torch.randn(8, dtype=torch.float64, device=device)
        pq = oct_mul(p, q)

        e_abs_sq = torch.abs(oct_abs_sq(pq) - oct_abs_sq(p) * oct_abs_sq(q)).item()
        e_abs = torch.abs(oct_abs(pq) - oct_abs(p) * oct_abs(q)).item()
        errs_abs_sq.append(e_abs_sq)
        errs_abs.append(e_abs)

    max_abs_sq = max(errs_abs_sq)
    max_abs = max(errs_abs)
    print(f"max multiplicativity abs_sq error: {max_abs_sq:.6e}")
    print(f"max multiplicativity abs error: {max_abs:.6e}")

    ok = (min_abs_sq >= -1e-12) and (max_abs_sq < 1e-10) and (max_abs < 1e-10)
    print(f"T06 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T06 failed")


if __name__ == "__main__":
    main()
