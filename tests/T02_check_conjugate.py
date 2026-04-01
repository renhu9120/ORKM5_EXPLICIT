import torch

from core.octonion_ops import oct_abs_sq, oct_conj, oct_real


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T02 conjugate check ===")
    print(f"selected device: {device}")

    z = torch.randn(64, 8, dtype=torch.float64, device=device)
    zcc = oct_conj(oct_conj(z))
    real_diff = oct_real(oct_conj(z)) - oct_real(z)
    abs_sq_diff = oct_abs_sq(oct_conj(z)) - oct_abs_sq(z)

    err_cc = torch.max(torch.abs(zcc - z)).item()
    err_real = torch.max(torch.abs(real_diff)).item()
    err_abs_sq = torch.max(torch.abs(abs_sq_diff)).item()

    print(f"max ||conj(conj(z)) - z||: {err_cc:.3e}")
    print(f"max real diff: {err_real:.3e}")
    print(f"max abs_sq diff: {err_abs_sq:.3e}")

    ok = err_cc < 1e-12 and err_real < 1e-12 and err_abs_sq < 1e-12
    print(f"T02 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T02 failed")


if __name__ == "__main__":
    main()
