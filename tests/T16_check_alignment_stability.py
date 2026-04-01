import torch

from core.octonion_align import right_aligned_distance
from core.octonion_metric import raw_distance
from core.octonion_ops import oct_normalize, oct_right_mul_global


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T16 alignment stability check ===")
    print(f"selected device: {device}")

    d = 40
    x_true = torch.randn(d, 8, dtype=torch.float64, device=device)

    q0 = torch.randn(8, dtype=torch.float64, device=device)
    q, valid = oct_normalize(q0)
    if not bool(valid.item()):
        raise RuntimeError("failed to sample valid unit octonion q")

    x_clean = oct_right_mul_global(x_true, q)
    noise = 1e-3 * torch.randn(d, 8, dtype=torch.float64, device=device)
    x_est = x_clean + noise

    d_raw = raw_distance(x_true, x_est)
    d_align = right_aligned_distance(x_true, x_est)

    print(f"d_raw:   {d_raw.item():.6e}")
    print(f"d_align: {d_align.item():.6e}")

    ok = d_align.item() < d_raw.item()
    print(f"T16 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T16 failed")


if __name__ == "__main__":
    main()
