import torch

from core.octonion_inner import row_inner_explicit
from core.octonion_ops import oct_conj, oct_mul


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T09 row inner single check ===")
    print(f"selected device: {device}")

    d = 7
    a_row = torch.randn(d, 8, dtype=torch.float64, device=device)
    x = torch.randn(d, 8, dtype=torch.float64, device=device)

    s_manual = torch.zeros(8, dtype=torch.float64, device=device)
    for j in range(d):
        s_manual = s_manual + oct_mul(oct_conj(a_row[j]), x[j])
    s_func = row_inner_explicit(a_row, x)
    diff = torch.norm(s_manual - s_func).item()

    print(f"s_manual: {s_manual}")
    print(f"s_func:   {s_func}")
    print(f"||diff||: {diff:.6e}")
    print(f"s_func shape: {tuple(s_func.shape)}, dtype: {s_func.dtype}, device: {s_func.device}")

    same_device = s_func.device.type == device.type
    if device.type == "cuda":
        same_device = same_device and (s_func.device.index == 0)
    ok = diff < 1e-12 and s_func.shape == (8,) and s_func.dtype == torch.float64 and same_device
    print(f"T09 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T09 failed")


if __name__ == "__main__":
    main()
