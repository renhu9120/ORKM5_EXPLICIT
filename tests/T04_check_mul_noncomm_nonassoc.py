import statistics

import torch

from core.octonion_ops import oct_mul


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T04 noncommutativity and nonassociativity check ===")
    print(f"selected device: {device}")

    count = 12
    noncomm_vals = []
    nonassoc_vals = []
    for k in range(count):
        p = torch.randn(8, dtype=torch.float64, device=device)
        q = torch.randn(8, dtype=torch.float64, device=device)
        r = torch.randn(8, dtype=torch.float64, device=device)

        pq = oct_mul(p, q)
        qp = oct_mul(q, p)
        e_noncomm = torch.norm(pq - qp).item()
        noncomm_vals.append(e_noncomm)

        left = oct_mul(oct_mul(p, q), r)
        right = oct_mul(p, oct_mul(q, r))
        e_nonassoc = torch.norm(left - right).item()
        nonassoc_vals.append(e_nonassoc)

        print(
            f"group {k:02d}: noncomm error={e_noncomm:.6e}, nonassoc error={e_nonassoc:.6e}"
        )

    print("noncomm stats:")
    print(f"  min={min(noncomm_vals):.6e}")
    print(f"  max={max(noncomm_vals):.6e}")
    print(f"  median={statistics.median(noncomm_vals):.6e}")
    print("nonassoc stats:")
    print(f"  min={min(nonassoc_vals):.6e}")
    print(f"  max={max(nonassoc_vals):.6e}")
    print(f"  median={statistics.median(nonassoc_vals):.6e}")

    ok = (max(noncomm_vals) > 1e-10) and (max(nonassoc_vals) > 1e-10)
    print(f"T04 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T04 failed")


if __name__ == "__main__":
    main()
