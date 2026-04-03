from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.octonion_sign import (
    absolute_inner_product_similarity,
    best_global_sign,
    sign_aligned_distance,
)


def _fmt_bool(x: bool) -> str:
    return "PASS" if x else "FAIL"


def main() -> None:
    torch.manual_seed(37)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(12, 8, dtype=torch.float64, device=device)
    y = torch.randn(12, 8, dtype=torch.float64, device=device)

    d_xx = float(sign_aligned_distance(x, x).item())
    d_xmx = float(sign_aligned_distance(x, -x).item())
    d_xy = float(sign_aligned_distance(x, y).item())
    d_manual = float(
        torch.minimum(
            torch.linalg.norm((x - y).reshape(-1), ord=2),
            torch.linalg.norm((x + y).reshape(-1), ord=2),
        ).item()
    )
    absip_xy = float(absolute_inner_product_similarity(x, y).item())
    absip_xx = float(absolute_inner_product_similarity(x, x).item())
    absip_xmx = float(absolute_inner_product_similarity(x, -x).item())
    s_xmx = best_global_sign(x, -x)

    checks = {
        "A_reflexive_d(x,x)=0": abs(d_xx) <= 1e-12,
        "B_sign_invariant_d(x,-x)=0": abs(d_xmx) <= 1e-12 and s_xmx == -1,
        "C_nonnegative_d(x,y)>=0": d_xy >= 0.0,
        "D_manual_formula_match": abs(d_xy - d_manual) <= 1e-12,
        "E_absip_range": (-1e-12 <= absip_xy <= 1.0 + 1e-12),
        "F_absip_signed_identity": abs(absip_xx - 1.0) <= 1e-12 and abs(absip_xmx - 1.0) <= 1e-12,
    }

    print("=== T37 sign metric sanity ===")
    print(f"device={device}")
    print(f"d(x,x)={d_xx:.6e}, d(x,-x)={d_xmx:.6e}, best_sign(x,-x)={s_xmx}")
    print(f"d(x,y)={d_xy:.6e}, d_manual={d_manual:.6e}, abs_err={abs(d_xy - d_manual):.3e}")
    print(f"absip(x,y)={absip_xy:.6e}, absip(x,x)={absip_xx:.6e}, absip(x,-x)={absip_xmx:.6e}")
    for name, ok in checks.items():
        print(f"{name}: {_fmt_bool(ok)}")

    all_ok = all(checks.values())
    print(f"T37 status: {_fmt_bool(all_ok)}")
    if not all_ok:
        raise RuntimeError("T37_sign_metric_sanity failed")


if __name__ == "__main__":
    main()
