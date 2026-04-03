from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.autograd.functional as autograd_F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prob_realpatch_common import (
    default_balloons_folder,
    default_device,
    load_one_center_patch,
    prob_milestone,
    sample_normalized_gaussian_oct,
)
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import normalize_oct_signal
from core.octonion_ops import oct_right_mul_global


def _build_x_cases(
        *,
        device: torch.device,
        dtype: torch.dtype,
        d: int,
        patch_size: int,
        roi_side: int,
        patch_index_in_roi: int,
        folder: str,
        smoke: bool,
) -> List[Tuple[str, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor]] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(999)
    out.append(("G", sample_normalized_gaussian_oct(d, device, dtype, gen)))

    raw = load_one_center_patch(
        folder=folder,
        obj_name_prefix="balloons",
        patch_size=patch_size,
        roi_side=roi_side,
        patch_index_in_roi=patch_index_in_roi,
        device=device,
        dtype=dtype,
    )
    out.append(("P", normalize_oct_signal(raw)))

    if smoke:
        return out

    gen_s = torch.Generator(device=device)
    gen_s.manual_seed(44)
    flat = raw.reshape(-1).clone()
    perm = torch.randperm(flat.numel(), device=flat.device, generator=gen_s)
    out.append(("S", normalize_oct_signal(flat[perm].view(raw.shape))))

    gen_r = torch.Generator(device=device)
    gen_r.manual_seed(45)
    fl = raw.reshape(-1)
    idx = torch.randint(0, fl.numel(), (fl.numel(),), device=fl.device, generator=gen_r)
    out.append(("R", normalize_oct_signal(fl[idx].view(raw.shape))))

    return out


def _phase_basis_q(x_true_cpu: torch.Tensor) -> np.ndarray:
    """Orthonormal basis for approximate right-phase tangent directions (7 imaginary units)."""
    cols = []
    for k in range(1, 8):
        ek = torch.zeros(8, dtype=torch.float64)
        ek[k] = 1.0
        vk = oct_right_mul_global(x_true_cpu, ek.to(device=x_true_cpu.device))
        cols.append(vk.reshape(-1).detach().numpy())
    V = np.stack(cols, axis=1)
    q, _ = np.linalg.qr(V, mode="reduced")
    return q


def _svd_report(J: np.ndarray, prefix: str, case: str) -> Dict[str, Any]:
    _, s, _ = np.linalg.svd(J, full_matrices=False)
    s_pos = s[s > 1e-30]
    if s_pos.size == 0:
        smin, smax = 0.0, 0.0
        cond = float("inf")
    else:
        smin = float(s_pos.min())
        smax = float(s_pos.max())
        cond = smax / smin if smin > 0 else float("inf")
    top10 = s[: min(10, s.size)]
    bot10 = s[max(0, s.size - 10) :]
    print(f"{prefix} case={case}")
    print(f"{prefix} sigma_max={smax:.6e}")
    print(f"{prefix} sigma_min={smin:.6e}")
    print(f"{prefix} cond={cond:.6e}")
    print(f"{prefix} top10 singular values={np.array2string(top10, precision=4, suppress_small=True)}")
    print(f"{prefix} bottom10 singular values={np.array2string(bot10, precision=4, suppress_small=True)}")
    return {"sigma_max": smax, "sigma_min": smin, "cond": cond, "s": s}


def run(
        *,
        smoke: bool = False,
        no_proj: bool = False,
        A_seed: int = 5,
        show_progress: bool = True,
) -> None:
    device = default_device()
    dtype = torch.float64
    patch_size = 8
    d = patch_size * patch_size
    roi_side = 3
    patch_index_in_roi = 4
    n_over_d = 20
    n = n_over_d * d
    A_seed = int(A_seed)

    folder = default_balloons_folder()
    if show_progress:
        prob_milestone(
            "PROB5",
            f"start Jacobian conditioning: n={n}, d_flat={d * 8}, A_seed={A_seed}, "
            f"smoke={smoke}, level2_proj={not no_proj}",
        )
    if show_progress:
        prob_milestone("PROB5", "building x_true cases (may load real patch) ...")
    cases = _build_x_cases(
        device=device,
        dtype=dtype,
        d=d,
        patch_size=patch_size,
        roi_side=roi_side,
        patch_index_in_roi=patch_index_in_roi,
        folder=folder,
        smoke=smoke,
    )
    case_names = [c[0] for c in cases]
    if show_progress:
        prob_milestone("PROB5", f"cases: {' '.join(case_names)} ({len(cases)} Jacobians)")

    torch.manual_seed(A_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(A_seed)
    A = torch.randn(n, d, 8, dtype=dtype, device=device)

    do_proj = not no_proj
    n_cases = len(cases)

    for ic, (case_name, x_true) in enumerate(cases):
        if show_progress:
            prob_milestone(
                "PROB5",
                f"case {ic + 1}/{n_cases} ({case_name}) - autograd Jacobian (CPU, forward-mode) ...",
            )
        t0 = time.time()
        A_cpu = A.detach().cpu().to(torch.float64)
        x_cpu = x_true.detach().cpu().to(torch.float64)
        x0 = x_cpu.reshape(-1).clone().requires_grad_(True)

        def F_flat(xf: torch.Tensor) -> torch.Tensor:
            x = xf.view(d, 8)
            return intensity_measurements_explicit(A_cpu, x)

        J = autograd_F.jacobian(F_flat, x0, vectorize=True, strategy="forward-mode")
        J_np = J.detach().numpy()
        elapsed = time.time() - t0
        if show_progress:
            prob_milestone(
                "PROB5",
                f"case {ic + 1}/{n_cases} ({case_name}) Level-1 Jacobian done in {elapsed:.1f}s, shape={tuple(J_np.shape)}",
            )
        print(f"[JAC] built Jacobian in {elapsed:.3f}s, shape={tuple(J_np.shape)}")
        _svd_report(J_np, "[JAC]", case_name)

        if do_proj:
            if show_progress:
                prob_milestone("PROB5", f"case {ic + 1}/{n_cases} ({case_name}) - Level-2 projected SVD ...")
            Q = _phase_basis_q(x_cpu)
            rank_phase = Q.shape[1]
            jq = J_np @ Q
            J_proj = J_np - jq @ Q.T
            print(f"[JAC-PROJ] case={case_name} rank_phase={rank_phase}")
            rep = _svd_report(J_proj, "[JAC-PROJ]", case_name)
            print(
                f"[JAC-PROJ] sigma_min_nontrivial={rep['sigma_min']:.6e}, "
                f"cond_nontrivial={rep['cond']:.6e}"
            )
        print()

    if show_progress:
        prob_milestone("PROB5", "finished.")


def main() -> None:
    p = argparse.ArgumentParser(description="TEST 5: Jacobian / local conditioning of F(x)=|Ax|^2")
    p.add_argument("--smoke", action="store_true", help="cases G,P only (still runs Level-2 unless --no-proj)")
    p.add_argument("--no-proj", action="store_true", help="skip Level-2 projected Jacobian")
    p.add_argument("--A-seed", type=int, default=5)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    run(
        smoke=args.smoke,
        no_proj=args.no_proj,
        A_seed=args.A_seed,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
