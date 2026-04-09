from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.algs.alg_orkm import alg_orkm
from utils.utils_img import (
    band_paths,
    read_png16_to_float64_01,
    select_8_indices,
)
from core.octonion_align import (
    apply_global_right_phase,
    estimate_global_right_phase,
    right_aligned_distance,
)
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import oct_array_norm, normalize_oct_signal


def _compute_center_roi_patch_grid(H: int, W: int, patch_size: int, roi_side: int) -> Tuple[int, int, int, int]:
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"H,W must be divisible by patch_size={patch_size}, got ({H},{W})")
    gh = H // patch_size
    gw = W // patch_size
    if gh < roi_side or gw < roi_side:
        raise ValueError(f"ROI patches side too large: gh={gh}, gw={gw}, roi_side={roi_side}")
    base_gr = (gh - roi_side) // 2
    base_gc = (gw - roi_side) // 2
    return gh, gw, base_gr, base_gc


def _patches8_to_x_true(patch8: torch.Tensor) -> torch.Tensor:
    """
    patch8: (8, patch, patch)
    return x_true: (d=patch*patch, 8)
    """
    if patch8.ndim != 3 or patch8.shape[0] != 8:
        raise ValueError(f"patch8 must be (8,patch,patch), got {tuple(patch8.shape)}")
    patch_size = int(patch8.shape[1])
    if patch8.shape[1] != patch8.shape[2]:
        raise ValueError(f"patch8 must be square in spatial dims, got {tuple(patch8.shape)}")
    return patch8.permute(1, 2, 0).contiguous().view(patch_size * patch_size, 8)


def _load_one_center_patch(
    folder: str,
    obj_name_prefix: str,
    patch_size: int,
    roi_side: int,
    patch_index_in_roi: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Load one patch using exactly the same patch-selection logic as run_stage1_roi_orkm.
    Returns x_true with shape (d, 8).
    """
    idx8 = select_8_indices()

    bands8_np: List[np.ndarray] = []
    for p in band_paths(folder, obj_name_prefix, idx8):
        bands8_np.append(read_png16_to_float64_01(p))
    S8_full = np.stack(bands8_np, axis=0).astype(np.float64)  # (8,H,W)
    H, W = int(S8_full.shape[1]), int(S8_full.shape[2])

    gh, gw, base_gr, base_gc = _compute_center_roi_patch_grid(H, W, patch_size, roi_side)
    _ = (gh, gw)

    if patch_index_in_roi < 0 or patch_index_in_roi >= roi_side * roi_side:
        raise ValueError(f"patch_index_in_roi must be in [0, {roi_side * roi_side - 1}]")

    pr = patch_index_in_roi // roi_side
    pc = patch_index_in_roi % roi_side

    gr = base_gr + pr
    gc = base_gc + pc
    y0 = gr * patch_size
    x0 = gc * patch_size

    patch8 = torch.as_tensor(
        S8_full[:, y0:y0 + patch_size, x0:x0 + patch_size],
        dtype=dtype,
        device=device,
    )
    x_true = _patches8_to_x_true(patch8)
    return x_true


def run_orkm_realpatch_probe() -> None:
    # ===== config =====
    folder = os.path.join(str(PROJECT_ROOT), "dataset", "complete_ms_data", "balloons_ms", "balloons_ms")
    obj_name_prefix = "balloons"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    patch_size = 8
    roi_side = 3
    patch_index_in_roi = 4   # 0,1,2,3 for roi_side=2

    seed = 1
    A_seed = 5
    n_over_d = 20
    d = patch_size * patch_size
    n = n_over_d * d
    passes = 80
    power_iters = 1
    dist_tol = 1e-5

    print(f"[realpatch_probe] device={device}, dtype=float64")
    print(f"[realpatch_probe] patch_size={patch_size}, roi_side={roi_side}, patch_index_in_roi={patch_index_in_roi}")
    print(f"[realpatch_probe] d={d}, n={n}, n/d={n_over_d}, passes={passes}, power_iters={power_iters}")

    # ===== load one real patch =====
    x_true_raw = _load_one_center_patch(
        folder=folder,
        obj_name_prefix=obj_name_prefix,
        patch_size=patch_size,
        roi_side=roi_side,
        patch_index_in_roi=patch_index_in_roi,
        device=device,
        dtype=dtype,
    )
    x_true_normed = normalize_oct_signal(x_true_raw)

    print(
        f"[realpatch_probe] raw patch norm={float(oct_array_norm(x_true_raw).item()):.6e}, "
        f"min={float(x_true_raw.min().item()):.6e}, "
        f"max={float(x_true_raw.max().item()):.6e}, "
        f"mean={float(x_true_raw.mean().item()):.6e}, "
        f"std={float(x_true_raw.std().item()):.6e}"
    )
    print(
        f"[realpatch_probe] normed patch norm={float(oct_array_norm(x_true_normed).item()):.6e}"
    )

    # ===== two cases: raw and normalized =====
    for case_name, x_true in [
        ("raw_patch", x_true_raw),
        ("normalized_patch", x_true_normed),
    ]:
        print("\n" + "=" * 80)
        print(f"[realpatch_probe] case={case_name}")
        print("=" * 80)

        # keep same style as run_orkm_main
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        torch.manual_seed(A_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(A_seed)
        A = torch.randn(n, d, 8, dtype=dtype, device=device)

        y = intensity_measurements_explicit(A, x_true)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        x_est, history = alg_orkm(
            A=A,
            y=y,
            T=passes,
            seed=seed,
            power_iters=power_iters,
            x_true_proc=x_true,
            verbose=True,
            progress_every=max(1, passes // 10),
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0

        q = estimate_global_right_phase(x_true, x_est)
        x_est_aligned = apply_global_right_phase(x_est, q)

        dist = float(right_aligned_distance(x_true, x_est).item())
        rel_l2 = float((oct_array_norm(x_true - x_est_aligned) / oct_array_norm(x_true)).item())

        y_hat = intensity_measurements_explicit(A, x_est_aligned)
        meas_rel = float((torch.linalg.norm(y_hat - y) / torch.linalg.norm(y)).item())

        print(
            f"[realpatch_probe] final case={case_name}, "
            f"dist={dist:.6e}, log(dist)={float(np.log(max(dist, 1e-300))):.6e}, "
            f"rel_l2={rel_l2:.6e}, meas_rel={meas_rel:.6e}, "
            f"success={bool(dist <= dist_tol)}, wall={wall:.3f}s"
        )


if __name__ == "__main__":
    run_orkm_realpatch_probe()