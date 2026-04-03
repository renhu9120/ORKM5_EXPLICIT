from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.balloons_hs_io import band_paths, read_png16_to_float64_01, select_8_indices
from core.octonion_align import apply_global_right_phase, estimate_global_right_phase, right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import oct_array_norm, normalize_oct_signal
from core.octonion_ops import oct_normalize, oct_right_mul_global


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_center_roi_patch_grid(H: int, W: int, patch_size: int, roi_side: int) -> Tuple[int, int, int, int]:
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"H,W must be divisible by patch_size={patch_size}, got ({H},{W})")
    gh = H // patch_size
    gw = W // patch_size
    if gh < roi_side or gw < roi_side:
        raise ValueError(f"ROI patches side too large: gh={gh}, gw={gw}, roi_side={roi_side}")
    base_gr = (gh - roi_side) // 2
    base_gc = (gw - roi_side) // 2
    return gh, gw, base_gr, base_gc


def patches8_to_x_true(patch8: Tensor) -> Tensor:
    if patch8.ndim != 3 or patch8.shape[0] != 8:
        raise ValueError(f"patch8 must be (8,patch,patch), got {tuple(patch8.shape)}")
    patch_size = int(patch8.shape[1])
    if patch8.shape[1] != patch8.shape[2]:
        raise ValueError(f"patch8 must be square in spatial dims, got {tuple(patch8.shape)}")
    return patch8.permute(1, 2, 0).contiguous().view(patch_size * patch_size, 8)


def load_one_center_patch(
        folder: str,
        obj_name_prefix: str,
        patch_size: int,
        roi_side: int,
        patch_index_in_roi: int,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
) -> Tensor:
    idx8 = select_8_indices()
    bands8_np: List[np.ndarray] = []
    for p in band_paths(folder, obj_name_prefix, idx8):
        bands8_np.append(read_png16_to_float64_01(p))
    S8_full = np.stack(bands8_np, axis=0).astype(np.float64)
    H, W = int(S8_full.shape[1]), int(S8_full.shape[2])

    gh, gw, base_gr, base_gc = compute_center_roi_patch_grid(H, W, patch_size, roi_side)
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
    return patches8_to_x_true(patch8)


def default_balloons_folder() -> str:
    return os.path.join(str(PROJECT_ROOT), "dataset", "complete_ms_data", "balloons_ms", "balloons_ms")


def sample_normalized_gaussian_oct(d: int, device: torch.device, dtype: torch.dtype, gen: torch.Generator) -> Tensor:
    x = torch.randn(d, 8, dtype=dtype, device=device, generator=gen)
    return normalize_oct_signal(x)


def make_measurement_matrix(A_seed: int, n: int, d: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    torch.manual_seed(A_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(A_seed)
    return torch.randn(n, d, 8, dtype=dtype, device=device)


def aligned_metrics(
        x_true: Tensor,
        x_est: Tensor,
        A: Tensor,
        y: Tensor,
) -> Dict[str, float]:
    dist = float(right_aligned_distance(x_true, x_est).item())
    q = estimate_global_right_phase(x_true, x_est)
    x_est_aligned = apply_global_right_phase(x_est, q)
    rel_l2 = float((oct_array_norm(x_true - x_est_aligned) / oct_array_norm(x_true)).item())
    y_hat = intensity_measurements_explicit(A, x_est_aligned)
    meas_rel = float((torch.linalg.norm(y_hat - y) / torch.linalg.norm(y)).item())
    return {
        "final_dist": dist,
        "log_final_dist": float(np.log(max(dist, 1e-300))),
        "final_rel_l2": rel_l2,
        "final_meas_rel": meas_rel,
    }


def log_final_line(prefix: str, m: Mapping[str, float], wall: float) -> None:
    print(
        f"{prefix} final_dist={m['final_dist']:.6e}, log(final_dist)={m['log_final_dist']:.6e}, "
        f"final_rel={m['final_rel_l2']:.6e}, final_meas_rel={m['final_meas_rel']:.6e}, wall={wall:.3f}s"
    )


def print_orbit_progress(
        info: Dict[str, Any],
        *,
        progress_stride: int = 8,
        prefix: str = "[PROG]",
) -> None:
    iters = info.get("orbit_log_iters", [])
    align = info.get("orbit_log_align", [])
    meas = info.get("orbit_log_meas_rel", [])
    if not iters:
        return
    max_it = max(iters)
    for i, it in enumerate(iters):
        if it == 0 or it == 1 or (it % progress_stride == 0) or it == max_it:
            a = align[i] if i < len(align) else float("nan")
            mr = meas[i] if i < len(meas) else float("nan")
            print(f"{prefix} iter={it}, dist_align={a:.6e}, meas_rel={mr:.6e}")


def append_csv(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def prob_milestone(test_id: str, message: str) -> None:
    """High-level progress line for long prob batches (flush for live logs / PyCharm)."""
    print(f"[{test_id}] {message}", flush=True)


def wall_synchronized(device: torch.device, t0: float) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def random_unit_octonion(device: torch.device, dtype: torch.dtype, gen: torch.Generator) -> Tensor:
    while True:
        q = torch.randn(8, dtype=dtype, device=device, generator=gen)
        qn, valid = oct_normalize(q)
        if bool(valid.item()):
            return qn


def warm_start_x0(
        x_true: Tensor,
        eps: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        gen: torch.Generator,
) -> Tuple[Tensor, float, float]:
    xi = torch.randn(x_true.shape, dtype=x_true.dtype, device=x_true.device, generator=gen)
    xi = xi / oct_array_norm(xi)
    q_rand = random_unit_octonion(device, dtype, gen)
    x_ref = apply_global_right_phase(x_true, q_rand)
    x0 = x_ref + float(eps) * xi
    x0 = normalize_oct_signal(x0)
    d0 = float(right_aligned_distance(x_true, x0).item())
    q0 = estimate_global_right_phase(x_true, x0)
    x0a = apply_global_right_phase(x0, q0)
    rel0 = float((oct_array_norm(x_true - x0a) / oct_array_norm(x_true)).item())
    return x0, d0, rel0
