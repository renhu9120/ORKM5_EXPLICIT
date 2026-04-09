"""
Shared hyperspectral (8-band) helpers for CAVE-style balloons experiments.

Kept separate from octonion algebra (`core/o_funcs.py`) because these are NumPy / IO utilities.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from PIL import Image


def read_png16_to_float64_01(path: str) -> np.ndarray:
    """Read a single-band PNG as float64; uint16/uint8 mapped to approximately [0, 1]."""
    im = Image.open(path)
    arr = np.asarray(im)
    if arr.ndim != 2:
        raise ValueError(f"expect grayscale band image, got {arr.shape} for {path}")
    if arr.dtype == np.uint16:
        return arr.astype(np.float64) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    mx = float(arr.max()) if arr.max() > 0 else 1.0
    return arr.astype(np.float64) / mx


def select_8_indices() -> List[int]:
    """Eight band indices (0..30); must stay aligned across experiments."""
    return torch.linspace(0, 30, steps=8).round().to(torch.long).tolist()


def idx_to_wavelength_nm(idx0: int) -> int:
    return 400 + 10 * int(idx0)


def pseudo_rgb_from_8bands(bands8: List[np.ndarray], idx8: List[int]) -> np.ndarray:
    """
    Pseudo-RGB by interpolating the eight hyperspectral bands at 450/550/650 nm.

    Output: ``(H, W, 3)`` float64 (not clipped).
    """
    wls = np.array([idx_to_wavelength_nm(i) for i in idx8], dtype=np.float64)
    order = np.argsort(wls)
    wl_sorted = wls[order]
    B_sorted = [bands8[i] for i in order]

    def interp_at(target_nm: float) -> np.ndarray:
        if target_nm <= wl_sorted[0]:
            return B_sorted[0]
        if target_nm >= wl_sorted[-1]:
            return B_sorted[-1]
        j = int(np.searchsorted(wl_sorted, target_nm))
        wl0, wl1 = float(wl_sorted[j - 1]), float(wl_sorted[j])
        t = (target_nm - wl0) / (wl1 - wl0)
        return (1.0 - t) * B_sorted[j - 1] + t * B_sorted[j]

    B = interp_at(450.0)
    G = interp_at(550.0)
    R = interp_at(650.0)
    return np.stack([R, G, B], axis=-1).astype(np.float64)


def band_paths(folder: str, obj_name_prefix: str, idx8: List[int]) -> List[str]:
    """PNG paths for the eight selected bands (1-based file index = idx + 1)."""
    paths = []
    for idx in idx8:
        band_id = idx + 1
        paths.append(os.path.join(folder, f"{obj_name_prefix}_ms_{band_id:02d}.png"))
    return paths
