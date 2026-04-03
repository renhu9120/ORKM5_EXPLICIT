from __future__ import annotations

"""
Patch-wise band whitening helpers shared by real-image ORKM scripts (main_4 / main_5).

Rows of X are spatial samples; columns are the 8 spectral bands (octonion components).
"""

from typing import Any, Dict

import torch
from torch import Tensor

from core.octonion_metric import normalize_oct_signal


def compute_patch_band_whitening(
        x: Tensor,
        eps: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    x: (d, 8)

    Returns:
        x_white : (d, 8) = x @ W
        W       : (8, 8)
        Winv    : (8, 8)
        sigma   : (8, 8) = (x^T x) / d
    """
    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError(f"x must have shape (d, 8), got {tuple(x.shape)}")
    d = int(x.shape[0])
    sigma = (x.T @ x) / float(d)
    evals, u = torch.linalg.eigh(sigma)
    evals = torch.clamp(evals, min=float(eps))
    w_mat = u @ torch.diag(torch.rsqrt(evals)) @ u.T
    winv = u @ torch.diag(torch.sqrt(evals)) @ u.T
    x_white = x @ w_mat
    return x_white, w_mat, winv, sigma


def prepare_x_true(
        x_patch_raw: Tensor,
        mode: str,
        eps: float = 1e-10,
) -> tuple[Tensor, Dict[str, Any]]:
    """
    mode: "raw" | "normalized" | "whitened"
    """
    aux: Dict[str, Any] = {"mode": mode}
    if mode == "raw":
        return x_patch_raw, aux
    if mode == "normalized":
        xn = normalize_oct_signal(x_patch_raw)
        aux["Sigma_raw"] = (xn.T @ xn) / float(xn.shape[0])
        return xn, aux
    if mode == "whitened":
        xn = normalize_oct_signal(x_patch_raw)
        xw, w_mat, winv, sigma = compute_patch_band_whitening(xn, eps=eps)
        xw = normalize_oct_signal(xw)
        aux["W"] = w_mat
        aux["Winv"] = winv
        aux["Sigma_raw"] = sigma
        aux["Sigma_white"] = (xw.T @ xw) / float(xw.shape[0])
        return xw, aux
    raise ValueError(f"Unknown mode: {mode!r} (expected raw|normalized|whitened)")


def recover_from_whitened(
        x_est_aligned: Tensor,
        signal_mode: str,
        aux: Dict[str, Any],
) -> Tensor:
    """Map ORKM reconstruction from whitening domain back to original band domain."""
    if signal_mode == "whitened":
        winv = aux["Winv"]
        return x_est_aligned @ winv
    return x_est_aligned
