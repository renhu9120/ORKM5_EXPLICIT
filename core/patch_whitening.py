from __future__ import annotations

from core.octonion_ops import normalize_oct_signal

"""
Patch-wise band whitening helpers shared by real-image ORKM scripts (main_4 / main_5).

Rows of X are spatial samples; columns are the 8 spectral bands (octonion components).
"""


from typing import Any, Dict, Tuple

import torch
from torch import Tensor

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




def _normalize_signal_batched(x: Tensor, *, eps: float) -> Tensor:
    if x.ndim != 3 or x.shape[-1] != 8:
        raise ValueError(f"x must be (B,d,8), got {tuple(x.shape)}")
    flat_sq = (x * x).sum(dim=(1, 2), keepdim=True)
    nrm = torch.sqrt(flat_sq.clamp(min=float(eps) ** 2))
    return x / nrm


def prepare_x_true_batched(
        X_patch_raw_all: Tensor,
        mode: str,
        eps: float = 1e-10,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Batched counterpart of ``prepare_x_true``.

    X_patch_raw_all: (B, d, 8)
    Returns:
        X_true_batch: (B, d, 8)
        aux_batch: dict with batched or stacked whitening metadata when applicable.
    """
    if X_patch_raw_all.ndim != 3 or X_patch_raw_all.shape[-1] != 8:
        raise ValueError(f"X_patch_raw_all must be (B,d,8), got {tuple(X_patch_raw_all.shape)}")
    B = int(X_patch_raw_all.shape[0])
    aux: Dict[str, Any] = {"mode": mode, "batch_size": B}

    if mode == "raw":
        return X_patch_raw_all, aux

    if mode == "normalized":
        out = _normalize_signal_batched(X_patch_raw_all, eps=1e-18)
        sigmas = torch.bmm(out.transpose(1, 2), out) / float(out.shape[1])
        aux["Sigma_raw"] = sigmas
        return out, aux

    if mode == "whitened":
        xn = _normalize_signal_batched(X_patch_raw_all, eps=1e-18)
        d = int(xn.shape[1])
        sigma = torch.bmm(xn.transpose(1, 2), xn) / float(d)
        evals, u = torch.linalg.eigh(sigma)
        evals = torch.clamp(evals, min=float(eps))
        rsqrt = torch.rsqrt(evals)
        sqrt_e = torch.sqrt(evals)
        w_mat = torch.bmm(torch.bmm(u, torch.diag_embed(rsqrt)), u.transpose(1, 2))
        winv = torch.bmm(torch.bmm(u, torch.diag_embed(sqrt_e)), u.transpose(1, 2))
        xw = torch.bmm(xn, w_mat)
        xw = _normalize_signal_batched(xw, eps=float(eps))
        aux["W"] = w_mat
        aux["Winv"] = winv
        aux["Sigma_raw"] = sigma
        aux["Sigma_white"] = torch.bmm(xw.transpose(1, 2), xw) / float(d)
        return xw, aux

    raise ValueError(f"Unknown mode: {mode!r} (expected raw|normalized|whitened)")


def recover_from_whitened_batched(
        x_est_aligned: Tensor,
        signal_mode: str,
        aux_batch: Dict[str, Any],
) -> Tensor:
    """Inverse whitening for batch (B, d, 8)."""
    if signal_mode == "whitened":
        winv = aux_batch["Winv"]
        if winv.ndim != 3:
            raise ValueError(f"expected Winv (B,8,8), got {tuple(winv.shape)}")
        return torch.bmm(x_est_aligned, winv)
    return x_est_aligned
