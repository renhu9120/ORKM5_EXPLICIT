from __future__ import annotations

import torch
from torch import Tensor

from core.octonion_base import ensure_octonion_tensor
from core.octonion_ops import oct_abs_sq


def oct_array_norm(x: Tensor, eps: float = 0.0) -> Tensor:
    """
    Euclidean norm over all octonion blocks in x with shape (..., 8).
    """
    x_std = ensure_octonion_tensor(x, name="x")
    total = oct_abs_sq(x_std).sum()
    if eps > 0.0:
        total = torch.clamp(total, min=float(eps) ** 2)
    return torch.sqrt(total)


def raw_distance(x_true: Tensor, x_est: Tensor) -> Tensor:
    """
    Raw distance d_raw = ||x_true - x_est||.
    """
    x_true_std = ensure_octonion_tensor(x_true, name="x_true")
    x_est_std = ensure_octonion_tensor(x_est, name="x_est")
    if x_true_std.shape != x_est_std.shape:
        raise ValueError(
            f"Shape mismatch: x_true{tuple(x_true_std.shape)} vs x_est{tuple(x_est_std.shape)}"
        )
    return oct_array_norm(x_true_std - x_est_std)


def relative_error(x_true: Tensor, x_est: Tensor, eps: float = 1e-18) -> Tensor:
    """
    Relative error = raw_distance(x_true, x_est) / ||x_true||.
    """
    num = raw_distance(x_true, x_est)
    den = oct_array_norm(x_true, eps=eps)
    return num / den


def optional_right_align_distance(
    x_true: Tensor, x_est: Tensor, *, mode: str = "not_implemented"
) -> Tensor:
    """
    Placeholder for right-aligned distance optimization.
    This metric is intentionally deferred to a later phase.
    """
    _ = (x_true, x_est, mode)
    raise NotImplementedError(
        "optional_right_align_distance is deferred and will be implemented in a later phase."
    )
