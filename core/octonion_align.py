from __future__ import annotations

import torch
from torch import Tensor

from core.octonion_base import ensure_octonion_tensor, oct_one, oct_zero
from core.octonion_metric import raw_distance
from core.octonion_ops import oct_conj, oct_mul, oct_normalize, oct_right_mul_global


def estimate_global_right_phase(x_true: Tensor, x_est: Tensor) -> Tensor:
    """
    Estimate global unit right phase q for x_est * q ~= x_true using:
        q ~ normalize(sum_i conj(x_est[i]) * x_true[i]).
    """
    x_true_std = ensure_octonion_tensor(x_true, name="x_true")
    x_est_std = ensure_octonion_tensor(x_est, name="x_est")
    if x_true_std.shape != x_est_std.shape:
        raise ValueError(
            f"Shape mismatch: x_true{tuple(x_true_std.shape)} vs x_est{tuple(x_est_std.shape)}"
        )
    if x_true_std.ndim != 2:
        raise ValueError(
            f"x_true and x_est must both have shape (d, 8), got {tuple(x_true_std.shape)}"
        )

    d = x_true_std.shape[0]
    acc = oct_zero(device=x_true_std.device)
    for i in range(d):
        term = oct_mul(oct_conj(x_est_std[i]), x_true_std[i])
        acc = acc + term

    q, valid = oct_normalize(acc)
    if not bool(valid.item()):
        return oct_one(device=x_true_std.device)
    return q


def apply_global_right_phase(x: Tensor, q: Tensor) -> Tensor:
    """
    Apply global right multiplication x_i -> x_i * q.
    """
    x_std = ensure_octonion_tensor(x, name="x")
    q_std = ensure_octonion_tensor(q, name="q")
    if q_std.ndim != 1 or q_std.shape[0] != 8:
        raise ValueError(f"q must have shape (8,), got {tuple(q_std.shape)}")
    return oct_right_mul_global(x_std, q_std)


def right_aligned_distance(x_true: Tensor, x_est: Tensor) -> Tensor:
    """
    Right-aligned distance:
        d_align = ||x_true - x_est * q*||,
    where q* is estimated global unit right phase.
    """
    q = estimate_global_right_phase(x_true, x_est)
    x_est_aligned = apply_global_right_phase(x_est, q)
    return raw_distance(x_true, x_est_aligned)
