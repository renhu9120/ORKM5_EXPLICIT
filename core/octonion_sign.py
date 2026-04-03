from __future__ import annotations

import torch
from torch import Tensor

from core.octonion_base import ensure_octonion_tensor


def flatten_real_vector(x: Tensor) -> Tensor:
    x_std = ensure_octonion_tensor(x, name="x")
    if x_std.ndim != 2 or x_std.shape[1] != 8:
        raise ValueError(f"x must have shape (d, 8), got {tuple(x_std.shape)}")
    return x_std.reshape(-1)


def sign_aligned_distance(x_true: Tensor, x_est: Tensor) -> Tensor:
    """
    Global-sign aligned distance:
        d_pm(x_true, x_est) = min(||x_true - x_est||_2, ||x_true + x_est||_2).
    This metric corresponds to the ambiguity x ~ -x only.
    """
    x_true_std = ensure_octonion_tensor(x_true, name="x_true")
    x_est_std = ensure_octonion_tensor(x_est, name="x_est")

    if x_true_std.shape != x_est_std.shape:
        raise ValueError(
            f"Shape mismatch: x_true{tuple(x_true_std.shape)} vs x_est{tuple(x_est_std.shape)}"
        )
    if x_true_std.ndim != 2 or x_true_std.shape[1] != 8:
        raise ValueError(
            f"x_true and x_est must both have shape (d, 8), got {tuple(x_true_std.shape)}"
        )

    d_plus = torch.linalg.norm((x_true_std - x_est_std).reshape(-1), ord=2)
    d_minus = torch.linalg.norm((x_true_std + x_est_std).reshape(-1), ord=2)
    return torch.minimum(d_plus, d_minus)


def best_global_sign(x_true: Tensor, x_est: Tensor) -> int:
    x_true_std = ensure_octonion_tensor(x_true, name="x_true")
    x_est_std = ensure_octonion_tensor(x_est, name="x_est")

    if x_true_std.shape != x_est_std.shape:
        raise ValueError(
            f"Shape mismatch: x_true{tuple(x_true_std.shape)} vs x_est{tuple(x_est_std.shape)}"
        )
    if x_true_std.ndim != 2 or x_true_std.shape[1] != 8:
        raise ValueError(
            f"x_true and x_est must both have shape (d, 8), got {tuple(x_true_std.shape)}"
        )

    d_plus = torch.linalg.norm((x_true_std - x_est_std).reshape(-1), ord=2)
    d_minus = torch.linalg.norm((x_true_std + x_est_std).reshape(-1), ord=2)
    return +1 if bool((d_plus <= d_minus).item()) else -1


def apply_global_sign(x: Tensor, sign: int | float) -> Tensor:
    x_std = ensure_octonion_tensor(x, name="x")
    if sign not in (+1, -1, 1.0, -1.0):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    return x_std * float(sign)


def sign_aligned_estimate(x_true: Tensor, x_est: Tensor) -> Tensor:
    sign = best_global_sign(x_true, x_est)
    return apply_global_sign(x_est, sign)


def absolute_inner_product_similarity(
    x_true: Tensor,
    x_est: Tensor,
    eps: float = 1e-18,
) -> Tensor:
    """
    Absolute inner-product similarity on the flattened real vectors:
        | <x_true/||x_true||, x_est/||x_est||> |.
    """
    v_true = flatten_real_vector(x_true)
    v_est = flatten_real_vector(x_est)

    n_true = torch.linalg.norm(v_true, ord=2).clamp_min(eps)
    n_est = torch.linalg.norm(v_est, ord=2).clamp_min(eps)

    v_true_u = v_true / n_true
    v_est_u = v_est / n_est

    return torch.abs(torch.dot(v_true_u, v_est_u))
