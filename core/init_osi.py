from __future__ import annotations

import torch
from torch import Tensor

from core.octonion_base import ensure_octonion_tensor
from core.octonion_inner import row_energy_explicit, row_inner_explicit
from core.octonion_metric import oct_array_norm
from core.octonion_ops import oct_mul


def normalize_oct_signal(x: Tensor, eps: float = 1e-18) -> Tensor:
    """
    Normalize explicit octonion signal x (shape: (d, 8)) to unit array norm.
    """
    x_std = ensure_octonion_tensor(x, name="x")
    if x_std.ndim != 2:
        raise ValueError(f"x must have shape (d, 8), got {tuple(x_std.shape)}")
    nrm = oct_array_norm(x_std)
    if bool(nrm <= eps):
        raise ValueError("cannot normalize near-zero octonion signal")
    return x_std / nrm


def init_osi(
    A: Tensor,
    y: Tensor,
    *,
    power_iters: int = 5,
    eps: float = 1e-18,
) -> Tensor:
    """
    OSI (Octonion Spectral Initialization), explicit and matrix-free.

    Iterative operator in fixed order:
        x_next[j] += (y_l / beta_l) * (a_lj * s_l(x))
    where
        s_l(x) = sum_t conj(a_lt) * x_t
        beta_l = sum_t |a_lt|^2
    """
    A_std = ensure_octonion_tensor(A, name="A")
    if A_std.ndim != 3:
        raise ValueError(f"A must have shape (n, d, 8), got {tuple(A_std.shape)}")
    n, d, _ = A_std.shape

    y_std = torch.as_tensor(y, dtype=torch.float64, device=A_std.device).reshape(-1)
    if y_std.shape[0] != n:
        raise ValueError(f"y must have shape ({n},), got {tuple(y_std.shape)}")

    x = torch.randn(d, 8, dtype=torch.float64, device=A_std.device)
    x = normalize_oct_signal(x, eps=eps)

    y_pos = torch.clamp(y_std, min=0.0)
    for _ in range(power_iters):
        x_next = torch.zeros_like(x, dtype=torch.float64, device=A_std.device)
        for l in range(n):
            a_row = A_std[l]
            beta_l = row_energy_explicit(a_row)
            if bool(beta_l <= eps):
                continue
            s_l = row_inner_explicit(a_row, x)
            weight = y_pos[l] / beta_l
            for j in range(d):
                x_next[j] = x_next[j] + weight * oct_mul(a_row[j], s_l)

        nrm_next = oct_array_norm(x_next)
        if bool(nrm_next <= eps):
            x_next = torch.randn(d, 8, dtype=torch.float64, device=A_std.device)
        x = normalize_oct_signal(x_next, eps=eps)

    # Real scalar scale estimate.
    scale = torch.sqrt(torch.mean(y_pos))
    return scale * x
