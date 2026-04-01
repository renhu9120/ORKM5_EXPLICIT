from __future__ import annotations

import torch
from torch import Tensor

from core.octonion_base import ensure_octonion_tensor, oct_zero_like


def oct_conj(z: Tensor) -> Tensor:
    """
    Octonion conjugate:
    conj(z0, z1, ..., z7) = (z0, -z1, ..., -z7).
    """
    z_std = ensure_octonion_tensor(z, name="z")
    out = z_std.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_conj(q: Tensor) -> Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_mul(u: Tensor, v: Tensor) -> Tensor:
    u0, u1, u2, u3 = u.unbind(dim=-1)
    v0, v1, v2, v3 = v.unbind(dim=-1)

    r0 = u0 * v0 - u1 * v1 - u2 * v2 - u3 * v3
    r1 = u0 * v1 + u1 * v0 + u2 * v3 - u3 * v2
    r2 = u0 * v2 - u1 * v3 + u2 * v0 + u3 * v1
    r3 = u0 * v3 + u1 * v2 - u2 * v1 + u3 * v0
    return torch.stack((r0, r1, r2, r3), dim=-1)


def oct_mul(p: Tensor, q: Tensor) -> Tensor:
    """
    Implements explicit octonion multiplication in a fixed basis.
    No lifting, embedding, or matrix representation is used.

    High-order expressions in this module must always follow explicit
    parenthesized order due to non-associativity.
    """
    p_std = ensure_octonion_tensor(p, name="p")
    q_std = ensure_octonion_tensor(q, name="q")

    a, b = p_std[..., :4], p_std[..., 4:]
    c, d = q_std[..., :4], q_std[..., 4:]

    left = _quat_mul(a, c) - _quat_mul(_quat_conj(d), b)
    right = _quat_mul(d, a) + _quat_mul(b, _quat_conj(c))
    return torch.cat((left, right), dim=-1)


def oct_square(z: Tensor) -> Tensor:
    """
    Return z * z.
    """
    z_std = ensure_octonion_tensor(z, name="z")
    return oct_mul(z_std, z_std)


def oct_real(z: Tensor) -> Tensor:
    """
    Return real part with shape (...,).
    """
    z_std = ensure_octonion_tensor(z, name="z")
    return z_std[..., 0]


def oct_imag(z: Tensor) -> Tensor:
    """
    Return imaginary components with shape (..., 7).
    """
    z_std = ensure_octonion_tensor(z, name="z")
    return z_std[..., 1:]


def oct_abs_sq(z: Tensor) -> Tensor:
    """
    Octonion squared magnitude computed by coordinate sum of squares.
    """
    z_std = ensure_octonion_tensor(z, name="z")
    return torch.sum(z_std * z_std, dim=-1)


def oct_abs(z: Tensor, eps: float = 0.0) -> Tensor:
    """
    Octonion magnitude with optional lower clamp epsilon.
    """
    z_abs_sq = oct_abs_sq(z)
    if eps > 0.0:
        z_abs_sq = torch.clamp(z_abs_sq, min=float(eps) ** 2)
    return torch.sqrt(z_abs_sq)


def oct_is_zero(z: Tensor, tol: float = 1e-18) -> Tensor:
    """
    Return boolean mask indicating near-zero octonions.
    """
    return oct_abs(z) <= tol


def oct_phase(z: Tensor, eps: float = 1e-18) -> tuple[Tensor, Tensor]:
    """
    Return (phase, valid_mask):
      phase(z) = z / |z| if |z| > eps, else 0.
    """
    z_std = ensure_octonion_tensor(z, name="z")
    mag = oct_abs(z_std)
    valid = mag > eps

    phase = oct_zero_like(z_std)
    if phase.numel() > 0:
        safe_mag = torch.where(valid, mag, torch.ones_like(mag))
        phase = z_std / safe_mag.unsqueeze(-1)
        phase = torch.where(valid.unsqueeze(-1), phase, torch.zeros_like(phase))
    return phase, valid


def oct_normalize(z: Tensor, eps: float = 1e-18) -> tuple[Tensor, Tensor]:
    """
    Geometry-oriented alias of oct_phase.
    """
    return oct_phase(z, eps=eps)


def oct_left_mul(a: Tensor, x: Tensor) -> Tensor:
    """
    Compute a * x.
    """
    return oct_mul(a, x)


def oct_right_mul(x: Tensor, a: Tensor) -> Tensor:
    """
    Compute x * a.
    """
    return oct_mul(x, a)


def oct_left_mul_global(a: Tensor, x: Tensor) -> Tensor:
    """
    Apply global left multiplication: a * x_i for each octonion block in x.
    """
    a_std = ensure_octonion_tensor(a, name="a")
    x_std = ensure_octonion_tensor(x, name="x")
    if a_std.ndim != 1 or a_std.shape[0] != 8:
        raise ValueError(f"a must have shape (8,), got {tuple(a_std.shape)}")
    return oct_mul(a_std, x_std)


def oct_right_mul_global(x: Tensor, a: Tensor) -> Tensor:
    """
    Apply global right multiplication: x_i * a for each octonion block in x.
    """
    x_std = ensure_octonion_tensor(x, name="x")
    a_std = ensure_octonion_tensor(a, name="a")
    if a_std.ndim != 1 or a_std.shape[0] != 8:
        raise ValueError(f"a must have shape (8,), got {tuple(a_std.shape)}")
    return oct_mul(x_std, a_std)
