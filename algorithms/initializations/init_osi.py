from __future__ import annotations

import torch
from torch import Tensor

from core.octonion_base import ensure_octonion_tensor
from core.octonion_inner import row_energy_batch, row_inner_batch_fast
from core.octonion_inner import row_energy_explicit, row_inner_fast
from core.octonion_ops import oct_abs_sq, oct_conj, oct_mul, normalize_oct_signal, oct_array_norm


def init_osi_cuda(
        A: Tensor,
        y: Tensor,
        *,
        power_iters: int = 5,
        beta: Tensor | None = None,
        eps: float = 1e-18,
        verbose: bool = False,
        progress_every: int = 1,
) -> Tensor:
    """
    OSI identical to ``init_osi``, but row energies may be supplied via ``beta`` and the inner
    accumulation over measurement rows is fully vectorized on the GPU.
    """
    A_std = ensure_octonion_tensor(A, name="A")
    if A_std.ndim != 3:
        raise ValueError(f"A must have shape (n, d, 8), got {tuple(A_std.shape)}")
    n, d, _ = A_std.shape

    y_std = torch.as_tensor(y, dtype=torch.float64, device=A_std.device).reshape(-1)
    if y_std.shape[0] != n:
        raise ValueError(f"y must have shape ({n},), got {tuple(y_std.shape)}")
    if power_iters < 0:
        raise ValueError(f"power_iters must be non-negative, got {power_iters}")
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")

    if beta is None:
        beta_vec = row_energy_batch(A_std).reshape(-1)
    else:
        beta_vec = torch.as_tensor(beta, dtype=torch.float64, device=A_std.device).reshape(-1)
        if beta_vec.shape[0] != n:
            raise ValueError(f"beta must have shape ({n},), got {tuple(beta_vec.shape)}")

    x = torch.randn(d, 8, dtype=torch.float64, device=A_std.device)
    x = normalize_oct_signal(x, eps=eps)
    y_pos = torch.clamp(y_std, min=0.0)
    if verbose:
        print(f"[init_osi_cuda] start: n={n}, d={d}, power_iters={power_iters}")

    for it in range(power_iters):
        s = row_inner_batch_fast(A_std, x)
        weight = torch.where(beta_vec > eps, y_pos / beta_vec, torch.zeros_like(y_pos))
        delta = oct_mul(A_std, s.unsqueeze(1))
        x_next = (weight.view(n, 1, 1) * delta).sum(dim=0)

        nrm_next = oct_array_norm(x_next)
        if bool(nrm_next <= eps):
            x_next = torch.randn(d, 8, dtype=torch.float64, device=A_std.device)
        x = normalize_oct_signal(x_next, eps=eps)
        if verbose and ((it + 1) % progress_every == 0 or (it + 1) == power_iters):
            print(
                f"[init_osi_cuda] iter={it + 1}/{power_iters}, x_norm={float(oct_array_norm(x).item()):.6e}"
            )

    scale = torch.sqrt(torch.mean(y_pos))
    if verbose:
        print(f"[init_osi_cuda] done: scale={float(scale.item()):.6e}")
    return scale * x


def _normalize_signal_batched(x: Tensor, *, eps: float) -> Tensor:
    """x: (B, d, 8) -> per-batch unit oct-array norm."""
    if x.ndim != 3 or x.shape[-1] != 8:
        raise ValueError(f"x must be (B,d,8), got {tuple(x.shape)}")
    flat_sq = (x * x).sum(dim=(1, 2), keepdim=True)
    nrm = torch.sqrt(flat_sq.clamp(min=float(eps) ** 2))
    return x / nrm


def init_osi_cuda_batched(
        A: Tensor,
        y: Tensor,
        *,
        power_iters: int = 5,
        beta: Tensor | None = None,
        eps: float = 1e-18,
        verbose: bool = False,
        progress_every: int = 1,
) -> Tensor:
    """
    Batched OSI: ``y`` is (B, n); returns ``x0`` of shape (B, d, 8).
    """
    A_std = ensure_octonion_tensor(A, name="A")
    if A_std.ndim != 3:
        raise ValueError(f"A must have shape (n, d, 8), got {tuple(A_std.shape)}")
    n, d, _ = A_std.shape

    y_std = torch.as_tensor(y, dtype=torch.float64, device=A_std.device)
    if y_std.ndim != 2 or y_std.shape[1] != n:
        raise ValueError(f"y must be (B, {n}), got {tuple(y_std.shape)}")
    B = int(y_std.shape[0])

    if beta is None:
        beta_vec = row_energy_batch(A_std).reshape(-1)
    else:
        beta_vec = torch.as_tensor(beta, dtype=torch.float64, device=A_std.device).reshape(-1)
        if beta_vec.shape[0] != n:
            raise ValueError(f"beta must have shape ({n},), got {tuple(beta_vec.shape)}")

    if power_iters < 0:
        raise ValueError(f"power_iters must be non-negative, got {power_iters}")
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")

    x = torch.randn(B, d, 8, dtype=torch.float64, device=A_std.device)
    x = _normalize_signal_batched(x, eps=eps)
    y_pos = torch.clamp(y_std, min=0.0)
    if verbose:
        print(f"[init_osi_cuda_batched] B={B}, n={n}, d={d}, power_iters={power_iters}")

    for it in range(power_iters):
        terms = oct_mul(oct_conj(A_std).unsqueeze(0), x.unsqueeze(1))
        s = terms.sum(dim=2)
        weight = torch.where(beta_vec.view(1, n) > eps, y_pos / beta_vec.view(1, n), torch.zeros_like(y_pos))
        delta = oct_mul(A_std.unsqueeze(0), s.unsqueeze(2))
        x_next = (weight.view(B, n, 1, 1) * delta).sum(dim=1)

        nrms = torch.sqrt((x_next * x_next).sum(dim=(1, 2)).clamp(min=float(eps) ** 2))
        bad = nrms <= eps
        if bool(bad.any().item()):
            x_next = x_next.clone()
            nb = int(bad.sum().item())
            x_next[bad] = torch.randn(nb, d, 8, dtype=torch.float64, device=A_std.device)
        x = _normalize_signal_batched(x_next, eps=eps)

        if verbose and ((it + 1) % progress_every == 0 or (it + 1) == power_iters):
            print(f"[init_osi_cuda_batched] iter={it + 1}/{power_iters}")

    scale = torch.sqrt(torch.mean(y_pos, dim=1)).view(B, 1, 1)
    if verbose:
        print("[init_osi_cuda_batched] done")
    return scale * x


def init_osi_cuda_batched_A(
        A: Tensor,
        y: Tensor,
        *,
        power_iters: int = 5,
        eps: float = 1e-18,
        verbose: bool = False,
        progress_every: int = 1,
) -> Tensor:
    """
    OSI with independent ``A[b]``: ``A`` is ``(B, n, d, 8)``, ``y`` is ``(B, n)``.
    """
    A_std = ensure_octonion_tensor(A, name="A")
    if A_std.ndim != 4:
        raise ValueError(f"A must be (B, n, d, 8), got {tuple(A_std.shape)}")
    B, n, d, _ = A_std.shape

    y_std = torch.as_tensor(y, dtype=torch.float64, device=A_std.device)
    if y_std.shape != (B, n):
        raise ValueError(f"y must be (B, n)=({B},{n}), got {tuple(y_std.shape)}")

    beta_bn = oct_abs_sq(A_std).sum(dim=2)
    if power_iters < 0:
        raise ValueError(f"power_iters must be non-negative, got {power_iters}")
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")

    x = torch.randn(B, d, 8, dtype=torch.float64, device=A_std.device)
    x = _normalize_signal_batched(x, eps=eps)
    y_pos = torch.clamp(y_std, min=0.0)
    if verbose:
        print(f"[init_osi_cuda_batched_A] B={B}, n={n}, d={d}, power_iters={power_iters}")

    for it in range(power_iters):
        terms = oct_mul(oct_conj(A_std), x.unsqueeze(1))
        s = terms.sum(dim=2)
        weight = torch.where(
            beta_bn > eps,
            y_pos / beta_bn,
            torch.zeros_like(y_pos),
        )
        delta = oct_mul(A_std, s.unsqueeze(2))
        x_next = (weight.unsqueeze(-1).unsqueeze(-1) * delta).sum(dim=1)

        nrms = torch.sqrt((x_next * x_next).sum(dim=(1, 2)).clamp(min=float(eps) ** 2))
        bad = nrms <= eps
        if bool(bad.any().item()):
            x_next = x_next.clone()
            nb = int(bad.sum().item())
            x_next[bad] = torch.randn(nb, d, 8, dtype=torch.float64, device=A_std.device)
        x = _normalize_signal_batched(x_next, eps=eps)

        if verbose and ((it + 1) % progress_every == 0 or (it + 1) == power_iters):
            print(f"[init_osi_cuda_batched_A] iter={it + 1}/{power_iters}")

    scale = torch.sqrt(torch.mean(y_pos, dim=1)).view(B, 1, 1)
    if verbose:
        print("[init_osi_cuda_batched_A] done")
    return scale * x


def init_osi(
        A: Tensor,
        y: Tensor,
        *,
        power_iters: int = 5,
        eps: float = 1e-18,
        verbose: bool = False,
        progress_every: int = 1,
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
    if power_iters < 0:
        raise ValueError(f"power_iters must be non-negative, got {power_iters}")
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")

    x = torch.randn(d, 8, dtype=torch.float64, device=A_std.device)
    x = normalize_oct_signal(x, eps=eps)
    y_pos = torch.clamp(y_std, min=0.0)
    if verbose:
        print(f"[init_osi] start: n={n}, d={d}, power_iters={power_iters}")

    for it in range(power_iters):
        x_next = torch.zeros_like(x, dtype=torch.float64, device=A_std.device)
        for l in range(n):
            a_row = A_std[l]
            beta_l = row_energy_explicit(a_row)
            if bool(beta_l <= eps):
                continue
            s_l = row_inner_fast(a_row, x)
            weight = y_pos[l] / beta_l
            x_next = x_next + weight * oct_mul(a_row, s_l)

        nrm_next = oct_array_norm(x_next)
        if bool(nrm_next <= eps):
            x_next = torch.randn(d, 8, dtype=torch.float64, device=A_std.device)
        x = normalize_oct_signal(x_next, eps=eps)
        if verbose and ((it + 1) % progress_every == 0 or (it + 1) == power_iters):
            print(
                f"[init_osi] iter={it + 1}/{power_iters}, x_norm={float(oct_array_norm(x).item()):.6e}"
            )

    # Real scalar scale estimate.
    scale = torch.sqrt(torch.mean(y_pos))
    if verbose:
        print(f"[init_osi] done: scale={float(scale.item()):.6e}")
    return scale * x
