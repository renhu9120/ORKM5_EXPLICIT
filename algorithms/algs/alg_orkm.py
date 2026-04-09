from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from algorithms.gradients.grad_orkm import (
    grad_orkm_cuda,
    grad_orkm_cuda_batched,
    grad_orkm_cuda_batched_independent_A,
)
from algorithms.initializations.init_osi import (
    init_osi_cuda,
    init_osi_cuda_batched,
    init_osi_cuda_batched_A,
)
from torch import Tensor

from algorithms.constants import DEFAULT_ORKM_OMEGA
from algorithms.gradients.grad_orkm import grad_orkm
from algorithms.initializations.init_osi import init_osi


@torch.no_grad()
def alg_orkm(
        A: Tensor,
        y: Tensor,
        T: int,
        *,
        seed: Optional[int] = None,
        passes: int = 50,
        power_iters: int = 5,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        record_meas_rel: bool = False,
        x_true_proc: Optional[Tensor] = None,
        stop_err: float = 0.0,
        verbose: bool = False,
        progress_every: int = 1,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    ORKM amplitude baseline driver (aligned with OWF stack layout):

      1) spectral initialization from intensity `y` (`init_osi`)
      2) lift to pseudo-real blocks `B = gamma_of_matrix_rows(A)`
      3) amplitude targets `b = sqrt(max(y,0))` from intensity
      4) ORKM iterations (`grad_orkm`)

    Real-image runs should keep ``stop_err=0`` (fixed ``T`` passes). If ``stop_err>0`` and
    ``x_true_proc`` is set, oracle early-stop uses ``sign_aligned_distance``.

    Returns:
      z_final: (8d,) pseudo-real iterate
      history: step diagnostics from `orkm_amplitude`, plus ``z0_proc`` (d,1,8) for metrics/logging.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if A.is_cuda:
            torch.cuda.manual_seed_all(seed)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    max_iters = int(T)

    x0 = init_osi(
        A,
        y,
        power_iters=power_iters,
        verbose=verbose,
        progress_every=progress_every,
    )

    x_est, info = grad_orkm(
        A=A,
        y=y,
        x0=x0,
        max_iters=max_iters,
        omega=omega,
        record_meas_rel=record_meas_rel,
        x_true_proc=x_true_proc,
        stop_err=stop_err,
        verbose=verbose,
        progress_every=progress_every,
        return_info=True,
    )
    info["z0_proc"] = x0
    return x_est, info



@torch.no_grad()
def alg_orkm_cuda(
        A: Tensor,
        y: Tensor,
        T: int,
        *,
        seed: Optional[int] = None,
        power_iters: int = 5,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        beta: Optional[Tensor] = None,
        x_true_proc: Optional[Tensor] = None,
        stop_err: float = 0.0,
        verbose: bool = False,
        progress_every: int = 1,
        record_orbit_metrics: bool = False,
        record_meas_rel: bool = False,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    CUDA/production ORKM driver: same measurement model and row updates as ``alg_orkm``,
    with optional cached ``beta`` and lightweight iteration (no orbit metrics unless requested).

    Real-image runs should keep ``stop_err=0`` (fixed ``T`` passes only). For success-rate
    oracle early-stop, set ``stop_err>0`` and pass ``x_true_proc``; stopping uses
    ``sign_aligned_distance``.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if A.is_cuda:
            torch.cuda.manual_seed_all(seed)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    max_iters = int(T)

    x0 = init_osi_cuda(
        A,
        y,
        power_iters=power_iters,
        beta=beta,
        verbose=verbose,
        progress_every=progress_every,
    )

    x_est, info = grad_orkm_cuda(
        A=A,
        y=y,
        x0=x0,
        max_iters=max_iters,
        beta=beta,
        omega=omega,
        x_true_proc=x_true_proc
        if (record_orbit_metrics or record_meas_rel or stop_err > 0.0)
        else None,
        record_orbit_metrics=record_orbit_metrics,
        record_meas_rel=record_meas_rel,
        stop_err=stop_err,
        verbose=verbose,
        progress_every=progress_every,
        return_info=True,
    )
    info["z0_proc"] = x0
    return x_est, info


@torch.no_grad()
def alg_orkm_cuda_success_rate_batched(
        A: Tensor,
        y: Tensor,
        T: int,
        *,
        seed: Optional[int] = None,
        power_iters: int = 5,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        x_true: Optional[Tensor] = None,
        stop_err: float = 1e-5,
        verbose: bool = False,
        progress_every: int = 1,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Batched driver with independent ``A[b]``: ``A`` is ``(B, n, d, 8)``.

    If ``stop_err > 0`` and ``x_true`` is provided, oracle early-stop uses
    ``sign_aligned_distance``. If ``stop_err <= 0``, runs ``T`` full epochs with
    no oracle stopping.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if A.is_cuda:
            torch.cuda.manual_seed_all(seed)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    max_iters = int(T)
    if stop_err > 0.0 and x_true is None:
        raise ValueError("x_true is required when stop_err > 0")

    x0 = init_osi_cuda_batched_A(
        A,
        y,
        power_iters=power_iters,
        verbose=verbose,
        progress_every=progress_every,
    )
    x_est, info = grad_orkm_cuda_batched_independent_A(
        A=A,
        y=y,
        x0=x0,
        max_iters=max_iters,
        x_true=x_true,
        stop_err=stop_err,
        omega=omega,
        verbose=verbose,
        progress_every=progress_every,
        return_info=True,
    )
    info["z0_proc"] = x0
    return x_est, info


@torch.no_grad()
def alg_orkm_cuda_batched(
        A: Tensor,
        y: Tensor,
        T: int,
        *,
        seed: Optional[int] = None,
        power_iters: int = 5,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        beta: Optional[Tensor] = None,
        verbose: bool = False,
        progress_every: int = 1,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Batched driver: ``y`` is (B, n), returns ``x_est`` (B, d, 8).
    """
    if seed is not None:
        torch.manual_seed(seed)
        if A.is_cuda:
            torch.cuda.manual_seed_all(seed)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    max_iters = int(T)

    x0 = init_osi_cuda_batched(
        A,
        y,
        power_iters=power_iters,
        beta=beta,
        verbose=verbose,
        progress_every=progress_every,
    )
    x_est, info = grad_orkm_cuda_batched(
        A=A,
        y=y,
        x0=x0,
        max_iters=max_iters,
        beta=beta,
        omega=omega,
        verbose=verbose,
        progress_every=progress_every,
        return_info=True,
    )
    info["z0_proc"] = x0
    return x_est, info
