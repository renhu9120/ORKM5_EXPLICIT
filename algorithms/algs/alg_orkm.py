from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
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
