from __future__ import annotations

import torch
from torch import Tensor

from algorithms.constants import DEFAULT_ORKM_OMEGA
from core.octonion_base import ensure_octonion_tensor
from core.octonion_inner import row_energy_explicit, row_inner_explicit
from core.octonion_ops import oct_mul, oct_phase


def _to_scalar_float64(value: float | Tensor, *, device: torch.device, name: str) -> Tensor:
    scalar = torch.as_tensor(value, dtype=torch.float64, device=device)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be a scalar, got shape {tuple(scalar.shape)}")
    return scalar.reshape(())


def _validate_positive_scalar(value: float | Tensor, *, device: torch.device, name: str) -> Tensor:
    scalar = _to_scalar_float64(value, device=device, name=name)
    if bool(scalar <= 0.0):
        raise ValueError(f"{name} must be > 0, got {float(scalar.item()):.6e}")
    return scalar


def orkm_single_row_update(
    x: Tensor,
    a_row: Tensor,
    y_l: float | Tensor,
    *,
    omega: float | Tensor = DEFAULT_ORKM_OMEGA,
    phase_eps: float = 1e-18,
    beta_eps: float = 1e-18,
    return_info: bool = False,
) -> Tensor | tuple[Tensor, dict[str, float | int | bool | str]]:
    """
    Explicit octonion Kaczmarz single-row update.

    Update rule:
        x_j <- x_j + (omega / beta_l) * (a_lj * r),
    where
        s = sum_j conj(a_lj) * x_j
        phase = oct_phase(s)
        s_target = sqrt(y_l) * phase
        r = s_target - s
        beta_l = sum_j |a_lj|^2
    """
    x_std = ensure_octonion_tensor(x, name="x")
    a_row_std = ensure_octonion_tensor(a_row, name="a_row")
    if x_std.ndim != 2 or a_row_std.ndim != 2:
        raise ValueError(
            f"x and a_row must both have shape (d, 8), got {tuple(x_std.shape)} and {tuple(a_row_std.shape)}"
        )
    if x_std.shape != a_row_std.shape:
        raise ValueError(
            f"Shape mismatch: x{tuple(x_std.shape)} vs a_row{tuple(a_row_std.shape)}"
        )

    y_scalar = _to_scalar_float64(y_l, device=x_std.device, name="y_l")
    omega_scalar = _validate_positive_scalar(omega, device=x_std.device, name="omega")
    b_l = torch.sqrt(torch.clamp(y_scalar, min=0.0))

    s = row_inner_explicit(a_row_std, x_std)
    phase_s, valid = oct_phase(s, eps=phase_eps)
    if not bool(valid.item()):
        x_out = x_std.clone()
        if return_info:
            info = {
                "skipped": True,
                "skip_reason": "phase_invalid",
                "phase_valid": False,
                "beta_value": float("nan"),
            }
            return x_out, info
        return x_out

    s_target = b_l * phase_s
    r = s_target - s

    beta_l = row_energy_explicit(a_row_std)
    if bool(beta_l <= beta_eps):
        x_out = x_std.clone()
        if return_info:
            info = {
                "skipped": True,
                "skip_reason": "beta_too_small",
                "phase_valid": True,
                "beta_value": float(beta_l.item()),
            }
            return x_out, info
        return x_out
    step_scale = omega_scalar / beta_l

    x_new = x_std.clone()
    d = x_new.shape[0]
    for j in range(d):
        delta_j = oct_mul(a_row_std[j], r)
        x_new[j] = x_new[j] + step_scale * delta_j

    if return_info:
        info = {
            "skipped": False,
            "skip_reason": "none",
            "phase_valid": True,
            "beta_value": float(beta_l.item()),
        }
        return x_new, info
    return x_new


def orkm_main(
    A: Tensor,
    y: Tensor,
    x0: Tensor,
    max_iters: int,
    *,
    omega: float | Tensor = DEFAULT_ORKM_OMEGA,
    return_info: bool = False,
) -> Tensor | tuple[Tensor, dict[str, float | int | list[int]]]:
    """
    Main explicit octonion Kaczmarz iterations.

    for k in range(max_iters):
        for l in random_permutation(n):
            x <- single_row_update(x, A[l], y[l])
    """
    A_std = ensure_octonion_tensor(A, name="A")
    x_std = ensure_octonion_tensor(x0, name="x0")
    if A_std.ndim != 3:
        raise ValueError(f"A must have shape (n, d, 8), got {tuple(A_std.shape)}")
    if x_std.ndim != 2:
        raise ValueError(f"x0 must have shape (d, 8), got {tuple(x_std.shape)}")
    if A_std.shape[1] != x_std.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: A.shape={tuple(A_std.shape)}, x0.shape={tuple(x_std.shape)}"
        )
    if max_iters < 0:
        raise ValueError(f"max_iters must be non-negative, got {max_iters}")
    _ = _validate_positive_scalar(omega, device=A_std.device, name="omega")

    n = A_std.shape[0]
    y_std = torch.as_tensor(y, dtype=torch.float64, device=A_std.device).reshape(-1)
    if y_std.shape[0] != n:
        raise ValueError(f"y must have shape ({n},), got {tuple(y_std.shape)}")

    x_est = x_std.clone()
    skip_count_total = 0
    total_row_updates = 0
    epoch_skip_counts: list[int] = []
    epoch_row_update_counts: list[int] = []

    for _ in range(max_iters):
        perm = torch.randperm(n, device=A_std.device)
        epoch_skips = 0
        epoch_updates = 0
        for l in perm.tolist():
            x_est, row_info = orkm_single_row_update(
                x_est,
                A_std[l],
                y_std[l],
                omega=omega,
                return_info=True,
            )
            epoch_updates += 1
            if bool(row_info["skipped"]):
                epoch_skips += 1

        total_row_updates += epoch_updates
        skip_count_total += epoch_skips
        epoch_skip_counts.append(epoch_skips)
        epoch_row_update_counts.append(epoch_updates)

    if not return_info:
        return x_est

    skip_ratio = 0.0 if total_row_updates == 0 else skip_count_total / total_row_updates
    info = {
        "skip_count_total": int(skip_count_total),
        "total_row_updates": int(total_row_updates),
        "skip_ratio": float(skip_ratio),
        "epoch_skip_counts": epoch_skip_counts,
        "epoch_row_update_counts": epoch_row_update_counts,
    }
    return x_est, info
