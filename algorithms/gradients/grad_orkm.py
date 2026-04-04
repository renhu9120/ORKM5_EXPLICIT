from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from core.octonion_base import ensure_octonion_tensor
from core.octonion_align import apply_global_right_phase, estimate_global_right_phase, right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit, row_energy_explicit, row_inner_fast
from core.octonion_metric import raw_distance
from core.octonion_ops import oct_mul, oct_phase
from core.octonion_sign import absolute_inner_product_similarity, sign_aligned_distance


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


def _normalize_perm(perm: Sequence[int] | Tensor, n: int, device: torch.device) -> Tensor:
    perm_tensor = torch.as_tensor(perm, dtype=torch.long, device=device).reshape(-1)
    if perm_tensor.numel() != n:
        raise ValueError(f"perm length must be {n}, got {perm_tensor.numel()}")
    if torch.any((perm_tensor < 0) | (perm_tensor >= n)):
        raise ValueError("perm contains out-of-range indices")
    return perm_tensor


def orkm_single_row_update_fast(
        x: Tensor,
        a_row: Tensor,
        y_l: float | Tensor,
        *,
        omega: float | Tensor = 1.0,
        phase_eps: float = 1e-18,
        beta_eps: float = 1e-18,
        return_info: bool = False,
) -> Tensor | tuple[Tensor, dict[str, float | int | bool | str]]:
    """
    Vectorized explicit octonion Kaczmarz single-row update.
    Mathematical rule is identical to orkm_single_row_update.
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

    s = row_inner_fast(a_row_std, x_std)
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
    delta = oct_mul(a_row_std, r)
    x_new = x_std + step_scale * delta

    if return_info:
        info = {
            "skipped": False,
            "skip_reason": "none",
            "phase_valid": True,
            "beta_value": float(beta_l.item()),
        }
        return x_new, info
    return x_new


def grad_orkm(
        A: Tensor,
        y: Tensor,
        x0: Tensor,
        max_iters: int,
        *,
        omega: float | Tensor = 1.0,
        x_true_proc: Tensor | None = None,
        record_meas_rel: bool = False,
        stop_err: float = 0.0,
        verbose: bool = False,
        progress_every: int = 1,
        return_info: bool = False,
) -> Tensor | tuple[Tensor, dict[str, object]]:
    """
    Strictly-equivalent main loop with vectorized single-row internals.
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
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")
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
    iter_hist: list[int] = [0]
    orbit_log_iters: list[int] = []
    orbit_log_ln: list[float] = []
    orbit_log_raw: list[float] = []
    orbit_log_align: list[float] = []
    orbit_log_pm: list[float] = []
    orbit_log_absip: list[float] = []
    orbit_log_meas_rel: list[float] = []

    if x_true_proc is not None:
        x_true_std = ensure_octonion_tensor(x_true_proc, name="x_true_proc")
        if x_true_std.shape != x_est.shape:
            raise ValueError(
                f"Shape mismatch: x_true_proc{tuple(x_true_std.shape)} vs x0{tuple(x_est.shape)}"
            )
        d0 = right_aligned_distance(x_true_std, x_est).item()
        d0_raw = raw_distance(x_true_std, x_est).item()
        d0_pm = sign_aligned_distance(x_true_std, x_est).item()
        absip0 = absolute_inner_product_similarity(x_true_std, x_est).item()
        orbit_log_iters.append(0)
        orbit_log_ln.append(float(torch.log(torch.tensor(max(d0, 1e-300), dtype=torch.float64)).item()))
        orbit_log_raw.append(float(d0_raw))
        orbit_log_align.append(float(d0))
        orbit_log_pm.append(float(d0_pm))
        orbit_log_absip.append(float(absip0))
        if record_meas_rel:
            q0 = estimate_global_right_phase(x_true_std, x_est)
            x0_al = apply_global_right_phase(x_est, q0)
            y_hat0 = intensity_measurements_explicit(A_std, x0_al)
            mr0 = float((torch.linalg.norm(y_hat0 - y_std) / torch.linalg.norm(y_std)).item())
            orbit_log_meas_rel.append(mr0)
    else:
        x_true_std = None
    if verbose:
        print(f"[grad_orkm] start: n={n}, max_iters={max_iters}, omega={float(torch.as_tensor(omega).item()):.3f}")

    for it_ in range(max_iters):
        perm = torch.randperm(n, device=A_std.device)
        epoch_skips = 0
        epoch_updates = 0
        for l in perm.tolist():
            x_est, row_info = orkm_single_row_update_fast(
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
        iter_hist.append(it_ + 1)

        d_align = None
        d_pm = None
        if x_true_std is not None:
            d_align = right_aligned_distance(x_true_std, x_est).item()
            d_raw = raw_distance(x_true_std, x_est).item()
            d_pm = sign_aligned_distance(x_true_std, x_est).item()
            absip = absolute_inner_product_similarity(x_true_std, x_est).item()
            orbit_log_iters.append(it_ + 1)
            orbit_log_ln.append(
                float(torch.log(torch.tensor(max(d_align, 1e-300), dtype=torch.float64)).item())
            )
            orbit_log_raw.append(float(d_raw))
            orbit_log_align.append(float(d_align))
            orbit_log_pm.append(float(d_pm))
            orbit_log_absip.append(float(absip))
            if record_meas_rel:
                q_it = estimate_global_right_phase(x_true_std, x_est)
                x_al = apply_global_right_phase(x_est, q_it)
                y_hat = intensity_measurements_explicit(A_std, x_al)
                mr = float((torch.linalg.norm(y_hat - y_std) / torch.linalg.norm(y_std)).item())
                orbit_log_meas_rel.append(mr)
            if stop_err > 0.0 and d_pm <= stop_err:
                if verbose:
                    log_align = float(
                        torch.log(
                            torch.tensor(max(d_align, 1e-300), dtype=torch.float64)
                        ).item()
                    )
                    log_pm_stop = float(
                        torch.log(torch.tensor(max(d_pm, 1e-300), dtype=torch.float64)).item()
                    )
                    print(
                        f"[grad_orkm] iter={it_ + 1}/{max_iters}, skip={epoch_skips}/{epoch_updates}, "
                        f"dist_sign={d_pm:.6e}, log(dist_sign)={log_pm_stop:.6e}, "
                        f"dist_align(ref)={d_align:.6e}, log(dist_align)={log_align:.6e}, early_stop=True"
                    )
                break
        if verbose and ((it_ + 1) % progress_every == 0 or (it_ + 1) == max_iters or it_ == 0):
            msg = f"[grad_orkm] iter={it_ + 1}/{max_iters}"
            if d_pm is not None:
                log_pm = float(
                    torch.log(torch.tensor(max(d_pm, 1e-300), dtype=torch.float64)).item()
                )
                msg += f", dist_sign={d_pm:.6e}, log(dist_sign)={log_pm:.6e}"
                if d_align is not None:
                    log_align = float(
                        torch.log(
                            torch.tensor(max(d_align, 1e-300), dtype=torch.float64)
                        ).item()
                    )
                    msg += f", dist_align(ref)={d_align:.6e}, log(dist_align)={log_align:.6e}"
                if orbit_log_meas_rel:
                    msg += f", meas_rel={orbit_log_meas_rel[-1]:.6e}"
            else:
                msg += ", dist_sign=N/A, log(dist_sign)=N/A"
            print(msg)

    if not return_info:
        return x_est

    skip_ratio = 0.0 if total_row_updates == 0 else skip_count_total / total_row_updates
    info = {
        "skip_count_total": int(skip_count_total),
        "total_row_updates": int(total_row_updates),
        "skip_ratio": float(skip_ratio),
        "epoch_skip_counts": epoch_skip_counts,
        "epoch_row_update_counts": epoch_row_update_counts,
        "iter": iter_hist,
        "orbit_log_iters": orbit_log_iters,
        "orbit_log_ln": orbit_log_ln,
        "orbit_log_raw": orbit_log_raw,
        "orbit_log_align": orbit_log_align,
        "orbit_log_pm": orbit_log_pm,
        "orbit_log_absip": orbit_log_absip,
        "orbit_log_meas_rel": orbit_log_meas_rel,
    }
    if verbose:
        print(
            f"[grad_orkm] done: executed_iters={len(epoch_row_update_counts)}, "
            f"total_skip_ratio={skip_ratio:.2%}"
        )
    return x_est, info
