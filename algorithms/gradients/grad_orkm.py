from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from algorithms.constants import DEFAULT_ORKM_OMEGA
from core.octonion_base import ensure_octonion_tensor
from core.octonion_inner import intensity_measurements_explicit, row_energy_batch, row_inner_fast
from core.octonion_inner import row_energy_explicit
from core.octonion_ops import raw_distance, estimate_global_right_phase, apply_global_right_phase
from core.octonion_ops import oct_abs_sq, oct_conj, oct_mul, oct_phase
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
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
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
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
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
    orbit_log_pm: list[float] = []
    orbit_log_absip: list[float] = []
    orbit_log_meas_rel: list[float] = []

    if x_true_proc is not None:
        x_true_std = ensure_octonion_tensor(x_true_proc, name="x_true_proc")
        if x_true_std.shape != x_est.shape:
            raise ValueError(
                f"Shape mismatch: x_true_proc{tuple(x_true_std.shape)} vs x0{tuple(x_est.shape)}"
            )
        d0 = sign_aligned_distance(x_true_std, x_est).item()
        d0_raw = raw_distance(x_true_std, x_est).item()
        absip0 = absolute_inner_product_similarity(x_true_std, x_est).item()
        orbit_log_iters.append(0)
        orbit_log_ln.append(float(torch.log(torch.tensor(max(d0, 1e-300), dtype=torch.float64)).item()))
        orbit_log_raw.append(float(d0_raw))
        orbit_log_pm.append(float(d0))
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

        d_pm = None
        if x_true_std is not None:
            d_pm = sign_aligned_distance(x_true_std, x_est).item()
            d_raw = raw_distance(x_true_std, x_est).item()
            absip = absolute_inner_product_similarity(x_true_std, x_est).item()
            orbit_log_iters.append(it_ + 1)
            orbit_log_ln.append(
                float(torch.log(torch.tensor(max(d_pm, 1e-300), dtype=torch.float64)).item())
            )
            orbit_log_raw.append(float(d_raw))
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
                    log_pm_stop = float(
                        torch.log(torch.tensor(max(d_pm, 1e-300), dtype=torch.float64)).item()
                    )
                    print(
                        f"[grad_orkm] iter={it_ + 1}/{max_iters}, skip={epoch_skips}/{epoch_updates}, "
                        f"dist_sign={d_pm:.6e}, log(dist_sign)={log_pm_stop:.6e}, "
                        f"early_stop=True"
                    )
                break
        if verbose and ((it_ + 1) % progress_every == 0 or (it_ + 1) == max_iters or it_ == 0):
            msg = f"[grad_orkm] iter={it_ + 1}/{max_iters}"
            if d_pm is not None:
                log_pm = float(
                    torch.log(torch.tensor(max(d_pm, 1e-300), dtype=torch.float64)).item()
                )
                msg += f", dist_sign={d_pm:.6e}, log(dist_sign)={log_pm:.6e}"
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





def orkm_single_row_update_fast_beta(
        x: Tensor,
        a_row: Tensor,
        y_l: float | Tensor,
        beta_l: float | Tensor,
        *,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        phase_eps: float = 1e-18,
        beta_eps: float = 1e-18,
        return_info: bool = False,
) -> Tensor | tuple[Tensor, dict[str, float | int | bool | str]]:
    """
    Same ORKM row update as ``orkm_single_row_update_fast``, but ``beta_l`` is supplied
    (row energy) instead of recomputed from ``a_row``.
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

    beta_tensor = torch.as_tensor(beta_l, dtype=torch.float64, device=x_std.device).reshape(())

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

    if bool(beta_tensor <= beta_eps):
        x_out = x_std.clone()
        if return_info:
            info = {
                "skipped": True,
                "skip_reason": "beta_too_small",
                "phase_valid": True,
                "beta_value": float(beta_tensor.item()),
            }
            return x_out, info
        return x_out

    step_scale = omega_scalar / beta_tensor
    delta = oct_mul(a_row_std, r)
    x_new = x_std + step_scale * delta

    if return_info:
        info = {
            "skipped": False,
            "skip_reason": "none",
            "phase_valid": True,
            "beta_value": float(beta_tensor.item()),
        }
        return x_new, info
    return x_new


def _row_inner_batched_for_fixed_row(x_batch: Tensor, a_row: Tensor) -> Tensor:
    """
    x_batch: (B, d, 8), a_row: (d, 8) -> s: (B, 8)
    s_b = sum_j conj(a_row[j]) * x_batch[b, j]
    """
    xb = ensure_octonion_tensor(x_batch, name="x_batch")
    ar = ensure_octonion_tensor(a_row, name="a_row")
    if xb.ndim != 3 or ar.ndim != 2:
        raise ValueError(f"expected x_batch (B,d,8) and a_row (d,8), got {tuple(xb.shape)}, {tuple(ar.shape)}")
    if xb.shape[1:] != ar.shape:
        raise ValueError(f"shape mismatch x_batch{tuple(xb.shape)} vs a_row{tuple(ar.shape)}")
    terms = oct_mul(oct_conj(ar).unsqueeze(0), xb)
    return terms.sum(dim=1)


def orkm_single_row_update_fast_beta_batched(
        x_batch: Tensor,
        a_row: Tensor,
        y_l: Tensor,
        beta_l: float | Tensor,
        *,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        phase_eps: float = 1e-18,
        beta_eps: float = 1e-18,
) -> tuple[Tensor, int]:
    """
    One ORKM row update for batch ``x_batch`` (B, d, 8), shared ``a_row`` (d, 8).
    ``y_l`` has shape (B,). Returns updated ``x`` and number of skipped batch elements.
    """
    xb = ensure_octonion_tensor(x_batch, name="x_batch")
    ar = ensure_octonion_tensor(a_row, name="a_row")
    B = int(xb.shape[0])
    yv = torch.as_tensor(y_l, dtype=torch.float64, device=xb.device).reshape(B)
    omega_scalar = _validate_positive_scalar(omega, device=xb.device, name="omega")
    beta_tensor = torch.as_tensor(beta_l, dtype=torch.float64, device=xb.device).reshape(())

    b_l = torch.sqrt(torch.clamp(yv, min=0.0))
    s = _row_inner_batched_for_fixed_row(xb, ar)
    phase_s, valid = oct_phase(s, eps=phase_eps)
    s_target = b_l.unsqueeze(-1) * phase_s
    r = s_target - s
    skipped = ~valid | (beta_tensor <= beta_eps)
    n_skipped = int(skipped.sum().item())
    if n_skipped == B:
        return xb.clone(), n_skipped

    step_scale = omega_scalar / beta_tensor
    delta = oct_mul(ar.unsqueeze(0), r.unsqueeze(1))
    upd = step_scale * delta
    x_new = torch.where(skipped.unsqueeze(-1).unsqueeze(-1), xb, xb + upd)
    return x_new, n_skipped


def orkm_single_row_update_fast_beta_batched_independent(
        x_batch: Tensor,
        a_rows: Tensor,
        y_l: Tensor,
        beta_row: Tensor,
        *,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        phase_eps: float = 1e-18,
        beta_eps: float = 1e-18,
) -> tuple[Tensor, int]:
    """
    ORKM row update for batch where each index has its own measurement row ``a_rows[b]`` (B, d, 8).
    """
    xb = ensure_octonion_tensor(x_batch, name="x_batch")
    ar = ensure_octonion_tensor(a_rows, name="a_rows")
    if xb.ndim != 3 or ar.ndim != 3:
        raise ValueError(f"expected (B,d,8), got x_batch{tuple(xb.shape)}, a_rows{tuple(ar.shape)}")
    if xb.shape != ar.shape:
        raise ValueError(f"shape mismatch x_batch{tuple(xb.shape)} vs a_rows{tuple(ar.shape)}")
    B = int(xb.shape[0])
    yv = torch.as_tensor(y_l, dtype=torch.float64, device=xb.device).reshape(B)
    omega_scalar = _validate_positive_scalar(omega, device=xb.device, name="omega")
    beta_b = torch.as_tensor(beta_row, dtype=torch.float64, device=xb.device).reshape(B)

    b_l = torch.sqrt(torch.clamp(yv, min=0.0))
    terms = oct_mul(oct_conj(ar), xb)
    s = terms.sum(dim=1)
    phase_s, valid = oct_phase(s, eps=phase_eps)
    s_target = b_l.unsqueeze(-1) * phase_s
    r = s_target - s
    skipped = ~valid | (beta_b <= beta_eps)
    n_skipped = int(skipped.sum().item())
    if n_skipped == B:
        return xb.clone(), n_skipped

    step_scale = (omega_scalar / beta_b).view(B, 1, 1)
    delta = oct_mul(ar, r.unsqueeze(1))
    upd = step_scale * delta
    x_new = torch.where(skipped.unsqueeze(-1).unsqueeze(-1), xb, xb + upd)
    return x_new, n_skipped


def _sign_aligned_distance_batched(x_true: Tensor, x_est: Tensor) -> Tensor:
    """x_true, x_est: (B, d, 8) -> (B,) float64."""
    xt = ensure_octonion_tensor(x_true, name="x_true")
    xe = ensure_octonion_tensor(x_est, name="x_est")
    if xt.shape != xe.shape or xt.ndim != 3 or xt.shape[-1] != 8:
        raise ValueError(f"expected matching (B,d,8), got {tuple(xt.shape)} vs {tuple(xe.shape)}")
    d_plus = torch.linalg.norm((xt - xe).reshape(xt.shape[0], -1), ord=2, dim=1)
    d_minus = torch.linalg.norm((xt + xe).reshape(xt.shape[0], -1), ord=2, dim=1)
    return torch.minimum(d_plus, d_minus)


def grad_orkm_cuda(
        A: Tensor,
        y: Tensor,
        x0: Tensor,
        max_iters: int,
        *,
        beta: Tensor | None = None,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        x_true_proc: Tensor | None = None,
        record_orbit_metrics: bool = False,
        record_meas_rel: bool = False,
        stop_err: float = 0.0,
        verbose: bool = False,
        progress_every: int = 1,
        return_info: bool = False,
) -> Tensor | tuple[Tensor, dict[str, object]]:
    """
    Production-oriented ORKM loop: optional cached ``beta``, no per-epoch diagnostics unless requested.
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

    if beta is None:
        beta_vec = row_energy_batch(A_std).reshape(-1)
    else:
        beta_vec = torch.as_tensor(beta, dtype=torch.float64, device=A_std.device).reshape(-1)
        if beta_vec.shape[0] != n:
            raise ValueError(f"beta must have shape ({n},), got {tuple(beta_vec.shape)}")

    x_est = x_std.clone()
    skip_count_total = 0
    total_row_updates = 0
    epoch_skip_counts: list[int] = []
    epoch_row_update_counts: list[int] = []
    iter_hist: list[int] = [0]
    orbit_log_iters: list[int] = []
    orbit_log_ln: list[float] = []
    orbit_log_raw: list[float] = []
    orbit_log_pm: list[float] = []
    orbit_log_absip: list[float] = []
    orbit_log_meas_rel: list[float] = []

    want_orbit = record_orbit_metrics and (x_true_proc is not None)
    want_meas = record_meas_rel and (x_true_proc is not None)
    need_x_true = x_true_proc is not None and (
            record_orbit_metrics or record_meas_rel or stop_err > 0.0
    )
    x_true_std: Tensor | None = None
    if need_x_true:
        x_true_std = ensure_octonion_tensor(x_true_proc, name="x_true_proc")
        if x_true_std.shape != x_est.shape:
            raise ValueError(
                f"Shape mismatch: x_true_proc{tuple(x_true_std.shape)} vs x0{tuple(x_est.shape)}"
            )
    if want_orbit:
        d0 = sign_aligned_distance(x_true_std, x_est).item()
        d0_raw = raw_distance(x_true_std, x_est).item()
        absip0 = absolute_inner_product_similarity(x_true_std, x_est).item()
        orbit_log_iters.append(0)
        orbit_log_ln.append(float(torch.log(torch.tensor(max(d0, 1e-300), dtype=torch.float64)).item()))
        orbit_log_raw.append(float(d0_raw))
        orbit_log_pm.append(float(d0))
        orbit_log_absip.append(float(absip0))
        if record_meas_rel:
            q0 = estimate_global_right_phase(x_true_std, x_est)
            x0_al = apply_global_right_phase(x_est, q0)
            y_hat0 = intensity_measurements_explicit(A_std, x0_al)
            mr0 = float((torch.linalg.norm(y_hat0 - y_std) / torch.linalg.norm(y_std)).item())
            orbit_log_meas_rel.append(mr0)
    elif want_meas:
        assert x_true_std is not None
        q0 = estimate_global_right_phase(x_true_std, x_est)
        x0_al = apply_global_right_phase(x_est, q0)
        y_hat0 = intensity_measurements_explicit(A_std, x0_al)
        mr0 = float((torch.linalg.norm(y_hat0 - y_std) / torch.linalg.norm(y_std)).item())
        orbit_log_meas_rel.append(mr0)

    if verbose:
        print(f"[grad_orkm_cuda] start: n={n}, max_iters={max_iters}, omega={float(torch.as_tensor(omega).item()):.3f}")

    for it_ in range(max_iters):
        perm = torch.randperm(n, device=A_std.device)
        epoch_skips = 0
        epoch_updates = 0
        for l in perm.tolist():
            x_est, row_info = orkm_single_row_update_fast_beta(
                x_est,
                A_std[l],
                y_std[l],
                beta_vec[l],
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

        d_pm = None
        if x_true_std is not None:
            d_pm = sign_aligned_distance(x_true_std, x_est).item()
            if want_orbit:
                assert x_true_std is not None
                d_raw = raw_distance(x_true_std, x_est).item()
                absip = absolute_inner_product_similarity(x_true_std, x_est).item()
                orbit_log_iters.append(it_ + 1)
                orbit_log_ln.append(
                    float(torch.log(torch.tensor(max(d_pm, 1e-300), dtype=torch.float64)).item())
                )
                orbit_log_raw.append(float(d_raw))
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
                    log_pm_stop = float(
                        torch.log(torch.tensor(max(d_pm, 1e-300), dtype=torch.float64)).item()
                    )
                    print(
                        f"[grad_orkm_cuda] iter={it_ + 1}/{max_iters}, skip={epoch_skips}/{epoch_updates}, "
                        f"dist_sign={d_pm:.6e}, log(dist_sign)={log_pm_stop:.6e}, early_stop=True"
                    )
                break
        if verbose and ((it_ + 1) % progress_every == 0 or (it_ + 1) == max_iters or it_ == 0):
            msg = f"[grad_orkm_cuda] iter={it_ + 1}/{max_iters}"
            if d_pm is not None:
                log_pm = float(
                    torch.log(torch.tensor(max(d_pm, 1e-300), dtype=torch.float64)).item()
                )
                msg += f", dist_sign={d_pm:.6e}, log(dist_sign)={log_pm:.6e}"
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
        "orbit_log_pm": orbit_log_pm,
        "orbit_log_absip": orbit_log_absip,
        "orbit_log_meas_rel": orbit_log_meas_rel,
    }
    if verbose:
        print(
            f"[grad_orkm_cuda] done: executed_iters={len(epoch_row_update_counts)}, "
            f"total_skip_ratio={skip_ratio:.2%}"
        )
    return x_est, info


def grad_orkm_cuda_batched(
        A: Tensor,
        y: Tensor,
        x0: Tensor,
        max_iters: int,
        *,
        beta: Tensor | None = None,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        record_orbit_metrics: bool = False,
        record_meas_rel: bool = False,
        stop_err: float = 0.0,
        verbose: bool = False,
        progress_every: int = 1,
        return_info: bool = False,
) -> Tensor | tuple[Tensor, dict[str, object]]:
    """
    Batched ORKM: ``x0`` / ``y`` have leading batch dim ``B``; all patches share the same ``A``.
    Each epoch uses one shared ``torch.randperm(n)`` for every batch element (valid Kaczmarz order).
    """
    if record_orbit_metrics or record_meas_rel or stop_err > 0.0:
        raise NotImplementedError(
            "grad_orkm_cuda_batched only supports production mode: "
            "record_orbit_metrics=False, record_meas_rel=False, stop_err=0"
        )

    A_std = ensure_octonion_tensor(A, name="A")
    if A_std.ndim != 3:
        raise ValueError(f"A must have shape (n, d, 8), got {tuple(A_std.shape)}")
    n, d, _ = A_std.shape

    xb = ensure_octonion_tensor(x0, name="x0")
    if xb.ndim != 3:
        raise ValueError(f"x0 must have shape (B, d, 8), got {tuple(xb.shape)}")
    if xb.shape[1] != d:
        raise ValueError(f"x0 d mismatch: got {xb.shape[1]}, A has d={d}")
    B = int(xb.shape[0])

    y_std = torch.as_tensor(y, dtype=torch.float64, device=A_std.device)
    if y_std.ndim != 2 or y_std.shape[0] != B or y_std.shape[1] != n:
        raise ValueError(f"y must be (B, {n}), got {tuple(y_std.shape)}")

    if max_iters < 0:
        raise ValueError(f"max_iters must be non-negative, got {max_iters}")
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")
    omega_scalar = _validate_positive_scalar(omega, device=A_std.device, name="omega")

    if beta is None:
        beta_vec = row_energy_batch(A_std).reshape(-1)
    else:
        beta_vec = torch.as_tensor(beta, dtype=torch.float64, device=A_std.device).reshape(-1)
        if beta_vec.shape[0] != n:
            raise ValueError(f"beta must have shape ({n},), got {tuple(beta_vec.shape)}")

    x_est = xb.clone()
    skip_count_total = 0
    total_row_updates = 0
    epoch_skip_counts: list[int] = []
    epoch_row_update_counts: list[int] = []

    if verbose:
        print(
            f"[grad_orkm_cuda_batched] B={B}, n={n}, d={d}, max_iters={max_iters}, "
            f"omega={float(omega_scalar.item()):.3f}"
        )

    for it_ in range(max_iters):
        perm = torch.randperm(n, device=A_std.device)
        epoch_skips = 0
        epoch_updates = 0
        for l in perm.tolist():
            x_est, n_sk = orkm_single_row_update_fast_beta_batched(
                x_est,
                A_std[l],
                y_std[:, l],
                beta_vec[l],
                omega=omega_scalar,
            )
            epoch_skips += n_sk
            epoch_updates += B

        total_row_updates += epoch_updates
        skip_count_total += epoch_skips
        epoch_skip_counts.append(epoch_skips)
        epoch_row_update_counts.append(epoch_updates)

        if verbose and ((it_ + 1) % progress_every == 0 or (it_ + 1) == max_iters or it_ == 0):
            print(f"[grad_orkm_cuda_batched] iter={it_ + 1}/{max_iters}, epoch_skips={epoch_skips}")

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


def grad_orkm_cuda_batched_independent_A(
        A: Tensor,
        y: Tensor,
        x0: Tensor,
        max_iters: int,
        *,
        x_true: Tensor | None = None,
        stop_err: float = 0.0,
        omega: float | Tensor = DEFAULT_ORKM_OMEGA,
        verbose: bool = False,
        progress_every: int = 1,
        return_info: bool = False,
) -> Tensor | tuple[Tensor, dict[str, object]]:
    """
    Batched ORKM with a **different** measurement tensor per batch index:
    ``A`` is ``(B, n, d, 8)``, ``y`` / ``x0`` are ``(B, n)`` / ``(B, d, 8)``.
    Each epoch uses one shared ``torch.randperm(n)`` across the batch.

    If ``stop_err > 0``, pass ``x_true`` (B, d, 8): oracle early-stop when
    ``sign_aligned_distance <= stop_err`` (per batch element). Converged
    indices stop receiving updates.
    """
    A_std = ensure_octonion_tensor(A, name="A")
    if A_std.ndim != 4:
        raise ValueError(f"A must have shape (B, n, d, 8), got {tuple(A_std.shape)}")
    B, n, d, _ = A_std.shape
    if _ != 8:
        raise ValueError(f"A last dim must be 8, got {tuple(A_std.shape)}")

    xb = ensure_octonion_tensor(x0, name="x0")
    if xb.shape != (B, d, 8):
        raise ValueError(f"x0 must be (B, d, 8) with B={B}, d={d}, got {tuple(xb.shape)}")

    y_std = torch.as_tensor(y, dtype=torch.float64, device=A_std.device)
    if y_std.shape != (B, n):
        raise ValueError(f"y must be (B, n)=({B},{n}), got {tuple(y_std.shape)}")

    if max_iters < 0:
        raise ValueError(f"max_iters must be non-negative, got {max_iters}")
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")
    omega_scalar = _validate_positive_scalar(omega, device=A_std.device, name="omega")

    want_stop = stop_err > 0.0
    x_true_std: Tensor | None = None
    if want_stop:
        if x_true is None:
            raise ValueError("x_true is required when stop_err > 0")
        x_true_std = ensure_octonion_tensor(x_true, name="x_true")
        if x_true_std.shape != (B, d, 8):
            raise ValueError(f"x_true must be (B, d, 8)=({B},{d},8), got {tuple(x_true_std.shape)}")

    beta_bn = oct_abs_sq(A_std).sum(dim=2)

    x_est = xb.clone()
    skip_count_total = 0
    total_row_updates = 0
    epoch_skip_counts: list[int] = []
    epoch_row_update_counts: list[int] = []

    done = torch.zeros(B, dtype=torch.bool, device=A_std.device)
    conv_epoch = torch.zeros(B, dtype=torch.int64, device=A_std.device)

    if verbose:
        print(
            f"[grad_orkm_cuda_batched_independent_A] B={B}, n={n}, d={d}, max_iters={max_iters}, "
            f"stop_err={stop_err}, omega={float(omega_scalar.item()):.3f}"
        )

    for it_ in range(max_iters):
        if want_stop and bool(done.all().item()):
            break
        perm = torch.randperm(n, device=A_std.device)
        epoch_skips = 0
        epoch_updates = 0
        for l in perm.tolist():
            active = ~done if want_stop else torch.ones(B, dtype=torch.bool, device=A_std.device)
            if want_stop and not bool(active.any().item()):
                break
            x_new, n_sk = orkm_single_row_update_fast_beta_batched_independent(
                x_est,
                A_std[:, l, :, :],
                y_std[:, l],
                beta_bn[:, l],
                omega=omega_scalar,
            )
            if want_stop:
                m = active.unsqueeze(-1).unsqueeze(-1)
                x_est = torch.where(m, x_new, x_est)
            else:
                x_est = x_new
            epoch_skips += n_sk
            epoch_updates += B

        total_row_updates += epoch_updates
        skip_count_total += epoch_skips
        epoch_skip_counts.append(epoch_skips)
        epoch_row_update_counts.append(epoch_updates)

        if want_stop and x_true_std is not None:
            d_pm = _sign_aligned_distance_batched(x_true_std, x_est)
            thr = torch.as_tensor(stop_err, dtype=torch.float64, device=A_std.device)
            newly = (~done) & (d_pm <= thr)
            if bool(newly.any().item()):
                conv_epoch = torch.where(newly, torch.full_like(conv_epoch, it_ + 1), conv_epoch)
                done = done | newly
            if bool(done.all().item()):
                if verbose:
                    print(f"[grad_orkm_cuda_batched_independent_A] all converged at iter={it_ + 1}")
                break

        if verbose and ((it_ + 1) % progress_every == 0 or (it_ + 1) == max_iters or it_ == 0):
            msg = f"[grad_orkm_cuda_batched_independent_A] iter={it_ + 1}/{max_iters}, epoch_skips={epoch_skips}"
            if want_stop and x_true_std is not None:
                dm = float(d_pm.max().item())  # type: ignore[name-defined]
                msg += f", max_dist_sign={dm:.6e}, done={int(done.sum().item())}/{B}"
            print(msg)

    if want_stop:
        not_done = ~done
        if bool(not_done.any().item()):
            conv_epoch = torch.where(not_done, torch.full_like(conv_epoch, max_iters), conv_epoch)

    if not return_info:
        return x_est

    skip_ratio = 0.0 if total_row_updates == 0 else skip_count_total / total_row_updates
    info: dict[str, object] = {
        "skip_count_total": int(skip_count_total),
        "total_row_updates": int(total_row_updates),
        "skip_ratio": float(skip_ratio),
        "epoch_skip_counts": epoch_skip_counts,
        "epoch_row_update_counts": epoch_row_update_counts,
        "per_trial_epochs": conv_epoch.detach().cpu().tolist() if want_stop else [max_iters] * B,
    }
    return x_est, info
