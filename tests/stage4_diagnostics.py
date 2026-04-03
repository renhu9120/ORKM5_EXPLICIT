from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from algorithms.gradients.grad_orkm import orkm_single_row_update_fast
from core.octonion_base import ensure_octonion_tensor, oct_zero
from core.octonion_inner import row_energy_explicit, row_inner_batch_fast, row_inner_fast
from core.octonion_metric import oct_array_norm
from core.octonion_ops import oct_abs, oct_conj, oct_mul, oct_phase


def compute_single_row_objects(
    x: Tensor,
    a_row: Tensor,
    y_l: float | Tensor,
    *,
    phase_eps: float = 1e-18,
) -> Dict[str, Tensor | bool | float]:
    x_std = ensure_octonion_tensor(x, name="x")
    a_row_std = ensure_octonion_tensor(a_row, name="a_row")
    y_scalar = torch.as_tensor(y_l, dtype=torch.float64, device=x_std.device).reshape(())
    b_l = torch.sqrt(torch.clamp(y_scalar, min=0.0))
    s = row_inner_fast(a_row_std, x_std)
    phase_s, valid = oct_phase(s, eps=phase_eps)
    s_target = b_l * phase_s if bool(valid.item()) else torch.zeros_like(s)
    e_l = s_target - s if bool(valid.item()) else torch.zeros_like(s)
    beta_l = row_energy_explicit(a_row_std)
    amp_s = oct_abs(s)
    r_amp = amp_s - b_l
    abs_r_amp = torch.abs(r_amp)
    abs_e = oct_abs(e_l)
    return {
        "s": s,
        "b_l": b_l,
        "phase_s": phase_s,
        "valid": bool(valid.item()),
        "s_target": s_target,
        "e_l": e_l,
        "beta_l": beta_l,
        "r_amp": r_amp,
        "abs_r_amp": abs_r_amp,
        "abs_e": abs_e,
    }


def compute_transfer_operator(A: Tensor, m: int, l: int, e: Tensor, beta_l: float | Tensor) -> Tensor:
    A_std = ensure_octonion_tensor(A, name="A")
    e_std = ensure_octonion_tensor(e, name="e")
    beta = torch.as_tensor(beta_l, dtype=torch.float64, device=A_std.device).reshape(())
    if A_std.ndim != 3:
        raise ValueError(f"A must be (n,d,8), got {tuple(A_std.shape)}")
    if e_std.ndim != 1:
        raise ValueError(f"e must be (8,), got {tuple(e_std.shape)}")
    d = A_std.shape[1]
    acc = oct_zero(device=A_std.device)
    for j in range(d):
        # strict bracketing: conj(a_mj) * (a_lj * e)
        term = oct_mul(oct_conj(A_std[m, j]), oct_mul(A_std[l, j], e_std))
        acc = acc + term
    return acc / beta


def compute_transfer_operator_wrong(A: Tensor, m: int, l: int, e: Tensor, beta_l: float | Tensor) -> Tensor:
    A_std = ensure_octonion_tensor(A, name="A")
    e_std = ensure_octonion_tensor(e, name="e")
    beta = torch.as_tensor(beta_l, dtype=torch.float64, device=A_std.device).reshape(())
    d = A_std.shape[1]
    acc = oct_zero(device=A_std.device)
    for j in range(d):
        # wrong bracketing on purpose: (conj(a_mj) * a_lj) * e
        term = oct_mul(oct_mul(oct_conj(A_std[m, j]), A_std[l, j]), e_std)
        acc = acc + term
    return acc / beta


def compute_eta_matrix(A: Tensor) -> Tuple[Tensor, Tensor]:
    A_std = ensure_octonion_tensor(A, name="A")
    if A_std.ndim != 3:
        raise ValueError(f"A must be (n,d,8), got {tuple(A_std.shape)}")
    n, d, _ = A_std.shape
    beta = torch.zeros(n, dtype=torch.float64, device=A_std.device)
    abs_rows = torch.zeros(n, d, dtype=torch.float64, device=A_std.device)
    for l in range(n):
        beta[l] = row_energy_explicit(A_std[l])
        abs_rows[l] = oct_abs(A_std[l])
    eta = torch.zeros(n, n, dtype=torch.float64, device=A_std.device)
    for m in range(n):
        for l in range(n):
            eta[m, l] = torch.sum(abs_rows[m] * abs_rows[l]) / beta[l]
    return eta, beta


def compute_residual_vector(A: Tensor, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
    A_std = ensure_octonion_tensor(A, name="A")
    x_std = ensure_octonion_tensor(x, name="x")
    y_std = torch.as_tensor(y, dtype=torch.float64, device=A_std.device).reshape(-1)
    s_all = row_inner_batch_fast(A_std, x_std)
    amp_all = oct_abs(s_all)
    b_all = torch.sqrt(torch.clamp(y_std, min=0.0))
    r_all = amp_all - b_all
    abs_r_all = torch.abs(r_all)
    return {
        "s_all": s_all,
        "amp_all": amp_all,
        "b_all": b_all,
        "r_all": r_all,
        "abs_r_all": abs_r_all,
    }


def compute_surrogate_E1(A: Tensor, x: Tensor, y: Tensor) -> float:
    out = compute_residual_vector(A, x, y)
    return float(out["abs_r_all"].sum().item())


def apply_single_row_update_with_info(
    x: Tensor,
    a_row: Tensor,
    y_l: float | Tensor,
    *,
    omega: float,
    phase_eps: float = 1e-18,
    beta_eps: float = 1e-18,
) -> Tuple[Tensor, Dict]:
    diag = compute_single_row_objects(x, a_row, y_l, phase_eps=phase_eps)
    x_new, row_info = orkm_single_row_update_fast(
        x,
        a_row,
        y_l,
        omega=omega,
        phase_eps=phase_eps,
        beta_eps=beta_eps,
        return_info=True,
    )
    merged = dict(row_info)
    merged.update(diag)
    return x_new, merged


def summarize_errors(errors: Sequence[float], tol: float) -> Dict[str, float]:
    if len(errors) == 0:
        return {
            "count": 0.0,
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "pass_rate": float("nan"),
        }
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "count": float(arr.size),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "pass_rate": float(np.mean(arr <= tol)),
    }


def pretty_print_stats(prefix: str, stats: Dict[str, float]) -> List[str]:
    lines = [prefix]
    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"{k}={v:.6e}")
        else:
            lines.append(f"{k}={v}")
    return lines


def save_txt_log(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv_rows(path: Path, headers: Sequence[str], rows: Iterable[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(headers))
        writer.writerows(rows)
