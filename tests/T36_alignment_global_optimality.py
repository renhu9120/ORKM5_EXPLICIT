from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.octonion_align import apply_global_right_phase
from core.octonion_base import ensure_octonion_tensor, oct_basis, oct_one
from core.octonion_metric import normalize_oct_signal, raw_distance
from core.octonion_ops import oct_abs, oct_conj, oct_mul
from utils.export_latex import plot_gap_distribution, save_table_csv, save_table_latex


def _fmt(value: float) -> str:
    return f"{float(value):.6e}"


def random_octonion(
    shape: Tuple[int, ...] | List[int],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    return torch.randn(*tuple(shape), 8, device=device, dtype=dtype)


def random_unit_octonion(
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-18,
) -> Tensor:
    q = torch.randn(8, device=device, dtype=dtype)
    nrm = torch.linalg.norm(q)
    if bool(nrm <= eps):
        return oct_one(device=device).to(dtype=dtype)
    return q / nrm


def alignment_objective(x_true: Tensor, x_est: Tensor, q: Tensor) -> float:
    x_true_std = ensure_octonion_tensor(x_true, name="x_true")
    x_est_std = ensure_octonion_tensor(x_est, name="x_est")
    q_std = ensure_octonion_tensor(q, name="q")
    if x_true_std.shape != x_est_std.shape:
        raise ValueError(
            f"Shape mismatch: x_true{tuple(x_true_std.shape)} vs x_est{tuple(x_est_std.shape)}"
        )
    if x_true_std.ndim != 2 or x_est_std.ndim != 2:
        raise ValueError(
            f"x_true and x_est must both be (d, 8), got {tuple(x_true_std.shape)} and {tuple(x_est_std.shape)}"
        )
    if q_std.ndim != 1 or q_std.shape[0] != 8:
        raise ValueError(f"q must be (8,), got {tuple(q_std.shape)}")
    x_aligned = apply_global_right_phase(x_est_std, q_std)
    return float(raw_distance(x_true_std, x_aligned).item())


def closed_form_q(
    x_true: Tensor,
    x_est: Tensor,
    *,
    eps: float = 1e-18,
) -> Tuple[Tensor, float, bool]:
    x_true_std = ensure_octonion_tensor(x_true, name="x_true")
    x_est_std = ensure_octonion_tensor(x_est, name="x_est")
    if x_true_std.shape != x_est_std.shape:
        raise ValueError(
            f"Shape mismatch: x_true{tuple(x_true_std.shape)} vs x_est{tuple(x_est_std.shape)}"
        )
    if x_true_std.ndim != 2:
        raise ValueError(f"x_true and x_est must both be (d, 8), got {tuple(x_true_std.shape)}")

    terms = oct_mul(oct_conj(x_est_std), x_true_std)
    c_vec = terms.sum(dim=0)
    c_abs = float(oct_abs(c_vec).item())
    if c_abs <= eps:
        return oct_one(device=x_true_std.device), c_abs, True
    return c_vec / c_abs, c_abs, False


def sample_random_unit_qs(
    nq: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-18,
) -> Tensor:
    if nq <= 0:
        raise ValueError(f"nq must be positive, got {nq}")
    qs = torch.randn(nq, 8, device=device, dtype=dtype)
    nrms = torch.linalg.norm(qs, dim=1, keepdim=True)
    nrms = torch.clamp(nrms, min=eps)
    return qs / nrms


def random_search_best_q(
    x_true: Tensor,
    x_est: Tensor,
    nq: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> Tuple[Tensor, float, float, float]:
    x_true_std = ensure_octonion_tensor(x_true, name="x_true")
    x_est_std = ensure_octonion_tensor(x_est, name="x_est")
    if x_true_std.shape != x_est_std.shape:
        raise ValueError(
            f"Shape mismatch: x_true{tuple(x_true_std.shape)} vs x_est{tuple(x_est_std.shape)}"
        )
    if x_true_std.ndim != 2:
        raise ValueError(f"x_true and x_est must both be (d, 8), got {tuple(x_true_std.shape)}")

    qs = sample_random_unit_qs(nq, device=device, dtype=dtype)
    x_est_batch = x_est_std.unsqueeze(0)
    q_batch = qs.unsqueeze(1)
    x_aligned_batch = oct_mul(x_est_batch, q_batch)
    diff = x_true_std.unsqueeze(0) - x_aligned_batch
    f_vals = torch.sqrt(torch.sum(diff * diff, dim=(1, 2)))

    best_idx = int(torch.argmin(f_vals).item())
    q_best = qs[best_idx]
    f_best = float(f_vals[best_idx].item())
    f_mean = float(f_vals.mean().item())
    f_std = float(f_vals.std(unbiased=False).item())
    return q_best, f_best, f_mean, f_std


def refine_on_sphere_from_random_inits(
    x_true: Tensor,
    x_est: Tensor,
    *,
    n_init: int,
    max_iters: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    tol: float = 1e-14,
    eps: float = 1e-18,
) -> Dict[str, float | Tensor]:
    x_true_std = ensure_octonion_tensor(x_true, name="x_true")
    x_est_std = ensure_octonion_tensor(x_est, name="x_est")
    if x_true_std.shape != x_est_std.shape:
        raise ValueError(
            f"Shape mismatch: x_true{tuple(x_true_std.shape)} vs x_est{tuple(x_est_std.shape)}"
        )

    terms = oct_mul(oct_conj(x_est_std), x_true_std)
    c_vec = terms.sum(dim=0)
    c_abs = float(oct_abs(c_vec).item())
    if c_abs > eps:
        c_step = c_vec / c_abs
    else:
        c_step = c_vec

    f_refined: List[float] = []
    q_refined: List[Tensor] = []
    for _ in range(n_init):
        q = random_unit_octonion(device=device, dtype=dtype, eps=eps)
        for _it in range(max_iters):
            # D1 on S^7: ascent on g(q)=<q,C> with tangent-space gradient then re-projection.
            dot_qc = torch.dot(q, c_step)
            grad_tan = c_step - dot_qc * q
            q_new = q + lr * grad_tan
            q_new_nrm = torch.linalg.norm(q_new)
            if bool(q_new_nrm <= eps):
                q_new = oct_one(device=device).to(dtype=dtype)
            else:
                q_new = q_new / q_new_nrm
            if float(torch.linalg.norm(q_new - q).item()) <= tol:
                q = q_new
                break
            q = q_new
        f_q = alignment_objective(x_true_std, x_est_std, q)
        f_refined.append(f_q)
        q_refined.append(q)

    f_arr = np.array(f_refined, dtype=np.float64)
    best_idx = int(np.argmin(f_arr))
    return {
        "best_refined": float(f_arr[best_idx]),
        "worst_refined": float(np.max(f_arr)),
        "mean_refined": float(np.mean(f_arr)),
        "q_best_refined": q_refined[best_idx],
        "c_abs": c_abs,
    }


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


@dataclass(frozen=True)
class ExperimentConfig:
    d_list_a: List[int]
    seeds_a: List[int]
    d_list_b: List[int]
    seeds_b: List[int]
    nq_b: int
    d_list_c: List[int]
    seeds_c: List[int]
    sigma_list_c: List[float]
    nq_c: int
    d_list_d: List[int]
    seeds_d: List[int]
    n_init_d: int
    max_iters_d: int
    lr_d: float
    d_list_e: List[int]
    seeds_e: List[int]
    alpha_list_e: List[float]
    nq_e: int
    n_init_e: int
    max_iters_e: int
    lr_e: float
    eps_deg_e: float
    seeds_f: List[int]
    n_init_f: int
    max_iters_f: int
    lr_f: float


def run_group_a(config: ExperimentConfig, *, device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    num_cases = 0
    num_pass = 0
    max_dist_q = 0.0
    max_f_cf = 0.0
    max_gap_exact = 0.0

    for d in config.d_list_a:
        for seed in config.seeds_a:
            _set_seed(seed * 1000 + d)
            x_est = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))
            q0 = random_unit_octonion(device=device, dtype=dtype)
            x_true = apply_global_right_phase(x_est, q0)

            q_cf, c_abs, degenerate = closed_form_q(x_true, x_est)
            f_q0 = alignment_objective(x_true, x_est, q0)
            f_cf = alignment_objective(x_true, x_est, q_cf)
            gap_exact = abs(f_cf - f_q0)
            dist_q = float(torch.linalg.norm(q_cf - q0).item())

            status = "PASS" if (
                (dist_q <= 1e-10) and (f_cf <= 1e-10) and (gap_exact <= 1e-12)
            ) else "FAIL"
            if status == "PASS":
                num_pass += 1
            num_cases += 1
            max_dist_q = max(max_dist_q, dist_q)
            max_f_cf = max(max_f_cf, f_cf)
            max_gap_exact = max(max_gap_exact, gap_exact)

            print(f"[ALIGN-A] seed={seed}, d={d}")
            print(f"q0_norm={_fmt(torch.linalg.norm(q0).item())}")
            print(f"qcf_norm={_fmt(torch.linalg.norm(q_cf).item())}")
            print(f"dist_q={_fmt(dist_q)}")
            print(f"f(q0)={_fmt(f_q0)}")
            print(f"f(q_cf)={_fmt(f_cf)}")
            print(f"gap_exact={_fmt(gap_exact)}")
            print(f"C_abs={_fmt(c_abs)}")
            print(f"degenerate={degenerate}")
            print(f"status={status}")

    summary = {
        "num_cases": float(num_cases),
        "num_pass": float(num_pass),
        "pass_rate": float(num_pass / max(1, num_cases)),
        "max_dist_q": max_dist_q,
        "max_f_cf": max_f_cf,
        "max_gap_exact": max_gap_exact,
    }
    print("[ALIGN-SUMMARY-A]")
    for k, v in summary.items():
        print(f"{k}={_fmt(v) if isinstance(v, float) else v}")
    return summary


def run_group_b(config: ExperimentConfig, *, device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    num_cases = 0
    num_pass = 0
    gaps: List[float] = []
    improves: List[float] = []

    for d in config.d_list_b:
        for seed in config.seeds_b:
            _set_seed(seed * 1000 + d)
            x_true = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))
            x_est = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))

            q_cf, c_abs, degenerate = closed_form_q(x_true, x_est)
            f_cf = alignment_objective(x_true, x_est, q_cf)
            _, f_rand_best, f_rand_mean, f_rand_std = random_search_best_q(
                x_true, x_est, config.nq_b, device=device, dtype=dtype
            )

            gap_rand = f_cf - f_rand_best
            improve_vs_mean = f_rand_mean - f_cf

            status = "PASS" if (gap_rand <= 1e-10) else "FAIL"
            if status == "PASS":
                num_pass += 1
            num_cases += 1
            gaps.append(gap_rand)
            improves.append(improve_vs_mean)

            print(f"[ALIGN-B] seed={seed}, d={d}, Nq={config.nq_b}")
            print(f"f_cf={_fmt(f_cf)}")
            print(f"f_rand_best={_fmt(f_rand_best)}")
            print(f"f_rand_mean={_fmt(f_rand_mean)}")
            print(f"f_rand_std={_fmt(f_rand_std)}")
            print(f"gap_rand={_fmt(gap_rand)}")
            print(f"improve_vs_mean={_fmt(improve_vs_mean)}")
            print(f"C_abs={_fmt(c_abs)}")
            print(f"degenerate={degenerate}")
            print(f"status={status}")

    gap_arr = np.array(gaps, dtype=np.float64) if gaps else np.array([0.0], dtype=np.float64)
    imp_arr = np.array(improves, dtype=np.float64) if improves else np.array([0.0], dtype=np.float64)
    summary = {
        "num_cases": float(num_cases),
        "num_pass": float(num_pass),
        "pass_rate": float(num_pass / max(1, num_cases)),
        "max_gap_rand": float(np.max(gap_arr)),
        "mean_gap_rand": float(np.mean(gap_arr)),
        "min_improve_vs_mean": float(np.min(imp_arr)),
        "gap_rand_values": [float(x) for x in gaps],
    }
    print("[ALIGN-SUMMARY-B]")
    for k, v in summary.items():
        print(f"{k}={_fmt(v) if isinstance(v, float) else v}")
    return summary


def run_group_c(config: ExperimentConfig, *, device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    num_cases = 0
    num_pass = 0
    gaps: List[float] = []

    print("[ALIGN-C-NOTE] normalization_scheme=normalize(x_true_clean + sigma * noise)")

    for d in config.d_list_c:
        for sigma in config.sigma_list_c:
            for seed in config.seeds_c:
                _set_seed(seed * 1000 + d + int(round(1e6 * sigma)))
                x_base = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))
                q0 = random_unit_octonion(device=device, dtype=dtype)
                x_true_clean = apply_global_right_phase(x_base, q0)

                noise = random_octonion((d,), device=device, dtype=dtype)
                x_true = normalize_oct_signal(x_true_clean + float(sigma) * noise)
                x_est = x_base

                q_cf, c_abs, degenerate = closed_form_q(x_true, x_est)
                f_cf = alignment_objective(x_true, x_est, q_cf)
                _, f_rand_best, _f_rand_mean, _f_rand_std = random_search_best_q(
                    x_true, x_est, config.nq_c, device=device, dtype=dtype
                )
                dist_q_to_q0 = float(torch.linalg.norm(q_cf - q0).item())
                gap_rand = f_cf - f_rand_best

                status = "PASS" if (gap_rand <= 1e-8) else "FAIL"
                if status == "PASS":
                    num_pass += 1
                num_cases += 1
                gaps.append(gap_rand)

                print(f"[ALIGN-C] seed={seed}, d={d}, sigma={sigma}, Nq={config.nq_c}")
                print(f"dist_q_to_q0={_fmt(dist_q_to_q0)}")
                print(f"f_cf={_fmt(f_cf)}")
                print(f"f_rand_best={_fmt(f_rand_best)}")
                print(f"gap_rand={_fmt(gap_rand)}")
                print(f"C_abs={_fmt(c_abs)}")
                print(f"degenerate={degenerate}")
                print(f"status={status}")

    gap_arr = np.array(gaps, dtype=np.float64) if gaps else np.array([0.0], dtype=np.float64)
    summary = {
        "num_cases": float(num_cases),
        "num_pass": float(num_pass),
        "pass_rate": float(num_pass / max(1, num_cases)),
        "max_gap_rand": float(np.max(gap_arr)),
        "mean_gap_rand": float(np.mean(gap_arr)),
        "gap_rand_values": [float(x) for x in gaps],
    }
    print("[ALIGN-SUMMARY-C]")
    for k, v in summary.items():
        print(f"{k}={_fmt(v) if isinstance(v, float) else v}")
    return summary


def run_group_d(config: ExperimentConfig, *, device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    num_cases = 0
    num_pass = 0
    gap_refines: List[float] = []
    dist_best_q_to_qcf_list: List[float] = []

    for d in config.d_list_d:
        for seed in config.seeds_d:
            _set_seed(seed * 1000 + d)
            x_true = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))
            x_est = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))
            q_cf, _c_abs, degenerate = closed_form_q(x_true, x_est)
            f_cf = alignment_objective(x_true, x_est, q_cf)

            refined = refine_on_sphere_from_random_inits(
                x_true,
                x_est,
                n_init=config.n_init_d,
                max_iters=config.max_iters_d,
                lr=config.lr_d,
                device=device,
                dtype=dtype,
            )
            best_refined = float(refined["best_refined"])
            worst_refined = float(refined["worst_refined"])
            mean_refined = float(refined["mean_refined"])
            q_best_refined = refined["q_best_refined"]
            dist_best_q_to_qcf = float(torch.linalg.norm(q_best_refined - q_cf).item())
            gap_refine = f_cf - best_refined

            pass_gap = abs(gap_refine) <= 1e-10
            pass_q = degenerate or (dist_best_q_to_qcf <= 1e-8) or (abs(gap_refine) <= 1e-10)
            status = "PASS" if (pass_gap and pass_q) else "FAIL"
            if status == "PASS":
                num_pass += 1
            num_cases += 1
            gap_refines.append(abs(gap_refine))
            dist_best_q_to_qcf_list.append(dist_best_q_to_qcf)

            print(f"[ALIGN-D] seed={seed}, d={d}, n_init={config.n_init_d}")
            print(f"f_cf={_fmt(f_cf)}")
            print(f"best_refined={_fmt(best_refined)}")
            print(f"gap_refine={_fmt(gap_refine)}")
            print(f"worst_refined={_fmt(worst_refined)}")
            print(f"mean_refined={_fmt(mean_refined)}")
            print(f"dist_best_q_to_qcf={_fmt(dist_best_q_to_qcf)}")
            print(f"status={status}")

    gap_arr = np.array(gap_refines, dtype=np.float64) if gap_refines else np.array([0.0], dtype=np.float64)
    dist_arr = (
        np.array(dist_best_q_to_qcf_list, dtype=np.float64)
        if dist_best_q_to_qcf_list
        else np.array([0.0], dtype=np.float64)
    )
    summary = {
        "num_cases": float(num_cases),
        "num_pass": float(num_pass),
        "pass_rate": float(num_pass / max(1, num_cases)),
        "max_gap_refine": float(np.max(gap_arr)),
        "mean_gap_refine": float(np.mean(gap_arr)),
        "max_dist_best_q_to_qcf": float(np.max(dist_arr)),
    }
    print("[ALIGN-SUMMARY-D]")
    for k, v in summary.items():
        print(f"{k}={_fmt(v) if isinstance(v, float) else v}")
    return summary


def run_group_e(config: ExperimentConfig, *, device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    num_cases = 0
    num_pass = 0
    c_abs_list: List[float] = []
    gap_refines: List[float] = []
    gap_rands: List[float] = []

    for d in config.d_list_e:
        for seed in config.seeds_e:
            _set_seed(seed * 1000 + d)
            x_est = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))

            # Build near-degenerate direction by orthogonalizing against
            # span{ x_est * e_k }_{k=0..7} in R^{8d}, which drives C toward zero.
            z_perp_raw = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))
            for k in range(8):
                e_k = oct_basis(k, device=device).to(dtype=dtype)
                v_k = apply_global_right_phase(x_est, e_k)
                denom = torch.sum(v_k * v_k)
                if bool(denom > 1e-18):
                    coeff = torch.sum(z_perp_raw * v_k) / denom
                    z_perp_raw = z_perp_raw - coeff * v_k
            z_perp_norm = torch.sqrt(torch.sum(z_perp_raw * z_perp_raw))
            if bool(z_perp_norm <= 1e-18):
                z_perp = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))
            else:
                z_perp = z_perp_raw / z_perp_norm
            q0 = random_unit_octonion(device=device, dtype=dtype)

            for alpha in config.alpha_list_e:
                mix = float(alpha) * apply_global_right_phase(x_est, q0) + np.sqrt(max(0.0, 1.0 - float(alpha) ** 2)) * z_perp
                x_true = normalize_oct_signal(mix)

                q_cf, c_abs, _degenerate = closed_form_q(x_true, x_est)
                f_cf = alignment_objective(x_true, x_est, q_cf)

                refined = refine_on_sphere_from_random_inits(
                    x_true,
                    x_est,
                    n_init=config.n_init_e,
                    max_iters=config.max_iters_e,
                    lr=config.lr_e,
                    device=device,
                    dtype=dtype,
                )
                best_refined = float(refined["best_refined"])
                gap_refine = f_cf - best_refined

                _q_rand_best, f_rand_best, f_rand_mean, f_rand_std = random_search_best_q(
                    x_true, x_est, config.nq_e, device=device, dtype=dtype
                )
                gap_rand = f_cf - f_rand_best
                degenerate_like = bool(c_abs <= config.eps_deg_e)

                status = "PASS" if (abs(gap_refine) <= 1e-10 and gap_rand <= 1e-8) else "FAIL"
                if status == "PASS":
                    num_pass += 1
                num_cases += 1

                c_abs_list.append(c_abs)
                gap_refines.append(abs(gap_refine))
                gap_rands.append(gap_rand)

                print(f"[ALIGN-E] seed={seed}, d={d}, alpha={alpha}")
                print(f"C_abs={_fmt(c_abs)}")
                print(f"f_cf={_fmt(f_cf)}")
                print(f"best_refined={_fmt(best_refined)}")
                print(f"gap_refine={_fmt(gap_refine)}")
                print(f"f_rand_best={_fmt(f_rand_best)}")
                print(f"gap_rand={_fmt(gap_rand)}")
                print(f"f_rand_mean={_fmt(f_rand_mean)}")
                print(f"f_rand_std={_fmt(f_rand_std)}")
                print(f"degenerate_like={degenerate_like}")
                print(f"status={status}")

    c_arr = np.array(c_abs_list, dtype=np.float64) if c_abs_list else np.array([0.0], dtype=np.float64)
    gr_arr = np.array(gap_refines, dtype=np.float64) if gap_refines else np.array([0.0], dtype=np.float64)
    gd_arr = np.array(gap_rands, dtype=np.float64) if gap_rands else np.array([0.0], dtype=np.float64)
    summary = {
        "num_cases": float(num_cases),
        "num_pass": float(num_pass),
        "pass_rate": float(num_pass / max(1, num_cases)),
        "min_C_abs": float(np.min(c_arr)),
        "max_gap_refine": float(np.max(gr_arr)),
        "mean_gap_refine": float(np.mean(gr_arr)),
        "max_gap_rand": float(np.max(gd_arr)),
        "mean_gap_rand": float(np.mean(gd_arr)),
        "gap_rand_values": [float(x) for x in gap_rands],
    }
    print("[ALIGN-SUMMARY-E]")
    for k, v in summary.items():
        print(f"{k}={_fmt(v) if isinstance(v, float) else v}")
    return summary


def run_group_f(config: ExperimentConfig, *, device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    num_cases = 0
    num_pass = 0
    gap_refines: List[float] = []
    dist_best_q_to_qcf_list: List[float] = []
    d = 32

    for seed in config.seeds_f:
        _set_seed(seed * 1000 + d)
        x_true = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))
        x_est = normalize_oct_signal(random_octonion((d,), device=device, dtype=dtype))
        q_cf, _c_abs, degenerate = closed_form_q(x_true, x_est)
        f_cf = alignment_objective(x_true, x_est, q_cf)

        refined = refine_on_sphere_from_random_inits(
            x_true,
            x_est,
            n_init=config.n_init_f,
            max_iters=config.max_iters_f,
            lr=config.lr_f,
            device=device,
            dtype=dtype,
        )
        best_refined = float(refined["best_refined"])
        worst_refined = float(refined["worst_refined"])
        mean_refined = float(refined["mean_refined"])
        q_best_refined = refined["q_best_refined"]
        dist_best_q_to_qcf = float(torch.linalg.norm(q_best_refined - q_cf).item())
        gap_refine = f_cf - best_refined

        pass_gap = abs(gap_refine) <= 1e-10
        pass_q = degenerate or (dist_best_q_to_qcf <= 1e-8) or (abs(gap_refine) <= 1e-10)
        status = "PASS" if (pass_gap and pass_q) else "FAIL"
        if status == "PASS":
            num_pass += 1
        num_cases += 1
        gap_refines.append(abs(gap_refine))
        dist_best_q_to_qcf_list.append(dist_best_q_to_qcf)

        print(f"[ALIGN-F] seed={seed}, d=32, n_init={config.n_init_f}")
        print(f"f_cf={_fmt(f_cf)}")
        print(f"best_refined={_fmt(best_refined)}")
        print(f"gap_refine={_fmt(gap_refine)}")
        print(f"worst_refined={_fmt(worst_refined)}")
        print(f"mean_refined={_fmt(mean_refined)}")
        print(f"dist_best_q_to_qcf={_fmt(dist_best_q_to_qcf)}")
        print(f"status={status}")

    gap_arr = np.array(gap_refines, dtype=np.float64) if gap_refines else np.array([0.0], dtype=np.float64)
    dist_arr = (
        np.array(dist_best_q_to_qcf_list, dtype=np.float64)
        if dist_best_q_to_qcf_list
        else np.array([0.0], dtype=np.float64)
    )
    summary = {
        "num_cases": float(num_cases),
        "num_pass": float(num_pass),
        "pass_rate": float(num_pass / max(1, num_cases)),
        "max_gap_refine": float(np.max(gap_arr)),
        "mean_gap_refine": float(np.mean(gap_arr)),
        "max_dist_best_q_to_qcf": float(np.max(dist_arr)),
    }
    print("[ALIGN-SUMMARY-F]")
    for k, v in summary.items():
        print(f"{k}={_fmt(v) if isinstance(v, float) else v}")
    return summary


def export_artifacts(out: Dict[str, Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_test_rows = [
        ["A", out["A"]["num_cases"], out["A"]["pass_rate"], out["A"]["max_dist_q"]],
        ["B", out["B"]["num_cases"], out["B"]["pass_rate"], out["B"]["mean_gap_rand"]],
        ["C", out["C"]["num_cases"], out["C"]["pass_rate"], out["C"]["mean_gap_rand"]],
        ["D", out["D"]["num_cases"], out["D"]["pass_rate"], out["D"]["max_gap_refine"]],
        ["E", out["E"]["num_cases"], out["E"]["pass_rate"], out["E"]["mean_gap_rand"]],
        ["F", out["F"]["num_cases"], out["F"]["pass_rate"], out["F"]["max_gap_refine"]],
    ]
    # Transposed layout for compact paper typesetting.
    headers = ["Metric", "A", "B", "C", "D", "E", "F"]
    rows = [
        ["Cases"] + [r[1] for r in per_test_rows],
        ["Pass Rate"] + [r[2] for r in per_test_rows],
        ["Metric"] + [r[3] for r in per_test_rows],
    ]
    tex_path = output_dir / "alignment_summary.tex"
    csv_path = output_dir / "alignment_summary.csv"
    fig_path = output_dir / "gap_hist.png"

    save_table_latex(
        tex_path,
        headers,
        rows,
        caption="Validation of global right alignment optimality",
        label="tab:alignment_validation",
    )
    save_table_csv(csv_path, ["Test", "Cases", "Pass Rate", "Metric"], per_test_rows)

    all_gap_rand: List[float] = []
    for key in ("B", "C", "E"):
        all_gap_rand.extend(out[key].get("gap_rand_values", []))
    if len(all_gap_rand) > 0:
        plot_gap_distribution(all_gap_rand, fig_path)
    else:
        plot_gap_distribution([0.0], fig_path)

    print(f"[ALIGN-EXPORT] latex={tex_path}")
    print(f"[ALIGN-EXPORT] csv={csv_path}")
    print(f"[ALIGN-EXPORT] fig={fig_path}")


def build_full_config() -> ExperimentConfig:
    return ExperimentConfig(
        d_list_a=[4, 8, 16, 32],
        seeds_a=list(range(20)),
        d_list_b=[4, 8, 16, 32],
        seeds_b=list(range(20)),
        nq_b=10000,
        d_list_c=[8, 16],
        seeds_c=list(range(10)),
        sigma_list_c=[1e-6, 1e-4, 1e-2, 1e-1],
        nq_c=10000,
        d_list_d=[4, 8, 16],
        seeds_d=list(range(10)),
        n_init_d=20,
        max_iters_d=2000,
        lr_d=0.1,
        d_list_e=[8, 16, 32],
        seeds_e=list(range(10)),
        alpha_list_e=[0.0, 1e-6, 1e-4, 1e-2, 1e-1],
        nq_e=10000,
        n_init_e=20,
        max_iters_e=2000,
        lr_e=0.1,
        eps_deg_e=1e-12,
        seeds_f=list(range(10)),
        n_init_f=20,
        max_iters_f=2000,
        lr_f=0.1,
    )


def build_quick_config() -> ExperimentConfig:
    return ExperimentConfig(
        d_list_a=[4, 8],
        seeds_a=[0, 1],
        d_list_b=[4],
        seeds_b=[0, 1],
        nq_b=2000,
        d_list_c=[8],
        seeds_c=[0, 1],
        sigma_list_c=[1e-6, 1e-2],
        nq_c=2000,
        d_list_d=[4],
        seeds_d=[0, 1],
        n_init_d=8,
        max_iters_d=400,
        lr_d=0.1,
        d_list_e=[8],
        seeds_e=[0, 1],
        alpha_list_e=[0.0, 1e-2],
        nq_e=2000,
        n_init_e=20,
        max_iters_e=2000,
        lr_e=0.1,
        eps_deg_e=1e-12,
        seeds_f=[0],
        n_init_f=20,
        max_iters_f=2000,
        lr_f=0.1,
    )


def run_all(config: ExperimentConfig) -> Dict[str, Dict[str, float]]:
    device = torch.device("cpu")
    dtype = torch.float64
    print(f"[ALIGN-START] device={device.type}, dtype={dtype}")

    out = {
        "A": run_group_a(config, device=device, dtype=dtype),
        "B": run_group_b(config, device=device, dtype=dtype),
        "C": run_group_c(config, device=device, dtype=dtype),
        "D": run_group_d(config, device=device, dtype=dtype),
        "E": run_group_e(config, device=device, dtype=dtype),
        "F": run_group_f(config, device=device, dtype=dtype),
    }
    export_artifacts(out, PROJECT_ROOT / "output" / "t36")
    print("[ALIGN-END] all groups completed")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numerical verification for alignment closed-form global optimality."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced smoke configuration for quick sanity checks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = build_quick_config() if args.quick else build_full_config()
    run_all(cfg)
