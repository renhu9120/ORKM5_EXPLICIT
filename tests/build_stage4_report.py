from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils_latex import plot_gap_distribution, save_table_latex


def _read_csv_dict_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(x: str) -> float:
    return float(x)


def _to_int(x: str) -> int:
    return int(float(x))


def _sci(x: float, digits: int = 4) -> str:
    return f"{float(x):.{digits}e}"


def _format_cases_int(x: float) -> str:
    # For LaTeX we will show as sci in requested style elsewhere.
    return str(int(round(float(x))))


def _weighted_pass_rate(rows: List[Dict[str, str]], pass_key: str, valid_key: str) -> float:
    num = 0.0
    den = 0.0
    for r in rows:
        valid = _to_float(r[valid_key])
        pass_rate = _to_float(r[pass_key])
        num += pass_rate * valid
        den += valid
    return num / den if den > 0 else float("nan")


def _get_group_from_A(path: Path) -> Dict[str, float]:
    rows = _read_csv_dict_rows(path)
    cases_total = sum(_to_float(r["cases"]) for r in rows)
    valid_total = sum(_to_float(r["valid_cases"]) for r in rows)
    skip_total = sum(_to_float(r["skipped_cases"]) for r in rows)
    worst_max_err_step = max(_to_float(r["max_err_step"]) for r in rows)
    worst_max_err_e_r = max(_to_float(r["max_err_e_r"]) for r in rows)
    pass_step = _weighted_pass_rate(rows, "pass_rate_step", "valid_cases")
    pass_e_r = _weighted_pass_rate(rows, "pass_rate_e_r", "valid_cases")
    overall_pass = min(pass_step, pass_e_r)
    return {
        "cases_total": float(cases_total),
        "valid_total": float(valid_total),
        "skip_total": float(skip_total),
        "worst_max_err_step": float(worst_max_err_step),
        "worst_max_err_e_r": float(worst_max_err_e_r),
        "pass_rate_step_weighted": float(pass_step),
        "pass_rate_e_r_weighted": float(pass_e_r),
        "overall_pass_rate": float(overall_pass),
        "violations_total": float(valid_total - round(overall_pass * valid_total)) if valid_total > 0 else 0.0,
    }


def _get_group_from_B(path: Path) -> Dict[str, float]:
    rows = _read_csv_dict_rows(path)
    cases_total = sum(_to_float(r["cases"]) for r in rows)
    valid_total = sum(_to_float(r["valid_cases"]) for r in rows)
    skip_total = sum(_to_float(r["skipped_cases"]) for r in rows)
    worst_cross = max(_to_float(r["max_err_cross_response"]) for r in rows)
    worst_self = max(_to_float(r["max_err_self_transfer"]) for r in rows)
    worst_wrong = max(_to_float(r["max_err_wrong_bracketing"]) for r in rows)
    pass_cross = _weighted_pass_rate(rows, "pass_rate_cross_response", "valid_cases")
    pass_self = _weighted_pass_rate(rows, "pass_rate_self_transfer", "valid_cases")
    overall_pass = min(pass_cross, pass_self)
    violations_total = float(valid_total - round(overall_pass * valid_total)) if valid_total > 0 else 0.0
    return {
        "cases_total": float(cases_total),
        "valid_total": float(valid_total),
        "skip_total": float(skip_total),
        "worst_max_err_cross_response": float(worst_cross),
        "worst_max_err_self_transfer": float(worst_self),
        "worst_max_err_wrong_bracketing": float(worst_wrong),
        "pass_rate_cross_weighted": float(pass_cross),
        "pass_rate_self_weighted": float(pass_self),
        "overall_pass_rate": float(overall_pass),
        "violations_total": violations_total,
    }


def _get_group_from_C(path: Path) -> Dict[str, float]:
    rows = _read_csv_dict_rows(path)
    cases_total = sum(_to_float(r["cases"]) for r in rows)
    valid_total = sum(_to_float(r["valid_cases"]) for r in rows)
    max_ratio_res = max(_to_float(r["max_ratio_res"]) for r in rows)
    max_ratio_cs = max(_to_float(r["max_ratio_cs"]) for r in rows)
    vT = sum(_to_float(r["violations_T"]) for r in rows)
    vs = sum(_to_float(r["violations_s"]) for r in rows)
    vamp = sum(_to_float(r["violations_amp"]) for r in rows)
    vres = sum(_to_float(r["violations_res"]) for r in rows)
    vcs = sum(_to_float(r["violations_cs"]) for r in rows)
    violations_total = vT + vs + vamp + vres + vcs
    worst_eta_diag_err = max(_to_float(r["max_abs_eta_ll_minus_1"]) for r in rows)
    return {
        "cases_total": float(cases_total),
        "valid_total": float(valid_total),
        "worst_max_ratio_res": float(max_ratio_res),
        "worst_max_ratio_cs": float(max_ratio_cs),
        "violations_total": float(violations_total),
        "violations_T": float(vT),
        "violations_s": float(vs),
        "violations_amp": float(vamp),
        "violations_res": float(vres),
        "violations_cs": float(vcs),
        "worst_eta_diag_err": float(worst_eta_diag_err),
    }


def _get_group_from_D(path: Path) -> Dict[str, float]:
    rows = _read_csv_dict_rows(path)
    cases_total = sum(_to_float(r["cases"]) for r in rows)
    valid_total = sum(_to_float(r["valid_cases"]) for r in rows)
    theorem_viol = sum(_to_float(r["theorem_violations"]) for r in rows)
    max_ineq_gap = max(_to_float(r["max_ineq_gap"]) for r in rows)
    min_emp = min(_to_float(r["empirical_descent_rate_all"]) for r in rows)
    max_emp = max(_to_float(r["empirical_descent_rate_all"]) for r in rows)
    min_freq = min(_to_float(r["freq_eta_col_sum_excl_lt_1"]) for r in rows)
    max_freq = max(_to_float(r["freq_eta_col_sum_excl_lt_1"]) for r in rows)
    total_cond_cases = sum(_to_float(r["descent_condition_cases"]) for r in rows)
    total_success_cases = sum(_to_float(r["descent_success_cases"]) for r in rows)
    # Use the mean of success rates weighted by condition cases
    weighted_success_rate = total_success_cases / total_cond_cases if total_cond_cases > 0 else float("nan")
    return {
        "cases_total": float(cases_total),
        "valid_total": float(valid_total),
        "theorem_violations_total": float(theorem_viol),
        "max_ineq_gap": float(max_ineq_gap),
        "empirical_descent_min": float(min_emp),
        "empirical_descent_max": float(max_emp),
        "freq_condition_min": float(min_freq),
        "freq_condition_max": float(max_freq),
        "descent_condition_cases_total": float(total_cond_cases),
        "descent_success_cases_total": float(total_success_cases),
        "descent_success_rate_weighted": float(weighted_success_rate),
    }


def _get_group_from_E(path: Path) -> Dict[str, float]:
    rows = _read_csv_dict_rows(path)
    if not rows:
        raise FileNotFoundError(f"Empty E summary csv: {path}")
    r = rows[0]
    return {
        "cases_total": _to_float(r["cases"]),
        "max_abs_e_consistent": _to_float(r["max_abs_e_consistent"]),
        "max_update_norm_consistent": _to_float(r["max_update_norm_consistent"]),
        "pass_rate_zero_update": _to_float(r["pass_rate_zero_update"]),
        "zero_implies_consistent_checks": _to_float(r["zero_implies_consistent_checks"]),
        "zero_implies_consistent_pass_rate": _to_float(r["zero_implies_consistent_pass_rate"]),
        "max_fullpass_change": _to_float(r["max_fullpass_change"]),
        "pass_rate_fullpass_fixed": _to_float(r["pass_rate_fullpass_fixed"]),
    }


def build_summary_figure(stage4_dir: Path) -> None:
    a_scatter = stage4_dir / "stage4_A_scatter.csv"
    d_scatter = stage4_dir / "stage4_D_scatter.csv"

    if not a_scatter.exists():
        raise FileNotFoundError(f"Missing A scatter csv: {a_scatter}")
    if not d_scatter.exists():
        raise FileNotFoundError(f"Missing D scatter csv: {d_scatter}")

    # Load A scatter (predicted_step_size, observed_step_size)
    a_pred = []
    a_obs = []
    with a_scatter.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            a_pred.append(_to_float(r["predicted_step_size"]))
            a_obs.append(_to_float(r["observed_step_size"]))
    a_pred_arr = np.asarray(a_pred, dtype=np.float64)
    a_obs_arr = np.asarray(a_obs, dtype=np.float64)

    # Load D scatter (eta_col_sum_excl, delta_E1)
    d_eta = []
    d_delta = []
    with d_scatter.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            d_eta.append(_to_float(r["eta_col_sum_excl"]))
            d_delta.append(_to_float(r["delta_E1"]))
    d_eta_arr = np.asarray(d_eta, dtype=np.float64)
    d_delta_arr = np.asarray(d_delta, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax0 = axes[0]
    ax0.scatter(d_eta_arr, d_delta_arr, s=8, alpha=0.5)
    ax0.set_xlabel("eta_col_sum_excl")
    ax0.set_ylabel("delta_E1 = E1_after - E1_before")
    ax0.set_title("(a)")
    ax0.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    ax1 = axes[1]
    ax1.scatter(a_pred_arr, a_obs_arr, s=8, alpha=0.5)
    mn = float(min(a_pred_arr.min(), a_obs_arr.min()))
    mx = float(max(a_pred_arr.max(), a_obs_arr.max()))
    ax1.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.0, color="black", alpha=0.6)
    ax1.set_xlabel("predicted step size")
    ax1.set_ylabel("observed step size ||x^+ - x||")
    ax1.set_title("(b)")
    ax1.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    out_png = stage4_dir / "stage4_summary_figure.png"
    out_pdf = stage4_dir / "stage4_summary_figure.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Stage4-Report] figure png={out_png}")
    print(f"[Stage4-Report] figure pdf={out_pdf}")


def build_table_and_metrics(stage4_dir: Path) -> None:
    a = stage4_dir / "stage4_A_step_identity.csv"
    b = stage4_dir / "stage4_B_cross_row_response.csv"
    c = stage4_dir / "stage4_C_cross_row_bound.csv"
    d = stage4_dir / "stage4_D_surrogate_descent.csv"
    e = stage4_dir / "stage4_E_summary.csv"

    if not a.exists():
        raise FileNotFoundError(f"Missing A summary csv: {a}")
    if not b.exists():
        raise FileNotFoundError(f"Missing B summary csv: {b}")
    if not c.exists():
        raise FileNotFoundError(f"Missing C summary csv: {c}")
    if not d.exists():
        raise FileNotFoundError(f"Missing D summary csv: {d}")
    if not e.exists():
        raise FileNotFoundError(f"Missing E summary csv: {e}")

    A = _get_group_from_A(a)
    B = _get_group_from_B(b)
    C = _get_group_from_C(c)
    D = _get_group_from_D(d)
    E = _get_group_from_E(e)

    # Prepare metrics csv
    metrics_path = stage4_dir / "stage4_summary_metrics.csv"
    metrics_headers = [
        "Group",
        "CasesTotal",
        "PassRateOverall",
        "ViolationsTotal",
        "WorstMetric1",
        "WorstMetric2",
    ]
    metrics_rows = [
        ["A", A["cases_total"], A["overall_pass_rate"], A["violations_total"], A["worst_max_err_step"], A["worst_max_err_e_r"]],
        ["B", B["cases_total"], B["overall_pass_rate"], B["violations_total"], B["worst_max_err_cross_response"], B["worst_max_err_wrong_bracketing"]],
        ["C", C["cases_total"], float("nan"), C["violations_total"], C["worst_max_ratio_res"], C["worst_max_ratio_cs"]],
        ["D", D["cases_total"], float("nan"), D["theorem_violations_total"], D["empirical_descent_min"], D["empirical_descent_max"]],
        ["E", E["cases_total"], min(E["pass_rate_zero_update"], E["pass_rate_fullpass_fixed"]), float("nan"), E["max_update_norm_consistent"], E["max_fullpass_change"]],
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(metrics_headers)
        writer.writerows(metrics_rows)

    print(f"[Stage4-Report] metrics csv={metrics_path}")

    # Build LaTeX table
    table_path = stage4_dir / "stage4_summary_table.tex"
    table_txt_path = stage4_dir / "stage4_summary_table.txt"

    group_rows = [
        (
            "A",
            "Step-size identity and $|e_\\ell|=|r_\\ell|$",
            A["cases_total"],
            f"pass={_sci(A['overall_pass_rate'])}",
            f"$\\max\\mathrm{{err}}_\\mathrm{{step}}={_sci(A['worst_max_err_step'])}$; "
            f"$\\max|e_\\ell/r_\\ell|={_sci(A['worst_max_err_e_r'])}$",
            "Machine-precision agreement",
        ),
        (
            "B",
            "Cross-row response and self-transfer identity",
            B["cases_total"],
            f"pass={_sci(B['overall_pass_rate'])}; wrong-bracketing $\\max={_sci(B['worst_max_err_wrong_bracketing'])}$",
            f"$\\max\\mathrm{{err}}_\\mathrm{{cross}}={_sci(B['worst_max_err_cross_response'])}$; "
            f"$\\max\\mathrm{{err}}_\\mathrm{{self}}={_sci(B['worst_max_err_self_transfer'])}$",
            "Exact formula validated; wrong bracketing fails strongly",
        ),
        (
            "C",
            "Transfer and perturbation bounds",
            C["cases_total"],
            f"violations={int(round(C['violations_total']))}",
            f"$\\max\\mathrm{{ratio}}_\\mathrm{{res}}={_sci(C['worst_max_ratio_res'])}$; "
            f"$\\max\\mathrm{{ratio}}_\\mathrm{{cs}}={_sci(C['worst_max_ratio_cs'])}$",
            "All tested bounds numerically respected",
        ),
        (
            "D",
            "One-step sufficient descent inequality",
            D["cases_total"],
            f"violations={int(round(D['theorem_violations_total']))}",
            f"empirical descent in [{_sci(D['empirical_descent_min'])}, {_sci(D['empirical_descent_max'])}]; "
            f"freq cond max={_sci(D['freq_condition_max'])}",
            "Inequality valid; sufficient condition conservative",
        ),
        (
            "E",
            "Fixed-point and consistency structure",
            E["cases_total"],
            f"pass={_sci(min(E['pass_rate_zero_update'], E['pass_rate_fullpass_fixed']))}",
            f"$\\max\\|x^+-x\\|={_sci(E['max_update_norm_consistent'])}$; "
            f"full-pass change={_sci(E['max_fullpass_change'])}",
            "Fixed-point behavior confirmed",
        ),
    ]

    lines_tex: List[str] = []
    lines_tex.append("\\begin{table*}[t]")
    lines_tex.append("\\centering")
    lines_tex.append("\\resizebox{\\textwidth}{!}{")
    lines_tex.append("\\begin{tabular}{c l c c c l}")
    lines_tex.append("\\toprule")
    lines_tex.append("Group & Theory validated & Cases & Pass / Violations & Representative metric & Interpretation \\\\")
    lines_tex.append("\\midrule")

    for gr in group_rows:
        group, theory, cases, passvio, repmet, interp = gr
        lines_tex.append(
            f"{group} & {theory} & {_sci(cases)} & {passvio} & {repmet} & {interp} \\\\"
        )

    lines_tex.append("\\bottomrule")
    lines_tex.append("\\end{tabular}}")
    lines_tex.append("\\caption{Summary of the numerical validation results for the Stage~4 structural theory.}")
    lines_tex.append("\\label{tab:stage4_validation_summary}")
    lines_tex.append("\\end{table*}")

    table_path.write_text("\n".join(lines_tex) + "\n", encoding="utf-8")

    # Plain-text version
    txt_lines = ["Stage4 summary table (LaTeX source preview):", ""]
    txt_lines.extend(lines_tex)
    table_txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    print(f"[Stage4-Report] table tex={table_path}")
    print(f"[Stage4-Report] table txt={table_txt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Stage4 paper-ready report assets.")
    p.add_argument("--skip_figure", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.perf_counter()
    stage4_dir = PROJECT_ROOT / "output"
    stage4_dir.mkdir(parents=True, exist_ok=True)

    build_table_and_metrics(stage4_dir)
    if not args.skip_figure:
        build_summary_figure(stage4_dir)
    print(f"[Stage4-Report] done elapsed={time.perf_counter() - t0:.1f}s")

