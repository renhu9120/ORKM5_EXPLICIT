import statistics

import torch

from algorithms.initializations.init_osi import normalize_oct_signal
from core.octonion_inner import intensity_measurements_explicit, row_inner_explicit
from core.octonion_ops import oct_abs
from discards.ork_m import orkm_single_row_update


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    torch.manual_seed(20)
    print("=== T20 single-row exactness batch check ===")
    print(f"device={device}, dtype=float64")

    groups = 8
    n, d = 12, 8
    exact_tol = 1e-12
    pass_max_after = 1e-10
    pass_exact_ratio = 0.95

    all_after: list[float] = []
    all_before: list[float] = []
    all_improve: list[float] = []
    total_rows = 0
    exact_rows = 0
    skipped_rows = 0

    for g in range(groups):
        A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
        x_true = normalize_oct_signal(torch.randn(d, 8, dtype=torch.float64, device=device))
        y = intensity_measurements_explicit(A, x_true)
        x0 = torch.randn(d, 8, dtype=torch.float64, device=device)

        before_vals: list[float] = []
        after_vals: list[float] = []
        improve_vals: list[float] = []
        exact_count_group = 0
        skipped_group = 0

        for l in range(n):
            s_before = row_inner_explicit(A[l], x0)
            b_l = torch.sqrt(torch.clamp(y[l], min=0.0))
            before = torch.abs(oct_abs(s_before) - b_l).item()

            x1, row_info = orkm_single_row_update(
                x0, A[l], y[l], omega=1.0, return_info=True
            )
            s_after = row_inner_explicit(A[l], x1)
            after = torch.abs(oct_abs(s_after) - b_l).item()

            before_vals.append(before)
            all_before.append(before)
            after_vals.append(after)
            all_after.append(after)
            improve_vals.append(before - after)
            all_improve.append(before - after)
            total_rows += 1

            if bool(row_info["skipped"]):
                skipped_group += 1
                skipped_rows += 1
            if after < exact_tol:
                exact_count_group += 1
                exact_rows += 1

        print(
            f"group={g:02d}, max_before={max(before_vals):.6e}, "
            f"max_after={max(after_vals):.6e}, median_after={statistics.median(after_vals):.6e}, "
            f"max_improve={max(improve_vals):.6e}, exact_count={exact_count_group}/{n}, "
            f"skip_count={skipped_group}/{n}"
        )

    overall_max_after = max(all_after)
    overall_median_after = statistics.median(all_after)
    overall_max_improve = max(all_improve)
    exact_ratio = 0.0 if total_rows == 0 else exact_rows / total_rows
    skip_ratio = 0.0 if total_rows == 0 else skipped_rows / total_rows

    print("---- overall summary ----")
    print(f"overall_rows={total_rows}, exact_rows={exact_rows}, skipped_rows={skipped_rows}")
    print(f"overall_max_after={overall_max_after:.6e}")
    print(f"overall_median_after={overall_median_after:.6e}")
    print(f"overall_max_improve={overall_max_improve:.6e}")
    print(f"exact_ratio={exact_ratio:.2%}, skip_ratio={skip_ratio:.2%}")

    ok = overall_max_after < pass_max_after and exact_ratio >= pass_exact_ratio
    print(
        f"T20 status: {'PASS' if ok else 'FAIL'} "
        f"(criteria: max_after<{pass_max_after:.1e}, exact_ratio>={pass_exact_ratio:.0%})"
    )


if __name__ == "__main__":
    main()
