from __future__ import annotations

"""
One-click runner for the five ORKM real-patch prob diagnostics (recommended order).

Mirrors tests/T37_stage4_theory_validation.py: path setup, argparse, per-stage timers,
and a final total wall time.

Progress: each prob script emits [PROBn] milestone lines (flush=True) when
show_progress is True (default). Use --verbose for per-iteration grad_orkm logs (slower).
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (PROJECT_ROOT, TESTS_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prob_1_warm_start_local_conv import run as run_prob1
from prob_2_omega_sweep_realpatch import run as run_prob2
from prob_3_Aseed_sweep_realpatch import run as run_prob3
from prob_4_matched_control_signals import run as run_prob4
from prob_5_jacobian_conditioning import run as run_prob5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T39: run prob tests 1..5 with timing (see docstring).")
    p.add_argument(
        "--quick",
        action="store_true",
        help="short smoke settings inside each prob (fewer passes / smaller sweeps).",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="do not append CSV files under scripts/output/prob.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="override CSV directory (default inside each prob: scripts/output/prob).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="forward to prob scripts: grad_orkm iteration logs + meas_rel orbit (much slower).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="disable [PROBn] milestone lines in prob scripts.",
    )
    p.add_argument(
        "--omega-prob3",
        type=float,
        default=1.0,
        help="omega for prob 3 (use best from prob 2 after you inspect results).",
    )
    p.add_argument(
        "--gaussian-only-prob1",
        action="store_true",
        help="only run Gaussian arm of prob 1 (skips real patch; for machines without dataset).",
    )
    p.add_argument(
        "--real-only-prob1",
        action="store_true",
        help="only run real-patch arm of prob 1 (skip Gaussian baseline).",
    )
    p.add_argument(
        "--no-proj-prob5",
        action="store_true",
        help="prob 5: skip Level-2 projected Jacobian (faster).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quick = args.quick
    out_dir = args.out_dir
    no_csv = args.no_csv
    verbose = args.verbose
    show_progress = not args.no_progress

    t0 = time.perf_counter()
    print("[T39] ORKM prob suite start (order: PROB1 -> PROB2 -> PROB3 -> PROB4 -> PROB5)", flush=True)
    print(
        f"[T39] flags: quick={quick}, no_csv={no_csv}, verbose={verbose}, show_progress={show_progress}",
        flush=True,
    )

    t1 = time.perf_counter()
    print("[T39] --- PROB 1/5: warm-start local convergence ---", flush=True)
    run_prob1(
        smoke=quick,
        no_csv=no_csv,
        out_dir=out_dir,
        verbose=verbose,
        gaussian_only=args.gaussian_only_prob1,
        real_only=args.real_only_prob1,
        show_progress=show_progress,
    )
    e1 = time.perf_counter() - t1
    print(f"[T39] PROB 1 done, elapsed={e1:.1f}s", flush=True)

    t2 = time.perf_counter()
    print("[T39] --- PROB 2/5: omega sweep ---", flush=True)
    run_prob2(
        smoke=quick,
        no_csv=no_csv,
        out_dir=out_dir,
        verbose=verbose,
        show_progress=show_progress,
    )
    e2 = time.perf_counter() - t2
    print(f"[T39] PROB 2 done, elapsed={e2:.1f}s", flush=True)

    t3 = time.perf_counter()
    print("[T39] --- PROB 3/5: A_seed sweep ---", flush=True)
    run_prob3(
        smoke=quick,
        no_csv=no_csv,
        out_dir=out_dir,
        omega=float(args.omega_prob3),
        verbose=verbose,
        show_progress=show_progress,
    )
    e3 = time.perf_counter() - t3
    print(f"[T39] PROB 3 done, elapsed={e3:.1f}s", flush=True)

    t4 = time.perf_counter()
    print("[T39] --- PROB 4/5: matched control signals ---", flush=True)
    run_prob4(
        smoke=quick,
        no_csv=no_csv,
        out_dir=out_dir,
        verbose=verbose,
        show_progress=show_progress,
    )
    e4 = time.perf_counter() - t4
    print(f"[T39] PROB 4 done, elapsed={e4:.1f}s", flush=True)

    t5 = time.perf_counter()
    print("[T39] --- PROB 5/5: Jacobian conditioning ---", flush=True)
    run_prob5(
        smoke=quick,
        no_proj=args.no_proj_prob5,
        show_progress=show_progress,
    )
    e5 = time.perf_counter() - t5
    print(f"[T39] PROB 5 done, elapsed={e5:.1f}s", flush=True)

    total = time.perf_counter() - t0
    print("", flush=True)
    print("[T39] ========== timing summary (seconds) ==========", flush=True)
    print(f"[T39] PROB1 warm-start     {e1:10.1f}", flush=True)
    print(f"[T39] PROB2 omega          {e2:10.1f}", flush=True)
    print(f"[T39] PROB3 A_seed         {e3:10.1f}", flush=True)
    print(f"[T39] PROB4 controls       {e4:10.1f}", flush=True)
    print(f"[T39] PROB5 Jacobian       {e5:10.1f}", flush=True)
    print(f"[T39] TOTAL                {total:10.1f}", flush=True)
    print("[T39] ORKM prob suite finished.", flush=True)
