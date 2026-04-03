from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_stage4_cross_row_bound import run as run_c
from test_stage4_cross_row_response import run as run_b
from test_stage4_fixed_point import run as run_e
from test_stage4_step_identity import run as run_a
from test_stage4_surrogate_descent import run as run_d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.perf_counter()
    print("[T37] Stage4 theory validation start")
    ta = time.perf_counter()
    run_a(quick=args.quick)
    print(f"[T37] Stage A done, elapsed={time.perf_counter() - ta:.1f}s")
    tb = time.perf_counter()
    run_b(quick=args.quick)
    print(f"[T37] Stage B done, elapsed={time.perf_counter() - tb:.1f}s")
    tc = time.perf_counter()
    run_c(quick=args.quick)
    print(f"[T37] Stage C done, elapsed={time.perf_counter() - tc:.1f}s")
    td = time.perf_counter()
    run_d(quick=args.quick)
    print(f"[T37] Stage D done, elapsed={time.perf_counter() - td:.1f}s")
    te = time.perf_counter()
    run_e(quick=args.quick)
    print(f"[T37] Stage E done, elapsed={time.perf_counter() - te:.1f}s")
    print(f"[T37] Stage4 theory validation done, total_elapsed={time.perf_counter() - t0:.1f}s")

    # Build paper-ready summary assets from produced CSVs.
    build_py = PROJECT_ROOT / "scripts" / "build_stage4_report.py"
    try:
        print(f"[T37] building stage4 report: {build_py}")
        subprocess.run([sys.executable, str(build_py)], cwd=str(PROJECT_ROOT), check=True)
        print("[T37] stage4 report build done")
    except Exception as ex:
        print(f"[T37] [warn] stage4 report build failed: {ex}")
