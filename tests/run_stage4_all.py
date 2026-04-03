from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    tests = [
        root / "tests" / "test_stage4_step_identity.py",
        root / "tests" / "test_stage4_cross_row_response.py",
        root / "tests" / "test_stage4_cross_row_bound.py",
        root / "tests" / "test_stage4_surrogate_descent.py",
        root / "tests" / "test_stage4_fixed_point.py",
    ]
    cmd_suffix = ["--quick"] if args.quick else []
    summary_lines = ["[Stage4-All] run summary"]
    for t in tests:
        cmd = [py, str(t)] + cmd_suffix
        print(f"[Stage4-All] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(root), check=False)
        summary_lines.append(f"{t.name}: return_code={proc.returncode}")
        if proc.returncode != 0:
            summary_lines.append(f"FAILED at {t.name}, stop.")
            break

    out = root / "output" / "stage4_all_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[Stage4-All] summary={out}")
