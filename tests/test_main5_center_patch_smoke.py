"""Smoke import/run for scripts/main_3_real_img_exp_single_patch.py (short passes, no GUI)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPONGES = ROOT / "dataset" / "complete_ms_data" / "sponges_ms" / "sponges_ms"
BALLOONS = ROOT / "dataset" / "complete_ms_data" / "balloons_ms" / "balloons_ms"
SCRIPT = ROOT / "scripts" / "main_3_real_img_exp_single_patch.py"


def _ms_folder_and_prefix() -> tuple[str, str] | None:
    if SPONGES.is_dir():
        return str(SPONGES), "sponges"
    if BALLOONS.is_dir():
        return str(BALLOONS), "balloons"
    return None


_MS = _ms_folder_and_prefix()


@pytest.mark.skipif(_MS is None, reason="neither sponges_ms nor balloons_ms dataset present")
def test_main5_center_patch_runs() -> None:
    assert _MS is not None
    folder, prefix = _MS
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLBACKEND", "Agg")
    out = ROOT / "output" / "test_main5_smoke"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--folder",
            folder,
            "--obj-name-prefix",
            prefix,
            "--signal-mode",
            "normalized",
            "--passes",
            "2",
            "--no-verbose",
            "--no-interactive-show",
            "--device",
            "cpu",
            "--out-dir",
            str(out),
        ],
        cwd=str(ROOT),
        check=True,
        env=env,
    )
