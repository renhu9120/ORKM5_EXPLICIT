"""Pathway: main_5 runs for raw / normalized / whitened (short passes, no GUI)."""

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


def _folder_prefix() -> tuple[str, str] | None:
    if SPONGES.is_dir():
        return str(SPONGES), "sponges"
    if BALLOONS.is_dir():
        return str(BALLOONS), "balloons"
    return None


_FP = _folder_prefix()


@pytest.mark.skipif(_FP is None, reason="no sponges_ms or balloons_ms dataset present")
@pytest.mark.parametrize("mode", ["raw", "normalized", "whitened"])
def test_main5_signal_mode_runs(mode: str) -> None:
    assert _FP is not None
    folder, prefix = _FP
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLBACKEND", "Agg")
    out = ROOT / "output" / f"test_main5_smoke_{mode}"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--folder",
            folder,
            "--obj-name-prefix",
            prefix,
            "--signal-mode",
            mode,
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
