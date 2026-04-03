"""Pathway: main_4 ROI with signal-mode + back-domain chain (short run)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "complete_ms_data" / "balloons_ms" / "balloons_ms"
SCRIPT = ROOT / "scripts" / "main_4_real_image_stage1_roi_orkm.py"


@pytest.mark.skipif(not DATA.is_dir(), reason="balloons_ms dataset not present")
@pytest.mark.parametrize("mode", ["raw", "normalized", "whitened"])
def test_main4_roi_signal_mode(mode: str) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLBACKEND", "Agg")
    out = ROOT / "output" / f"test_main4_smoke_{mode}"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--signal-mode",
            mode,
            "--max-patches",
            "1",
            "--passes",
            "2",
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
