"""
Lightweight smoke: run prob diagnostic scripts with --smoke (short passes, reduced sweeps).
Skips if the balloons real-patch dataset path is missing (prob scripts need it except prob_1 --gaussian-only).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROB_DATASET = ROOT / "dataset" / "complete_ms_data" / "balloons_ms" / "balloons_ms"


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), "--smoke", "--no-csv"]
    if extra:
        cmd.extend(extra)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


@pytest.mark.skipif(not PROB_DATASET.is_dir(), reason="balloons_ms dataset not present")
def test_prob_1_warm_start_smoke_real() -> None:
    _run("prob_1_warm_start_local_conv.py", ["--real-only"])


@pytest.mark.skipif(not PROB_DATASET.is_dir(), reason="balloons_ms dataset not present")
def test_prob_2_omega_smoke() -> None:
    _run("prob_2_omega_sweep_realpatch.py")


@pytest.mark.skipif(not PROB_DATASET.is_dir(), reason="balloons_ms dataset not present")
def test_prob_3_Aseed_smoke() -> None:
    _run("prob_3_Aseed_sweep_realpatch.py")


@pytest.mark.skipif(not PROB_DATASET.is_dir(), reason="balloons_ms dataset not present")
def test_prob_4_control_smoke() -> None:
    _run("prob_4_matched_control_signals.py")


@pytest.mark.skipif(not PROB_DATASET.is_dir(), reason="balloons_ms dataset not present")
def test_prob_5_jacobian_smoke() -> None:
    _run("prob_5_jacobian_conditioning.py")


def test_prob_1_warm_start_smoke_gaussian_only_no_dataset() -> None:
    _run("prob_1_warm_start_local_conv.py", ["--gaussian-only"])
