from __future__ import annotations

import os
from pathlib import Path
import sys
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.octonion_sign import (
    absolute_inner_product_similarity,
    best_global_sign,
    sign_aligned_distance,
)


def _rand_x(d: int = 12) -> torch.Tensor:
    return torch.randn(d, 8, dtype=torch.float64)


def test_sign_aligned_reflexive() -> None:
    x = _rand_x()
    dist = sign_aligned_distance(x, x)
    assert dist.ndim == 0
    assert float(dist.item()) <= 1e-12


def test_sign_invariance() -> None:
    x = _rand_x()
    dist = sign_aligned_distance(x, -x)
    assert float(dist.item()) <= 1e-12
    assert best_global_sign(x, -x) == -1


def test_nonnegative() -> None:
    x = _rand_x()
    y = _rand_x()
    dist = sign_aligned_distance(x, y)
    assert float(dist.item()) >= 0.0


def test_matches_manual_min_definition() -> None:
    x = _rand_x(d=7)
    y = _rand_x(d=7)
    got = sign_aligned_distance(x, y)
    d_plus = torch.linalg.norm((x - y).reshape(-1), ord=2)
    d_minus = torch.linalg.norm((x + y).reshape(-1), ord=2)
    ref = torch.minimum(d_plus, d_minus)
    assert torch.allclose(got, ref, atol=1e-12, rtol=0.0)


def test_absip_range_and_signed_identity() -> None:
    x = _rand_x()
    y = _rand_x()
    absip_xy = absolute_inner_product_similarity(x, y)
    assert float(absip_xy.item()) >= -1e-12
    assert float(absip_xy.item()) <= 1.0 + 1e-12

    absip_xx = absolute_inner_product_similarity(x, x)
    absip_xmx = absolute_inner_product_similarity(x, -x)
    assert torch.allclose(absip_xx, torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=0.0)
    assert torch.allclose(absip_xmx, torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_shape_mismatch_and_invalid_input() -> None:
    x = _rand_x(d=6)
    y = _rand_x(d=7)
    with pytest.raises(ValueError):
        _ = sign_aligned_distance(x, y)

    x_bad = torch.randn(6, 7, dtype=torch.float64)
    with pytest.raises(ValueError):
        _ = sign_aligned_distance(x_bad, x_bad)
