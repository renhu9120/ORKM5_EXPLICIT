from __future__ import annotations

import torch
from torch import Tensor

from core.octonion_base import ensure_octonion_tensor, oct_zero
from core.octonion_ops import oct_abs, oct_abs_sq, oct_conj, oct_mul


def row_inner_explicit(a_row: Tensor, x: Tensor) -> Tensor:
    """
    Fixed-parenthesized explicit inner product:
        s = sum_j (conj(a_j) * x_j)
    """
    a_row_std = ensure_octonion_tensor(a_row, name="a_row")
    x_std = ensure_octonion_tensor(x, name="x")
    if a_row_std.ndim != 2 or x_std.ndim != 2:
        raise ValueError(
            f"a_row and x must both have shape (d, 8), got {a_row_std.shape} and {x_std.shape}"
        )
    if a_row_std.shape != x_std.shape:
        raise ValueError(
            f"Shape mismatch: a_row{tuple(a_row_std.shape)} vs x{tuple(x_std.shape)}"
        )

    d = a_row_std.shape[0]
    acc = oct_zero(device=a_row_std.device)
    for j in range(d):
        term = oct_mul(oct_conj(a_row_std[j]), x_std[j])
        acc = acc + term
    return acc


def row_inner_batch(A: Tensor, x: Tensor) -> Tensor:
    """
    Batch explicit inner product:
        s_l = sum_j (conj(a_lj) * x_j)
    """
    A_std = ensure_octonion_tensor(A, name="A")
    x_std = ensure_octonion_tensor(x, name="x")
    if A_std.ndim != 3 or x_std.ndim != 2:
        raise ValueError(
            f"A must be (n, d, 8) and x must be (d, 8), got {A_std.shape} and {x_std.shape}"
        )
    if A_std.shape[1] != x_std.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: A.shape={tuple(A_std.shape)}, x.shape={tuple(x_std.shape)}"
        )

    n = A_std.shape[0]
    out = torch.zeros((n, 8), dtype=torch.float64, device=A_std.device)
    for l in range(n):
        out[l] = row_inner_explicit(A_std[l], x_std)
    return out


def row_inner_fast(a_row: Tensor, x: Tensor) -> Tensor:
    """
    Vectorized explicit inner product equivalent to row_inner_explicit:
        s = sum_j (conj(a_j) * x_j)
    """
    a_row_std = ensure_octonion_tensor(a_row, name="a_row")
    x_std = ensure_octonion_tensor(x, name="x")
    if a_row_std.ndim != 2 or x_std.ndim != 2:
        raise ValueError(
            f"a_row and x must both have shape (d, 8), got {a_row_std.shape} and {x_std.shape}"
        )
    if a_row_std.shape != x_std.shape:
        raise ValueError(
            f"Shape mismatch: a_row{tuple(a_row_std.shape)} vs x{tuple(x_std.shape)}"
        )

    terms = oct_mul(oct_conj(a_row_std), x_std)
    return terms.sum(dim=0)


def row_inner_batch_fast(A: Tensor, x: Tensor) -> Tensor:
    """
    Vectorized batch explicit inner product equivalent to row_inner_batch:
        s_l = sum_j (conj(a_lj) * x_j)
    """
    A_std = ensure_octonion_tensor(A, name="A")
    x_std = ensure_octonion_tensor(x, name="x")
    if A_std.ndim != 3 or x_std.ndim != 2:
        raise ValueError(
            f"A must be (n, d, 8) and x must be (d, 8), got {A_std.shape} and {x_std.shape}"
        )
    if A_std.shape[1] != x_std.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: A.shape={tuple(A_std.shape)}, x.shape={tuple(x_std.shape)}"
        )

    terms = oct_mul(oct_conj(A_std), x_std.unsqueeze(0))
    return terms.sum(dim=1)


def row_intensity_explicit(a_row: Tensor, x: Tensor) -> Tensor:
    """
    Single-row intensity:
        y = |sum_j(conj(a_j) * x_j)|^2
    """
    s = row_inner_explicit(a_row, x)
    return oct_abs_sq(s)


def intensity_measurements_explicit(A: Tensor, x: Tensor) -> Tensor:
    """
    All-row intensities:
        y_l = |s_l(x)|^2
    """
    # Keep the explicit mathematical definition while using a vectorized backend.
    s = row_inner_batch_fast(A, x)
    return oct_abs_sq(s)


def row_intensity_fast(a_row: Tensor, x: Tensor) -> Tensor:
    """
    Single-row intensity via row_inner_fast:
        y = |sum_j(conj(a_j) * x_j)|^2
    """
    s = row_inner_fast(a_row, x)
    return oct_abs_sq(s)


def intensity_measurements_fast(A: Tensor, x: Tensor) -> Tensor:
    """
    All-row intensities via row_inner_batch_fast:
        y_l = |s_l(x)|^2
    """
    s = row_inner_batch_fast(A, x)
    return oct_abs_sq(s)


def intensity_measurements_independent_batches(A: Tensor, X: Tensor) -> Tensor:
    """
    Batched intensities when each batch element has its own ``A[b]``.

    A: (B, n, d, 8)
    X: (B, d, 8)
    returns y: (B, n)
    """
    A_std = ensure_octonion_tensor(A, name="A")
    X_std = ensure_octonion_tensor(X, name="X")
    if A_std.ndim != 4:
        raise ValueError(f"A must be (B, n, d, 8), got {tuple(A_std.shape)}")
    if X_std.ndim != 3:
        raise ValueError(f"X must be (B, d, 8), got {tuple(X_std.shape)}")
    if A_std.shape[0] != X_std.shape[0] or A_std.shape[2] != X_std.shape[1]:
        raise ValueError(f"shape mismatch A{tuple(A_std.shape)} vs X{tuple(X_std.shape)}")
    terms = oct_mul(oct_conj(A_std), X_std.unsqueeze(1))
    s = terms.sum(dim=2)
    return oct_abs_sq(s)


def intensity_measurements_batched(A: Tensor, X: Tensor) -> Tensor:
    """
    Batched intensities for a shared measurement operator ``A``.

    A: (n, d, 8)
    X: (B, d, 8)
    returns y: (B, n) with y[b, l] = |sum_j conj(a_lj) * x[b, j]|^2
    """
    A_std = ensure_octonion_tensor(A, name="A")
    X_std = ensure_octonion_tensor(X, name="X")
    if A_std.ndim != 3:
        raise ValueError(f"A must be (n, d, 8), got {tuple(A_std.shape)}")
    if X_std.ndim != 3:
        raise ValueError(f"X must be (B, d, 8), got {tuple(X_std.shape)}")
    if A_std.shape[1] != X_std.shape[1]:
        raise ValueError(
            f"Incompatible d: A{tuple(A_std.shape)} vs X{tuple(X_std.shape)}"
        )
    terms = oct_mul(oct_conj(A_std).unsqueeze(0), X_std.unsqueeze(1))
    s = terms.sum(dim=2)
    return oct_abs_sq(s)


def row_amplitude_explicit(a_row: Tensor, x: Tensor) -> Tensor:
    """
    Single-row amplitude:
        |s_l(x)|
    """
    s = row_inner_explicit(a_row, x)
    return oct_abs(s)


def amplitude_measurements_explicit(A: Tensor, x: Tensor) -> Tensor:
    """
    All-row amplitudes:
        b_l = |s_l(x)|
    """
    s = row_inner_batch(A, x)
    return oct_abs(s)


def row_amplitude_fast(a_row: Tensor, x: Tensor) -> Tensor:
    """
    Single-row amplitude via row_inner_fast:
        |s_l(x)|
    """
    s = row_inner_fast(a_row, x)
    return oct_abs(s)


def amplitude_measurements_fast(A: Tensor, x: Tensor) -> Tensor:
    """
    All-row amplitudes via row_inner_batch_fast:
        b_l = |s_l(x)|
    """
    s = row_inner_batch_fast(A, x)
    return oct_abs(s)


def row_energy_explicit(a_row: Tensor) -> Tensor:
    """
    Row energy:
        beta_l = sum_j |a_lj|^2
    """
    a_row_std = ensure_octonion_tensor(a_row, name="a_row")
    if a_row_std.ndim != 2:
        raise ValueError(f"a_row must be (d, 8), got {tuple(a_row_std.shape)}")
    return oct_abs_sq(a_row_std).sum()


def row_energy_batch(A: Tensor) -> Tensor:
    """
    Batch row energy for A of shape (n, d, 8).
    """
    A_std = ensure_octonion_tensor(A, name="A")
    if A_std.ndim != 3:
        raise ValueError(f"A must be (n, d, 8), got {tuple(A_std.shape)}")
    return oct_abs_sq(A_std).sum(dim=1)
