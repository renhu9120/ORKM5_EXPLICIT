from __future__ import annotations

import torch
from torch import Tensor


def ensure_octonion_tensor(x: Tensor, name: str = "x") -> Tensor:
    """
    Standardize input as an explicit octonion tensor with shape (..., 8).
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.ndim == 0 or x.shape[-1] != 8:
        raise ValueError(f"{name} must have shape (..., 8), got {tuple(x.shape)}")
    return x.to(dtype=torch.float64).contiguous()


def assert_same_shape(
    x: Tensor, y: Tensor, x_name: str = "x", y_name: str = "y"
) -> None:
    """
    Assert two octonion tensors have identical shapes.
    """
    if tuple(x.shape) != tuple(y.shape):
        raise ValueError(
            f"Shape mismatch: {x_name}{tuple(x.shape)} vs {y_name}{tuple(y.shape)}"
        )


def oct_zero_like(x: Tensor) -> Tensor:
    """
    Return a zero octonion tensor with the same shape as x.
    """
    x_std = ensure_octonion_tensor(x, name="x")
    return torch.zeros_like(x_std, dtype=torch.float64, device=x_std.device)


def oct_zero(*shape: int, device=None) -> Tensor:
    """
    Create a zero octonion tensor with shape (*shape, 8).
    If shape is empty, returns shape (8,).
    """
    target_shape = (8,) if len(shape) == 0 else (*shape, 8)
    return torch.zeros(target_shape, dtype=torch.float64, device=device)


def oct_one(device=None) -> Tensor:
    """
    Return octonion multiplicative identity: (1, 0, ..., 0).
    """
    out = oct_zero(device=device)
    out[0] = 1.0
    return out


def oct_basis(index: int, device=None) -> Tensor:
    """
    Return the basis element with index 0..7.
    index=0 is scalar 1, index=1..7 are imaginary basis elements.
    """
    if not isinstance(index, int):
        raise TypeError(f"index must be int, got {type(index)}")
    if index < 0 or index > 7:
        raise ValueError(f"index must be in [0, 7], got {index}")
    out = oct_zero(device=device)
    out[index] = 1.0
    return out


def oct_stack_basis(device=None) -> Tensor:
    """
    Return all basis elements stacked as shape (8, 8), one per row.
    """
    basis = [oct_basis(i, device=device) for i in range(8)]
    return torch.stack(basis, dim=0).to(dtype=torch.float64)
