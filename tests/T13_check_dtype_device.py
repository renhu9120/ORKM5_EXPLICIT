import torch

from core.octonion_base import ensure_octonion_tensor, oct_one, oct_zero
from core.octonion_inner import (
    amplitude_measurements_explicit,
    intensity_measurements_explicit,
    row_energy_batch,
    row_inner_batch,
)
from core.octonion_metric import oct_array_norm, raw_distance, relative_error
from core.octonion_ops import (
    oct_abs,
    oct_abs_sq,
    oct_conj,
    oct_left_mul,
    oct_left_mul_global,
    oct_mul,
    oct_normalize,
    oct_phase,
    oct_real,
    oct_right_mul,
    oct_right_mul_global,
)


def same_device(out_device: torch.device, expected_device: torch.device) -> bool:
    if out_device.type != expected_device.type:
        return False
    if expected_device.type == "cuda":
        return out_device.index == 0
    return True


def check_item(name: str, out: torch.Tensor, expected_device: torch.device) -> bool:
    ok = (out.dtype == torch.float64) and same_device(out.device, expected_device)
    print(
        f"{name:34s} | dtype={out.dtype} | device={out.device} | status={'OK' if ok else 'FAIL'}"
    )
    return ok


def run_for_device(device: torch.device) -> None:
    print(f"--- running checks on device: {device} ---")
    ok_all = True

    x = torch.randn(5, 8, dtype=torch.float64, device=device)
    y = torch.randn(5, 8, dtype=torch.float64, device=device)
    a = torch.randn(8, dtype=torch.float64, device=device)
    A = torch.randn(4, 5, 8, dtype=torch.float64, device=device)

    ok_all &= check_item("ensure_octonion_tensor", ensure_octonion_tensor(x), device)
    ok_all &= check_item("oct_zero", oct_zero(3, device=device), device)
    ok_all &= check_item("oct_one", oct_one(device=device), device)
    ok_all &= check_item("oct_conj", oct_conj(x), device)
    ok_all &= check_item("oct_mul", oct_mul(x, y), device)
    ok_all &= check_item("oct_abs_sq", oct_abs_sq(x), device)
    ok_all &= check_item("oct_abs", oct_abs(x), device)
    ok_all &= check_item("oct_real", oct_real(x), device)
    ok_all &= check_item("oct_left_mul", oct_left_mul(a, x), device)
    ok_all &= check_item("oct_right_mul", oct_right_mul(x, a), device)
    ok_all &= check_item("oct_left_mul_global", oct_left_mul_global(a, x), device)
    ok_all &= check_item("oct_right_mul_global", oct_right_mul_global(x, a), device)

    phase, valid = oct_phase(x)
    ok_all &= check_item("oct_phase.phase", phase, device)
    valid_ok = same_device(valid.device, device)
    print(
        f"{'oct_phase.valid':34s} | dtype={valid.dtype} | device={valid.device} | status={'OK' if valid_ok else 'FAIL'}"
    )
    ok_all &= valid_ok

    normed, valid2 = oct_normalize(x)
    ok_all &= check_item("oct_normalize.normed", normed, device)
    valid2_ok = same_device(valid2.device, device)
    print(
        f"{'oct_normalize.valid':34s} | dtype={valid2.dtype} | device={valid2.device} | status={'OK' if valid2_ok else 'FAIL'}"
    )
    ok_all &= valid2_ok

    ok_all &= check_item("row_inner_batch", row_inner_batch(A, x), device)
    ok_all &= check_item("intensity_measurements", intensity_measurements_explicit(A, x), device)
    ok_all &= check_item("amplitude_measurements", amplitude_measurements_explicit(A, x), device)
    ok_all &= check_item("row_energy_batch", row_energy_batch(A), device)
    ok_all &= check_item("oct_array_norm", oct_array_norm(x), device)
    ok_all &= check_item("raw_distance", raw_distance(x, y), device)
    ok_all &= check_item("relative_error", relative_error(x, y), device)

    print(f"device {device} status: {'PASS' if ok_all else 'FAIL'}")
    if not ok_all:
        raise RuntimeError(f"dtype/device check failed on {device}")


def main() -> None:
    print("=== T13 dtype/device consistency check ===")
    run_for_device(torch.device("cpu"))
    if torch.cuda.is_available():
        run_for_device(torch.device("cuda"))
    print("T13 status: PASS")


if __name__ == "__main__":
    main()
