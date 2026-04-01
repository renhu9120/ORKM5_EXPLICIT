import torch

from core.octonion_ops import oct_abs, oct_normalize, oct_phase


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T07 phase/normalize check ===")
    print(f"selected device: {device}")

    z = torch.randn(64, 8, dtype=torch.float64, device=device)
    z[0] = 0.0
    z[5] = 0.0

    phase, valid = oct_phase(z, eps=1e-18)
    phase2, valid2 = oct_normalize(z, eps=1e-18)

    valid_count = int(valid.sum().item())
    invalid_count = int((~valid).sum().item())
    print(f"valid count: {valid_count}")
    print(f"invalid count: {invalid_count}")

    if valid_count > 0:
        phase_norm = oct_abs(phase[valid])
        max_unit_err = torch.max(torch.abs(phase_norm - 1.0)).item()
    else:
        max_unit_err = 0.0

    if invalid_count > 0:
        zero_phase_norm = torch.max(oct_abs(phase[~valid])).item()
    else:
        zero_phase_norm = 0.0

    norm_diff = torch.max(torch.abs(phase - phase2)).item()
    valid_diff = int(torch.sum(valid != valid2).item())

    print(f"max ||abs(phase)-1|| on valid: {max_unit_err:.6e}")
    print(f"zero phase norm on invalid: {zero_phase_norm:.6e}")
    print(f"max phase diff (phase vs normalize): {norm_diff:.6e}")
    print(f"valid mask mismatch count: {valid_diff}")

    ok = max_unit_err < 1e-10 and zero_phase_norm < 1e-12 and norm_diff < 1e-12 and valid_diff == 0
    print(f"T07 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T07 failed")


if __name__ == "__main__":
    main()
