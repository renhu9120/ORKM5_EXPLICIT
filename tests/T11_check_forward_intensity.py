import torch

from core.octonion_inner import (
    amplitude_measurements_explicit,
    intensity_measurements_explicit,
    row_amplitude_explicit,
    row_energy_batch,
    row_energy_explicit,
    row_intensity_explicit,
)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T11 forward intensity check ===")
    print(f"selected device: {device}")

    n, d = 10, 12
    A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
    x = torch.randn(d, 8, dtype=torch.float64, device=device)

    y_batch = intensity_measurements_explicit(A, x)
    b_batch = amplitude_measurements_explicit(A, x)
    beta_batch = row_energy_batch(A)

    y_single = torch.zeros(n, dtype=torch.float64, device=device)
    b_single = torch.zeros(n, dtype=torch.float64, device=device)
    beta_single = torch.zeros(n, dtype=torch.float64, device=device)
    for l in range(n):
        y_single[l] = row_intensity_explicit(A[l], x)
        b_single[l] = row_amplitude_explicit(A[l], x)
        beta_single[l] = row_energy_explicit(A[l])

    e_intensity = torch.max(torch.abs(y_batch - y_single)).item()
    e_amp = torch.max(torch.abs(b_batch * b_batch - y_batch)).item()
    e_energy = torch.max(torch.abs(beta_batch - beta_single)).item()
    y_min = torch.min(y_batch).item()
    y_max = torch.max(y_batch).item()
    y_mean = torch.mean(y_batch).item()

    print(f"max intensity batch/single diff: {e_intensity:.6e}")
    print(f"max |amp^2 - intensity|: {e_amp:.6e}")
    print(f"max energy diff: {e_energy:.6e}")
    print(f"intensity stats: min={y_min:.6e}, max={y_max:.6e}, mean={y_mean:.6e}")

    ok = e_intensity < 1e-12 and e_amp < 1e-10 and e_energy < 1e-12 and y_min >= -1e-12
    print(f"T11 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T11 failed")


if __name__ == "__main__":
    main()
