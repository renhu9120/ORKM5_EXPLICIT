import torch

from core.octonion_metric import oct_array_norm, raw_distance, relative_error


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T12 metric raw check ===")
    print(f"selected device: {device}")

    x = torch.randn(20, 8, dtype=torch.float64, device=device)
    noise = 0.01 * torch.randn(20, 8, dtype=torch.float64, device=device)
    x2 = x + noise

    nx = oct_array_norm(x)
    d_same = raw_distance(x, x)
    d_noise = raw_distance(x, x2)
    rel = relative_error(x, x2)

    print(f"norm(x): {nx.item():.6e}")
    print(f"raw_distance(x, x): {d_same.item():.6e}")
    print(f"raw_distance(x, x+noise): {d_noise.item():.6e}")
    print(f"relative_error(x, x+noise): {rel.item():.6e}")

    ok = d_same.item() < 1e-12 and d_noise.item() > 0.0 and rel.item() > 0.0
    print(f"T12 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T12 failed")


if __name__ == "__main__":
    main()
