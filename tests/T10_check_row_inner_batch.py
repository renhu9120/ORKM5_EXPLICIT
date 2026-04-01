import torch

from core.octonion_inner import row_inner_batch, row_inner_explicit


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    device = pick_device()
    print("=== T10 row inner batch check ===")
    print(f"selected device: {device}")

    groups = 5
    max_err_all = 0.0
    for g in range(groups):
        n = 6
        d = 9
        A = torch.randn(n, d, 8, dtype=torch.float64, device=device)
        x = torch.randn(d, 8, dtype=torch.float64, device=device)

        s_batch = row_inner_batch(A, x)
        errs = []
        for l in range(n):
            s_single = row_inner_explicit(A[l], x)
            err = torch.norm(s_batch[l] - s_single).item()
            errs.append(err)
            print(f"group {g:02d}, row {l:02d}: error={err:.6e}")

        max_err_group = max(errs)
        max_err_all = max(max_err_all, max_err_group)
        print(f"group {g:02d}: max row error={max_err_group:.6e}")

    print(f"overall max row error: {max_err_all:.6e}")
    ok = max_err_all < 1e-12
    print(f"T10 status: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("T10 failed")


if __name__ == "__main__":
    main()
