from __future__ import annotations

"""
Single-strip real-image ORKM on CAVE-style 8-band PNGs (same layout as main_4).

  - Same band file selection / PNG read / S8_full (8,H,W).
  - Octonion packing: patch8 (8, ph, pw) -> x_true (ph*pw, 8), row-major over spatial grid
    (index i = r * pw + c for pixel (r,c) inside the crop).

Defaults target sponges_ms (dataset/complete_ms_data/sponges_ms/sponges_ms, prefix sponges).

Spatial crop (replacing square patches):

  - One horizontal strip 1 x 100 pixels (1 row, 100 columns), centered in the image:
      y0 = (H - 1) // 2,  x0 = (W - 100) // 2
  - Requires W >= 100 and H >= 1. Still d = 100 so n/d and A shapes match the previous 10x10 setup.

Signal preprocessing (``--signal-mode``) before ORKM, same A and measurement model:

  - raw: use crop as-is
  - normalized: ``normalize_oct_signal(x_patch_raw)``
  - whitened: normalize -> patch-wise band Gram whitening -> normalize again

When ``signal_mode == "whitened"``, ORKM runs in the whitening domain; recon is inverse-whitened
and metrics / pseudo-RGB use the back-domain (original band) signal per ``patch_whitening`` design.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.algs.alg_orkm import alg_orkm
from core.balloons_hs_io import (
    band_paths,
    pseudo_rgb_from_8bands,
    read_png16_to_float64_01,
    select_8_indices,
)
from core.octonion_align import apply_global_right_phase, estimate_global_right_phase, right_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import oct_array_norm, normalize_oct_signal
from core.patch_whitening import prepare_x_true, recover_from_whitened


@dataclass(frozen=True)
class CenterStripConfig:
    patch_h: int = 1
    patch_w: int = 100
    n_over_d: int = 20
    passes: int = 80
    A_seed: int = 3
    seed_fixed: int = 2
    power_iters: int = 1
    stop_err: float = 0.0


def _center_rect_origin(H: int, W: int, ph: int, pw: int) -> Tuple[int, int]:
    if H < ph or W < pw:
        raise ValueError(f"image too small for crop: (H,W)=({H},{W}), crop=({ph},{pw})")
    y0 = (H - ph) // 2
    x0 = (W - pw) // 2
    return y0, x0


def _patches8_to_x_true(patch8: torch.Tensor) -> torch.Tensor:
    """(8, ph, pw) -> (ph*pw, 8), row-major: idx = r * pw + c."""
    if patch8.ndim != 3 or patch8.shape[0] != 8:
        raise ValueError(f"patch8 must be (8,ph,pw), got {tuple(patch8.shape)}")
    ph, pw = int(patch8.shape[1]), int(patch8.shape[2])
    return patch8.permute(1, 2, 0).contiguous().view(ph * pw, 8)


def _x_true_to_patches8(x_true: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """(ph*pw, 8) -> (8, ph, pw)."""
    if x_true.ndim != 2 or x_true.shape[1] != 8:
        raise ValueError(f"x_true must be (d,8), got {tuple(x_true.shape)}")
    d_expected = ph * pw
    if x_true.shape[0] != d_expected:
        raise ValueError(f"x_true has d={x_true.shape[0]}, expected {d_expected}")
    return x_true.view(ph, pw, 8).permute(2, 0, 1).contiguous()


def _save_patch_rgb_images(
        rgb_gt: np.ndarray,
        rgb_rec: np.ndarray,
        out_png: Path,
        out_np: Path,
        *,
        interactive_show: bool,
        title_suffix: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_np.parent.mkdir(parents=True, exist_ok=True)
    vmin = float(min(rgb_gt.min(), rgb_rec.min()))
    vmax = float(max(rgb_gt.max(), rgb_rec.max()))
    denom = (vmax - vmin + 1e-30)

    rgb_gt_disp = np.clip((rgb_gt - vmin) / denom, 0.0, 1.0)
    rgb_rec_disp = np.clip((rgb_rec - vmin) / denom, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 2.2), dpi=140)
    axes[0].imshow(rgb_gt_disp, aspect="auto")
    axes[0].set_title(f"GT pseudo-RGB ({title_suffix}, scaled)")
    axes[0].axis("off")
    axes[1].imshow(rgb_rec_disp, aspect="auto")
    axes[1].set_title(f"Recon pseudo-RGB ({title_suffix}, scaled)")
    axes[1].axis("off")
    plt.tight_layout()
    if interactive_show:
        plt.show()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    np.save(out_np, {"rgb_gt": rgb_gt, "rgb_rec": rgb_rec})


def run_center_patch_orkm(
        *,
        folder: str,
        obj_name_prefix: str,
        device: str,
        out_dir: str,
        passes: int,
        interactive_show: bool,
        verbose: bool,
        progress_every: int,
        record_meas_rel: bool,
        signal_mode: str,
        whiten_eps: float,
) -> None:
    base = CenterStripConfig()
    ph, pw = base.patch_h, base.patch_w
    d = ph * pw
    n = d * base.n_over_d
    T = int(passes)

    device_t = torch.device(device)
    dtype = torch.float64

    idx8 = select_8_indices()
    bands8_np: List[np.ndarray] = []
    for p in band_paths(folder, obj_name_prefix, idx8):
        bands8_np.append(read_png16_to_float64_01(p))
    S8_full = np.stack(bands8_np, axis=0).astype(np.float64)
    H, W = int(S8_full.shape[1]), int(S8_full.shape[2])

    y0, x0 = _center_rect_origin(H, W, ph, pw)
    y1, x1 = y0 + ph, x0 + pw

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = (
        f"{obj_name_prefix}_center_strip{ph}x{pw}_sm{signal_mode}_n{n}_d{d}_T{T}_"
        f"Aseed{base.A_seed}_seed{base.seed_fixed}"
    )

    torch.manual_seed(base.seed_fixed)
    if device_t.type == "cuda":
        torch.cuda.manual_seed_all(base.seed_fixed)
    torch.manual_seed(base.A_seed)
    if device_t.type == "cuda":
        torch.cuda.manual_seed_all(base.A_seed)
    A = torch.randn((n, d, 8), dtype=dtype, device=device_t)

    print(f"[Main5-CenterPatch] device={device_t.type}, dtype=float64")
    print(f"[Main5-CenterPatch] folder={folder}, obj_name_prefix={obj_name_prefix}")
    print(
        f"[Main5-CenterPatch] image(H,W)=({H},{W}), strip={ph}x{pw} (centered), "
        f"pixel ROI y[{y0}:{y1}], x[{x0}:{x1}]"
    )
    print(f"[Main5-CenterPatch] d={d}, n={n}, n/d={base.n_over_d}, T={T}, power_iters={base.power_iters}")
    print(f"[Main5-CenterPatch] A_seed={base.A_seed}, seed_fixed={base.seed_fixed} (before A); alg_orkm seed=None")
    print(f"[Main5-CenterPatch] signal_mode={signal_mode}, whiten_eps={whiten_eps:.6e}")
    print(
        f"[Main5-CenterPatch] iteration logging: verbose={verbose}, progress_every={progress_every}, "
        f"record_meas_rel={record_meas_rel}"
    )

    patch8 = torch.as_tensor(S8_full[:, y0:y1, x0:x1], dtype=dtype, device=device_t)
    x_patch_raw = _patches8_to_x_true(patch8)
    x_true, aux = prepare_x_true(x_patch_raw, mode=signal_mode, eps=whiten_eps)

    print(f"[Main5-CenterPatch] raw_patch_norm={float(oct_array_norm(x_patch_raw).item()):.6e}")
    print(f"[Main5-CenterPatch] x_true_used_norm={float(oct_array_norm(x_true).item()):.6e}")

    if signal_mode == "normalized":
        sig = aux["Sigma_raw"]
        eig_n = torch.linalg.eigvalsh(sig).detach().cpu().numpy()
        print(f"[Main5-CenterPatch] eig(Sigma_normed)={np.array2string(eig_n, precision=4, suppress_small=True)}")
    if signal_mode == "whitened":
        eig_b = torch.linalg.eigvalsh(aux["Sigma_raw"]).detach().cpu().numpy()
        eig_a = torch.linalg.eigvalsh(aux["Sigma_white"]).detach().cpu().numpy()
        print(
            f"[Main5-CenterPatch] eig(Sigma_before_whiten)="
            f"{np.array2string(eig_b, precision=4, suppress_small=True)}"
        )
        print(
            f"[Main5-CenterPatch] eig(Sigma_after_whiten)="
            f"{np.array2string(eig_a, precision=4, suppress_small=True)}"
        )

    y_meas = intensity_measurements_explicit(A, x_true)

    if verbose:
        print("[Main5-CenterPatch] ----- init_osi + grad_orkm iteration log (below) -----", flush=True)
    t0 = time.perf_counter()
    x_est, _info = alg_orkm(
        A=A,
        y=y_meas,
        T=T,
        seed=None,
        power_iters=base.power_iters,
        omega=1.0,
        record_meas_rel=record_meas_rel,
        x_true_proc=x_true,
        stop_err=base.stop_err,
        verbose=verbose,
        progress_every=progress_every,
    )
    wall = time.perf_counter() - t0

    if x_est.shape != x_true.shape:
        raise RuntimeError(f"Unexpected x_est shape {tuple(x_est.shape)}, expected {tuple(x_true.shape)}")

    q_w = estimate_global_right_phase(x_true, x_est)
    x_est_aligned = apply_global_right_phase(x_est, q_w)

    dist_w = float(right_aligned_distance(x_true, x_est).item())
    rel_l2_w = float((oct_array_norm(x_true - x_est_aligned) / oct_array_norm(x_true)).item())
    log_dist_w = float(np.log(max(dist_w, 1e-300)))
    y_hat_w = intensity_measurements_explicit(A, x_est_aligned)
    meas_rel_w = float((torch.linalg.norm(y_hat_w - y_meas) / torch.linalg.norm(y_meas)).item())

    print("[Main5-CenterPatch] white_domain:")
    print(
        f"  dist={dist_w:.6e}, log(dist)={log_dist_w:.6e}, rel_l2={rel_l2_w:.6e}, meas_rel={meas_rel_w:.6e}"
    )

    x_rec_for_image = recover_from_whitened(x_est_aligned, signal_mode, aux)
    if signal_mode == "whitened":
        x_ref_back = normalize_oct_signal(x_patch_raw)
        x_rec_back = normalize_oct_signal(x_rec_for_image)
    else:
        x_ref_back = x_true
        x_rec_back = x_rec_for_image

    q_b = estimate_global_right_phase(x_ref_back, x_rec_back)
    x_rec_back_aligned = apply_global_right_phase(x_rec_back, q_b)
    dist_b = float(right_aligned_distance(x_ref_back, x_rec_back).item())
    rel_l2_b = float(
        (oct_array_norm(x_ref_back - x_rec_back_aligned) / oct_array_norm(x_ref_back)).item()
    )
    log_dist_b = float(np.log(max(dist_b, 1e-300)))

    print("[Main5-CenterPatch] back_domain:")
    print(f"  dist_back={dist_b:.6e}, log(dist_back)={log_dist_b:.6e}, rel_l2_back={rel_l2_b:.6e}")

    patch8_gt_back = _x_true_to_patches8(x_ref_back, ph, pw).detach().cpu().numpy()
    patch8_rec = _x_true_to_patches8(x_rec_back_aligned, ph, pw).detach().cpu().numpy()

    S8_rec = np.zeros_like(S8_full, dtype=np.float64)
    S8_rec[:, y0:y1, x0:x1] = patch8_rec

    rgb_gt = pseudo_rgb_from_8bands([patch8_gt_back[k] for k in range(8)], idx8)
    rgb_rec = pseudo_rgb_from_8bands([patch8_rec[k] for k in range(8)], idx8)
    psnr = float(peak_signal_noise_ratio(rgb_gt, rgb_rec, data_range=1.0))
    try:
        ssim = float(structural_similarity(rgb_gt, rgb_rec, data_range=1.0, channel_axis=-1))
    except ValueError as ex:
        ssim = float("nan")
        print(f"[Main5-CenterPatch] SSIM skipped (thin ROI): {ex}")

    out_png = out_root / f"center_strip_recon_{stamp}.png"
    out_np = out_root / f"center_strip_recon_{stamp}.npy"
    _save_patch_rgb_images(
        rgb_gt,
        rgb_rec,
        out_png,
        out_np,
        interactive_show=interactive_show,
        title_suffix=f"{ph}x{pw} strip back-domain",
    )

    print("[Main5-CenterPatch] pseudo-RGB(back-domain):")
    if np.isfinite(ssim):
        print(f"  PSNR={psnr:.3f}, SSIM={ssim:.6f}")
    else:
        print(f"  PSNR={psnr:.3f}, SSIM=n/a")
    print(f"[Main5-CenterPatch] wall_orkm_sec={wall:.2f}s")
    print(f"[Main5-CenterPatch] saved: {out_png}")
    print(f"[Main5-CenterPatch] saved: {out_np}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single centered 1x100 strip ORKM on CAVE-style 8-band PNGs (d=100)."
    )
    p.add_argument(
        "--folder",
        type=str,
        default=os.path.join(str(PROJECT_ROOT), "dataset", "complete_ms_data", "sponges_ms", "sponges_ms"),
    )
    p.add_argument("--obj-name-prefix", type=str, default="sponges")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "output" / "real_sponges_center_patch_orkm"))
    p.add_argument("--passes", type=int, default=80, help="ORKM iterations T (default 80).")
    p.add_argument(
        "--no-interactive-show",
        action="store_true",
        help="do not call plt.show() (recommended for batch / CI).",
    )
    p.add_argument(
        "--no-verbose",
        action="store_true",
        help="disable init_osi / grad_orkm iteration prints.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=8,
        help="print grad_orkm line every this many epochs (also used for init_osi when it has multiple iters).",
    )
    p.add_argument(
        "--verbose-meas",
        action="store_true",
        help="when verbose, also log meas_rel each recorded iter (adds a full intensity forward per iter; slower).",
    )
    p.add_argument(
        "--signal-mode",
        type=str,
        default="whitened",
        choices=["raw", "normalized", "whitened"],
        help="signal preprocessing before ORKM (raw | normalized | whitened).",
    )
    p.add_argument(
        "--whiten-eps",
        type=float,
        default=1e-10,
        help="eigenvalue floor for band whitening (whitened mode only).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verbose = not args.no_verbose
    run_center_patch_orkm(
        folder=args.folder,
        obj_name_prefix=args.obj_name_prefix,
        device=args.device,
        out_dir=args.out_dir,
        passes=args.passes,
        interactive_show=not args.no_interactive_show,
        verbose=verbose,
        progress_every=max(1, int(args.progress_every)),
        record_meas_rel=bool(verbose and args.verbose_meas),
        signal_mode=str(args.signal_mode),
        whiten_eps=float(args.whiten_eps),
    )
