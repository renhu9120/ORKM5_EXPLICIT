from __future__ import annotations

"""
Single centered 8×8 patch — real-image ORKM demo (CAVE-style 8-band PNGs).

  - Dataset: same ``--folder`` / ``--obj-name-prefix`` convention as ``main_3_real_img_exp``.
  - One square patch (8×8), centered in the image.
  - ``patch8`` (8, 8, 8) → ``x_true`` (64, 8), row-major (idx = r * 8 + c).
  - ``A ~ randn`` with **no fixed seed** (natural RNG unless you seed externally).

End figure: 1×3 subplots — full-scene pseudo-RGB with patch box, GT patch, reconstruction.
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
from matplotlib.patches import Rectangle
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.constants import DEFAULT_ORKM_OMEGA
from algorithms.algs.alg_orkm import alg_orkm
from core.balloons_hs_io import (
    band_paths,
    pseudo_rgb_from_8bands,
    read_png16_to_float64_01,
    select_8_indices,
)
from core.octonion_align import apply_global_right_phase, estimate_global_right_phase
from core.octonion_sign import sign_aligned_distance
from core.octonion_inner import intensity_measurements_explicit
from core.octonion_metric import oct_array_norm, normalize_oct_signal
from core.patch_whitening import prepare_x_true, recover_from_whitened


@dataclass(frozen=True)
class CenterPatchConfig:
    patch_size: int = 16
    n_over_d: int = 20
    passes: int = 80
    power_iters: int = 1
    # Real-image runs use fixed ``passes`` only; keep oracle early-stop off.
    stop_err: float = 0.0


def _center_square_origin(H: int, W: int, patch_size: int) -> Tuple[int, int]:
    if H < patch_size or W < patch_size:
        raise ValueError(f"image too small for patch: (H,W)=({H},{W}), patch_size={patch_size}")
    y0 = (H - patch_size) // 2
    x0 = (W - patch_size) // 2
    return y0, x0


def _patches8_to_x_true(patch8: torch.Tensor) -> torch.Tensor:
    """(8, p, p) -> (p*p, 8), row-major: idx = r * p + c."""
    if patch8.ndim != 3 or patch8.shape[0] != 8:
        raise ValueError(f"patch8 must be (8,p,p), got {tuple(patch8.shape)}")
    p = int(patch8.shape[1])
    if patch8.shape[2] != p:
        raise ValueError(f"patch8 must be square, got {tuple(patch8.shape)}")
    return patch8.permute(1, 2, 0).contiguous().view(p * p, 8)


def _x_true_to_patches8(x_true: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(p*p, 8) -> (8, p, p)."""
    if x_true.ndim != 2 or x_true.shape[1] != 8:
        raise ValueError(f"x_true must be (d,8), got {tuple(x_true.shape)}")
    d_expected = patch_size * patch_size
    if x_true.shape[0] != d_expected:
        raise ValueError(f"x_true has d={x_true.shape[0]}, expected {d_expected}")
    return x_true.view(patch_size, patch_size, 8).permute(2, 0, 1).contiguous()


def _scale_01(a: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = vmax - vmin + 1e-30
    return np.clip((a - vmin) / denom, 0.0, 1.0)


def _save_demo_figure(
        *,
        rgb_full: np.ndarray,
        rgb_patch_gt: np.ndarray,
        rgb_patch_rec: np.ndarray,
        y0: int,
        x0: int,
        patch_size: int,
        out_png: Path,
        out_npz: Path,
        interactive_show: bool,
        p_sz: int,
) -> None:
    """subplot(1,3): full image + patch box | GT patch | recon patch."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    H, W = rgb_full.shape[0], rgb_full.shape[1]
    vmin_f = float(rgb_full.min())
    vmax_f = float(rgb_full.max())
    full_disp = _scale_01(rgb_full, vmin_f, vmax_f)

    vmin_p = float(min(rgb_patch_gt.min(), rgb_patch_rec.min()))
    vmax_p = float(max(rgb_patch_gt.max(), rgb_patch_rec.max()))
    gt_disp = _scale_01(rgb_patch_gt, vmin_p, vmax_p)
    rec_disp = _scale_01(rgb_patch_rec, vmin_p, vmax_p)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), dpi=160)
    ax0, ax1, ax2 = axes[0], axes[1], axes[2]

    ax0.imshow(full_disp)
    rect = Rectangle(
        (x0, y0),
        patch_size,
        patch_size,
        linewidth=2.5,
        edgecolor="lime",
        facecolor="none",
    )
    ax0.add_patch(rect)
    ax0.set_title(f"Full pseudo-RGB (H={H}, W={W})\ngreen box = centered {p_sz}×{p_sz} patch")
    ax0.axis("off")

    ax1.imshow(gt_disp)
    ax1.set_title("GT patch (back-domain pseudo-RGB, scaled)")
    ax1.axis("off")

    ax2.imshow(rec_disp)
    ax2.set_title("Reconstruction (same scaling as GT patch)")
    ax2.axis("off")

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    if interactive_show:
        print("[Main5-CenterPatch] showing figure (close window to continue)...")
        plt.show(block=True)
    plt.close(fig)

    np.savez_compressed(
        out_npz,
        rgb_full=rgb_full,
        rgb_patch_gt=rgb_patch_gt,
        rgb_patch_rec=rgb_patch_rec,
        patch_y0=y0,
        patch_x0=x0,
        patch_y1=y0 + patch_size,
        patch_x1=x0 + patch_size,
    )


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
    base = CenterPatchConfig()
    p_sz = base.patch_size
    d = p_sz * p_sz
    n = d * base.n_over_d
    T = int(passes)

    device_t = torch.device(device)
    dtype = torch.float64

    idx8 = select_8_indices()
    bands8_np: List[np.ndarray] = []
    for path in band_paths(folder, obj_name_prefix, idx8):
        bands8_np.append(read_png16_to_float64_01(path))
    S8_full = np.stack(bands8_np, axis=0).astype(np.float64)
    H, W = int(S8_full.shape[1]), int(S8_full.shape[2])

    y0, x0 = _center_square_origin(H, W, p_sz)
    y1, x1 = y0 + p_sz, x0 + p_sz

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = (
        f"{obj_name_prefix}_center_patch{p_sz}x{p_sz}_sm{signal_mode}_n{n}_d{d}_T{T}_"
        f"pi{base.power_iters}"
    )

    A = torch.randn((n, d, 8), dtype=dtype, device=device_t)

    print(f"[Main5-CenterPatch] device={device_t.type}, dtype=float64")
    print(f"[Main5-CenterPatch] folder={folder}, obj_name_prefix={obj_name_prefix}")
    print(
        f"[Main5-CenterPatch] image(H,W)=({H},{W}), patch={p_sz}x{p_sz} (centered), "
        f"pixel ROI y[{y0}:{y1}], x[{x0}:{x1}]"
    )
    print(f"[Main5-CenterPatch] d={d}, n={n}, n/d={base.n_over_d}, T={T}, power_iters={base.power_iters}")
    print("[Main5-CenterPatch] A ~ randn (no fixed seed); alg_orkm(..., seed=None)")
    print(f"[Main5-CenterPatch] signal_mode={signal_mode}, whiten_eps={whiten_eps:.6e}")
    print(
        f"[Main5-CenterPatch] iteration logging: verbose={verbose}, progress_every={progress_every}, "
        f"record_meas_rel={record_meas_rel}"
    )

    rgb_full = pseudo_rgb_from_8bands([S8_full[k] for k in range(8)], idx8)

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
        omega=DEFAULT_ORKM_OMEGA,
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

    dist_sign_w = float(sign_aligned_distance(x_true, x_est).item())
    rel_l2_w = float((oct_array_norm(x_true - x_est_aligned) / oct_array_norm(x_true)).item())
    log_dist_w = float(np.log(max(dist_sign_w, 1e-300)))
    y_hat_w = intensity_measurements_explicit(A, x_est_aligned)
    meas_rel_w = float((torch.linalg.norm(y_hat_w - y_meas) / torch.linalg.norm(y_meas)).item())

    print("[Main5-CenterPatch] white_domain:")
    print(
        f"  dist_sign={dist_sign_w:.6e}, log(dist_sign)={log_dist_w:.6e}, "
        f"rel_l2={rel_l2_w:.6e}, meas_rel={meas_rel_w:.6e}"
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
    dist_sign_b = float(sign_aligned_distance(x_ref_back, x_rec_back).item())
    rel_l2_b = float(
        (oct_array_norm(x_ref_back - x_rec_back_aligned) / oct_array_norm(x_ref_back)).item()
    )
    log_dist_b = float(np.log(max(dist_sign_b, 1e-300)))

    print("[Main5-CenterPatch] back_domain:")
    print(
        f"  dist_sign_back={dist_sign_b:.6e}, log(dist_sign)={log_dist_b:.6e}, rel_l2_back={rel_l2_b:.6e}"
    )

    patch8_gt_back = _x_true_to_patches8(x_ref_back, p_sz).detach().cpu().numpy()
    patch8_rec = _x_true_to_patches8(x_rec_back_aligned, p_sz).detach().cpu().numpy()

    rgb_gt = pseudo_rgb_from_8bands([patch8_gt_back[k] for k in range(8)], idx8)
    rgb_rec = pseudo_rgb_from_8bands([patch8_rec[k] for k in range(8)], idx8)
    psnr = float(peak_signal_noise_ratio(rgb_gt, rgb_rec, data_range=1.0))
    ssim = float(structural_similarity(rgb_gt, rgb_rec, data_range=1.0, channel_axis=-1))

    out_png = out_root / f"center_patch_demo_{stamp}.png"
    out_npz = out_root / f"center_patch_demo_{stamp}.npz"
    _save_demo_figure(
        rgb_full=rgb_full,
        rgb_patch_gt=rgb_gt,
        rgb_patch_rec=rgb_rec,
        y0=y0,
        x0=x0,
        patch_size=p_sz,
        out_png=out_png,
        out_npz=out_npz,
        interactive_show=interactive_show,
        p_sz=p_sz,
    )

    print("[Main5-CenterPatch] pseudo-RGB(back-domain patch):")
    print(f"  PSNR={psnr:.3f}, SSIM={ssim:.6f}")
    print(f"[Main5-CenterPatch] wall_orkm_sec={wall:.2f}s")
    print(f"[Main5-CenterPatch] saved: {out_png}")
    print(f"[Main5-CenterPatch] saved: {out_npz}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Single centered 8×8 patch ORKM demo (same --folder / --obj-name-prefix as main_3; "
            "1×3 figure: full scene + patch box | GT patch | recon)."
        ),
    )
    p.add_argument(
        "--folder",
        type=str,
        default=os.path.join(str(PROJECT_ROOT), "dataset", "complete_ms_data", "balloons_ms", "balloons_ms"),
    )
    p.add_argument("--obj-name-prefix", type=str, default="balloons")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "output" / "real_center_patch_orkm_demo"),
    )
    p.add_argument("--passes", type=int, default=80, help="ORKM iterations T (default 80).")
    p.add_argument(
        "--no-interactive-show",
        action="store_true",
        help="do not open matplotlib window (PNG/NPY still saved).",
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
