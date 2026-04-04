from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.algs.alg_orkm import alg_orkm
from algorithms.algs.alg_orkm_cuda import alg_orkm_cuda, alg_orkm_cuda_batched
from core.balloons_hs_io import (
    band_paths,
    pseudo_rgb_from_8bands,
    read_png16_to_float64_01,
    select_8_indices,
)
from core.octonion_align import apply_global_right_phase, estimate_global_right_phase, right_aligned_distance
from core.octonion_inner import intensity_measurements_batched, intensity_measurements_explicit, row_energy_batch
from core.octonion_metric import oct_array_norm, normalize_oct_signal
from core.patch_whitening import prepare_x_true, recover_from_whitened
from core.patch_whitening_batched import prepare_x_true_batched


@dataclass(frozen=True)
class ROIConfig:
    patch_size: int = 8
    roi_patches_side: int = 4  # 4x4 patches => 16 patches
    n_over_d: int = 16
    passes: int = 8  # T=passes=80 in current ORKM implementation
    A_seed: int = 32
    seed_fixed: int = 12
    power_iters: int = 12
    stop_err: float = 0.0


def _compute_center_roi_patch_grid(H: int, W: int, patch_size: int, roi_side: int) -> Tuple[int, int, int, int]:
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"H,W must be divisible by patch_size={patch_size}, got ({H},{W})")
    gh = H // patch_size
    gw = W // patch_size
    if gh < roi_side or gw < roi_side:
        raise ValueError(f"ROI patches side too large: gh={gh}, gw={gw}, roi_side={roi_side}")
    base_gr = (gh - roi_side) // 2
    base_gc = (gw - roi_side) // 2
    return gh, gw, base_gr, base_gc


def _flatten_patch_idx_r_c(patch_size: int, r_in: int, c_in: int) -> int:
    # Required convention: idx = r*patch_size + c (row-major).
    return r_in * patch_size + c_in


def _patches8_to_x_true(patch8: torch.Tensor) -> torch.Tensor:
    """
    patch8: (8, patch, patch)
    return x_true: (d=patch*patch, 8)
    idx corresponds to pixel (r,c) with idx=r*patch + c
    """
    if patch8.ndim != 3 or patch8.shape[0] != 8:
        raise ValueError(f"patch8 must be (8,patch,patch), got {tuple(patch8.shape)}")
    patch_size = int(patch8.shape[1])
    if patch8.shape[1] != patch8.shape[2]:
        raise ValueError(f"patch8 must be square in spatial dims, got {tuple(patch8.shape)}")
    # (8,patch,patch) -> (patch,patch,8) -> (d,8)
    x_true = patch8.permute(1, 2, 0).contiguous().view(patch_size * patch_size, 8)
    return x_true


def _aux_for_batch_index(aux_batch: dict, b: int, signal_mode: str) -> dict:
    if signal_mode == "raw":
        return {"mode": "raw"}
    if signal_mode == "normalized":
        return {"mode": "normalized", "Sigma_raw": aux_batch["Sigma_raw"][b]}
    if signal_mode == "whitened":
        return {
            "mode": "whitened",
            "W": aux_batch["W"][b],
            "Winv": aux_batch["Winv"][b],
            "Sigma_raw": aux_batch["Sigma_raw"][b],
            "Sigma_white": aux_batch["Sigma_white"][b],
        }
    raise ValueError(signal_mode)


def _x_true_to_patches8(x_true: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    x_true: (d=patch*patch,8) -> patch8: (8,patch,patch)
    """
    if x_true.ndim != 2 or x_true.shape[1] != 8:
        raise ValueError(f"x_true must be (d,8), got {tuple(x_true.shape)}")
    d_expected = patch_size * patch_size
    if x_true.shape[0] != d_expected:
        raise ValueError(f"x_true has d={x_true.shape[0]}, expected {d_expected}")
    patch8 = x_true.view(patch_size, patch_size, 8).permute(2, 0, 1).contiguous()
    return patch8


def _save_roi_images(
        rgb_gt: np.ndarray,
        rgb_rec: np.ndarray,
        out_png: Path,
        out_np: Path,
        *,
        interactive_show: bool,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_np.parent.mkdir(parents=True, exist_ok=True)
    # Clip only for display; PSNR/SSIM should use raw arrays.
    vmin = float(min(rgb_gt.min(), rgb_rec.min()))
    vmax = float(max(rgb_gt.max(), rgb_rec.max()))
    denom = (vmax - vmin + 1e-30)

    rgb_gt_disp = np.clip((rgb_gt - vmin) / denom, 0.0, 1.0)
    rgb_rec_disp = np.clip((rgb_rec - vmin) / denom, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=160)
    axes[0].imshow(rgb_gt_disp)
    axes[0].set_title("GT pseudo-RGB (display scaled)")
    axes[0].axis("off")

    psnr_dummy = 0.0
    _ = psnr_dummy
    axes[1].imshow(rgb_rec_disp)
    axes[1].set_title("Recon pseudo-RGB (display scaled)")
    axes[1].axis("off")
    plt.tight_layout()
    if interactive_show:
        plt.show()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    np.save(out_np, {"rgb_gt": rgb_gt, "rgb_rec": rgb_rec})


def run_stage1_roi_orkm(
    *,
    folder: str,
    obj_name_prefix: str,
    device: str,
    out_dir: str,
    max_patches: int,
    signal_mode: str = "raw",
    whiten_eps: float = 1e-10,
    passes: Optional[int] = None,
    interactive_show: bool = True,
    use_cuda_orkm: bool = False,
    patch_batch_size: int = 1,
) -> None:
    cfg = ROIConfig()
    patch_size = cfg.patch_size
    roi_side = cfg.roi_patches_side
    d = patch_size * patch_size
    n = d * cfg.n_over_d
    T = int(cfg.passes if passes is None else passes)

    device_t = torch.device(device)
    dtype = torch.float64

    idx8 = select_8_indices()

    # Load 8 bands.
    bands8_np: List[np.ndarray] = []
    for p in band_paths(folder, obj_name_prefix, idx8):
        bands8_np.append(read_png16_to_float64_01(p))
    S8_full = np.stack(bands8_np, axis=0).astype(np.float64)  # (8,H,W)
    H, W = int(S8_full.shape[1]), int(S8_full.shape[2])
    S8_full_t = torch.as_tensor(S8_full, dtype=dtype, device=device_t)

    gh, gw, base_gr, base_gc = _compute_center_roi_patch_grid(H, W, patch_size, roi_side)
    roi_gr0 = base_gr
    roi_gc0 = base_gc

    roi_pix_y0 = roi_gr0 * patch_size
    roi_pix_x0 = roi_gc0 * patch_size
    roi_pix_y1 = roi_pix_y0 + roi_side * patch_size
    roi_pix_x1 = roi_pix_x0 + roi_side * patch_size

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = (
        f"ps{patch_size}_roi{roi_side}x{roi_side}_sm{signal_mode}_n{n}_d{d}_T{T}_"
        f"Aseed{cfg.A_seed}_seed{cfg.seed_fixed}"
    )

    # Fixed measurement matrix A.
    # Important: set RNG seed ONCE globally to keep ORKM's per-iter randperm random
    # (within each patch we must not reset RNG before each iter).
    torch.manual_seed(cfg.seed_fixed)
    if device_t.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed_fixed)

    torch.manual_seed(cfg.A_seed)
    if device_t.type == "cuda":
        torch.cuda.manual_seed_all(cfg.A_seed)
    A = torch.randn((n, d, 8), dtype=dtype, device=device_t)
    beta = row_energy_batch(A)

    print(f"[Stage1-ROI-ORKM] device={device_t.type}, dtype=float64")
    print(f"[Stage1-ROI-ORKM] image(H,W)=({H},{W}), patch_size={patch_size}, roi_side={roi_side}, ROI pixel=({roi_pix_y0}:{roi_pix_y1},{roi_pix_x0}:{roi_pix_x1})")
    print(f"[Stage1-ROI-ORKM] d={d}, n={n}, n/d={cfg.n_over_d}, T={T}, power_iters={cfg.power_iters}")
    print(f"[Stage1-ROI-ORKM] A fixed with A_seed={cfg.A_seed}")
    print(f"[Stage1-ROI-ORKM] signal_mode={signal_mode}, whiten_eps={whiten_eps:.6e}")
    print(
        f"[Stage1-ROI-ORKM] engine={'cuda_batched' if use_cuda_orkm and patch_batch_size > 1 else ('cuda_patch' if use_cuda_orkm else 'reference')}, "
        f"patch_batch_size={patch_batch_size}"
    )

    # Prepare reconstruction buffers.
    S8_rec = np.zeros_like(S8_full, dtype=np.float64)
    S8_gt_roi = S8_full[:, roi_pix_y0:roi_pix_y1, roi_pix_x0:roi_pix_x1]
    S8_ref_back = np.zeros_like(S8_gt_roi, dtype=np.float64)

    patch_white_dist: List[float] = []
    patch_white_rel: List[float] = []
    patch_white_meas: List[float] = []
    patch_back_dist: List[float] = []
    patch_back_rel: List[float] = []

    patch_jobs: List[Tuple[int, int, int, int, int]] = []
    for pr in range(roi_side):
        for pc in range(roi_side):
            patch_index = pr * roi_side + pc
            gr = roi_gr0 + pr
            gc = roi_gc0 + pc
            y0 = gr * patch_size
            x0p = gc * patch_size
            patch_jobs.append((patch_index, pr, pc, y0, x0p))
            if len(patch_jobs) >= max_patches:
                break
        if len(patch_jobs) >= max_patches:
            break

    effective_bs = int(patch_batch_size) if (use_cuda_orkm and patch_batch_size > 1) else 1
    if effective_bs < 1:
        raise ValueError("patch_batch_size must be >= 1")

    t0 = time.perf_counter()
    patch_counter = 0
    job_cursor = 0
    while job_cursor < len(patch_jobs):
        batch_slice = patch_jobs[job_cursor : job_cursor + effective_bs]
        job_cursor += len(batch_slice)

        if len(batch_slice) == 1:
            patch_index, pr, pc, y0, x0p = batch_slice[0]
            patch8 = S8_full_t[:, y0 : y0 + patch_size, x0p : x0p + patch_size]
            x_patch_raw = _patches8_to_x_true(patch8)
            x_true, aux = prepare_x_true(x_patch_raw, mode=signal_mode, eps=whiten_eps)
            y = intensity_measurements_explicit(A, x_true)
            if use_cuda_orkm:
                x_est, _info = alg_orkm_cuda(
                    A=A,
                    y=y,
                    T=T,
                    seed=None,
                    power_iters=cfg.power_iters,
                    omega=1.0,
                    beta=beta,
                    x_true_proc=None,
                    stop_err=cfg.stop_err,
                    verbose=False,
                    record_orbit_metrics=False,
                    record_meas_rel=False,
                )
            else:
                x_est, _info = alg_orkm(
                    A=A,
                    y=y,
                    T=T,
                    seed=None,
                    power_iters=cfg.power_iters,
                    x_true_proc=x_true,
                    stop_err=cfg.stop_err,
                    verbose=False,
                )
            if x_est.shape != x_true.shape:
                raise RuntimeError(f"Unexpected x_est shape {tuple(x_est.shape)}, expected {tuple(x_true.shape)}")

            q_w = estimate_global_right_phase(x_true, x_est)
            x_est_aligned = apply_global_right_phase(x_est, q_w)

            dist_w = float(right_aligned_distance(x_true, x_est).item())
            rel_l2_w = float((oct_array_norm(x_true - x_est_aligned) / oct_array_norm(x_true)).item())
            log_dist_w = float(np.log(max(dist_w, 1e-300)))
            y_hat_w = intensity_measurements_explicit(A, x_est_aligned)
            meas_rel_w = float((torch.linalg.norm(y_hat_w - y) / torch.linalg.norm(y)).item())

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

            patch_white_dist.append(dist_w)
            patch_white_rel.append(rel_l2_w)
            patch_white_meas.append(meas_rel_w)
            patch_back_dist.append(dist_b)
            patch_back_rel.append(rel_l2_b)

            patch8_rec = _x_true_to_patches8(x_rec_back_aligned, patch_size).detach().cpu().numpy()
            S8_rec[:, y0 : y0 + patch_size, x0p : x0p + patch_size] = patch8_rec

            patch8_ref = _x_true_to_patches8(x_ref_back, patch_size).detach().cpu().numpy()
            S8_ref_back[
                :,
                y0 - roi_pix_y0 : y0 - roi_pix_y0 + patch_size,
                x0p - roi_pix_x0 : x0p - roi_pix_x0 + patch_size,
            ] = patch8_ref

            print(
                f"[Stage1-ROI-ORKM] patch_index={patch_index:02d} (pr={pr},pc={pc})\n"
                f"  white_dist={dist_w:.6e}, white_log_dist={log_dist_w:.6e}, "
                f"white_rel_l2={rel_l2_w:.6e}, white_meas_rel={meas_rel_w:.6e}\n"
                f"  back_dist={dist_b:.6e}, back_log_dist={log_dist_b:.6e}, back_rel_l2={rel_l2_b:.6e}"
            )
            patch_counter += 1
        else:
            patch8_batch = torch.stack(
                [S8_full_t[:, y0 : y0 + patch_size, x0p : x0p + patch_size] for _, _, _, y0, x0p in batch_slice],
                dim=0,
            )
            X_raw = torch.stack([_patches8_to_x_true(patch8_batch[b]) for b in range(len(batch_slice))], dim=0)
            x_true_b, aux_b = prepare_x_true_batched(X_raw, mode=signal_mode, eps=whiten_eps)
            y_b = intensity_measurements_batched(A, x_true_b)
            x_est_b, _ = alg_orkm_cuda_batched(
                A=A,
                y=y_b,
                T=T,
                seed=None,
                power_iters=cfg.power_iters,
                omega=1.0,
                beta=beta,
                verbose=False,
            )
            if x_est_b.shape != x_true_b.shape:
                raise RuntimeError(
                    f"Unexpected batched x_est shape {tuple(x_est_b.shape)}, expected {tuple(x_true_b.shape)}"
                )
            for b, (patch_index, pr, pc, y0, x0p) in enumerate(batch_slice):
                x_true = x_true_b[b]
                x_est = x_est_b[b]
                aux = _aux_for_batch_index(aux_b, b, signal_mode)
                y = y_b[b]
                q_w = estimate_global_right_phase(x_true, x_est)
                x_est_aligned = apply_global_right_phase(x_est, q_w)
                dist_w = float(right_aligned_distance(x_true, x_est).item())
                rel_l2_w = float((oct_array_norm(x_true - x_est_aligned) / oct_array_norm(x_true)).item())
                log_dist_w = float(np.log(max(dist_w, 1e-300)))
                y_hat_w = intensity_measurements_explicit(A, x_est_aligned)
                meas_rel_w = float((torch.linalg.norm(y_hat_w - y) / torch.linalg.norm(y)).item())
                x_rec_for_image = recover_from_whitened(x_est_aligned, signal_mode, aux)
                x_patch_raw = X_raw[b]
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
                patch_white_dist.append(dist_w)
                patch_white_rel.append(rel_l2_w)
                patch_white_meas.append(meas_rel_w)
                patch_back_dist.append(dist_b)
                patch_back_rel.append(rel_l2_b)
                patch8_rec = _x_true_to_patches8(x_rec_back_aligned, patch_size).detach().cpu().numpy()
                S8_rec[:, y0 : y0 + patch_size, x0p : x0p + patch_size] = patch8_rec
                patch8_ref = _x_true_to_patches8(x_ref_back, patch_size).detach().cpu().numpy()
                S8_ref_back[
                    :,
                    y0 - roi_pix_y0 : y0 - roi_pix_y0 + patch_size,
                    x0p - roi_pix_x0 : x0p - roi_pix_x0 + patch_size,
                ] = patch8_ref
                print(
                    f"[Stage1-ROI-ORKM] patch_index={patch_index:02d} (pr={pr},pc={pc})\n"
                    f"  white_dist={dist_w:.6e}, white_log_dist={log_dist_w:.6e}, "
                    f"white_rel_l2={rel_l2_w:.6e}, white_meas_rel={meas_rel_w:.6e}\n"
                    f"  back_dist={dist_b:.6e}, back_log_dist={log_dist_b:.6e}, back_rel_l2={rel_l2_b:.6e}"
                )
            patch_counter += len(batch_slice)

    # ROI metrics/visualization (back-domain reference vs back-domain reconstruction).
    S8_rec_roi = S8_rec[:, roi_pix_y0:roi_pix_y1, roi_pix_x0:roi_pix_x1]
    rgb_gt = pseudo_rgb_from_8bands([S8_ref_back[k] for k in range(8)], idx8)
    rgb_rec = pseudo_rgb_from_8bands([S8_rec_roi[k] for k in range(8)], idx8)
    psnr = float(peak_signal_noise_ratio(rgb_gt, rgb_rec, data_range=1.0))
    ssim = float(structural_similarity(rgb_gt, rgb_rec, data_range=1.0, channel_axis=-1))

    roi_psnr_path = out_root / f"stage1_roi_recon_{stamp}.png"
    roi_np_path = out_root / f"stage1_roi_recon_{stamp}.npy"
    _save_roi_images(rgb_gt, rgb_rec, roi_psnr_path, roi_np_path, interactive_show=interactive_show)

    t1 = time.perf_counter()
    print("[Stage1-ROI-ORKM] done")
    print(f"  wall_time_sec={t1 - t0:.2f}s")
    print(
        f"  white_domain dist: mean={np.mean(patch_white_dist):.6e}, "
        f"median={np.median(patch_white_dist):.6e}, max={np.max(patch_white_dist):.6e}"
    )
    print(
        f"  white_domain meas_rel: mean={np.mean(patch_white_meas):.6e}, "
        f"median={np.median(patch_white_meas):.6e}, max={np.max(patch_white_meas):.6e}"
    )
    print(
        f"  back_domain dist: mean={np.mean(patch_back_dist):.6e}, "
        f"median={np.median(patch_back_dist):.6e}, max={np.max(patch_back_dist):.6e}"
    )
    print(
        f"  back_domain rel_l2: mean={np.mean(patch_back_rel):.6e}, "
        f"median={np.median(patch_back_rel):.6e}, max={np.max(patch_back_rel):.6e}"
    )
    print(f"  ROI pseudo-RGB(back-domain): PSNR={psnr:.3f}, SSIM={ssim:.6f}")
    print(f"  saved: {roi_psnr_path}")
    print(f"  saved npy: {roi_np_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 real image ROI reconstruction (data-consistent ORKM).")
    p.add_argument(
        "--folder",
        type=str,
        default=os.path.join(str(PROJECT_ROOT), "dataset", "complete_ms_data", "balloons_ms", "balloons_ms"),
    )
    p.add_argument("--obj-name-prefix", type=str, default="balloons")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "output" / "real_balloons_stage1_roi_orkm"))
    p.add_argument(
        "--max-patches",
        type=int,
        default=16,
        help="For smoke tests only. Default 16 means full 4x4 ROI.",
    )
    p.add_argument(
        "--signal-mode",
        type=str,
        default="whitened",
        choices=["raw", "normalized", "whitened"],
        help="per-patch preprocessing before ORKM (same as main_5). Default raw matches legacy main_4.",
    )
    p.add_argument("--whiten-eps", type=float, default=1e-10, help="eigenvalue floor for whitening.")
    p.add_argument(
        "--passes",
        type=int,
        default=None,
        help="override ORKM passes T (default: ROIConfig.passes).",
    )
    p.add_argument(
        "--no-interactive-show",
        action="store_true",
        help="do not call plt.show() when saving ROI figure.",
    )
    p.add_argument(
        "--use-cuda-orkm",
        action="store_true",
        help="use alg_orkm_cuda (cached beta, no per-epoch orbit metrics) for ORKM iterations.",
    )
    p.add_argument(
        "--patch-batch-size",
        type=int,
        default=16,
        help="when >1 with --use-cuda-orkm, run batched patches per ORKM solve (shared A, shared row perm).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage1_roi_orkm(
        folder=args.folder,
        obj_name_prefix=args.obj_name_prefix,
        device=args.device,
        out_dir=args.out_dir,
        max_patches=args.max_patches,
        signal_mode=str(args.signal_mode),
        whiten_eps=float(args.whiten_eps),
        passes=args.passes,
        interactive_show=not args.no_interactive_show,
        use_cuda_orkm=bool(args.use_cuda_orkm),
        patch_batch_size=int(args.patch_batch_size),
    )

