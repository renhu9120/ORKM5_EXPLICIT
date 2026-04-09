"""
Full-image real-data stage-2 reconstruction — **ORKM baseline, engineering-equivalent acceleration**.

This script is **not** a new ORKM algorithm. It runs the same ORKM baseline as
``real_image_exp_orkm.py`` (single-measurement uniform sampling, same block projection)
using:

1. **Cached projectors** ``P_i = B_i^T (B_i B_i^T)^{-1}`` built once from ``gamma(A)``.
2. **Patch-level parallel** execution: independent trajectories per patch, batched tensor ops.

Gauge-fix, beta, inverse preprocess, and metrics are unchanged from the baseline stage-2
protocol. See ``docs/baselines/ORKM.md`` (Engineering-equivalent acceleration).

Primary control: ``passes``; internal ``T = round(passes * n)``.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


from utils.utils_img import (
    band_paths,
    pseudo_rgb_from_8bands,
    read_png16_to_float64_01,
    select_8_indices,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)



@dataclass
class Config:
    folder: str
    obj_name_prefix: str = "balloons"
    patch_size: int = 8
    n_over_d: float = 12.0

    # ORKM real-image primary control: passes through the n measurements (see logs for derived T).
    passes: float = 1.0
    max_power_iter: int = 8
    A_scale_div8: bool = True
    seed_exp: int = 122

    device: str = "cuda"
    batch_size: int = 8

    show_plot: bool = True
    save_dir: str = "output/real_balloons_stage2_orkm_v2"
    save_per_patch_metrics: bool = False


def run_stage2_orkm_v2(cfg: Config) -> dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if cfg.device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[warn] CUDA not available, fallback to cpu.")
        cfg.device = "cpu"

    t_total_start = time.time()
    device = torch.device(cfg.device)
    dtype = torch.float64

    idx8 = select_8_indices()

    t_load_start = time.time()
    bands8: List[np.ndarray] = []
    for p in band_paths(cfg.folder, cfg.obj_name_prefix, idx8):
        bands8.append(read_png16_to_float64_01(p))

    S8_full = np.stack(bands8, axis=0).astype(np.float64)
    t_load_end = time.time()
    H, W = S8_full.shape[1], S8_full.shape[2]

    if H % cfg.patch_size != 0 or W % cfg.patch_size != 0:
        raise ValueError(f"H,W must be divisible by patch_size={cfg.patch_size}, got ({H},{W})")

    gh, gw = H // cfg.patch_size, W // cfg.patch_size
    P = gh * gw
    d = cfg.patch_size * cfg.patch_size

    n = int(np.floor(cfg.n_over_d * d))
    if n <= 0:
        raise ValueError(f"invalid n={n} from n_over_d={cfg.n_over_d}, d={d}")

    passes = float(cfg.passes)
    T = compute_total_steps_from_passes(passes, n)

    rgb_full = pseudo_rgb_from_8bands([S8_full[k] for k in range(8)], idx8)

    torch.manual_seed(cfg.seed_exp)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed_exp)

    A = torch.randn((n, d, 8), device=device, dtype=dtype)
    if cfg.A_scale_div8:
        A = A / 8.0

    t_pre_start = time.time()
    GA = gamma_of_matrix_rows(A)
    t_proj_start = time.time()
    orkm_cache = precompute_orkm_projectors(GA)
    t_proj_end = time.time()
    t_pre_end = time.time()

    S8_recon = np.zeros_like(S8_full, dtype=np.float64)

    print(f"[Stage2-ORKM-v2] (H,W)=({H},{W}), patch={cfg.patch_size}, gh={gh}, gw={gw}, P={P}")
    print(f"[Stage2-ORKM-v2] d={d}, n={n}, n/d={cfg.n_over_d}")
    print(f"[Stage2-ORKM-v2] passes={passes}")
    print(f"[Stage2-ORKM-v2] derived total iteration steps T = round(passes * n) = {T}")
    print(f"[Stage2-ORKM-v2] max_power_iter={cfg.max_power_iter}, device={device}, batch_size={cfg.batch_size}")
    print(
        "[Stage2-ORKM-v2] Baseline ORKM with cached projectors + patch-parallel (engineering only; not a new algorithm)."
    )
    print(
        f"[Stage2-ORKM-v2] orkm_projector_precompute_sec = {t_proj_end - t_proj_start:.3f} "
        f"(one-time, shared across all patches)"
    )
    print(
        "[Stage2-ORKM-v2] Raw-domain preview/metrics use truth-assisted gauge fixing (not blind reconstruction). "
        "See docs/baselines/ORKM_realdata_protocol.md."
    )

    batch_starts = list(range(0, P, cfg.batch_size))
    t_batches_start = time.time()
    all_beta: List[np.ndarray] = []
    all_orbit: List[np.ndarray] = []
    all_proc_before: List[np.ndarray] = []
    all_proc_after: List[np.ndarray] = []
    all_skip_count: List[np.ndarray] = []
    all_skip_ratio: List[np.ndarray] = []

    for bi, start in enumerate(batch_starts, start=1):
        end = min(P, start + cfg.batch_size)
        B = end - start
        patches8 = np.zeros((B, 8, cfg.patch_size, cfg.patch_size), dtype=np.float64)
        pids = []
        for local_i, pid in enumerate(range(start, end)):
            r = pid // gw
            c = pid % gw
            patches8[local_i] = S8_full[:, r * cfg.patch_size: (r + 1) * cfg.patch_size,
            c * cfg.patch_size: (c + 1) * cfg.patch_size]
            pids.append(pid)

        x_raw_batch = patch_to_x_batch(patches8).to(device=device, dtype=dtype)
        x_proc_batch, meta_batch = preprocess_patch_B_batch(x_raw_batch)
        y_batch = synth_y_batch_from_GA(GA, x_proc_batch)

        seeds = torch.tensor(
            [cfg.seed_exp + int(pid) for pid in pids],
            device=device,
            dtype=torch.long,
        )
        zT_batch, hist_pb = alg_orkm_parallel_batch(
            A,
            y_batch,
            T,
            seeds,
            max_power_iter=cfg.max_power_iter,
            eps_skip=1e-12,
            orkm_cache=orkm_cache,
        )
        skip_c = hist_pb["skip_count"].numpy().astype(np.int64)
        skip_r = hist_pb["skip_ratio"].numpy().astype(np.float64)
        x_rec_proc_batch = zT_batch.reshape(B, d, 1, 8).contiguous()
        x_true_proc_batch = x_proc_batch

        z_batch, orbit_dist_batch = estimate_optimal_right_phase_batch(
            x_rec_proc_batch, x_true_proc_batch
        )
        x_rec_fixed_proc_batch = gauge_fix_estimate_by_true_phase_batch(x_rec_proc_batch, z_batch)

        rel_bef, rel_aft = proc_rel_l2_before_after_batch(
            x_rec_proc_batch, x_true_proc_batch, x_rec_fixed_proc_batch
        )
        all_orbit.append(orbit_dist_batch.detach().cpu().numpy())
        all_proc_before.append(rel_bef.detach().cpu().numpy())
        all_proc_after.append(rel_aft.detach().cpu().numpy())
        all_skip_count.append(skip_c)
        all_skip_ratio.append(skip_r)

        beta_batch = estimate_beta_batch(GA, y_batch, x_rec_fixed_proc_batch)
        all_beta.append(beta_batch.detach().cpu().numpy())

        x_rec_raw_batch = inverse_preprocess_patch_B_batch(x_rec_fixed_proc_batch, meta_batch, scale=beta_batch)

        x_flat = x_rec_raw_batch.reshape(B, d, 8)
        patch_rec8 = x_flat.reshape(B, cfg.patch_size, cfg.patch_size, 8).permute(0, 3, 1, 2).contiguous()
        patch_rec8_cpu = patch_rec8.detach().cpu().numpy()
        for local_i, pid in enumerate(pids):
            r = pid // gw
            c = pid % gw
            S8_recon[:, r * cfg.patch_size: (r + 1) * cfg.patch_size,
            c * cfg.patch_size: (c + 1) * cfg.patch_size] = patch_rec8_cpu[local_i]

        if bi % 10 == 0 or bi == len(batch_starts):
            print(f"[Stage2-ORKM-v2] batch {bi}/{len(batch_starts)} done. processed patches {end}/{P}")

    t_batches_end = time.time()

    beta_all = np.concatenate(all_beta, axis=0)
    orbit_all = np.concatenate(all_orbit, axis=0)
    proc_before_all = np.concatenate(all_proc_before, axis=0)
    proc_after_all = np.concatenate(all_proc_after, axis=0)
    skip_count_all = np.concatenate(all_skip_count, axis=0)
    skip_ratio_all = np.concatenate(all_skip_ratio, axis=0)
    beta_stats = summarize_beta(beta_all)

    total_skip_count = int(np.sum(skip_count_all))
    mean_skip_ratio = float(np.mean(skip_ratio_all))
    max_skip_ratio = float(np.max(skip_ratio_all))

    def _tri(arr: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    orbit_summ = _tri(orbit_all)
    proc_after_summ = _tri(proc_after_all)

    rgb_recon = pseudo_rgb_from_8bands([S8_recon[k] for k in range(8)], idx8)
    psnr = float(peak_signal_noise_ratio(rgb_full, rgb_recon, data_range=1.0))
    ssim = float(structural_similarity(rgb_full, rgb_recon, data_range=1.0, channel_axis=-1))
    raw8 = raw_eightband_cube_metrics(S8_recon, S8_full, data_range=1.0)

    print("\n[Stage2-ORKM-v2][summary: engineering / stage2 protocol — not paper blind recon]")
    print(f"  passes={passes}")
    print(f"  n={n}")
    print(f"  T={T}")
    print(
        f"  beta: mean={beta_stats['mean']:.6e} std={beta_stats['std']:.6e} "
        f"min={beta_stats['min']:.6e} max={beta_stats['max']:.6e}"
    )
    print(
        f"  ORKM skips: total_skip_count={total_skip_count}, "
        f"mean_skip_ratio_per_patch={mean_skip_ratio:.6e}, max_skip_ratio_per_patch={max_skip_ratio:.6e}"
    )
    print(
        f"  orbit_dist: mean={orbit_summ['mean']:.6e} median={orbit_summ['median']:.6e} max={orbit_summ['max']:.6e}"
    )
    print(
        "  proc_rel_l2 after gauge-fix: "
        f"mean={proc_after_summ['mean']:.6e} median={proc_after_summ['median']:.6e} "
        f"max={proc_after_summ['max']:.6e}"
    )
    print(
        "  proc_rel_l2 before gauge-fix: "
        f"mean={float(np.mean(proc_before_all)):.6e} median={float(np.median(proc_before_all)):.6e} "
        f"max={float(np.max(proc_before_all)):.6e}"
    )

    print("\n[Stage2-ORKM-v2][raw 8-band cube: truth-assisted gauge-fixed, data_range=1.0]")
    print(
        f"  rel_l2={raw8['raw_rel_l2']:.6e} MSE={raw8['raw_mse']:.6e} PSNR={raw8['raw_psnr']:.4f}"
    )
    for k in range(8):
        print(
            f"  band {k}: rel_l2={raw8['per_band_rel_l2'][k]:.6e} "
            f"MSE={raw8['per_band_mse'][k]:.6e} PSNR={raw8['per_band_psnr'][k]:.4f}"
        )

    print("\n[Stage2-ORKM-v2][pseudo-RGB preview: auxiliary display metric, data_range=1.0]")
    print(f"  PSNR={psnr:.4f}, SSIM={ssim:.6f}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    passes_slug = f"{passes:g}".replace(".", "p")
    tag = f"H{H}_W{W}_ps{cfg.patch_size}_passes{passes_slug}_T{T}_v2eng"
    recon_rgb_path = os.path.join(cfg.save_dir, f"orkm_v2_balloons_recon_rgb_{tag}.png")
    npy_path = os.path.join(cfg.save_dir, f"orkm_v2_balloons_S8_recon_{tag}.npy")
    metrics_json_path = os.path.join(cfg.save_dir, f"orkm_v2_balloons_metrics_{tag}.json")
    perband_csv_path = os.path.join(cfg.save_dir, f"orkm_v2_balloons_perband_metrics_{tag}.csv")
    full_npz_path = os.path.join(cfg.save_dir, f"orkm_v2_balloons_stage2_full_{tag}.npz")

    vmin = float(min(rgb_full.min(), rgb_recon.min()))
    vmax = float(max(rgb_full.max(), rgb_recon.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].imshow(np.clip((rgb_full - vmin) / (vmax - vmin + 1e-30), 0, 1))
    axes[0].set_title("GT pseudo-RGB (display scaled)")
    axes[0].axis("off")
    axes[1].imshow(np.clip((rgb_recon - vmin) / (vmax - vmin + 1e-30), 0, 1))
    axes[1].set_title(f"Recon pseudo-RGB (ORKM baseline, v2 eng.)\nPSNR={psnr:.2f}, SSIM={ssim:.4f}")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(recon_rgb_path, bbox_inches="tight")
    print(f"[Stage2-ORKM-v2] saved preview to: {recon_rgb_path}")

    np.save(npy_path, S8_recon)
    print(f"[Stage2-ORKM-v2] saved S8_recon to: {npy_path}")

    if cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)

    t_total_end = time.time()

    print("[Stage2-ORKM-v2][Timing breakdown]")
    print(f"  load_full_cube_sec     = {t_load_end - t_load_start:.3f}")
    print(f"  precompute_operators   = {t_pre_end - t_pre_start:.3f}")
    print(f"    (includes gamma + projector cache; projector_sec = {t_proj_end - t_proj_start:.3f})")
    print(f"  orkm_reconstruct_total = {t_batches_end - t_batches_start:.3f}")
    print(f"  postprocess_and_save   = {t_total_end - t_batches_end:.3f}")
    print(f"  total_wall_time_sec    = {t_total_end - t_total_start:.3f}")

    summary = {
        "algorithm": "ORKM_amplitude_baseline_engineering_cached_parallel",
        "implementation_note": (
            "Same ORKM baseline math as real_image_exp_orkm.py; cached projectors + patch-level parallel execution only."
        ),
        "orkm_projector_precompute_sec": float(t_proj_end - t_proj_start),
        "primary_iteration_unit": "passes",
        "passes": float(passes),
        "n": int(n),
        "T": int(T),
        "d": int(d),
        "n_over_d": float(cfg.n_over_d),
        "psnr_pseudo_rgb": float(psnr),
        "ssim_pseudo_rgb": float(ssim),
        "recon_rgb_path": recon_rgb_path,
        "recon_cube_path": npy_path,
        "total_wall_time_sec": float(t_total_end - t_total_start),
        "num_patches": int(P),
        "truth_assisted_gauge_fix_disclaimer": (
            "Raw-domain preview/metrics use truth-assisted gauge fixing (not blind reconstruction)."
        ),
        "beta_stats": beta_stats,
        "orkm_skip": {
            "total_skip_count": total_skip_count,
            "mean_skip_ratio_per_patch": mean_skip_ratio,
            "max_skip_ratio_per_patch": max_skip_ratio,
        },
        "A_orbit_distance_proc": orbit_summ,
        "B_proc_rel_l2_gauge_fixed": proc_after_summ,
        "B_proc_rel_l2_before_gauge_fix": _tri(proc_before_all),
        "C_raw_eightband": {
            "data_range": 1.0,
            "rel_l2": raw8["raw_rel_l2"],
            "mse": raw8["raw_mse"],
            "psnr": raw8["raw_psnr"],
            "per_band_rel_l2": raw8["per_band_rel_l2"],
            "per_band_mse": raw8["per_band_mse"],
            "per_band_psnr": raw8["per_band_psnr"],
        },
        "D_pseudo_rgb": {"psnr": float(psnr), "ssim": float(ssim)},
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({"config": {**cfg.__dict__, "dtype": "torch.float64"}, "summary": summary}, f, ensure_ascii=False,
                  indent=2)
    print(f"[Stage2-ORKM-v2] saved metrics JSON to: {metrics_json_path}")

    with open(perband_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band", "rel_l2", "mse", "psnr"])
        for k in range(8):
            w.writerow(
                [
                    k,
                    raw8["per_band_rel_l2"][k],
                    raw8["per_band_mse"][k],
                    raw8["per_band_psnr"][k],
                ]
            )
    print(f"[Stage2-ORKM-v2] saved per-band CSV to: {perband_csv_path}")

    np.savez(
        full_npz_path,
        orbit_dist=orbit_all,
        proc_rel_l2_before=proc_before_all,
        proc_rel_l2_after=proc_after_all,
        beta=beta_all,
        orkm_skip_count=skip_count_all,
        orkm_skip_ratio=skip_ratio_all,
    )
    print(f"[Stage2-ORKM-v2] saved full per-patch arrays to: {full_npz_path}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg.save_dir, f"real_full_orkm_v2_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    if cfg.save_per_patch_metrics:
        per_patch_path = os.path.join(run_dir, "per_patch_metrics.npz")
        np.savez(
            per_patch_path,
            patch_id=np.arange(P, dtype=np.int32),
            beta=beta_all,
            orbit_dist=orbit_all,
            proc_rel_l2_before=proc_before_all,
            proc_rel_l2_after=proc_after_all,
            orkm_skip_count=skip_count_all,
            orkm_skip_ratio=skip_ratio_all,
        )
        summary["per_patch_metrics_npz"] = per_patch_path
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {**cfg.__dict__, "dtype": "torch.float64"},
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary["metrics_json_path"] = metrics_json_path
    summary["perband_csv_path"] = perband_csv_path
    summary["full_npz_path"] = full_npz_path
    summary["run_meta_json"] = os.path.join(run_dir, "meta.json")
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-image full ORKM stage-2 v2: baseline + cached projectors + patch-parallel (engineering)."
    )
    p.add_argument(
        "--folder",
        type=str,
        default=os.path.join(ROOT, "dataset", "complete_ms_data", "balloons_ms", "balloons_ms"),
        help="Directory containing balloons_ms PNG bands.",
    )
    p.add_argument("--obj-name-prefix", type=str, default="balloons")
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--n-over-d", type=float, default=12.0, dest="n_over_d")
    p.add_argument(
        "--passes",
        type=float,
        nargs="+",
        default=[1.0],
        help="Primary ORKM budget: expected scans in units of n (T = round(passes*n) internally). "
             "Pass multiple values to sweep, e.g. --passes 1 5 10.",
    )
    p.add_argument("--max-power-iter", type=int, default=8, dest="max_power_iter")
    p.add_argument("--seed", type=int, default=11, dest="seed_exp")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--save-dir", type=str, default="output/real_balloons_stage2_orkm_v2")
    p.add_argument("--no-plot", action="store_true", help="Do not call plt.show()")
    p.add_argument("--save-per-patch", action="store_true", dest="save_per_patch_metrics")
    return p


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    summaries: List[dict] = []
    for pval in args.passes:
        cfg = Config(
            folder=args.folder,
            obj_name_prefix=args.obj_name_prefix,
            patch_size=args.patch_size,
            n_over_d=args.n_over_d,
            passes=float(pval),
            max_power_iter=args.max_power_iter,
            seed_exp=args.seed_exp,
            device=args.device,
            batch_size=args.batch_size,
            show_plot=not args.no_plot,
            save_dir=args.save_dir,
            save_per_patch_metrics=args.save_per_patch_metrics,
        )
        summaries.append(run_stage2_orkm_v2(cfg))
    if len(summaries) > 1:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
