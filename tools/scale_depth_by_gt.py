#!/usr/bin/env python3
"""
Scale a predicted depth map using sparse ground-truth correspondences (CSV).

Input:
 - `--pred` path to a .npy depth map (or directory containing `_raw_depth_meter.npy`).
 - `--pairs-file` CSV with rows `u,v,gt` (pixel coordinates u,v and ground-truth depth in meters).

Supported mapping methods: `median`, `lsq`, `affine`, `poly`, `reciprocal`,
`reciprocal_offset`, `binning`.

Example:
    python tools/scale_depth_by_gt.py --pred results_raw/first_billboard1_raw_depth_meter.npy \
        --pairs-file pairs.csv --method affine --out results_raw/first_billboard1_scaled.npy
"""

import argparse
import csv
import os
from typing import List, Tuple

import numpy as np


def load_pairs_from_csv(path: str) -> List[Tuple[int, int, float]]:
    pairs = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            if len(row) < 3:
                continue
            u = int(float(row[0])); v = int(float(row[1])); gt = float(row[2])
            pairs.append((u, v, gt))
    return pairs


def gather_pairs(args) -> List[Tuple[int, int, float]]:
    # Only CSV input is supported now
    if not args.pairs_file:
        raise RuntimeError('Please provide correspondences via --pairs-file')
    return load_pairs_from_csv(args.pairs_file)


def compute_scale(pred: np.ndarray, pairs: List[Tuple[int,int,float]], method: str = 'median', degree: int = 2, bins: int = 10):
    """
    Fit mapping from predicted values to ground-truth using several methods.

    Returns the transformed depth map (same shape as pred) where original zeros are preserved.
    """
    H, W = pred.shape[:2]
    xs = []
    ys = []
    for (u, v, gt) in pairs:
        if not (0 <= v < H and 0 <= u < W):
            continue
        p = float(pred[v, u])
        if p <= 0 or not np.isfinite(p) or gt <= 0 or not np.isfinite(gt):
            continue
        xs.append(p)
        ys.append(gt)
    if len(xs) == 0:
        raise RuntimeError('No valid correspondences (nonzero pred & gt)')
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    # helper to preserve zeros
    mask = (pred > 0) & np.isfinite(pred)

    if method == 'median':
        s = float(np.median(ys / xs))
        out = np.zeros_like(pred, dtype=np.float32)
        out[mask] = pred[mask] * s
        return out

    if method == 'lsq':
        num = float((xs * ys).sum())
        den = float((xs * xs).sum())
        if den == 0:
            raise RuntimeError('Denominator zero in LSQ')
        s = num / den
        out = np.zeros_like(pred, dtype=np.float32)
        out[mask] = pred[mask] * s
        return out

    if method == 'affine':
        # fit gt = a*p + b
        A = np.vstack([xs, np.ones_like(xs)]).T
        a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        out = np.zeros_like(pred, dtype=np.float32)
        out[mask] = (a * pred[mask] + b).astype(np.float32)
        return out

    if method == 'poly':
        coeffs = np.polyfit(xs, ys, deg=degree)
        out = np.zeros_like(pred, dtype=np.float32)
        if np.any(mask):
            vals = np.polyval(coeffs, pred[mask])
            out[mask] = vals.astype(np.float32)
        return out

    if method == 'reciprocal':
        # k / pred
        k = float(np.median(ys * xs))
        out = np.zeros_like(pred, dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[mask] = (k / pred[mask]).astype(np.float32)
        return out

    if method == 'reciprocal_offset':
        # fit gt = a + b*(1/p)  -> linear in [1, 1/p]
        invx = 1.0 / xs
        A = np.vstack([np.ones_like(invx), invx]).T
        a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        out = np.zeros_like(pred, dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[mask] = (a + b / pred[mask]).astype(np.float32)
        return out

    if method == 'binning':
        # piecewise median mapping: compute median gt for bins of pred
        mins = float(np.min(xs))
        maxs = float(np.max(xs))
        if mins == maxs:
            out = np.zeros_like(pred, dtype=np.float32)
            out[mask] = pred[mask] * float(np.median(ys / xs))
            return out
        bins_edges = np.linspace(mins, maxs, bins + 1)
        bin_idx = np.digitize(xs, bins_edges) - 1
        centers = []
        medians = []
        for i in range(bins):
            sel = bin_idx == i
            if not np.any(sel):
                continue
            centers.append(np.median(xs[sel]))
            medians.append(np.median(ys[sel]))
        if len(centers) < 2:
            # fallback to median scaling
            s = float(np.median(ys / xs))
            out = np.zeros_like(pred, dtype=np.float32)
            out[mask] = pred[mask] * s
            return out
        centers = np.array(centers)
        medians = np.array(medians)
        # interpolate per-pixel
        flat = pred.flatten()
        mapped = np.interp(flat, centers, medians, left=medians[0], right=medians[-1])
        out = mapped.reshape(pred.shape).astype(np.float32)
        out[~mask] = 0.0
        return out

    raise RuntimeError('Unknown method')


def main():
    p = argparse.ArgumentParser(description='Scale predicted depth map using sparse GT')
    p.add_argument('--pred', required=True, help='predicted depth .npy file (or directory containing _raw_depth_meter.npy)')
    p.add_argument('--out', help='output path for scaled .npy')
    p.add_argument('--method', choices=['median','lsq','affine','poly','reciprocal','reciprocal_offset','binning'], default='median')
    p.add_argument('--degree', type=int, default=2, help='degree for poly method')
    p.add_argument('--bins', type=int, default=10, help='number of bins for binning method')
    p.add_argument('--invert', choices=['none','flip','reciprocal','auto'], default='none',
                   help='handle inverse-style predictions before fitting: "flip" = max-pred, "reciprocal" = k/pred, "auto" tries to detect')
    p.add_argument('--pairs-file', required=True, help='CSV file with rows: u,v,gt')
    args = p.parse_args()

    pred_path = args.pred
    if os.path.isdir(pred_path):
        # try to find *_raw_depth_meter.npy
        files = [f for f in os.listdir(pred_path) if f.endswith('_raw_depth_meter.npy')]
        if not files:
            raise RuntimeError('No _raw_depth_meter.npy found in directory')
        pred_path = os.path.join(pred_path, files[0])

    pred = np.load(pred_path).astype(np.float32)

    # optionally handle inverse-style predictions before scaling
    def handle_inversion(pred_map: np.ndarray, pairs, mode: str):
        if mode == 'none':
            return pred_map, None
        if mode == 'flip':
            # flip linearly around max value: pred' = max - pred
            mx = float(np.nanmax(pred_map))
            mapped = (mx - pred_map)
            # preserve original zeros as zeros
            mapped = np.where(pred_map == 0, 0.0, mapped)
            return mapped, None
        if mode == 'reciprocal':
            # Reciprocal model: pred = k * (1/Z) -> Z = k / pred
            # Estimate k from correspondences: k_i = gt_i * pred_i
            vals = []
            H, W = pred_map.shape[:2]
            for (u, v, gt) in pairs:
                if not (0 <= v < H and 0 <= u < W):
                    continue
                p = float(pred_map[v, u])
                if p <= 0 or not np.isfinite(p) or gt <= 0 or not np.isfinite(gt):
                    continue
                vals.append(gt * p)
            if len(vals) == 0:
                raise RuntimeError('No valid correspondences to estimate reciprocal factor')
            k = float(np.median(vals))
            # return transformed depth (metric)
            with np.errstate(divide='ignore', invalid='ignore'):
                zmap = np.where(pred_map > 0, k / pred_map, 0.0).astype(np.float32)
            return zmap, k
        if mode == 'auto':
            # Try to detect if reciprocal fits: check if gt * pred is roughly constant
            H, W = pred_map.shape[:2]
            vals = []
            ratios = []
            for (u, v, gt) in pairs:
                if not (0 <= v < H and 0 <= u < W):
                    continue
                p = float(pred_map[v, u])
                if p <= 0 or not np.isfinite(p) or gt <= 0 or not np.isfinite(gt):
                    continue
                vals.append(gt * p)
                ratios.append(gt / p)
            if len(vals) >= 3:
                vals = np.array(vals)
                rel_std = float(np.std(vals) / (np.mean(vals) + 1e-12))
                if rel_std < 0.2:
                    # reciprocal model seems consistent
                    k = float(np.median(vals))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        zmap = np.where(pred_map > 0, k / pred_map, 0.0).astype(np.float32)
                    return zmap, k
            # fallback: flip sign (linear inversion) — flip around max
            mx = float(np.nanmax(pred_map))
            return (mx - pred_map), None
        raise RuntimeError('Unknown invert mode')

    pairs = gather_pairs(args)

    # Preprocess inversion if needed (uses the provided correspondences)
    if args.invert != 'none':
        pred_proc, aux = handle_inversion(pred, pairs, args.invert)
        if args.invert == 'reciprocal' or (args.invert == 'auto' and aux is not None):
            # pred_proc is already metric depth (Z = k / pred), save and exit
            pred_scaled = pred_proc
            print(f'Applied reciprocal inversion with factor k = {aux:.6f}')
            out_path = args.out if args.out else os.path.splitext(pred_path)[0] + '_scaled.npy'
            np.save(out_path, pred_scaled)
            print('Saved scaled depth to', out_path)
            return
        else:
            pred = pred_proc

    # compute the mapped depth map using selected method
    mapped = compute_scale(pred, pairs, method=args.method, degree=args.degree, bins=args.bins)

    out_path = args.out if args.out else os.path.splitext(pred_path)[0] + '_scaled.npy'
    np.save(out_path, mapped)
    print('Saved mapped depth to', out_path)

    # print some statistics
    print('Before: min {:.4f} median {:.4f} max {:.4f}'.format(np.nanmin(pred[pred>0]) if np.any(pred>0) else np.nan,
                                                               np.nanmedian(pred[pred>0]) if np.any(pred>0) else np.nan,
                                                               np.nanmax(pred)))
    print('After : min {:.4f} median {:.4f} max {:.4f}'.format(np.nanmin(mapped[mapped>0]) if np.any(mapped>0) else np.nan,
                                                               np.nanmedian(mapped[mapped>0]) if np.any(mapped>0) else np.nan,
                                                               np.nanmax(mapped)))


if __name__ == '__main__':
    main()
