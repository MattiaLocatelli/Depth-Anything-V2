#!/usr/bin/env python3
import argparse
import csv
import os
from typing import List, Tuple

import cv2
import numpy as np


def parse_coord(s: str) -> Tuple[int, int]:
    s = s.strip()
    if ',' in s:
        u, v = s.split(',')
    else:
        parts = s.split()
        if len(parts) == 2:
            u, v = parts
        else:
            raise argparse.ArgumentTypeError(f'Invalid coord format: {s}')
    return int(float(u)), int(float(v))


def load_depth(path: str, max_depth: float = None) -> np.ndarray:
    if path.endswith('.npy'):
        d = np.load(path).astype(np.float32)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'Cannot read depth file: {path}')
        img = img.astype(np.float32)
        if img.max() > 1 and max_depth is None:
            raise RuntimeError('For image depth inputs, please pass --max-depth to scale values to meters')
        if max_depth is not None:
            if img.max() <= 1.0:
                d = img * max_depth
            else:
                d = (img / 255.0) * max_depth
        else:
            d = img
    return d


def backproject(u: int, v: int, z: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def main():
    p = argparse.ArgumentParser(description='Report depth (meters) for one or more pixel coordinates')
    p.add_argument('--depth-file', required=True, help='path to _raw_depth_meter.npy (preferred) or depth image')
    p.add_argument('--coord', dest='coords', action='append', type=parse_coord,
                   help='pixel coord as "u,v" or "u v"; can be repeated')
    p.add_argument('--coord-file', help='text file with coords per line: u,v or u v')
    p.add_argument('--backproject', action='store_true', help='also compute 3D camera coordinates (requires intrinsics)')
    p.add_argument('--intrinsics', nargs=4, type=float, metavar=('fx','fy','cx','cy'),
                   help='camera intrinsics for backprojection')
    p.add_argument('--max-depth', type=float, default=None, help='when depth-file is an image: scale to meters with this max depth')
    p.add_argument('--output-csv', help='optional output CSV path to save results')
    args = p.parse_args()

    coords: List[Tuple[int,int]] = []
    if args.coords:
        coords.extend(args.coords)
    if args.coord_file:
        if not os.path.exists(args.coord_file):
            raise RuntimeError('coord-file not found')
        with open(args.coord_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                coords.append(parse_coord(line))

    if len(coords) == 0:
        raise RuntimeError('No coordinates provided; use --coord or --coord-file')

    depth = load_depth(args.depth_file, max_depth=args.max_depth)

    H, W = depth.shape[:2]

    intr = None
    if args.backproject:
        if args.intrinsics is None:
            raise RuntimeError('Backprojection requested but --intrinsics not provided')
        fx, fy, cx, cy = args.intrinsics
        intr = (fx, fy, cx, cy)

    rows = []
    for (u, v) in coords:
        if not (0 <= v < H and 0 <= u < W):
            print(f'coord ({u},{v}) out of bounds for depth size {W}x{H}; skipping')
            continue
        z = float(depth[v, u])
        if z == 0 or not np.isfinite(z):
            print(f'coord ({u},{v}): invalid depth {z}')
            rows.append({'u':u,'v':v,'depth_m':float('nan')})
            continue
        if intr is None:
            print(f'coord ({u},{v}): depth = {z:.4f} m')
            rows.append({'u':u,'v':v,'depth_m':z})
        else:
            x,y,z3 = backproject(u, v, z, *intr)
            print(f'coord ({u},{v}): depth = {z:.4f} m -> point_cam = ({x:.4f}, {y:.4f}, {z3:.4f}) m')
            rows.append({'u':u,'v':v,'depth_m':z,'x_m':x,'y_m':y,'z_m':z3})

    if args.output_csv:
        keys = list(rows[0].keys()) if rows else ['u','v','depth_m']
        with open(args.output_csv, 'w', newline='') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print('Saved results to', args.output_csv)


if __name__ == '__main__':
    main()
