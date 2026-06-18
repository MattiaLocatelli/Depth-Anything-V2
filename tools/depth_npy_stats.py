#!/usr/bin/env python3
"""Quick CLI to inspect .npy depth maps and print statistics."""
import argparse
import glob
import json
import os
import csv
import sys
import numpy as np

DEFAULT_PERCENTILES = [1, 5, 25, 50, 75, 95, 99]

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except Exception:
    plt = None
    cm = None


def create_colormap_image(arr, cmap_name='viridis', vmin=None, vmax=None, nan_color=(0, 0, 0)):
    """Convert a 2D depth array to an HxWx3 uint8 RGB image using a colormap.

    - `arr` may contain NaNs; NaNs will be painted with `nan_color`.
    - `vmin`/`vmax`: explicit scaling range (floats) or None to use finite min/max.
    """
    if cm is None:
        raise RuntimeError('matplotlib is required for visualization (install matplotlib)')
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError('Expected 2D array for depth visualization')
    finite_mask = np.isfinite(arr)
    h, w = arr.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if not finite_mask.any():
        img[:] = np.array(nan_color, dtype=np.uint8)
        return img
    vals = arr[finite_mask]
    if vmin is None:
        vmin = float(np.min(vals))
    if vmax is None:
        vmax = float(np.max(vals))
    norm = np.zeros_like(arr, dtype=np.float32)
    if vmax == vmin:
        norm[finite_mask] = 1.0
    else:
        norm[finite_mask] = (arr[finite_mask] - vmin) / (vmax - vmin)
        norm[finite_mask] = np.clip(norm[finite_mask], 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)  # H x W x 4 floats in [0,1]
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    # Paint NaNs
    nan_mask = ~finite_mask
    if nan_mask.any():
        rgb[nan_mask] = np.array(nan_color, dtype=np.uint8)
    return rgb


def save_colormap_image(img, out_path):
    """Save an HxWx3 uint8 RGB image to `out_path` using matplotlib or PIL fallback."""
    if plt is not None:
        try:
            plt.imsave(out_path, img)
            return
        except Exception:
            pass
    try:
        from PIL import Image

        Image.fromarray(img).save(out_path)
        return
    except Exception:
        pass
    raise RuntimeError('Unable to save image: install matplotlib or pillow')

def collect_npy_files(paths, recursive=False):
    files = []
    for p in paths:
        if os.path.isdir(p):
            pattern = os.path.join(p, '**', '*.npy') if recursive else os.path.join(p, '*.npy')
            files.extend(glob.glob(pattern, recursive=recursive))
        else:
            matches = glob.glob(p, recursive=recursive)
            if matches:
                files.extend(matches)
            else:
                if os.path.isfile(p):
                    files.append(p)
    return sorted(set(files))

def analyze_file(path, percentiles, hist_bins, topk_max):
    arr = np.load(path, allow_pickle=False)
    flat = arr.ravel()
    n_total = int(flat.size)
    finite_mask = np.isfinite(flat)
    n_finite = int(finite_mask.sum())
    n_nan = n_total - n_finite
    res = {
        'file': path,
        'shape': tuple(arr.shape),
        'dtype': str(arr.dtype),
        'total': n_total,
        'finite': n_finite,
        'nan': n_nan,
        'nan_pct': 100.0 * n_nan / n_total if n_total else 0.0,
    }
    if n_finite == 0:
        return res
    finite_vals = flat[finite_mask]
    res.update({
        'min': float(np.min(finite_vals)),
        'max': float(np.max(finite_vals)),
        'mean': float(np.mean(finite_vals)),
        'median': float(np.median(finite_vals)),
        'std': float(np.std(finite_vals)),
        'percentiles': {str(int(p)): float(np.percentile(finite_vals, p)) for p in percentiles},
        'zeros': int((finite_vals == 0).sum()),
        'zeros_pct': 100.0 * int((finite_vals == 0).sum()) / n_finite if n_finite else 0.0,
    })
    if n_finite <= 2_000_000:
        res['unique'] = int(np.unique(finite_vals).size)
    else:
        res['unique'] = None
    if hist_bins and hist_bins > 0:
        counts, edges = np.histogram(finite_vals, bins=hist_bins)
        res['hist_counts'] = counts.tolist()
        res['hist_edges'] = edges.tolist()
    if topk_max and topk_max > 0:
        mval = res['max']
        positions = np.argwhere(np.isclose(arr, mval, rtol=1e-8, atol=1e-12))
        res['max_positions'] = [tuple(map(int, p)) for p in positions[:topk_max]]
    return res

def print_summary(res, percentiles, hist_bins, topk_max, json_out=False):
    if json_out:
        print(json.dumps(res, ensure_ascii=False))
        return
    print(f"File: {res['file']}")
    print(f"  shape: {res['shape']}, dtype: {res['dtype']}")
    print(f"  total: {res['total']}, finite: {res['finite']} ({100.0*res['finite']/res['total']:.2f}%), NaN: {res['nan']} ({res['nan_pct']:.2f}%)")
    if res['finite'] == 0:
        print('  (no finite values)\n')
        return
    print(f"  min: {res['min']:.6g}, max: {res['max']:.6g}, mean: {res['mean']:.6g}, median: {res['median']:.6g}, std: {res['std']:.6g}")
    p_str = ', '.join([f"{p}%={res['percentiles'][str(int(p))]:.6g}" for p in percentiles])
    print(f"  percentiles: {p_str}")
    print(f"  zeros: {res['zeros']} ({res['zeros_pct']:.2f}%)")
    if res.get('unique') is not None:
        print(f"  unique values: {res['unique']}")
    if hist_bins and hist_bins > 0 and 'hist_counts' in res:
        print("  histogram:")
        edges = res['hist_edges']
        counts = res['hist_counts']
        for i in range(len(counts)):
            print(f"    [{edges[i]:.6g}, {edges[i+1]:.6g}): {counts[i]}")
    if topk_max and res.get('max_positions'):
        print(f"  max positions (first {len(res['max_positions'])}): {res['max_positions']}")
    print("")

def get_depth_at_coords(arr, y, x):
    """Extract depth value at specific coordinates (y, x).
    
    Returns a dict with the depth value and coordinate validation info.
    """
    h, w = arr.shape
    result = {
        'coords': (y, x),
        'shape': (h, w),
        'valid_coords': False,
        'value': None,
        'is_nan': False,
    }
    
    if not (0 <= y < h and 0 <= x < w):
        result['error'] = f'Coordinates ({y}, {x}) out of bounds. Array shape: ({h}, {w})'
        return result
    
    result['valid_coords'] = True
    val = arr[y, x]
    result['value'] = float(val) if np.isfinite(val) else val
    result['is_nan'] = bool(np.isnan(val))
    
    return result

def print_depth_at_coords(path, coords):
    """Load .npy file and print depth value at given coordinates."""
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f'Error reading {path}: {e}', file=sys.stderr)
        return False
    
    if arr.ndim != 2:
        print(f'Error: Expected 2D array, got shape {arr.shape}', file=sys.stderr)
        return False
    
    y, x = coords
    res = get_depth_at_coords(arr, y, x)
    
    print(f"File: {path}")
    print(f"  Array shape: {res['shape']}")
    
    if not res['valid_coords']:
        print(f"  {res['error']}")
        return False
    
    print(f"  Coordinates (Y, X): ({y}, {x})")
    if res['is_nan']:
        print(f"  Depth value: NaN")
    else:
        print(f"  Depth value: {res['value']:.6g}")
    
    return True

def write_csv(results, out_csv):
    headers = ['file', 'shape', 'dtype', 'total', 'finite', 'nan', 'nan_pct', 'min', 'max', 'mean', 'median', 'std', 'zeros', 'zeros_pct', 'unique']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in headers}
            row['shape'] = 'x'.join(map(str, r.get('shape', '')))
            w.writerow(row)

def parse_args():
    p = argparse.ArgumentParser(description='Inspect .npy depth maps and print statistics.')
    p.add_argument('paths', nargs='+', help='file(s), directory(ies) or glob pattern(s) to inspect')
    p.add_argument('--recursive', action='store_true', help='recursively search directories for .npy files')
    p.add_argument('--percentiles', nargs='*', type=float, default=DEFAULT_PERCENTILES, help='percentiles to compute (default 1 5 25 50 75 95 99)')
    p.add_argument('--hist-bins', type=int, default=0, help='compute histogram with this many bins (0 to skip)')
    p.add_argument('--out-csv', default=None, help='write summary CSV to this path')
    p.add_argument('--json', action='store_true', help='print per-file result as JSON')
    p.add_argument('--topk-max', type=int, default=5, help='show up to N coordinates of the max value')
    p.add_argument('--vis', action='store_true', help='display colorized visualization using matplotlib')
    p.add_argument('--save-vis', default=None, help='directory to write colorized PNGs for each .npy (creates dir if needed)')
    p.add_argument('--cmap', default='viridis', help='matplotlib colormap name for visualization')
    p.add_argument('--vmin', type=float, default=None, help='minimum depth value for colormap scaling')
    p.add_argument('--vmax', type=float, default=None, help='maximum depth value for colormap scaling')
    p.add_argument('--coords', nargs=2, type=int, metavar=('Y', 'X'), help='get depth value at specific coordinates (row, col)')
    return p.parse_args()

def main():
    args = parse_args()
    
    # Handle coordinate query mode
    if args.coords:
        files = collect_npy_files(args.paths, recursive=args.recursive)
        if not files:
            print('No .npy files found for the given paths.', file=sys.stderr)
            return 1
        
        print(f"Querying depth value at coordinates (Y={args.coords[0]}, X={args.coords[1]})")
        print()
        
        success_count = 0
        for fpath in files:
            if print_depth_at_coords(fpath, args.coords):
                success_count += 1
            print()
        
        if success_count == 0:
            return 1
        return 0
    
    # Normal analysis mode
    files = collect_npy_files(args.paths, recursive=args.recursive)
    if not files:
        print('No .npy files found for the given paths.', file=sys.stderr)
        return 1
    results = []
    for fpath in files:
        try:
            res = analyze_file(fpath, args.percentiles, args.hist_bins, args.topk_max)
        except Exception as e:
            print(f'Error reading {fpath}: {e}', file=sys.stderr)
            continue
        print_summary(res, args.percentiles, args.hist_bins, args.topk_max, json_out=args.json)
        results.append(res)
        # Visualization / save colored PNG
        if args.vis or args.save_vis:
            try:
                arr = np.load(fpath, allow_pickle=False)
            except Exception as e:
                print(f'  could not load for visualization: {e}', file=sys.stderr)
                continue
            try:
                img = create_colormap_image(arr, cmap_name=args.cmap, vmin=args.vmin, vmax=args.vmax)
            except Exception as e:
                print(f'  visualization error: {e}', file=sys.stderr)
                continue
            if args.save_vis:
                out_dir = args.save_vis
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    print(f'  cannot create/save to {out_dir}', file=sys.stderr)
                base = os.path.splitext(os.path.basename(fpath))[0]
                out_path = os.path.join(out_dir, base + '_vis.png')
                try:
                    save_colormap_image(img, out_path)
                    print(f'  saved visualization to {out_path}')
                except Exception as e:
                    print(f'  failed saving visualization: {e}', file=sys.stderr)
            if args.vis:
                if plt is None:
                    print('  matplotlib not available: cannot show visualization', file=sys.stderr)
                else:
                    try:
                        plt.figure(figsize=(8, 6))
                        plt.imshow(img)
                        plt.title(os.path.basename(fpath))
                        plt.axis('off')
                        plt.show()
                    except Exception as e:
                        print(f'  error showing visualization: {e}', file=sys.stderr)
    if args.out_csv:
        write_csv(results, args.out_csv)
        print(f'Wrote CSV summary to {args.out_csv}')
    return 0

if __name__ == '__main__':
    sys.exit(main())