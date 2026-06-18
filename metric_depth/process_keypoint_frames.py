import argparse
import csv
import cv2
import glob
import numpy as np
import os
import re
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def infer_and_save_depths(image_dir, outdir, depth_anything, args, image_type='live', relative_depth_map=None, mask_tol=1e-6):
    """
    Process all images in image_dir, compute depth maps, save as .npy
    Returns mapping: {image_basename -> path_to_npy}
    """
    if not os.path.isdir(image_dir):
        print(f"Warning: {image_type} image directory {image_dir} not found")
        return {}
    
    image_files = sorted(glob.glob(os.path.join(image_dir, '**/*'), recursive=True))
    image_files = [f for f in image_files if os.path.isfile(f) and 
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"\nProcessing {len(image_files)} {image_type} images...")
    
    os.makedirs(outdir, exist_ok=True)
    depth_map = {}  # basename -> npy_path
    
    for k, filename in enumerate(image_files):
        print(f'  [{image_type}] {k+1}/{len(image_files)}: {os.path.basename(filename)}')

        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"    Warning: could not read {filename}")
            continue

        depth = depth_anything.infer_image(raw_image, args.input_size)

        # Optional: clip to a maximum metric depth if provided
        if getattr(args, 'max_depth', None) is not None:
            try:
                depth = np.clip(depth, 0.0, float(args.max_depth))
            except Exception:
                pass

        # Save numpy depth (apply optional relative-mask if provided)
        img_base = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(outdir, f'{img_base}_raw_depth_meter.npy')

        rel_path = None
        if relative_depth_map:
            rel_path = relative_depth_map.get(img_base)

        if rel_path:
            try:
                rel = np.load(rel_path)
                # Resize relative to metric if shapes differ (nearest to preserve mask)
                if rel.shape != depth.shape:
                    rel_resized = cv2.resize(rel, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    rel_resized = rel

                rel_min = np.nanmin(rel_resized)
                if np.isfinite(rel_min):
                    mask = (rel_resized <= (rel_min + mask_tol))
                    mask = mask.astype(bool)
                    finite = np.isfinite(depth)
                    if finite.any():
                        metric_max = np.nanmax(depth[finite])
                    else:
                        metric_max = np.nanmax(depth)
                    depth[mask] = metric_max
                    masked_output_path = os.path.join(outdir, f'{img_base}_raw_depth_meter_masked.npy')
                    np.save(masked_output_path, depth)
                    depth_map[img_base] = masked_output_path
                else:
                    np.save(output_path, depth)
                    depth_map[img_base] = output_path
            except Exception as e:
                print(f"    Warning: failed to apply relative mask for {img_base}: {e}")
                np.save(output_path, depth)
                depth_map[img_base] = output_path
        else:
            np.save(output_path, depth)
            depth_map[img_base] = output_path
    
    return depth_map


def load_depth_maps_from_directory(depth_dir, image_type='live'):
    """
    Load all .npy depth maps from a directory.
    Returns mapping: {image_basename -> path_to_npy}
    """
    if not os.path.isdir(depth_dir):
        print(f"Warning: {image_type} depth directory {depth_dir} not found")
        return {}
    
    depth_map = {}
    npy_files = sorted(glob.glob(os.path.join(depth_dir, '**/*.npy'), recursive=True))
    
    print(f"\nLoading {len(npy_files)} {image_type} depth maps from {depth_dir}...")
    
    for npy_file in npy_files:
        # Extract basename (remove _raw_depth_meter.npy suffix if present)
        base = os.path.splitext(os.path.basename(npy_file))[0]
        if base.endswith('_raw_depth_meter'):
            img_base = base[:-len('_raw_depth_meter')]
        else:
            img_base = base
        depth_map[img_base] = npy_file
    
    print(f"  → Loaded {len(depth_map)} depth maps")
    return depth_map


def apply_relative_mask_to_metric_array(metric_arr, rel_arr, mask_tol=1e-6):
    """
    Apply mask derived from relative depth array onto metric array in-memory.
    NOTE: for the relative maps used in this project larger values correspond
    to nearer points. Therefore the mask is built from the relative map's
    minimum values (points with lowest relative depth are considered farthest
    and will be transferred to the metric map as maximum-depth pixels).
    Pixels that are at (or within `mask_tol` of) the relative map's min
    will be set to the metric map's max.
    """
    try:
        if rel_arr.shape != metric_arr.shape:
            rel_resized = cv2.resize(rel_arr, (metric_arr.shape[1], metric_arr.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            rel_resized = rel_arr

        rel_min = np.nanmin(rel_resized)
        if not np.isfinite(rel_min):
            return metric_arr

        mask = (rel_resized <= (rel_min + mask_tol)).astype(bool)

        finite = np.isfinite(metric_arr)
        if finite.any():
            metric_max = np.nanmax(metric_arr[finite])
        else:
            metric_max = np.nanmax(metric_arr)

        metric_arr[mask] = metric_max
        return metric_arr
    except Exception as e:
        print(f"Warning: failed to apply mask to metric array: {e}")
        return metric_arr


def apply_mask_to_npy_file(metric_npy_path, rel_npy_path, mask_tol=1e-6, max_depth=None):
    """
    Load metric and relative .npy files, apply mask and save a masked copy
    next to the original metric file. Returns path to masked file (or original on error).
    """
    try:
        metric = np.load(metric_npy_path)
        rel = np.load(rel_npy_path)
        masked = apply_relative_mask_to_metric_array(metric, rel, mask_tol=mask_tol)
        if max_depth is not None:
            try:
                masked = np.clip(masked, 0.0, float(max_depth))
            except Exception:
                pass
        base = os.path.splitext(os.path.basename(metric_npy_path))[0]
        dirn = os.path.dirname(metric_npy_path)
        masked_filename = f"{base}_masked.npy"
        masked_path = os.path.join(dirn, masked_filename)
        np.save(masked_path, masked)
        return masked_path
    except Exception as e:
        print(f"Warning: failed to apply mask to file {metric_npy_path}: {e}")
        return metric_npy_path


def find_depth_map_for_keypoint_row(row, live_depth_map, kf_depth_map, kf_name_pattern=None):
    """
    Given a CSV row with live/keyframe identifiers, find corresponding depth maps.
    Returns (live_depth_npy_path, kf_depth_npy_path) or (None, None) if not found.
    """
    live_depth_npy = None
    kf_depth_npy = None
    
    # Try to find live frame depth map
    if 'live_frame_path' in row and row['live_frame_path']:
        live_basename = os.path.splitext(os.path.basename(row['live_frame_path']))[0]
        if live_basename in live_depth_map:
            live_depth_npy = live_depth_map[live_basename]
        else:
            # Fuzzy match
            for k, v in live_depth_map.items():
                if live_basename in k or k in live_basename:
                    live_depth_npy = v
                    break
    
    # Try to find keyframe depth map
    # Could be identified by matched_keyframe_idx or keyframe_idx
    kf_idx = row.get('matched_keyframe_idx') or row.get('keyframe_idx')
    if kf_idx:
        # Look for files matching the keyframe index pattern
        # Pattern might be 'kf{idx}' or 'keyframe_{idx}' etc.
        for k, v in kf_depth_map.items():
            # Extract numeric parts
            numbers = re.findall(r'\d+', k)
            if str(kf_idx) in numbers:
                kf_depth_npy = v
                break
    
    # Also try explicit keyframe_path if available
    if kf_depth_npy is None and 'keyframe_path' in row and row['keyframe_path']:
        kf_basename = os.path.splitext(os.path.basename(row['keyframe_path']))[0]
        if kf_basename in kf_depth_map:
            kf_depth_npy = kf_depth_map[kf_basename]
        else:
            # Fuzzy match
            for k, v in kf_depth_map.items():
                if kf_basename in k or k in kf_basename:
                    kf_depth_npy = v
                    break
    
    return live_depth_npy, kf_depth_npy


def sample_depth_at_keypoint(depth_npy_path, x, y):
    """
    Load depth map and sample at (x, y) pixel coordinates.
    Returns depth value or None if out of bounds.
    """
    if depth_npy_path is None:
        return None
    
    try:
        d = np.load(depth_npy_path)
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= yi < d.shape[0] and 0 <= xi < d.shape[1]:
            return float(d[yi, xi])
        else:
            return None
    except Exception as e:
        print(f"Warning: failed to sample depth: {e}")
        return None


def process_keypoints_csv(csv_path, live_depth_map, kf_depth_map, output_csv_path):
    """
    Read CSV with keypoint matches, sample depths, write output CSV.
    """
    print(f"\nProcessing keypoints CSV: {csv_path}")
    
    with open(csv_path, 'r', newline='') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        
        # Add output columns if not present
        new_columns = ['live_depth', 'keyframe_depth', 'depth_diff', 'notes']
        for col in new_columns:
            if col not in fieldnames:
                fieldnames.append(col)
        
        rows = list(reader)
    
    # Process rows and sample depths
    processed_rows = []
    for i, row in enumerate(rows):
        if i % max(1, len(rows) // 10) == 0:
            print(f"  Sampling: {i+1}/{len(rows)}")
        
        row['live_depth'] = ''
        row['keyframe_depth'] = ''
        row['depth_diff'] = ''
        row['notes'] = ''
        
        # Find depth maps for this row
        live_npy, kf_npy = find_depth_map_for_keypoint_row(row, live_depth_map, kf_depth_map)
        
        # Sample live point depth
        live_depth = None
        if live_npy and 'live_x' in row and 'live_y' in row:
            try:
                lx = float(row['live_x'])
                ly = float(row['live_y'])
                live_depth = sample_depth_at_keypoint(live_npy, lx, ly)
                if live_depth is not None:
                    row['live_depth'] = f"{live_depth:.4f}"
                else:
                    row['notes'] = 'live keypoint out of bounds'
            except Exception as e:
                row['notes'] = f'live sampling error: {e}'
        elif not live_npy:
            row['notes'] = 'live depth map not found'
        
        # Sample keyframe point depth
        kf_depth = None
        if kf_npy and 'kf_x' in row and 'kf_y' in row:
            try:
                kx = float(row['kf_x'])
                ky = float(row['kf_y'])
                kf_depth = sample_depth_at_keypoint(kf_npy, kx, ky)
                if kf_depth is not None:
                    row['keyframe_depth'] = f"{kf_depth:.4f}"
                else:
                    if row['notes']:
                        row['notes'] += '; '
                    row['notes'] += 'kf keypoint out of bounds'
            except Exception as e:
                if row['notes']:
                    row['notes'] += '; '
                row['notes'] += f'kf sampling error: {e}'
        elif not kf_npy:
            if row['notes']:
                row['notes'] += '; '
            row['notes'] += 'kf depth map not found'
        
        # Compute depth difference if both available
        if live_depth is not None and kf_depth is not None:
            row['depth_diff'] = f"{abs(live_depth - kf_depth):.4f}"
        
        processed_rows.append(row)
    
    # Write output CSV
    with open(output_csv_path, 'w', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)
    
    print(f"\nWrote keypoint depths to: {output_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute depth maps (optional) and sample keypoint depths from .npy depth maps'
    )
    
    # Optional: compute depth maps
    parser.add_argument('--live-frames', type=str, default=None, help='directory with live frame images (to compute depths)')
    parser.add_argument('--keyframes', type=str, default=None, help='directory with offline keyframe images (to compute depths)')
    
    # Optional: use pre-computed depth maps
    parser.add_argument('--live-depths', type=str, default=None, help='directory with pre-computed live frame .npy depth maps')
    parser.add_argument('--keyframe-depths', type=str, default=None, help='directory with pre-computed keyframe .npy depth maps')
    
    # Output directories for computed depths
    parser.add_argument('--live-depths-output', type=str, default=None, help='where to save computed live frame depths (default: <outdir>/Frame_live)')
    parser.add_argument('--keyframe-depths-output', type=str, default=None, help='where to save computed keyframe depths (default: <outdir>/Keyframe_offline)')
    
    # CSV and general options
    parser.add_argument('--keypoints-csv', type=str, required=True, help='CSV with keypoint matches and coordinates')
    parser.add_argument('--kp-output', type=str, default=None, help='output CSV path with depth values (defaults to same dir as input CSV with _with_depths suffix)')
    
    # Model options
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'], help='encoder for depth model')
    parser.add_argument('--input-size', type=int, default=518, help='input size for depth inference')
    parser.add_argument('--outdir', type=str, default='./depth_results', help='base output directory')
    parser.add_argument('--relative-depths', type=str, default=None, help='directory with pre-computed relative depth .npy maps (or parent dir with subfolders)')
    parser.add_argument('--relative-depths-live', type=str, default=None, help='directory with pre-computed relative depth .npy maps for live frames')
    parser.add_argument('--relative-depths-keyframe', type=str, default=None, help='directory with pre-computed relative depth .npy maps for keyframes')
    parser.add_argument('--mask-tol', type=float, default=1e-6, help='tolerance when comparing to relative depth max')
    parser.add_argument('--max-depth', type=float, default=800.0, help='maximum depth value (meters) used for clipping/visualization')
    parser.add_argument('--model-type', type=str, choices=['base', 'metric'], default='base', help='choose checkpoint type to load: base or metric')
    
    args = parser.parse_args()
    
    # Determine which depth maps to use/compute
    live_depth_map = {}
    kf_depth_map = {}
    
    # Load model if needed
    model = None
    device = None
    
    # Load relative depth maps (if provided) to use as mask source
    relative_live_map = {}
    relative_kf_map = {}

    # Explicit separate directories take precedence
    if args.relative_depths_live:
        relative_live_map = load_depth_maps_from_directory(args.relative_depths_live, image_type='relative_live')
        print(f"  → Loaded {len(relative_live_map)} relative live depth maps from: {args.relative_depths_live}")
    if args.relative_depths_keyframe:
        relative_kf_map = load_depth_maps_from_directory(args.relative_depths_keyframe, image_type='relative_keyframe')
        print(f"  → Loaded {len(relative_kf_map)} relative keyframe depth maps from: {args.relative_depths_keyframe}")

    # If a single parent directory is provided, try to detect subfolders, otherwise use it for both
    if args.relative_depths and not (relative_live_map or relative_kf_map):
        base_rel = args.relative_depths
        live_candidates = ['Frame_live', 'frame_live', 'FrameLive', 'live', 'Live']
        kf_candidates = ['Keyframe_offline', 'keyframe_offline', 'KeyframeOffline', 'Keyframe_offline', 'Keyframe', 'keyframe', 'offline', 'Offline']
        found_live = False
        found_kf = False
        for c in live_candidates:
            p = os.path.join(base_rel, c)
            if os.path.isdir(p):
                relative_live_map = load_depth_maps_from_directory(p, image_type='relative_live')
                print(f"  → Loaded {len(relative_live_map)} relative live depth maps from: {p}")
                found_live = True
                break
        for c in kf_candidates:
            p = os.path.join(base_rel, c)
            if os.path.isdir(p):
                relative_kf_map = load_depth_maps_from_directory(p, image_type='relative_keyframe')
                print(f"  → Loaded {len(relative_kf_map)} relative keyframe depth maps from: {p}")
                found_kf = True
                break
        if not (found_live or found_kf):
            # fallback: load all into both maps
            rel_all = load_depth_maps_from_directory(base_rel, image_type='relative')
            if rel_all:
                relative_live_map = rel_all.copy()
                relative_kf_map = rel_all.copy()
                print(f"  → Loaded {len(rel_all)} relative depth maps from: {base_rel} (using for both live and keyframe)")
    
    if args.live_frames or args.keyframes:
        # Need to compute depths
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {device}")

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        print(f"\nLoading Depth Anything V2 ({args.encoder})...")

        # Choose checkpoint according to model type
        if args.model_type == 'metric':
            candidates = [
                f'metric_depth/checkpoints/depth_anything_v2_metric_{args.encoder}.pth',
                f'metric_depth/checkpoints/depth_anything_v2_metric_hypersim_{args.encoder}.pth',
                f'metric_depth/checkpoints/depth_anything_v2_metric_vkitti_{args.encoder}.pth',
            ]
            ckpt_path = None
            for p in candidates:
                if os.path.exists(p):
                    ckpt_path = p
                    break
            if ckpt_path is None:
                # fallback to base checkpoint if present
                fallback = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
                if os.path.exists(fallback):
                    print(f'Warning: no metric checkpoint found; falling back to base checkpoint: {fallback}')
                    ckpt_path = fallback
                else:
                    print('Error: no metric nor base checkpoint found in checkpoints/. Please download a checkpoint or set up files accordingly.')
                    raise FileNotFoundError('checkpoint not found')
        else:
            ckpt_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
            if not os.path.exists(ckpt_path):
                # try metric candidates as possible alternatives
                alt = f'checkpoints/depth_anything_v2_metric_{args.encoder}.pth'
                if os.path.exists(alt):
                    print(f'Warning: base checkpoint not found; using metric checkpoint: {alt}')
                    ckpt_path = alt
                else:
                    print(f'Error: checkpoint not found: {ckpt_path}')
                    raise FileNotFoundError(ckpt_path)

        print(f'  -> Loading checkpoint: {ckpt_path}')

        # Instantiate the correct model implementation.
        # Metric models in `metric_depth/depth_anything_v2` use a Sigmoid head
        # and multiply by a `max_depth` value; import that implementation when
        # requested so outputs are in meters as expected.
        if args.model_type == 'metric':
            try:
                from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as MetricDepthAnythingV2
                print(f'  -> Using metric model implementation (metric_depth) with max_depth={args.max_depth}')
                model = MetricDepthAnythingV2(**model_configs[args.encoder], max_depth=args.max_depth)
            except Exception as e:
                print(f'Warning: failed to import metric model implementation: {e}. Falling back to base class.')
                model = DepthAnythingV2(**model_configs[args.encoder])
        else:
            model = DepthAnythingV2(**model_configs[args.encoder])

        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model = model.to(device).eval()
    
    # Compute or load live frame depths
    if args.live_frames:
        live_out = args.live_depths_output if args.live_depths_output else os.path.join(args.outdir, 'Frame_live')
        os.makedirs(live_out, exist_ok=True)
        live_depth_map = infer_and_save_depths(
            args.live_frames, live_out, model, args, image_type='live',
            relative_depth_map=relative_live_map, mask_tol=args.mask_tol
        )
        print(f"  → Saved {len(live_depth_map)} live frame depth maps to: {live_out}")
    elif args.live_depths:
        live_depth_map = load_depth_maps_from_directory(args.live_depths, image_type='live')
        # Apply relative mask to precomputed metric maps if available
        if relative_live_map:
            for base, metric_path in list(live_depth_map.items()):
                rel_path = relative_live_map.get(base)
                if rel_path:
                    masked_path = apply_mask_to_npy_file(metric_path, rel_path, mask_tol=args.mask_tol, max_depth=args.max_depth)
                    live_depth_map[base] = masked_path
    
    # Compute or load keyframe depths
    if args.keyframes:
        kf_out = args.keyframe_depths_output if args.keyframe_depths_output else os.path.join(args.outdir, 'Keyframe_offline')
        os.makedirs(kf_out, exist_ok=True)
        kf_depth_map = infer_and_save_depths(
            args.keyframes, kf_out, model, args, image_type='keyframe',
            relative_depth_map=relative_kf_map, mask_tol=args.mask_tol
        )
        print(f"  → Saved {len(kf_depth_map)} keyframe depth maps to: {kf_out}")
    elif args.keyframe_depths:
        kf_depth_map = load_depth_maps_from_directory(args.keyframe_depths, image_type='keyframe')
        # Apply relative mask to precomputed keyframe metric maps if available
        if relative_kf_map:
            for base, metric_path in list(kf_depth_map.items()):
                rel_path = relative_kf_map.get(base)
                if rel_path:
                    masked_path = apply_mask_to_npy_file(metric_path, rel_path, mask_tol=args.mask_tol, max_depth=args.max_depth)
                    kf_depth_map[base] = masked_path
    
    # Process CSV and sample depths
    if args.kp_output is None:
        csv_dir = os.path.dirname(args.keypoints_csv) if os.path.dirname(args.keypoints_csv) else '.'
        csv_base = os.path.splitext(os.path.basename(args.keypoints_csv))[0]
        args.kp_output = os.path.join(csv_dir, f'{csv_base}_with_depths.csv')
    
    process_keypoints_csv(args.keypoints_csv, live_depth_map, kf_depth_map, args.kp_output)
    
    print(f"\n✓ Done! Results in: {args.kp_output}")
