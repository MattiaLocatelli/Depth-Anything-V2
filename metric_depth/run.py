import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import csv

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vits.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    parser.add_argument('--mask-dir', type=str, default=None, help='directory with mask .npy or .png files (0 values will be masked)')
    parser.add_argument('--bit-depth', type=int, choices=[8,16], default=8, help='output bit depth for visualization PNG (8 or 16)')
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--keypoints-csv', dest='keypoints_csv', type=str, default=None, help='CSV with keypoint coordinates to sample depth from')
    parser.add_argument('--kp-output', dest='kp_output', type=str, default=None, help='output CSV path for keypoint depths (defaults to <outdir>/keypoint_depths.csv)')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)

        # If a mask directory is provided, try to find a corresponding mask and apply it.
        if args.mask_dir is not None:
            def find_mask_file(src_name, mask_dir):
                stem = os.path.splitext(os.path.basename(src_name))[0]
                # common candidate patterns
                candidates = [
                    f"{stem}_raw_depth_meter.npy",
                    f"{stem}_raw_depth.npy",
                    f"{stem}.npy",
                    f"{stem}_mask.npy",
                    f"{stem}.png",
                    f"{stem}_mask.png",
                ]
                for c in candidates:
                    p = os.path.join(mask_dir, c)
                    if os.path.exists(p):
                        return p
                # fallback: any file starting with the stem
                globbed = glob.glob(os.path.join(mask_dir, stem + '*'))
                if len(globbed) > 0:
                    return globbed[0]
                return None

            mask_path = find_mask_file(filename, args.mask_dir)
            if mask_path is not None:
                try:
                    if mask_path.lower().endswith('.npy'):
                        mask_arr = np.load(mask_path)
                        # reduce to single channel if needed
                        if mask_arr.ndim == 3:
                            mask_arr = mask_arr[..., 0]
                        mask_bool = (mask_arr == 0)
                    else:
                        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                        if mask_img is None:
                            mask_bool = None
                        else:
                            if mask_img.ndim == 3:
                                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                            mask_bool = (mask_img == 0)

                    if mask_bool is not None:
                        # resize mask if it doesn't match depth shape
                        if mask_bool.shape != depth.shape:
                            mask_bool = cv2.resize(mask_bool.astype('uint8'), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                        # apply mask: set masked pixels to 0 (preserve zeros semantics)
                        depth[mask_bool] = args.max_depth
                        print(f'Applied mask from {mask_path} — masked {np.count_nonzero(mask_bool)} pixels')
                except Exception as e:
                    print(f'Warning: failed to load/apply mask {mask_path}: {e}')

        # If user requested keypoint sampling, ensure raw numpy depth maps are saved for lookup
        if args.keypoints_csv:
            args.save_numpy = True

        if args.save_numpy:
                output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
                np.save(output_path, depth)

        # Visualize depth using a fixed color scale matched to args.max_depth
        # This preserves metric meaning: 0 -> near, max_depth -> far (clipped)
        depth_clipped = np.clip(depth, 0.0, args.max_depth)
        depth_norm = (depth_clipped / float(args.max_depth))

        # Support both 8-bit and 16-bit visualizations
        if args.bit_depth == 16:
            if args.grayscale:
                vis16 = (depth_norm * 65535.0).astype(np.uint16)
                vis = np.repeat(vis16[..., np.newaxis], 3, axis=-1)
            else:
                cmap_rgba = cmap(depth_norm)[:, :, :3]
                vis = (cmap_rgba * 65535.0).astype(np.uint16)
                vis = vis[:, :, ::-1]
        else:
            depth_8u = (depth_norm * 255.0).astype(np.uint8)
            if args.grayscale:
                vis = np.repeat(depth_8u[..., np.newaxis], 3, axis=-1)
            else:
                vis = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, vis)
        else:
            # Prepare display images with matching dtype for concatenation
            if args.bit_depth == 16:
                raw_disp = (raw_image.astype(np.uint16) * 257)
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint16) * 65535
                vis_disp = vis
            else:
                raw_disp = raw_image
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                vis_disp = vis

            # Ensure vis has same height as raw_disp for hconcat
            if vis_disp.shape[0] != raw_disp.shape[0] or vis_disp.shape[1] != raw_disp.shape[1]:
                h, w = raw_disp.shape[0], raw_disp.shape[1]
                interp = cv2.INTER_NEAREST if vis_disp.dtype == np.uint16 else cv2.INTER_LINEAR
                vis_disp = cv2.resize(vis_disp, (w, h), interpolation=interp)

            combined_result = cv2.hconcat([raw_disp, split_region, vis_disp])

            cv2.imwrite(output_path, combined_result)

    # End of image loop

    # If keypoints CSV was provided, sample depths for each keypoint now that all images were processed
    if args.keypoints_csv:
        kp_out = args.kp_output if args.kp_output is not None else os.path.join(args.outdir, 'keypoint_depths.csv')

        # Build a quick map of available depth numpy files in outdir by basename
        depth_files = {}
        for p in glob.glob(os.path.join(args.outdir, '*_raw_depth_meter.npy')):
            base = os.path.splitext(os.path.basename(p))[0]
            # base is like '<name>_raw_depth_meter' -> remove suffix
            if base.endswith('_raw_depth_meter'):
                img_base = base[:-len('_raw_depth_meter')]
            else:
                img_base = base
            depth_files[img_base] = p

        def find_depth_npy_for_path(path_str):
            if path_str is None or path_str == '':
                return None
            # if path exists and corresponds to one of processed images, try basename
            name = os.path.splitext(os.path.basename(path_str))[0]
            # direct match
            if name in depth_files:
                return depth_files[name]
            # try fuzzy match: find any key that contains the name or viceversa
            for k, v in depth_files.items():
                if name in k or k in name:
                    return v
            return None

        # Read CSV and append depth columns
        with open(args.keypoints_csv, 'r', newline='') as f_in, open(kp_out, 'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames[:] if reader.fieldnames is not None else []
            # add output columns
            if 'live_depth' not in fieldnames:
                fieldnames.append('live_depth')
            if 'keyframe_depth' not in fieldnames:
                fieldnames.append('keyframe_depth')

            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            # helper to choose column name
            def choose(cols, candidates):
                for c in candidates:
                    if c in cols:
                        return c
                return None

            cols = reader.fieldnames if reader.fieldnames is not None else []
            live_x_col = choose(cols, ['live_x', 'live_u', 'lx', 'live_x_px'])
            live_y_col = choose(cols, ['live_y', 'live_v', 'ly', 'live_y_px'])
            kf_x_col = choose(cols, ['kf_x', 'keyframe_x', 'kf_u', 'kf_x_px'])
            kf_y_col = choose(cols, ['kf_y', 'keyframe_y', 'kf_v', 'kf_y_px'])
            live_path_col = choose(cols, ['live_frame_path', 'live_path', 'live_frame', 'live_image'])
            kf_path_col = choose(cols, ['keyframe_path', 'kf_frame_path', 'kf_path', 'keyframe_image'])
            keyframe_idx_col = choose(cols, ['keyframe_idx', 'matched_keyframe_idx', 'kf_idx'])

            for row in reader:
                # default depths
                row['live_depth'] = ''
                row['keyframe_depth'] = ''

                # LIVE point
                try:
                    if live_x_col and live_y_col:
                        lx = float(row.get(live_x_col, 'nan'))
                        ly = float(row.get(live_y_col, 'nan'))
                        live_img_path = row.get(live_path_col) if live_path_col else None
                        depth_npy = find_depth_npy_for_path(live_img_path) if live_img_path else None
                        if depth_npy is None and live_img_path is not None:
                            # try using basename only
                            depth_npy = find_depth_npy_for_path(os.path.basename(live_img_path))
                        if depth_npy is not None and not np.isnan(lx) and not np.isnan(ly):
                            d = np.load(depth_npy)
                            xi = int(round(lx))
                            yi = int(round(ly))
                            if 0 <= yi < d.shape[0] and 0 <= xi < d.shape[1]:
                                row['live_depth'] = float(d[yi, xi])
                            else:
                                row['live_depth'] = ''
                                print(f"Warning: live keypoint out of bounds ({xi},{yi}) for image {live_img_path}")
                        else:
                            if live_img_path is None:
                                print('Warning: no live image path column found for row; skipping live depth')
                    else:
                        print('Warning: live_x/live_y columns not found; skipping live depth sampling')
                except Exception as e:
                    print(f'Warning: failed sampling live depth: {e}')

                # KEYFRAME point
                try:
                    has_kf_coords = (kf_x_col is not None and kf_y_col is not None)
                    if has_kf_coords:
                        kx = float(row.get(kf_x_col, 'nan'))
                        ky = float(row.get(kf_y_col, 'nan'))
                        kf_img_path = None
                        if kf_path_col:
                            kf_img_path = row.get(kf_path_col)
                        # try to derive from keyframe index if explicit path not available
                        if kf_img_path is None and keyframe_idx_col is not None:
                            idx = row.get(keyframe_idx_col)
                            if idx:
                                # search for any processed image filename containing the index
                                for k in depth_files.keys():
                                    if f'kf{idx}' in k or f'{idx}' in k:
                                        kf_img_path = k
                                        break

                        depth_npy_kf = find_depth_npy_for_path(kf_img_path) if kf_img_path else None
                        if depth_npy_kf is not None and not np.isnan(kx) and not np.isnan(ky):
                            d_k = np.load(depth_npy_kf)
                            xi = int(round(kx))
                            yi = int(round(ky))
                            if 0 <= yi < d_k.shape[0] and 0 <= xi < d_k.shape[1]:
                                row['keyframe_depth'] = float(d_k[yi, xi])
                            else:
                                row['keyframe_depth'] = ''
                                print(f"Warning: keyframe keypoint out of bounds ({xi},{yi}) for image {kf_img_path}")
                        else:
                            if kf_img_path is None:
                                print('Warning: no keyframe image path found for row; skipping keyframe depth')
                    else:
                        print('Warning: kf_x/kf_y columns not found; skipping keyframe depth sampling')
                except Exception as e:
                    print(f'Warning: failed sampling keyframe depth: {e}')

                writer.writerow(row)

        print(f'Wrote keypoint depths to {kp_out}')