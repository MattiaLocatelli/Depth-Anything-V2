import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=100)
    parser.add_argument('--ref-dir', type=str, default=None, help='directory with outputs (guide depth and masks) from the general model')
    parser.add_argument('--guide-weight', type=float, default=1.0, help='blend weight for guide depth in overexposed areas (0..1)')
    parser.add_argument('--overexp-thresh', type=int, default=240, help='grayscale threshold (0-255) to detect overexposure')
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
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

        # If a reference directory is provided, attempt to find and apply a guide depth
        if args.ref_dir is not None:
            def find_guide_file(src_name, guide_dir):
                stem = os.path.splitext(os.path.basename(src_name))[0]
                candidates = [
                    f"{stem}_raw_depth_meter.npy",
                    f"{stem}_raw_depth.npy",
                    f"{stem}.npy",
                    f"{stem}.png",
                    f"{stem}.jpg",
                ]
                for c in candidates:
                    p = os.path.join(guide_dir, c)
                    if os.path.exists(p):
                        return p
                globbed = glob.glob(os.path.join(guide_dir, stem + '*'))
                if len(globbed) > 0:
                    return globbed[0]
                return None

            guide_path = find_guide_file(filename, args.ref_dir)
            if guide_path is not None:
                try:
                    guide = None
                    if guide_path.lower().endswith('.npy'):
                        guide = np.load(guide_path).astype(np.float32)
                        if guide.ndim == 3:
                            guide = guide[..., 0]
                    else:
                        gimg = cv2.imread(guide_path, cv2.IMREAD_UNCHANGED)
                        if gimg is not None:
                            if gimg.ndim == 3:
                                gimg = cv2.cvtColor(gimg, cv2.COLOR_BGR2GRAY)
                            guide = gimg.astype(np.float32)

                    if guide is not None:
                        gmin, gmax = np.nanmin(guide), np.nanmax(guide)
                        if gmax - gmin < 1e-6:
                            print(f'Skipping guide {guide_path}: constant or invalid values')
                        else:
                            guide_norm = (guide - gmin) / (gmax - gmin)
                            # Invert mapping: run.py low depth = farther, while metric low = closer
                            guide_mapped = (1.0 - guide_norm) * args.max_depth
                            if guide_mapped.shape != depth.shape:
                                guide_mapped = cv2.resize(guide_mapped.astype('float32'), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

                            gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
                            over_mask = gray >= args.overexp_thresh

                            if args.guide_weight >= 1.0:
                                depth[over_mask] = guide_mapped[over_mask]
                            else:
                                # blend only on overexposed pixels
                                w = float(np.clip(args.guide_weight, 0.0, 1.0))
                                blended = depth.copy()
                                blended[over_mask] = depth[over_mask] * (1.0 - w) + guide_mapped[over_mask] * w
                                depth = blended

                            print(f'Applied guide {guide_path} — overexposed pixels: {np.count_nonzero(over_mask)} (weight {args.guide_weight})')
                except Exception as e:
                    print(f'Warning: failed to load/apply guide {guide_path}: {e}')
            else:
                print(f'No guide file found for {filename} in {args.ref_dir}')

        # If a reference directory is provided, try to find a corresponding mask and apply it.
        mask_dir_effective = args.ref_dir
        if mask_dir_effective is not None:
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

            mask_path = find_mask_file(filename, mask_dir_effective)
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

        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(output_path, combined_result)