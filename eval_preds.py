import argparse
import glob
import os

import cv2
import numpy as np
import torch

from metric_depth.util.metric import eval_depth


def load_gt(path, scale=1.0):
    if path.endswith('.npy'):
        gt = np.load(path).astype(np.float32)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'Cannot read GT file: {path}')
        if img.dtype == np.uint16:
            gt = img.astype(np.float32) * scale
        else:
            gt = img.astype(np.float32) * scale
    return gt


def main():
    parser = argparse.ArgumentParser(description='Evaluate depth predictions against GT')
    parser.add_argument('--pred-dir', required=True, help='directory with predicted .npy files')
    parser.add_argument('--gt-dir', required=True, help='directory with ground-truth depth files (npy/png)')
    parser.add_argument('--gt-ext', default=None, help='force GT extension (e.g. .npy, .png)')
    parser.add_argument('--gt-scale', type=float, default=1.0, help='multiply GT by this to convert to meters')
    parser.add_argument('--max-depth', type=float, default=100.0, help='clip values greater than this')
    args = parser.parse_args()

    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, '*_raw_depth_meter.npy')))
    if len(pred_paths) == 0:
        raise RuntimeError('No prediction .npy files found in pred-dir')

    sum_metrics = {}
    n = 0

    for ppath in pred_paths:
        name = os.path.splitext(os.path.basename(ppath))[0].replace('_raw_depth_meter', '')

        # find GT
        if args.gt_ext:
            candidates = [os.path.join(args.gt_dir, name + args.gt_ext)]
        else:
            candidates = [os.path.join(args.gt_dir, name + ext) for ext in ('.npy', '.png', '.png16', '.jpg')]

        gt_path = None
        for c in candidates:
            if os.path.exists(c):
                gt_path = c
                break

        if gt_path is None:
            print(f'Skipping {name}: no GT found')
            continue

        pred = np.load(ppath).astype(np.float32)
        gt = load_gt(gt_path, scale=args.gt_scale)

        if pred.shape != gt.shape:
            print(f'Skipping {name}: shape mismatch pred {pred.shape} gt {gt.shape}')
            continue

        # valid mask
        mask = (gt > 0) & (pred > 0) & (gt < args.max_depth)
        if mask.sum() == 0:
            print(f'Skipping {name}: empty valid mask')
            continue

        pred_t = torch.from_numpy(pred[mask]).float().cuda() if torch.cuda.is_available() else torch.from_numpy(pred[mask]).float()
        gt_t = torch.from_numpy(gt[mask]).float().cuda() if torch.cuda.is_available() else torch.from_numpy(gt[mask]).float()

        metrics = eval_depth(pred_t, gt_t)

        print(name, metrics)

        for k, v in metrics.items():
            sum_metrics[k] = sum_metrics.get(k, 0.0) + v
        n += 1

    if n == 0:
        print('No evaluated images')
        return

    avg = {k: v / n for k, v in sum_metrics.items()}
    print('\nAverage over', n, 'images:')
    for k, v in avg.items():
        print(f'{k}: {v:.4f}')


if __name__ == '__main__':
    main()
