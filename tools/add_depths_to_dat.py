#!/usr/bin/env python3
"""
Add depth samples to .dat keypoint files using precomputed depth maps (.npy).

Usage examples:
  # Match depth .npy files by .dat basename (default)
  python tools/add_depths_to_dat.py --dat-dir path/to/dat --depth-dir path/to/depths --out-dir path/to/out

  # If .dat files have a header and columns 'x' and 'y':
  python tools/add_depths_to_dat.py --dat-dir path/to/dat --depth-dir path/to/depths --header --x-col x --y-col y --out-dir out

  # Use bilinear interpolation and overwrite originals (make backup)
  python tools/add_depths_to_dat.py --dat-dir path/to/dat --depth-dir path/to/depths --interp bilinear --inplace

This script is conservative: it will write augmented files to --out-dir by default
and will not modify originals unless --inplace is passed.
"""
from __future__ import annotations
import argparse
import os
import sys
import shutil
import csv
from typing import Optional, Tuple

import numpy as np
try:
    from scipy.ndimage import map_coordinates
except Exception:
    map_coordinates = None
import struct
from typing import List


def sample_depth(depth_arr: np.ndarray, x: float, y: float, method: str = "nearest") -> Optional[float]:
    """Sample depth array at pixel coordinates (x=col, y=row).

    Coordinates: x is horizontal (column), y is vertical (row).
    """
    if np.isnan(x) or np.isnan(y):
        return None
    h, w = depth_arr.shape[:2]
    if method == "nearest":
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= iy < h and 0 <= ix < w:
            return float(depth_arr[iy, ix])
        return None
    elif method in ("bilinear", "linear"):
        if map_coordinates is None:
            raise RuntimeError("scipy is required for bilinear interpolation (scipy.ndimage.map_coordinates)")
        # coords are (row_coords, col_coords)
        coords = np.array([[y], [x]])
        try:
            val = map_coordinates(depth_arr, coords, order=1, mode="nearest")[0]
            return float(val)
        except Exception:
            return None
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def find_depth_file_for_dat(dat_path: str, depth_dir: str, depth_ext: str = ".npy") -> Optional[str]:
    """Try to find depth file matching dat basename in depth_dir.

    Returns absolute path or None.
    """
    base = os.path.splitext(os.path.basename(dat_path))[0]
    candidate = os.path.join(depth_dir, base + depth_ext)
    if os.path.exists(candidate):
        return candidate
    # fallback: try any file containing the base
    for f in os.listdir(depth_dir):
        if base in f and f.endswith(depth_ext):
            return os.path.join(depth_dir, f)
    return None


def parse_line_tokens(line: str, delim: Optional[str]) -> list[str]:
    if delim:
        return line.strip().split(delim)
    return line.strip().split()


def process_dat_file(dat_path: str,
                     depth_path: Optional[str],
                     out_path: str,
                     header: bool,
                     delim: Optional[str],
                     x_col: Optional[str],
                     y_col: Optional[str],
                     x_index: Optional[int],
                     y_index: Optional[int],
                     file_col: Optional[str],
                     file_col_index: Optional[int],
                     interp: str = "nearest") -> Tuple[int, int]:
    """Read dat file, sample depths, write augmented file.

    Returns (n_rows_processed, n_depths_sampled)
    """
    with open(dat_path, "r", newline="") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0

    # detect output directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # If depth_path provided, load depth array once
    depth_arr = None
    if depth_path and os.path.exists(depth_path):
        depth_arr = np.load(depth_path)

    out_lines: list[str] = []
    header_tokens = None
    start_idx = 0

    if header:
        header_tokens = parse_line_tokens(lines[0], delim)
        start_idx = 1

    sampled = 0
    total = 0

    # prepare output header if needed
    if header and header_tokens is not None:
        out_header = header_tokens + ["depth"]
        if delim:
            out_lines.append(delim.join(out_header) + "\n")
        else:
            out_lines.append(" ".join(out_header) + "\n")

    for li in range(start_idx, len(lines)):
        line = lines[li]
        if not line.strip():
            out_lines.append(line)
            continue
        # preserve comments
        if line.lstrip().startswith("#"):
            out_lines.append(line)
            continue

        tokens = parse_line_tokens(line, delim)
        total += 1

        # get x,y
        x = None
        y = None
        if header and x_col and y_col and header_tokens is not None:
            try:
                xi = header_tokens.index(x_col)
                yi = header_tokens.index(y_col)
                x = float(tokens[xi])
                y = float(tokens[yi])
            except Exception:
                x = y = None
        else:
            # use indices
            try:
                if x_index is not None and y_index is not None:
                    x = float(tokens[x_index])
                    y = float(tokens[y_index])
            except Exception:
                x = y = None

        depth_val = None
        # if depth array is loaded and coordinates are present, sample
        if depth_arr is not None and x is not None and y is not None:
            depth_val = sample_depth(depth_arr, x, y, method=interp)
            if depth_val is not None:
                sampled += 1

        # create output line: if header, preserve tokens order and add depth; else append depth
        try:
            if header and header_tokens is not None:
                # reconstruct line with tokens; maintain delim
                if delim:
                    out_lines.append(delim.join(tokens + ["" if depth_val is None else f"{depth_val:.6f}"]) + "\n")
                else:
                    out_lines.append(" ".join(tokens + ["" if depth_val is None else f"{depth_val:.6f}"]) + "\n")
            else:
                if delim:
                    out_lines.append(delim.join(tokens + ["" if depth_val is None else f"{depth_val:.6f}"]) + "\n")
                else:
                    out_lines.append(" ".join(tokens + ["" if depth_val is None else f"{depth_val:.6f}"]) + "\n")
        except Exception:
            # fallback: write original line and a trailing depth if possible
            out_lines.append(line.rstrip('\n') + (" " + (f"{depth_val:.6f}" if depth_val is not None else "")) + "\n")

    # write output
    with open(out_path, "w", newline="") as f:
        f.writelines(out_lines)

    return total, sampled


def process_dat_file_binary(dat_path: str,
                            depth_path: Optional[str],
                            out_dir: str,
                            size_t_bytes: int = 8,
                            endian: str = 'little',
                            interp: str = 'nearest',
                            write_npy: bool = True,
                            write_csv: bool = False,
                            embed: bool = False,
                            verbose: bool = False) -> Tuple[int, int, int]:
    """Parse binary .dat files written by SaveKeypoints and sample depths.

    Returns (frames_parsed, total_keypoints, total_sampled_depths)
    """
    if depth_path is None or not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    # load depth array once
    depth_arr = np.load(depth_path)

    endian_char = '<' if endian == 'little' else '>'
    size_fmt = 'Q' if size_t_bytes == 8 else 'I'

    total_frames = 0
    total_kps = 0
    total_sampled = 0

    base = os.path.splitext(os.path.basename(dat_path))[0]

    with open(dat_path, 'rb') as fh:
        file_size = os.path.getsize(dat_path)
        if verbose:
            print(f"Binary file size: {file_size} bytes")

        frame_idx = 0
        while True:
            start_pos = fh.tell()
            # read header
            b = fh.read(1)
            if not b or len(b) < 1:
                break
            try:
                type_val = struct.unpack(endian_char + 'B', b)[0]
            except Exception:
                break

            b = fh.read(4)
            if not b or len(b) < 4:
                break
            closest_idx = struct.unpack(endian_char + 'i', b)[0]

            b = fh.read(size_t_bytes)
            if not b or len(b) < size_t_bytes:
                break
            keypoints_size = struct.unpack(endian_char + size_fmt, b)[0]

            # read keypoints
            kp_bytes_len = keypoints_size * (4 * 2)  # Point2f = 2 floats
            kp_bytes = fh.read(kp_bytes_len)
            if not kp_bytes or len(kp_bytes) < kp_bytes_len:
                break
            kp_dtype = np.dtype(endian_char + 'f4')
            try:
                kp_arr = np.frombuffer(kp_bytes, dtype=kp_dtype).reshape(-1, 2)
            except Exception:
                # malformed
                break

            # descriptors size
            b = fh.read(size_t_bytes)
            if not b or len(b) < size_t_bytes:
                break
            desc_size = struct.unpack(endian_char + size_fmt, b)[0]

            # descriptors
            desc_bytes_len = desc_size * 4
            desc_bytes = fh.read(desc_bytes_len)
            if desc_size and (not desc_bytes or len(desc_bytes) < desc_bytes_len):
                break

            # rotation (9 floats)
            rot_bytes = fh.read(9 * 4)
            if not rot_bytes or len(rot_bytes) < 9 * 4:
                break
            rot = np.frombuffer(rot_bytes, dtype=kp_dtype).reshape(3, 3)

            # translation (3 floats)
            trans_bytes = fh.read(3 * 4)
            if not trans_bytes or len(trans_bytes) < 3 * 4:
                break
            trans = np.frombuffer(trans_bytes, dtype=kp_dtype)

            # sample depths for keypoints
            depths = np.full((keypoints_size,), np.nan, dtype=np.float32)
            sampled = 0
            for i, (x, y) in enumerate(kp_arr):
                d = sample_depth(depth_arr, float(x), float(y), method=interp)
                if d is not None:
                    depths[i] = d
                    sampled += 1

            # save outputs
            out_items = []
            if write_npy:
                out_npy = os.path.join(out_dir, f"{base}_frame{frame_idx}_depths.npy")
                np.save(out_npy, depths)
                out_items.append(out_npy)
            if write_csv:
                out_csv = os.path.join(out_dir, f"{base}_frame{frame_idx}_depths.csv")
                with open(out_csv, 'w', newline='') as cf:
                    cf.write('idx,x,y,depth\n')
                    for i, (x, y) in enumerate(kp_arr):
                        depth_str = '' if np.isnan(depths[i]) else f"{depths[i]:.6f}"
                        cf.write(f"{i},{x},{y},{depth_str}\n")
                out_items.append(out_csv)

            if embed:
                # write a copy and append depth block with magic
                # create embedded filename
                # create embedded filename
                emb_name = os.path.join(out_dir, os.path.basename(dat_path) + '.embedded')
                shutil.copy2(dat_path, emb_name)
                with open(emb_name, 'ab') as ef:
                    # magic 'KPD0' + version byte
                    ef.write(b'KPD0')
                    ef.write(struct.pack(endian_char + 'B', 1))
                    # write keypoints count as size_t
                    ef.write(struct.pack(endian_char + size_fmt, keypoints_size))
                    # write depths as float32 with chosen endian
                    if endian_char == '<':
                        ef.write(depths.astype('<f4').tobytes())
                    else:
                        ef.write(depths.astype('>f4').tobytes())
                out_items.append(emb_name)

            if verbose:
                print(f"frame {frame_idx}: type={type_val}, closest_idx={closest_idx}, kps={keypoints_size}, sampled={sampled}, outputs={out_items}")

            total_frames += 1
            total_kps += keypoints_size
            total_sampled += sampled
            frame_idx += 1

    return total_frames, total_kps, total_sampled


def main():
    parser = argparse.ArgumentParser(description="Add depth samples to .dat keypoint files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dat-dir", type=str, help="Directory containing .dat files")
    group.add_argument("--dat-file", type=str, help="Single .dat file to process")
    parser.add_argument("--depth-dir", type=str, required=True, help="Directory with .npy depth maps")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to write augmented .dat files (default: dat-dir/aug)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite original .dat files (a .bak copy will be created)")
    parser.add_argument("--header", action="store_true", help="Treat .dat files as having a header line with column names")
    parser.add_argument("--delimiter", type=str, default=None, help="Delimiter for .dat (default: any whitespace)")
    parser.add_argument("--x-col", type=str, default=None, help="Column name for X (header mode)")
    parser.add_argument("--y-col", type=str, default=None, help="Column name for Y (header mode)")
    parser.add_argument("--x-index", type=int, default=None, help="0-based token index for X (non-header mode or preferred)")
    parser.add_argument("--y-index", type=int, default=None, help="0-based token index for Y (non-header mode or preferred)")
    parser.add_argument("--depth-ext", type=str, default=".npy", help="Extension for depth files (default: .npy)")
    parser.add_argument("--interp", choices=["nearest", "bilinear"], default="nearest", help="Interpolation method for sampling depth")
    # Binary SaveKeypoints support
    parser.add_argument("--binary-savekeypoints", action="store_true", help="Parse .dat files written with SaveKeypoints (binary)")
    parser.add_argument("--size-t", type=int, default=8, help="Size of C++ size_t in bytes when reading binary (4 or 8). Default 8")
    parser.add_argument("--endian", choices=["little", "big"], default="little", help="Endian of binary file (default: little)")
    parser.add_argument("--write-npy", action="store_true", help="Write per-keypoint depths to .npy files (default: True)")
    parser.add_argument("--write-csv", action="store_true", help="Write per-keypoint depths to CSV files alongside .npy")
    parser.add_argument("--embed", action="store_true", help="Embed depths into a copy of the binary .dat by appending a depth block (non-destructive unless --inplace)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.dat_file:
        dat_files = [args.dat_file]
        base_dat_dir = os.path.dirname(args.dat_file) or "."
    else:
        base_dat_dir = args.dat_dir
        dat_files = [os.path.join(base_dat_dir, f) for f in os.listdir(base_dat_dir) if f.lower().endswith(".dat")]

    if len(dat_files) == 0:
        print("No .dat files found to process.")
        sys.exit(1)

    out_dir = args.out_dir if args.out_dir else os.path.join(base_dat_dir, "augmented_dat")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {len(dat_files)} .dat files")

    summary = []
    for dat_path in dat_files:
        if args.verbose:
            print(f"\nProcessing {dat_path}")

        depth_file = find_depth_file_for_dat(dat_path, args.depth_dir, args.depth_ext)
        if depth_file is None:
            print(f"Warning: no depth file found for {dat_path} (expected basename match). Skipping.")
            continue

        if args.inplace:
            out_path = dat_path
            bak_path = dat_path + ".bak"
            if not os.path.exists(bak_path):
                shutil.copy2(dat_path, bak_path)
        else:
            out_path = os.path.join(out_dir, os.path.basename(dat_path))
        if args.binary_savekeypoints:
            # process binary SaveKeypoints format
            try:
                total_frames, total_kps, total_sampled = process_dat_file_binary(
                    dat_path,
                    depth_file,
                    out_dir,
                    size_t_bytes=args.size_t,
                    endian=args.endian,
                    interp=args.interp,
                    write_npy=(True if args.write_npy else False),
                    write_csv=args.write_csv,
                    embed=args.embed,
                    verbose=args.verbose,
                )
                summary.append((dat_path, depth_file, total_kps, total_sampled, out_path))
                print(f"Processed binary {dat_path} — frames: {total_frames}, keypoints: {total_kps}, depths sampled: {total_sampled}")
            except Exception as e:
                print(f"Error processing binary {dat_path}: {e}")
        else:
            total, sampled = process_dat_file(
                dat_path,
                depth_file,
                out_path,
                header=args.header,
                delim=args.delimiter,
                x_col=args.x_col,
                y_col=args.y_col,
                x_index=args.x_index,
                y_index=args.y_index,
                file_col=None,
                file_col_index=None,
                interp=args.interp,
            )
            summary.append((dat_path, depth_file, total, sampled, out_path))
            print(f"Wrote {out_path} — rows: {total}, depths sampled: {sampled}")

    print("\nSummary:")
    for datp, dfile, total, sampled, outp in summary:
        print(f"{os.path.basename(datp)} -> {os.path.basename(dfile) if dfile else 'NONE'} | rows {total}, depths {sampled} -> {outp}")


if __name__ == "__main__":
    main()
