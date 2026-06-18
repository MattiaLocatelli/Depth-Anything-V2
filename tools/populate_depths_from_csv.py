#!/usr/bin/env python3
"""
Script to populate keypoint depths in .dat files from CSV data.

Usage:
    python3 populate_depths_from_csv.py <csv_file> <data_cam_folder> <output_folder>

Example:
    python3 populate_depths_from_csv.py keypoint_matches_filtered.csv ~/code/map/data_cam1/ ~/code/map/data_cam1/merged_kfs_with_depths/
"""

import struct
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd


def read_keyframe_data(filepath):
    """Read a keyframe .dat file and return the keyframe data."""
    with open(filepath, 'rb') as f:
        # Read type and idx
        type_val = struct.unpack('B', f.read(1))[0]
        closest_idx = struct.unpack('i', f.read(4))[0]
        
        # Read keypoints
        keypoints_size = struct.unpack('Q', f.read(8))[0]
        keypoints = []
        for _ in range(keypoints_size):
            x, y = struct.unpack('ff', f.read(8))
            keypoints.append((x, y))
        
        # Read descriptors
        descriptors_size = struct.unpack('Q', f.read(8))[0]
        descriptors = struct.unpack(f'{descriptors_size}f', f.read(descriptors_size * 4))
        
        # Detect format based on remaining bytes
        pos_start = f.tell()
        f.seek(0, 2)
        end = f.tell()
        f.seek(pos_start)
        
        # 36 (RotM) + 12 (Trans) = 48 bytes
        # If there's more than 48, it must have a depths_size field
        remaining = end - pos_start
        if remaining > 48:
            depths_size = struct.unpack('Q', f.read(8))[0]
            depths = list(struct.unpack(f'{depths_size}f', f.read(depths_size * 4)))
        else:
            depths = []
        
        # Read rotation matrix (3x3)
        rotm = list(struct.unpack('9f', f.read(36)))
        
        # Read translation (3 floats)
        translation = list(struct.unpack('3f', f.read(12)))
    
    return {
        'type': type_val,
        'closest_idx': closest_idx,
        'keypoints': keypoints,
        'descriptors': descriptors,
        'depths': depths,
        'rotm': rotm,
        'translation': translation
    }


def write_keyframe_data(filepath, data):
    """Write keyframe data to a .dat file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        # Write type and idx
        f.write(struct.pack('B', data['type']))
        f.write(struct.pack('i', data['closest_idx']))
        
        # Write keypoints
        f.write(struct.pack('Q', len(data['keypoints'])))
        for x, y in data['keypoints']:
            f.write(struct.pack('ff', x, y))
        
        # Write descriptors
        f.write(struct.pack('Q', len(data['descriptors'])))
        f.write(struct.pack(f'{len(data["descriptors"])}f', *data['descriptors']))
        
        # Write depths
        f.write(struct.pack('Q', len(data['depths'])))
        f.write(struct.pack(f'{len(data["depths"])}f', *data['depths']))
        
        # Write rotation matrix
        f.write(struct.pack('9f', *data['rotm']))
        
        # Write translation
        f.write(struct.pack('3f', *data['translation']))


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 populate_depths_from_csv.py <csv_file> <data_cam_folder> <output_folder> <npy_folder>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    data_cam_folder = sys.argv[2]
    output_folder = sys.argv[3]
    npy_folder = Path(sys.argv[4])
    
    # Load and update CSV
    print(f"Reading and updating CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    
    type_map = {0: 'L', 1: 'C', 2: 'R'}
    # Helper to map type characters back to 0, 1, 2 if needed for .dat file logic
    
    # Cache for npy files
    npy_cache = {}

    # Update CSV with depths from npy files
    print("Updating CSV depths from npy files...")
    # Add a column for depth if it doesn't exist
    if 'depth' not in df.columns:
        df['depth'] = 0.0
        
    for idx, row in df.iterrows():
        kf_idx = row['keyframe_idx']
        kf_type_char = type_map.get(row['kf_type'], 'C')
        

        filename = f"{kf_type_char}{kf_idx}_raw_depth_meter_masked.npy"
        
        if filename not in npy_cache:
            npy_path = npy_folder / filename
            npy_cache[filename] = np.load(npy_path) if npy_path.exists() else None
            
        depth_map = npy_cache[filename]
        
        if depth_map is not None:
            iy, ix = int(round(row['y'])), int(round(row['x']))
            if 0 <= iy < depth_map.shape[0] and 0 <= ix < depth_map.shape[1]:
                depth_val = float(depth_map[iy, ix])
                df.at[idx, 'depth'] = depth_val
                # print(f"DEBUG: Setting depth for {filename} at ({ix}, {iy}) to {depth_val}")

    
    updated_csv_path = csv_file.replace('.csv', '_updated.csv')
    df.to_csv(updated_csv_path, index=False)
    print(f"Updated CSV saved as {updated_csv_path}")
    
    # Find all .dat files in data_cam_folder
    data_cam_path = Path(data_cam_folder)
    dat_files = list(data_cam_path.rglob("*.dat"))
    
    if not dat_files:
        print(f"No .dat files found in {data_cam_folder}")
        sys.exit(1)
    
    print(f"Found {len(dat_files)} .dat files")
    
    # Process each file to update .dat with depths
    updated_count = 0
    for dat_file in dat_files:
        try:
            kf_data = read_keyframe_data(str(dat_file))
            kf_idx = kf_data['closest_idx']
            kf_type = type_map.get(kf_data['type'], 'C')
            filename = f"{kf_type}{kf_idx}_raw_depth_meter_masked.npy"
            depth_map = npy_cache.get(filename)
            
            # Populate depths
            depths_array = []
            if depth_map is not None:
                for x, y in kf_data['keypoints']:
                    iy, ix = int(round(y)), int(round(x))
                    if 0 <= iy < depth_map.shape[0] and 0 <= ix < depth_map.shape[1]:
                        depths_array.append(float(depth_map[iy, ix]))
                    else:
                        depths_array.append(0.0)
            else:
                depths_array = [0.0] * len(kf_data['keypoints'])
            
            kf_data['depths'] = depths_array
            
            # Write updated file to output
            rel_path = dat_file.relative_to(data_cam_path)
            output_file = Path(output_folder) / rel_path
            write_keyframe_data(str(output_file), kf_data)
            updated_count += 1
            
            if updated_count % 10 == 0:
                print(f"Processed {updated_count} keyframes...")
        
        except Exception as e:
            print(f"Error processing {dat_file}: {e}")
            continue
    
    print(f"\nSuccessfully processed {updated_count} keyframes")
    print(f"Output saved to: {output_folder}")


if __name__ == "__main__":
    main()
