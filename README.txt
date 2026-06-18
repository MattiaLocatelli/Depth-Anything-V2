Extract and analyze Keypoint Match data from ROS 2 bag files
Compare Pixel Distances of Keypoint Matches to Keypoint Depths from Depth-Anything-V2 model

Description
-----------
Script: scripts/extract_keypoint_matches.py

This script extracts and analyzes `KeypointMatch` messages contained in ROS 2
bag files (sqlite3 storage). It reads the bag in streaming mode, deserializes
messages, and writes results to a CSV stream to avoid excessive memory usage.
Optionally, it saves live frame images annotated with keypoints for inspection,
and can generate plots and filtered CSVs.


Main outputs
------------
- `keypoint_matches.csv`: one row per matched keypoint. Columns include:
  `live_frame_id`, `timestamp`, `camera_id`, `keyframe_idx`,
  `matched_keyframe_idx`, `match_id`, `live_kp_idx`, `kf_kp_idx`,
  `live_x`, `live_y`, `kf_x`, `kf_y`, `pixel_distance`, `match_score`,
  `is_inlier`, `total_matches`, `total_inliers`, `mean_distance`,
  `max_distance`, `live_frame_path`.
- `keypoint_frames/`: directory with saved images of live frames containing
  keypoints (if present in messages).


Second phase: keyframe depth maps and comparative analysis
---------------------------------------------------------
Goal
- Compute depth maps (relative and/or metric) for keyframes and live frames,
  sample depth values at matched keypoints, and analyze differences and
  correlations between relative and metric depths (in meters).

Pipeline (overview)
- Main script for this phase: `metric_depth/process_keypoint_frames.py`.
  - It can compute depth maps from images (`--live-frames`, `--keyframes`) or use precomputed maps (`--live-depths`, `--keyframe-depths`).
  - It supports using relative depth maps as a source for masks (`--relative-depths-live`, `--relative-depths-keyframe` or `--relative-depths`).
  - Important parameters: `--model-type` (base|metric) and `--max-depth`
  - When `--model-type metric` the script uses the metric implementation in `metric_depth/depth_anything_v2` (Sigmoid head multiplied by `max_depth`) to obtain meter-scaled outputs consistent with the checkpoint.
  - Map outputs: `<basename>_raw_depth_meter.npy`. If a mask is applied the script also saves a masked copy (`_raw_depth_meter_masked.npy`) or, for precomputed maps, `<basename>_masked.npy`.

Masking
- Use the precision of the relative depth model to mask the sky pixels (minimun relative depth) in the metric depth maps, which presents more noise in the sky portion
- Load the corresponding relative depth map (resized with `INTER_NEAREST` to the metric map resolution to preserve discrete mask boundaries).
- Compute `rel_min = nanmin(relative_map)` and build the boolean mask:
  `mask = (relative_map >= rel_max - mask_tol)`.
- Compute `metric_max = nanmax(metric_map[finite_values])` and set `metric_map[mask] = metric_max`.

Sampling and CSV
- After generating (or loading) the metric `.npy` maps, the script samples depth at each keypoint and writes an output CSV (`<input_csv>_with_depths.csv`) that adds the columns: `live_depth`, `keyframe_depth`, `depth_diff`, `notes`.
- This CSV is the input for comparative analysis and plotting.

Analysis and visualization
- Main analysis script: `plot_keypoint_analysis.py` — produces scatter plots and 3D density surfaces to compare:
  - Pixel distance vs Depth (live or keyframe)
  - 3D density surfaces

Outputs and naming conventions
- Computed maps: `<basename>_raw_depth_meter.npy`
- Masked maps (inference): `<basename>_raw_depth_meter_masked.npy`
- Masked maps (precomputed): `<basename>_masked.npy`
- Keypoint CSV with depths: `<input_csv>_with_depths.csv` (contains `live_depth`,
  `keyframe_depth`, `depth_diff`, `notes`)

