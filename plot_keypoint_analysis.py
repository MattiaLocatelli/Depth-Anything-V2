import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def get_output_path(output_dir, filename):
    """
    Helper function to get output path.
    If output_dir is provided, saves to that directory.
    Otherwise saves to current working directory.
    
    Args:
        output_dir: Output directory (can be None)
        filename: Filename for the plot (without path)
    
    Returns:
        Full output path
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    else:
        return filename


def calculate_r_squared(y_true, y_pred):
    """
    Calculate R² (coefficient of determination).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        R² value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def plot_depth_vs_pixel_distance(csv_path, output_dir=None, use_keyframe=False, poly_degree=2, show_linear=False, show_trend=False, show_stats=False):
    """
    Create scatter plot with inverted axes: Depth on X-axis, Pixel Distance on Y-axis.
    Shows distribution of pixel distances across depth values.
    
    Args:
        csv_path: Path to CSV with keypoint matches and depths
        output_dir: Directory to save the plot (default: current directory)
        use_keyframe: If True, use keyframe_depth; else use live_depth
        poly_degree: Degree of polynomial trend line (default: 2)
    """
    
    # Read CSV
    pixel_distances = []
    depths = []
    valid_rows = 0
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        depth_col = 'keyframe_depth' if use_keyframe else 'live_depth'
        
        for row in reader:
            try:
                # Get pixel distance
                if 'pixel_distance' not in row or not row['pixel_distance']:
                    continue
                pixel_dist = float(row['pixel_distance'])
                
                # Get depth
                if depth_col not in row or not row[depth_col]:
                    continue
                depth = float(row[depth_col])
                
                pixel_distances.append(pixel_dist)
                depths.append(depth)
                valid_rows += 1
            except (ValueError, KeyError):
                continue
    
    if valid_rows == 0:
        print(f"Error: No valid rows with pixel_distance and {depth_col}")
        return
    
    print(f"Loaded {valid_rows} valid data points for inverted plot")
    
    # Convert to numpy arrays
    pixel_distances = np.array(pixel_distances)
    depths = np.array(depths)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Compute point density for coloring (density of points in depth vs pixel_distance)
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([depths, pixel_distances])
        kde = gaussian_kde(xy)
        densities = kde(xy)
        # plot points sorted by density so that dense points are plotted on top
        idx = densities.argsort()
        depths_s = depths[idx]
        pixel_dist_s = pixel_distances[idx]
        dens_s = densities[idx]
        scatter = ax.scatter(depths_s, pixel_dist_s, c=dens_s, s=20, cmap='viridis')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Point density', fontsize=11)
    except Exception:
        # fallback: color by pixel distance
        scatter = ax.scatter(depths, pixel_distances, alpha=0.5, s=20, c=pixel_distances, cmap='viridis')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pixel Distance', fontsize=11)

    # Add vertical lines marking depth quartiles (25%, 50%, 75%) and annotate
    try:
        quartiles = np.percentile(depths, [25, 50, 75])
        # small horizontal offset to place the label to the right of the line (in data units)
        dx = (depths.max() - depths.min()) * 0.01 if depths.max() != depths.min() else 1e-6
        for q_val, pct in zip(quartiles, [25, 50, 75]):
            ax.axvline(q_val, color='#ff0000', linestyle='--', linewidth=1.2, alpha=0.95)
            # label with percentile and numeric depth value (e.g. '25%\n0.123')
            label = f"{pct}%\n{q_val:.1f}"
            if pct == 25:
                text_x = q_val - dx
                ha = 'right'
            else:
                text_x = q_val + dx
                ha = 'left'
            ax.text(text_x, 0.01, label, transform=ax.get_xaxis_transform(), ha=ha, va='top', fontsize=9, color='#ff0000')
    except Exception:
        pass

    # Polynomial trend (optional)
    if show_trend:
        z = np.polyfit(depths, pixel_distances, poly_degree)
        p = np.poly1d(z)
        x_smooth = np.linspace(depths.min(), depths.max(), 100)
        y_smooth = p(x_smooth)
        r2_poly = calculate_r_squared(pixel_distances, p(depths))
        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Trend (degree {poly_degree}, R²={r2_poly:.4f})')

    # Linear fit for reference (optional)
    r_value = None
    if show_linear:
        slope, intercept, r_value, p_value, std_err = stats.linregress(depths, pixel_distances)
        y_linear = slope * depths + intercept
        ax.plot(depths, y_linear, 'g--', linewidth=1.5, alpha=0.7, label=f'Linear fit (R²={r_value**2:.4f})')
    
    # Labels and title
    depth_type = "Keyframe" if use_keyframe else "Live Frame"
    ax.set_xlabel(f'{depth_type} Depth (Relative)', fontsize=12)
    ax.set_ylabel('Pixel Distance (pixels)', fontsize=12)
    ax.set_title(f'Pixel Distance vs {depth_type} Depth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if show_trend or show_linear:
        ax.legend(fontsize=11)
    
    # Set X-axis to start from 0
    ax.set_xlim(left=0)
    
    # Statistics box (optional)
    if show_stats:
        stats_lines = [
            f"Samples: {valid_rows}",
            f"{depth_type} Depth - Min: {depths.min():.4f}, Max: {depths.max():.4f}, Mean: {depths.mean():.4f}",
            f"Pixel Dist - Min: {pixel_distances.min():.2f}, Max: {pixel_distances.max():.2f}, Mean: {pixel_distances.mean():.2f}",
        ]
        if show_linear and (r_value is not None):
            stats_lines.append(f"Linear R²: {r_value**2:.4f}")
        stats_lines.append(f"Correlation: {np.corrcoef(depths, pixel_distances)[0, 1]:.4f}")
        stats_text = "Statistics:\n" + "\n".join(stats_lines)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
    
    plt.tight_layout()
    
    # Save
    depth_type_short = 'keyframe' if use_keyframe else 'live'
    filename = f'pixel_distance_vs_{depth_type_short}_depth.png'
    output_path = get_output_path(output_dir, filename)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")


def plot_depth_diff_vs_pixel_distance_density_3d(csv_path, output_dir=None, show_stats=False):
    """
    3D density surface for Depth Difference vs Pixel Distance.
    X-axis: Depth Difference |live_depth - keyframe_depth|
    Y-axis: Pixel Distance
    Z-axis: Density
    """
    from scipy.stats import gaussian_kde
    from mpl_toolkits.mplot3d import Axes3D

    pixel_distances = []
    depth_diffs = []
    valid_rows = 0

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if 'pixel_distance' not in row or not row['pixel_distance']:
                    continue
                if 'depth_diff' not in row or not row['depth_diff']:
                    continue
                pixel_dist = float(row['pixel_distance'])
                depth_diff = float(row['depth_diff'])
                pixel_distances.append(pixel_dist)
                depth_diffs.append(depth_diff)
                valid_rows += 1
            except (ValueError, KeyError):
                continue

    if valid_rows == 0:
        print("Note: No valid data for depth_diff 3D plot")
        return

    print(f"Creating 3D density surface plot (depth_diff) with {valid_rows} data points")

    pixel_distances = np.array(pixel_distances)
    depth_diffs = np.array(depth_diffs)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = depth_diffs.min(), depth_diffs.max()
    y_min, y_max = pixel_distances.min(), pixel_distances.max()
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    values = np.vstack([depth_diffs, pixel_distances])
    kernel = gaussian_kde(values)
    zz = np.reshape(kernel(positions).T, xx.shape)

    surf = ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.85, edgecolor='none', antialiased=True)
    ax.scatter(depth_diffs, pixel_distances, 0, alpha=0.3, s=10, c='red', label='Data points')

    ax.set_xlabel('Depth Difference |live_depth - keyframe_depth|', fontsize=11)
    ax.set_ylabel('Pixel Distance (pixels)', fontsize=11)
    ax.set_zlabel('Density', fontsize=11)
    ax.set_title('3D Density Surface: Depth Difference vs Pixel Distance', fontsize=13, fontweight='bold')

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Density', fontsize=10)

    if show_stats:
        stats_text = f'''Samples: {valid_rows}\nCorrelation: {np.corrcoef(depth_diffs, pixel_distances)[0, 1]:.4f}'''
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                  family='monospace')

    ax.view_init(elev=25, azim=45)
    plt.tight_layout()

    filename = 'depth_diff_vs_pixel_distance_density_3d.png'
    output_path = get_output_path(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 3D density surface plot (depth_diff) saved to: {output_path}")

def plot_depth_diff_vs_pixel_distance(csv_path, output_dir=None, show_linear=False, show_trend=False, show_stats=False, poly_degree=2):
    """
    Create scatter plot: depth difference on X-axis, pixel distance on Y-axis.
    Optional: linear fit, polynomial trend, statistics box (controlled via args).
    """
    pixel_distances = []
    depth_diffs = []
    valid_rows = 0
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if 'pixel_distance' not in row or not row['pixel_distance']:
                    continue
                if 'depth_diff' not in row or not row['depth_diff']:
                    continue
                pixel_dist = float(row['pixel_distance'])
                depth_diff = float(row['depth_diff'])
                pixel_distances.append(pixel_dist)
                depth_diffs.append(depth_diff)
                valid_rows += 1
            except (ValueError, KeyError):
                continue
    if valid_rows == 0:
        print("Note: No valid data for depth_diff plot")
        return
    pixel_distances = np.array(pixel_distances)
    depth_diffs = np.array(depth_diffs)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Compute point density for coloring (depth_diff vs pixel_distance)
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([depth_diffs, pixel_distances])
        kde = gaussian_kde(xy)
        densities = kde(xy)
        idx = densities.argsort()
        dd_s = depth_diffs[idx]
        pd_s = pixel_distances[idx]
        dens_s = densities[idx]
        scatter = ax.scatter(dd_s, pd_s, c=dens_s, s=20, cmap='viridis')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Point density', fontsize=11)
    except Exception:
        scatter = ax.scatter(depth_diffs, pixel_distances, alpha=0.5, s=20, c=pixel_distances, cmap='viridis')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pixel Distance', fontsize=11)

    # Add vertical lines marking depth-diff quartiles (25%, 50%, 75%) and annotate
    try:
        quartiles = np.percentile(depth_diffs, [25, 50, 75])
        dx = (depth_diffs.max() - depth_diffs.min()) * 0.01 if depth_diffs.max() != depth_diffs.min() else 1e-6
        for q_val, pct in zip(quartiles, [25, 50, 75]):
            ax.axvline(q_val, color='#ff0000', linestyle='--', linewidth=1.2, alpha=0.95)
            # label with percentile and numeric depth-diff value
            label = f"{pct}%\n{q_val:.1f}"
            if pct == 25:
                text_x = q_val - dx
                ha = 'right'
            else:
                text_x = q_val + dx
                ha = 'left'
            ax.text(text_x, 0.01, label, transform=ax.get_xaxis_transform(), ha=ha, va='top', fontsize=7, color='#ff0000')
    except Exception:
        pass

    # Polynomial trend (optional)
    if show_trend:
        z = np.polyfit(depth_diffs, pixel_distances, poly_degree)
        p = np.poly1d(z)
        x_smooth = np.linspace(depth_diffs.min(), depth_diffs.max(), 200)
        y_smooth = p(x_smooth)
        r2_poly = calculate_r_squared(pixel_distances, p(depth_diffs))
        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Trend (deg {poly_degree}, R²={r2_poly:.4f})')

    # Linear fit (optional)
    if show_linear:
        slope, intercept, r_value, p_value, std_err = stats.linregress(depth_diffs, pixel_distances)
        x_vals = np.array([depth_diffs.min(), depth_diffs.max()])
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, 'g--', linewidth=1.5, alpha=0.8, label=f'Linear fit (R²={r_value**2:.4f})')

    # Labels and layout
    ax.set_xlabel('Depth Difference |live_depth - keyframe_depth|', fontsize=12)
    ax.set_ylabel('Pixel Distance (pixels)', fontsize=12)
    ax.set_title('Pixel Distance vs Depth Difference', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if show_trend or show_linear:
        ax.legend(fontsize=11)

    # Statistics box (optional)
    if show_stats:
        stats_text = f'''Statistics:\nSamples: {valid_rows}\nDepth Diff - Min: {depth_diffs.min():.4f}, Max: {depth_diffs.max():.4f}, Mean: {depth_diffs.mean():.4f}\nPixel Dist - Min: {pixel_distances.min():.2f}, Max: {pixel_distances.max():.2f}, Mean: {pixel_distances.mean():.2f}\nCorrelation: {np.corrcoef(depth_diffs, pixel_distances)[0, 1]:.4f}'''
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='monospace')

    plt.tight_layout()
    filename = 'depth_difference_vs_pixel_distance.png'
    output_path = get_output_path(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")


def plot_depth_vs_pixel_distance_density_3d(csv_path, output_dir=None, use_keyframe=False, show_stats=False):
    """
    Create 3D surface plot showing distribution density.
    X-axis: Depth (live or keyframe)
    Y-axis: Pixel Distance
    Z-axis: Density (height of the surface)
    """
    from scipy.stats import gaussian_kde
    from mpl_toolkits.mplot3d import Axes3D
    
    pixel_distances = []
    depths = []
    valid_rows = 0
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        depth_col = 'keyframe_depth' if use_keyframe else 'live_depth'
        
        for row in reader:
            try:
                if 'pixel_distance' not in row or not row['pixel_distance']:
                    continue
                pixel_dist = float(row['pixel_distance'])
                
                if depth_col not in row or not row[depth_col]:
                    continue
                depth = float(row[depth_col])
                
                pixel_distances.append(pixel_dist)
                depths.append(depth)
                valid_rows += 1
            except (ValueError, KeyError):
                continue
    
    if valid_rows == 0:
        print("Note: No valid data for 3D density plot")
        return
    
    print(f"Creating 3D density surface plot with {valid_rows} data points")
    
    pixel_distances = np.array(pixel_distances)
    depths = np.array(depths)
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 2D grid for density estimation
    x_min, x_max = depths.min(), depths.max()
    y_min, y_max = pixel_distances.min(), pixel_distances.max()
    
    # Grid resolution
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # 2D KDE
    values = np.vstack([depths, pixel_distances])
    kernel = gaussian_kde(values)
    zz = np.reshape(kernel(positions).T, xx.shape)
    
    # Plot 3D surface
    surf = ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.85, edgecolor='none', antialiased=True)
    
    # Scatter plot overlay (optional, shows raw data points projected on bottom)
    ax.scatter(depths, pixel_distances, 0, alpha=0.3, s=10, c='red', label='Data points')
    
    # Labels and title
    depth_type = "Keyframe" if use_keyframe else "Live"
    ax.set_xlabel(f'{depth_type} Depth', fontsize=11)
    ax.set_ylabel('Pixel Distance (pixels)', fontsize=11)
    ax.set_zlabel('Density', fontsize=11)
    ax.set_title(f'3D Density Surface: {depth_type} Depth vs Pixel Distance', fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Density', fontsize=10)
    
    # Statistics box (optional)
    if show_stats:
        stats_text = f'''Samples: {valid_rows}
Correlation: {np.corrcoef(depths, pixel_distances)[0, 1]:.4f}'''
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                  family='monospace')
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    # Save
    depth_suffix = 'keyframe' if use_keyframe else 'live'
    filename = f'{depth_suffix}_depth_vs_pixel_distance_density_3d.png'
    output_path = get_output_path(output_dir, filename)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 3D density surface plot saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze keypoint depth relationships with scatter plots and 3D density visualization'
    )
    
    parser.add_argument('csv_path', type=str, help='CSV with keypoint matches and depths')
    parser.add_argument('--output', type=str, default=None, help='output directory for plots (default: current directory)')
    parser.add_argument('--use-keyframe', dest='use_keyframe', action='store_true',
                       help='use keyframe_depth instead of live_depth')
    parser.add_argument('--poly-degree', type=int, default=2,
                       help='polynomial degree for trend line (default: 2). Use higher values for more complex curves')
    parser.add_argument('--3d', dest='plot_3d', action='store_true',
                       help='generate 3D density surface plot instead of scatter plot')
    parser.add_argument('--with-fits', dest='with_fits', action='store_true',
                       help='Enable linear fit, polynomial trend and statistics across supported plots')
    
    args = parser.parse_args()
    
    # Generate scatter plot with depth on X-axis and pixel distance on Y-axis
    if args.plot_3d:
        plot_depth_vs_pixel_distance_density_3d(args.csv_path, args.output, args.use_keyframe, show_stats=args.with_fits)
    else:
        plot_depth_vs_pixel_distance(args.csv_path, args.output, args.use_keyframe, args.poly_degree,
                                     show_linear=args.with_fits, show_trend=args.with_fits, show_stats=args.with_fits)

# Generate depth-diff plot (2D or 3D depending on --3d)
if args.plot_3d:
    plot_depth_diff_vs_pixel_distance_density_3d(args.csv_path, args.output, show_stats=args.with_fits)
else:
    plot_depth_diff_vs_pixel_distance(args.csv_path, args.output,
                                      show_linear=args.with_fits,
                                      show_trend=args.with_fits,
                                      show_stats=args.with_fits,
                                      poly_degree=args.poly_degree)
