"""Compare SfM initialization points with trained 3DGS gaussians."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

from .utils import write_ply, colormap_by_value


def load_ply_points(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load point positions and colors from a PLY file.

    Args:
        ply_path: Path to PLY file

    Returns:
        Tuple of (positions, colors) where colors may be None
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    positions = np.stack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ], axis=1)

    colors = None
    if 'red' in vertex.data.dtype.names:
        colors = np.stack([
            vertex['red'],
            vertex['green'],
            vertex['blue']
        ], axis=1)

    return positions, colors


def compare_sfm_vs_3dgs(
    sfm_ply_path: Path,
    dgs_ply_path: Path,
    output_dir: Path
) -> dict:
    """
    Compare SfM initialization points with trained 3DGS gaussians.

    Args:
        sfm_ply_path: Path to SfM point cloud PLY
        dgs_ply_path: Path to 3DGS splat PLY
        output_dir: Output directory for visualizations

    Returns:
        Summary statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SfM points...")
    sfm_positions, sfm_colors = load_ply_points(sfm_ply_path)
    print(f"  {len(sfm_positions)} points")

    print("Loading 3DGS points...")
    dgs_positions, dgs_colors = load_ply_points(dgs_ply_path)
    print(f"  {len(dgs_positions)} points")

    stats = {
        "sfm_points": len(sfm_positions),
        "dgs_points": len(dgs_positions),
        "growth_factor": len(dgs_positions) / len(sfm_positions) if len(sfm_positions) > 0 else 0,
    }

    # Compute nearest neighbor distances from SfM to 3DGS
    print("Computing SfM -> 3DGS distances...")
    from scipy.spatial import cKDTree

    dgs_tree = cKDTree(dgs_positions)
    sfm_to_dgs_dist, _ = dgs_tree.query(sfm_positions, k=1)

    stats["sfm_to_dgs"] = {
        "min": float(sfm_to_dgs_dist.min()),
        "max": float(sfm_to_dgs_dist.max()),
        "mean": float(sfm_to_dgs_dist.mean()),
        "median": float(np.median(sfm_to_dgs_dist)),
        "std": float(sfm_to_dgs_dist.std()),
    }

    # Compute nearest neighbor distances from 3DGS to SfM
    print("Computing 3DGS -> SfM distances...")
    sfm_tree = cKDTree(sfm_positions)
    dgs_to_sfm_dist, _ = sfm_tree.query(dgs_positions, k=1)

    stats["dgs_to_sfm"] = {
        "min": float(dgs_to_sfm_dist.min()),
        "max": float(dgs_to_sfm_dist.max()),
        "mean": float(dgs_to_sfm_dist.mean()),
        "median": float(np.median(dgs_to_sfm_dist)),
        "std": float(dgs_to_sfm_dist.std()),
    }

    # Generate histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SfM -> 3DGS histogram
    ax = axes[0]
    ax.hist(sfm_to_dgs_dist, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Number of SfM Points")
    ax.set_title("SfM Point Distance to Nearest 3DGS Gaussian")

    p95 = np.percentile(sfm_to_dgs_dist, 95)
    ax.axvline(p95, color='r', linestyle='--', label=f'95th pct: {p95:.3f}m')
    ax.legend()

    # 3DGS -> SfM histogram
    ax = axes[1]
    ax.hist(dgs_to_sfm_dist, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Number of 3DGS Gaussians")
    ax.set_title("3DGS Gaussian Distance to Nearest SfM Point")

    p95 = np.percentile(dgs_to_sfm_dist, 95)
    ax.axvline(p95, color='r', linestyle='--', label=f'95th pct: {p95:.3f}m')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "distance_histogram.png", dpi=150)
    plt.close()

    # Color SfM points by distance to nearest 3DGS
    sfm_distance_colors = colormap_by_value(
        sfm_to_dgs_dist,
        vmin=0,
        vmax=np.percentile(sfm_to_dgs_dist, 95),
        cmap="RdYlGn_r"  # Green=close, Red=far
    )
    write_ply(output_dir / "sfm_by_dgs_distance.ply", sfm_positions, sfm_distance_colors)

    # Color 3DGS points by distance to nearest SfM
    dgs_distance_colors = colormap_by_value(
        dgs_to_sfm_dist,
        vmin=0,
        vmax=np.percentile(dgs_to_sfm_dist, 95),
        cmap="RdYlGn_r"  # Green=close, Red=far
    )
    write_ply(output_dir / "dgs_by_sfm_distance.ply", dgs_positions, dgs_distance_colors)

    # Identify regions where 3DGS diverged significantly
    divergence_threshold = np.percentile(dgs_to_sfm_dist, 90)
    diverged_mask = dgs_to_sfm_dist > divergence_threshold
    if diverged_mask.sum() > 0:
        write_ply(
            output_dir / "dgs_diverged.ply",
            dgs_positions[diverged_mask],
            np.full((diverged_mask.sum(), 3), [255, 0, 0], dtype=np.uint8)
        )
        stats["diverged_points"] = int(diverged_mask.sum())
        stats["divergence_threshold"] = float(divergence_threshold)
        print(f"Exported {diverged_mask.sum()} diverged 3DGS points (>{divergence_threshold:.3f}m from SfM)")

    # Create combined overlay PLY for visualization
    # SfM points in blue, 3DGS points in red
    combined_positions = np.vstack([sfm_positions, dgs_positions])
    combined_colors = np.vstack([
        np.full((len(sfm_positions), 3), [0, 0, 255], dtype=np.uint8),  # Blue for SfM
        np.full((len(dgs_positions), 3), [255, 0, 0], dtype=np.uint8),  # Red for 3DGS
    ])
    write_ply(output_dir / "sfm_3dgs_overlay.ply", combined_positions, combined_colors)

    # Print summary
    print(f"\nComparison Summary:")
    print(f"  SfM points: {stats['sfm_points']}")
    print(f"  3DGS points: {stats['dgs_points']} ({stats['growth_factor']:.1f}x growth)")
    print(f"  SfM -> 3DGS distance: mean={stats['sfm_to_dgs']['mean']:.4f}m, "
          f"median={stats['sfm_to_dgs']['median']:.4f}m")
    print(f"  3DGS -> SfM distance: mean={stats['dgs_to_sfm']['mean']:.4f}m, "
          f"median={stats['dgs_to_sfm']['median']:.4f}m")

    if stats["dgs_to_sfm"]["mean"] > 0.1:
        print(f"\n⚠️  WARNING: High average distance from 3DGS to SfM ({stats['dgs_to_sfm']['mean']:.3f}m)")
        print("   3DGS may have diverged significantly from initialization.")

    return stats


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compare SfM vs 3DGS")
    parser.add_argument("sfm_ply", type=Path, help="Path to SfM point cloud PLY")
    parser.add_argument("dgs_ply", type=Path, help="Path to 3DGS splat PLY")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: sfm_ply.parent/debug/comparison)")

    args = parser.parse_args()

    output_dir = args.output or args.sfm_ply.parent / "debug" / "comparison"
    stats = compare_sfm_vs_3dgs(args.sfm_ply, args.dgs_ply, output_dir)

    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
