"""Analyze and visualize SfM point quality via reprojection error."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    load_reconstruction,
    write_ply,
    colormap_by_value,
)


def analyze_sfm_quality(
    dataset_dir: Path,
    output_dir: Path
) -> dict:
    """
    Analyze SfM reconstruction quality.

    Args:
        dataset_dir: Path to OpenSfM dataset directory
        output_dir: Output directory for visualizations

    Returns:
        Summary statistics
    """
    reconstruction_path = dataset_dir / "reconstruction.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reconstruction...")
    recon = load_reconstruction(reconstruction_path)

    points = recon.get("points", {})
    shots = recon.get("shots", {})
    cameras = recon.get("cameras", {})

    print(f"Points: {len(points)}, Shots: {len(shots)}, Cameras: {len(cameras)}")

    # Collect point data
    positions = []
    colors = []
    reproj_errors = []

    for point_id, point_data in points.items():
        coords = point_data.get("coordinates", [0, 0, 0])
        color = point_data.get("color", [128, 128, 128])
        reproj_error = point_data.get("reprojection_error")

        positions.append(coords)
        colors.append(color)
        reproj_errors.append(reproj_error if reproj_error is not None else np.nan)

    positions = np.array(positions)
    colors = np.array(colors, dtype=np.uint8)
    reproj_errors = np.array(reproj_errors)

    # Filter out NaN errors
    valid_mask = ~np.isnan(reproj_errors)
    valid_errors = reproj_errors[valid_mask]

    stats = {
        "total_points": len(points),
        "points_with_error": int(valid_mask.sum()),
    }

    if len(valid_errors) > 0:
        stats["reprojection_error"] = {
            "min": float(valid_errors.min()),
            "max": float(valid_errors.max()),
            "mean": float(valid_errors.mean()),
            "median": float(np.median(valid_errors)),
            "std": float(valid_errors.std()),
        }

        # Generate histogram
        plt.figure(figsize=(10, 6))

        # Use log scale for x-axis if range is large
        if valid_errors.max() / (valid_errors.min() + 1e-10) > 100:
            plt.hist(np.log10(valid_errors + 1e-6), bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel("log10(Reprojection Error)")
        else:
            plt.hist(valid_errors, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel("Reprojection Error (pixels)")

        plt.ylabel("Number of Points")
        plt.title("Reprojection Error Distribution")

        # Add statistics text
        stats_text = (f"Total points: {len(points)}\n"
                      f"Min error: {valid_errors.min():.3f}\n"
                      f"Max error: {valid_errors.max():.3f}\n"
                      f"Mean error: {valid_errors.mean():.3f}\n"
                      f"Median error: {np.median(valid_errors):.3f}")
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / "reproj_error_histogram.png", dpi=150)
        plt.close()

        # Color points by reprojection error
        error_colors = colormap_by_value(
            reproj_errors,
            vmin=np.nanpercentile(reproj_errors, 5),
            vmax=np.nanpercentile(reproj_errors, 95),
            cmap="RdYlGn_r"  # Green=low error, Red=high error
        )

        # Replace NaN colors with gray
        nan_mask = np.isnan(reproj_errors)
        error_colors[nan_mask] = [128, 128, 128]

        write_ply(output_dir / "points_by_reproj_error.ply", positions, error_colors)

        # Export low-error points only (good quality)
        low_error_threshold = np.nanpercentile(reproj_errors, 50)
        low_error_mask = reproj_errors < low_error_threshold
        if low_error_mask.sum() > 0:
            write_ply(
                output_dir / "points_low_error.ply",
                positions[low_error_mask],
                colors[low_error_mask]
            )
            print(f"Exported {low_error_mask.sum()} low-error points (< {low_error_threshold:.3f})")

        # Export high-error points (suspicious)
        high_error_threshold = np.nanpercentile(reproj_errors, 95)
        high_error_mask = reproj_errors > high_error_threshold
        if high_error_mask.sum() > 0:
            write_ply(
                output_dir / "points_high_error.ply",
                positions[high_error_mask],
                np.full((high_error_mask.sum(), 3), [255, 0, 0], dtype=np.uint8)
            )
            print(f"Exported {high_error_mask.sum()} high-error points (> {high_error_threshold:.3f})")

    else:
        print("No reprojection error data available in reconstruction")

    # Analyze point distribution
    if len(positions) > 0:
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)

        stats["spatial_distribution"] = {
            "centroid": centroid.tolist(),
            "distance_to_centroid": {
                "min": float(distances.min()),
                "max": float(distances.max()),
                "mean": float(distances.mean()),
                "std": float(distances.std()),
            }
        }

        # Distance histogram
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel("Distance from Centroid (meters)")
        plt.ylabel("Number of Points")
        plt.title("Point Distance Distribution")

        # Mark outlier threshold
        p95 = np.percentile(distances, 95)
        plt.axvline(p95, color='r', linestyle='--', label=f'95th percentile: {p95:.1f}m')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "distance_histogram.png", dpi=150)
        plt.close()

    return stats


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze SfM quality")
    parser.add_argument("dataset_dir", type=Path, help="OpenSfM dataset directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: dataset_dir/../debug/sfm)")

    args = parser.parse_args()

    output_dir = args.output or args.dataset_dir.parent / "debug" / "sfm"
    stats = analyze_sfm_quality(args.dataset_dir, output_dir)

    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
