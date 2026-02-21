"""Analyze and visualize camera poses and distribution."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    load_reconstruction,
    rotation_from_angle_axis,
    get_camera_center,
    write_ply,
)


def analyze_cameras(
    dataset_dir: Path,
    output_dir: Path
) -> dict:
    """
    Analyze camera pose distribution and quality.

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

    shots = recon.get("shots", {})

    if not shots:
        print("No camera shots found")
        return {"error": "no shots"}

    # Extract camera positions and orientations
    shot_names = sorted(shots.keys())
    positions = []
    orientations = []  # Forward direction vectors

    for shot_name in shot_names:
        shot = shots[shot_name]
        center = get_camera_center(shot)
        positions.append(center)

        # Get forward direction (z-axis in camera frame)
        R = rotation_from_angle_axis(shot.get("rotation", [0, 0, 0]))
        forward = R.T[:, 2]  # Third column of R^T
        orientations.append(forward)

    positions = np.array(positions)
    orientations = np.array(orientations)

    # Compute centroid and radius
    centroid = positions.mean(axis=0)
    radii = np.linalg.norm(positions - centroid, axis=1)

    # Compute baselines between adjacent cameras
    baselines = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    # Compute loop closure distance
    loop_closure_dist = np.linalg.norm(positions[-1] - positions[0])
    path_length = baselines.sum()

    stats = {
        "n_cameras": len(positions),
        "centroid": centroid.tolist(),
        "radii": {
            "min": float(radii.min()),
            "max": float(radii.max()),
            "mean": float(radii.mean()),
            "std": float(radii.std()),
        },
        "baselines": {
            "min": float(baselines.min()),
            "max": float(baselines.max()),
            "mean": float(baselines.mean()),
            "std": float(baselines.std()),
        },
        "path_length": float(path_length),
        "loop_closure_distance": float(loop_closure_dist),
        "is_circular": bool(loop_closure_dist < path_length * 0.3),
    }

    # Print summary
    print(f"\nCamera Analysis:")
    print(f"  Cameras: {stats['n_cameras']}")
    print(f"  Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    print(f"  Radius: {radii.min():.2f}m - {radii.max():.2f}m (std: {radii.std():.2f}m)")
    print(f"  Baseline: {baselines.min():.3f}m - {baselines.max():.3f}m (mean: {baselines.mean():.3f}m)")
    print(f"  Path length: {path_length:.2f}m")
    print(f"  Loop closure: {loop_closure_dist:.2f}m")
    print(f"  Path type: {'CIRCULAR' if stats['is_circular'] else 'LINEAR'}")

    # Warnings
    if baselines.mean() < 0.01:
        print(f"\n⚠️  WARNING: Very small baseline ({baselines.mean()*100:.1f}cm)")
        print("   This may cause poor depth estimation.")

    if radii.std() > radii.mean() * 0.3:
        print(f"\n⚠️  WARNING: Inconsistent camera distances from centroid")
        print(f"   Radius std ({radii.std():.2f}m) is >30% of mean ({radii.mean():.2f}m)")

    if stats["is_circular"] and loop_closure_dist > baselines.mean() * 3:
        print(f"\n⚠️  WARNING: Poor loop closure")
        print(f"   Gap ({loop_closure_dist:.2f}m) is {loop_closure_dist/baselines.mean():.1f}x baseline")

    # Generate baseline histogram
    plt.figure(figsize=(10, 6))
    plt.hist(baselines, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Baseline Distance (meters)")
    plt.ylabel("Count")
    plt.title("Inter-Camera Baseline Distribution")

    stats_text = (f"Min: {baselines.min():.3f}m\n"
                  f"Max: {baselines.max():.3f}m\n"
                  f"Mean: {baselines.mean():.3f}m\n"
                  f"Std: {baselines.std():.3f}m")
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "baseline_histogram.png", dpi=150)
    plt.close()

    # Generate radius histogram
    plt.figure(figsize=(10, 6))
    plt.hist(radii, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Distance from Centroid (meters)")
    plt.ylabel("Count")
    plt.title("Camera Distance from Scene Centroid")

    plt.tight_layout()
    plt.savefig(output_dir / "radius_histogram.png", dpi=150)
    plt.close()

    # Generate interactive 3D plot with plotly
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Camera positions
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+lines',
            marker=dict(size=5, color=np.arange(len(positions)), colorscale='Viridis'),
            line=dict(color='gray', width=2),
            name='Camera Path',
            text=[f"Camera {i}: {name}" for i, name in enumerate(shot_names)],
        ))

        # Camera orientations (forward vectors)
        arrow_scale = baselines.mean() * 0.5
        for i, (pos, fwd) in enumerate(zip(positions, orientations)):
            end = pos + fwd * arrow_scale
            fig.add_trace(go.Scatter3d(
                x=[pos[0], end[0]],
                y=[pos[1], end[1]],
                z=[pos[2], end[2]],
                mode='lines',
                line=dict(color='red', width=3),
                showlegend=False,
            ))

        # Centroid marker
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2]],
            mode='markers',
            marker=dict(size=10, color='orange', symbol='x'),
            name='Centroid',
        ))

        # Start and end markers
        fig.add_trace(go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='diamond'),
            name='Start',
        ))
        fig.add_trace(go.Scatter3d(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            z=[positions[-1, 2]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='square'),
            name='End',
        ))

        fig.update_layout(
            title="Camera Path 3D Visualization",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
            ),
            width=1000,
            height=800,
        )

        fig.write_html(output_dir / "camera_path_3d.html")
        print(f"Saved interactive 3D plot to {output_dir / 'camera_path_3d.html'}")

    except ImportError:
        print("Plotly not available, skipping 3D interactive plot")

    # Export camera path as PLY
    # Color by sequence (rainbow)
    n = len(positions)
    colors = plt.cm.viridis(np.linspace(0, 1, n))[:, :3] * 255
    write_ply(output_dir / "camera_positions.ply", positions, colors.astype(np.uint8))

    # Export camera frustums (simplified as lines)
    frustum_positions = []
    frustum_colors = []
    arrow_len = baselines.mean() * 0.3

    for i, (pos, fwd) in enumerate(zip(positions, orientations)):
        # Add camera center
        frustum_positions.append(pos)
        frustum_colors.append([0, 255, 0])  # Green

        # Add forward point
        frustum_positions.append(pos + fwd * arrow_len)
        frustum_colors.append([255, 0, 0])  # Red

    write_ply(
        output_dir / "camera_orientations.ply",
        np.array(frustum_positions),
        np.array(frustum_colors, dtype=np.uint8)
    )

    return stats


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze camera poses")
    parser.add_argument("dataset_dir", type=Path, help="OpenSfM dataset directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: dataset_dir/../debug/cameras)")

    args = parser.parse_args()

    output_dir = args.output or args.dataset_dir.parent / "debug" / "cameras"
    stats = analyze_cameras(args.dataset_dir, output_dir)

    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
