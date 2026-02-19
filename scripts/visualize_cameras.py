#!/usr/bin/env python3
"""
Visualize camera positions from transforms.json or OpenSfM reconstruction.

Creates a PLY file of camera positions for visualization in MeshLab or LichtFeld.

Usage:
    python scripts/visualize_cameras.py output_full/
    python scripts/visualize_cameras.py output_full/transforms.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def rotation_from_angle_axis(angle_axis: list[float]) -> np.ndarray:
    """Convert angle-axis rotation to rotation matrix using Rodrigues' formula."""
    angle_axis = np.array(angle_axis)
    theta = np.linalg.norm(angle_axis)

    if theta < 1e-10:
        return np.eye(3)

    k = angle_axis / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def load_cameras_from_transforms(transforms_path: Path) -> list[tuple[float, float, float]]:
    """Load camera positions from transforms.json."""
    with open(transforms_path) as f:
        data = json.load(f)

    positions = []
    for frame in data['frames']:
        m = frame['transform_matrix']
        positions.append((m[0][3], m[1][3], m[2][3]))

    return positions


def load_cameras_from_opensfm(reconstruction_path: Path) -> list[tuple[float, float, float]]:
    """Load camera positions from OpenSfM reconstruction.json."""
    with open(reconstruction_path) as f:
        reconstructions = json.load(f)

    if not reconstructions:
        raise ValueError("No reconstruction found")

    recon = reconstructions[0]
    shots = recon.get("shots", {})

    positions = []
    for shot_id, shot in sorted(shots.items()):
        rotation = shot.get("rotation", [0, 0, 0])
        translation = shot.get("translation", [0, 0, 0])

        R = rotation_from_angle_axis(rotation)
        t = np.array(translation)
        C = -R.T @ t  # Camera center in world coordinates

        positions.append((C[0], C[1], C[2]))

    return positions


def analyze_camera_path(positions: list[tuple[float, float, float]]) -> dict:
    """Analyze camera path and return metrics."""
    if not positions:
        return {}

    positions_arr = np.array(positions)

    # Centroid
    cx, cy, cz = positions_arr.mean(axis=0)

    # Radii from centroid
    radii = np.linalg.norm(positions_arr - [cx, cy, cz], axis=1)

    # Total path length
    total_dist = sum(
        math.sqrt(
            (positions[i][0] - positions[i-1][0])**2 +
            (positions[i][1] - positions[i-1][1])**2 +
            (positions[i][2] - positions[i-1][2])**2
        )
        for i in range(1, len(positions))
    )

    # Start-end distance
    start_end = math.sqrt(
        (positions[-1][0] - positions[0][0])**2 +
        (positions[-1][1] - positions[0][1])**2 +
        (positions[-1][2] - positions[0][2])**2
    )

    # Bounding box
    min_coords = positions_arr.min(axis=0)
    max_coords = positions_arr.max(axis=0)
    bbox_size = max_coords - min_coords

    return {
        "n_cameras": len(positions),
        "centroid": (cx, cy, cz),
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "radius_mean": float(radii.mean()),
        "radius_std": float(radii.std()),
        "path_length": total_dist,
        "start_end_dist": start_end,
        "is_circular": start_end < total_dist * 0.3,
        "bbox_size": tuple(bbox_size),
    }


def export_camera_path_ply(
    positions: list[tuple[float, float, float]],
    output_path: Path,
    include_lines: bool = True
) -> None:
    """Export camera positions to PLY file with color gradient."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_vertices = len(positions)
    n_edges = len(positions) - 1 if include_lines else 0

    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if include_lines:
            f.write(f"element edge {n_edges}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
        f.write("end_header\n")

        # Vertices with color gradient (red -> blue)
        for i, (x, y, z) in enumerate(positions):
            t = i / max(1, len(positions) - 1)
            r = int(255 * (1 - t))
            g = 0
            b = int(255 * t)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

        # Edges
        if include_lines:
            for i in range(n_edges):
                f.write(f"{i} {i + 1}\n")


def export_camera_frustums_ply(
    positions: list[tuple[float, float, float]],
    output_path: Path,
    frustum_size: float = 0.3
) -> None:
    """Export camera frustums (pyramids) to PLY file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Each camera has 5 vertices (apex + 4 corners)
    n_cameras = len(positions)
    n_vertices = n_cameras * 5
    n_faces = n_cameras * 4  # 4 triangular faces per frustum

    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertices
        s = frustum_size
        for i, (x, y, z) in enumerate(positions):
            t = i / max(1, len(positions) - 1)
            r = int(255 * (1 - t))
            g = 0
            b = int(255 * t)

            # Apex (camera position)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
            # Four corners of frustum (simplified, facing -Z)
            f.write(f"{x - s} {y - s} {z - s*2} {r} {g} {b}\n")
            f.write(f"{x + s} {y - s} {z - s*2} {r} {g} {b}\n")
            f.write(f"{x + s} {y + s} {z - s*2} {r} {g} {b}\n")
            f.write(f"{x - s} {y + s} {z - s*2} {r} {g} {b}\n")

        # Faces (triangles from apex to corners)
        for i in range(n_cameras):
            base = i * 5
            f.write(f"3 {base} {base + 1} {base + 2}\n")
            f.write(f"3 {base} {base + 2} {base + 3}\n")
            f.write(f"3 {base} {base + 3} {base + 4}\n")
            f.write(f"3 {base} {base + 4} {base + 1}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize camera positions from transforms.json or OpenSfM reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/visualize_cameras.py output_full/
    python scripts/visualize_cameras.py output_full/transforms.json
    python scripts/visualize_cameras.py output_full/opensfm/reconstruction.json

Output:
    Creates camera_path.ply in the same directory as the input file.
    View in MeshLab, CloudCompare, or LichtFeld Studio.
"""
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to transforms.json, reconstruction.json, or dataset directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output PLY file path (default: camera_path.ply in same dir)"
    )
    parser.add_argument(
        "--frustums",
        action="store_true",
        help="Also export camera frustums visualization"
    )

    args = parser.parse_args()

    # Find input file
    input_path = args.path
    if input_path.is_dir():
        # Try transforms.json first, then opensfm reconstruction
        transforms_path = input_path / "transforms.json"
        recon_path = input_path / "opensfm" / "reconstruction.json"

        if transforms_path.exists():
            input_path = transforms_path
        elif recon_path.exists():
            input_path = recon_path
        else:
            print(f"Error: No transforms.json or reconstruction.json found in {args.path}")
            return 1

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    # Load camera positions
    if "reconstruction" in input_path.name:
        print(f"Loading OpenSfM reconstruction: {input_path}")
        positions = load_cameras_from_opensfm(input_path)
    else:
        print(f"Loading transforms: {input_path}")
        positions = load_cameras_from_transforms(input_path)

    if not positions:
        print("Error: No cameras found")
        return 1

    # Analyze path
    metrics = analyze_camera_path(positions)

    print(f"\nCamera Path Analysis:")
    print(f"  Cameras: {metrics['n_cameras']}")
    print(f"  Centroid: ({metrics['centroid'][0]:.2f}, {metrics['centroid'][1]:.2f}, {metrics['centroid'][2]:.2f})")
    print(f"  Radius: {metrics['radius_min']:.2f}m - {metrics['radius_max']:.2f}m")
    print(f"  Radius mean: {metrics['radius_mean']:.2f}m, std: {metrics['radius_std']:.2f}m")
    print(f"  Path length: {metrics['path_length']:.2f}m")
    print(f"  Start-end distance: {metrics['start_end_dist']:.2f}m")
    print(f"  Bounding box: {metrics['bbox_size'][0]:.2f} x {metrics['bbox_size'][1]:.2f} x {metrics['bbox_size'][2]:.2f}m")
    print(f"  Path type: {'CIRCULAR' if metrics['is_circular'] else 'LINEAR'}")

    # Quality warnings
    if metrics['radius_std'] > metrics['radius_mean'] * 0.3:
        print(f"\n⚠️  WARNING: High radius variance (std={metrics['radius_std']:.2f}m)")
        print("   This suggests inconsistent camera pose estimation.")
        print("   Expected for circular motion: consistent radius from subject.")

    if not metrics['is_circular']:
        print(f"\n⚠️  WARNING: Path detected as LINEAR (start-end: {metrics['start_end_dist']:.2f}m)")
        print("   If this was circular video, SfM may have failed to close the loop.")

    # Export PLY
    output_path = args.output or input_path.parent / "camera_path.ply"
    export_camera_path_ply(positions, output_path)
    print(f"\nExported camera path to: {output_path}")

    if args.frustums:
        frustums_path = output_path.parent / "camera_frustums.ply"
        export_camera_frustums_ply(positions, frustums_path)
        print(f"Exported camera frustums to: {frustums_path}")

    return 0


if __name__ == "__main__":
    exit(main())
