#!/usr/bin/env python3
"""
Convert OpenSfM reconstruction to LichtFeld Studio transforms.json format.

This script reads OpenSfM's reconstruction.json and outputs a transforms.json
file compatible with LichtFeld Studio's EQUIRECTANGULAR camera model.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Any


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


def opensfm_to_transforms(
    reconstruction_path: Path,
    output_path: Path,
    images_dir: str = "images"
) -> dict[str, Any]:
    """
    Convert OpenSfM reconstruction to LichtFeld transforms.json format.

    Args:
        reconstruction_path: Path to OpenSfM reconstruction.json
        output_path: Path to output transforms.json
        images_dir: Relative path to images directory

    Returns:
        The transforms dictionary
    """
    with open(reconstruction_path) as f:
        reconstructions = json.load(f)

    if not reconstructions:
        raise ValueError("No reconstruction found in OpenSfM output")

    # Use the first (main) reconstruction
    recon = reconstructions[0]

    # Get camera parameters
    cameras = recon.get("cameras", {})
    shots = recon.get("shots", {})

    if not shots:
        raise ValueError("No shots found in reconstruction")

    # Find the equirectangular camera
    equirect_camera = None
    for cam in cameras.values():
        if cam.get("projection_type") == "equirectangular":
            equirect_camera = cam
            break

    if equirect_camera is None:
        # Fall back to first camera if no equirectangular found
        equirect_camera = next(iter(cameras.values())) if cameras else {}
        print("Warning: No equirectangular camera found, using first available camera")

    # Extract image dimensions
    width = equirect_camera.get("width", 3840)
    height = equirect_camera.get("height", 1920)

    # Build transforms.json structure
    transforms = {
        "camera_model": "EQUIRECTANGULAR",
        "w": width,
        "h": height,
        "frames": []
    }

    # Process each shot
    for shot_id, shot in sorted(shots.items()):
        # OpenSfM uses angle-axis rotation
        rotation = shot.get("rotation", [0, 0, 0])
        translation = shot.get("translation", [0, 0, 0])

        # Convert to rotation matrix
        R = rotation_from_angle_axis(rotation)
        t = np.array(translation)

        # OpenSfM stores camera-to-world, but we need world-to-camera
        # Actually, OpenSfM stores rotation and translation such that:
        # X_camera = R * X_world + t
        # We need the inverse: transform_matrix that goes from camera to world

        # Camera center in world coordinates: C = -R^T * t
        C = -R.T @ t

        # Build 4x4 transform matrix (camera-to-world)
        transform = np.eye(4)
        transform[:3, :3] = R.T  # Transpose for camera-to-world
        transform[:3, 3] = C

        frame = {
            "file_path": f"{images_dir}/{shot_id}",
            "transform_matrix": transform.tolist()
        }
        transforms["frames"].append(frame)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)

    print(f"Converted {len(transforms['frames'])} frames to {output_path}")
    return transforms


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenSfM reconstruction to LichtFeld transforms.json"
    )
    parser.add_argument(
        "reconstruction",
        type=Path,
        help="Path to OpenSfM reconstruction.json"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("transforms.json"),
        help="Output transforms.json path (default: transforms.json)"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Relative path to images directory (default: images)"
    )

    args = parser.parse_args()

    if not args.reconstruction.exists():
        raise FileNotFoundError(f"Reconstruction file not found: {args.reconstruction}")

    opensfm_to_transforms(args.reconstruction, args.output, args.images_dir)


if __name__ == "__main__":
    main()
