"""Utility functions for debug visualization."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_reconstruction(reconstruction_path: Path) -> dict[str, Any]:
    """
    Load OpenSfM reconstruction.json.

    Args:
        reconstruction_path: Path to reconstruction.json

    Returns:
        First reconstruction dict (cameras, shots, points)
    """
    with open(reconstruction_path) as f:
        reconstructions = json.load(f)

    if not reconstructions:
        raise ValueError("No reconstruction found")

    return reconstructions[0]


def rotation_from_angle_axis(angle_axis: list[float] | np.ndarray) -> np.ndarray:
    """
    Convert angle-axis rotation to rotation matrix using Rodrigues' formula.

    Args:
        angle_axis: 3-element rotation vector (axis * angle)

    Returns:
        3x3 rotation matrix
    """
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


def spherical_to_equirect(
    theta: np.ndarray,
    phi: np.ndarray,
    width: int,
    height: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to equirectangular pixel coordinates.

    Args:
        theta: Azimuth angle in radians [-pi, pi]
        phi: Elevation angle in radians [-pi/2, pi/2]
        width: Image width
        height: Image height

    Returns:
        Tuple of (x, y) pixel coordinates
    """
    # Normalize to [0, 1]
    u = (theta + np.pi) / (2 * np.pi)
    v = (np.pi / 2 - phi) / np.pi

    x = u * width
    y = v * height

    return x, y


def equirect_to_spherical(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert equirectangular pixel coordinates to spherical coordinates.

    Args:
        x: Pixel x coordinates
        y: Pixel y coordinates
        width: Image width
        height: Image height

    Returns:
        Tuple of (theta, phi) in radians
    """
    u = x / width
    v = y / height

    theta = u * 2 * np.pi - np.pi
    phi = np.pi / 2 - v * np.pi

    return theta, phi


def project_point_to_equirect(
    point_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int
) -> tuple[float, float, bool]:
    """
    Project a 3D point onto equirectangular image.

    Args:
        point_3d: 3D point in world coordinates
        R: Camera rotation matrix (world to camera)
        t: Camera translation
        width: Image width
        height: Image height

    Returns:
        Tuple of (x, y, is_valid) where is_valid indicates if point is in front of camera
    """
    # Transform to camera coordinates
    point_cam = R @ point_3d + t

    # Convert to spherical coordinates
    x, y, z = point_cam
    r = np.linalg.norm(point_cam)

    if r < 1e-10:
        return 0, 0, False

    # Azimuth (theta) and elevation (phi)
    theta = np.arctan2(x, z)  # Note: OpenSfM uses z-forward
    phi = np.arcsin(y / r)

    # Convert to pixel coordinates
    px, py = spherical_to_equirect(np.array([theta]), np.array([phi]), width, height)

    return float(px[0]), float(py[0]), True


def get_camera_center(shot: dict) -> np.ndarray:
    """
    Get camera center in world coordinates from shot data.

    Args:
        shot: OpenSfM shot dictionary

    Returns:
        Camera center as 3D numpy array
    """
    rotation = shot.get("rotation", [0, 0, 0])
    translation = shot.get("translation", [0, 0, 0])

    R = rotation_from_angle_axis(rotation)
    t = np.array(translation)

    # Camera center: C = -R^T * t
    return -R.T @ t


def load_tracks(tracks_path: Path) -> dict[str, list[tuple[str, int]]]:
    """
    Load tracks from OpenSfM tracks.csv.

    Args:
        tracks_path: Path to tracks.csv

    Returns:
        Dictionary mapping track_id to list of (image_name, feature_id)
    """
    tracks = {}

    with open(tracks_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                image_name = parts[0]
                track_id = parts[1]
                feature_id = int(parts[2])

                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append((image_name, feature_id))

    return tracks


def write_ply(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray | None = None
) -> None:
    """
    Write a PLY point cloud file.

    Args:
        path: Output path
        positions: Nx3 array of positions
        colors: Nx3 array of RGB colors (0-255), optional
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = len(positions)

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if colors is not None:
            for pos, col in zip(positions, colors):
                f.write(f"{pos[0]} {pos[1]} {pos[2]} {int(col[0])} {int(col[1])} {int(col[2])}\n")
        else:
            for pos in positions:
                f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")


def colormap_by_value(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis"
) -> np.ndarray:
    """
    Map values to RGB colors using a colormap.

    Args:
        values: 1D array of values
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        cmap: Colormap name

    Returns:
        Nx3 array of RGB colors (0-255)
    """
    import matplotlib.pyplot as plt

    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()

    # Normalize to [0, 1]
    norm_values = (values - vmin) / (vmax - vmin + 1e-10)
    norm_values = np.clip(norm_values, 0, 1)

    # Apply colormap
    cm = plt.get_cmap(cmap)
    colors = cm(norm_values)[:, :3] * 255

    return colors.astype(np.uint8)
