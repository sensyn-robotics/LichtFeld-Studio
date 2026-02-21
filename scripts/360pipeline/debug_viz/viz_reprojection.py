"""Overlay SfM points on original images using equirectangular projection."""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .utils import (
    load_reconstruction,
    rotation_from_angle_axis,
    project_point_to_equirect,
    colormap_by_value,
)


def visualize_reprojection(
    dataset_dir: Path,
    output_dir: Path,
    max_images: int = 20,
    max_points: int = 5000,
) -> dict:
    """
    Overlay SfM points on original images.

    Args:
        dataset_dir: Path to OpenSfM dataset directory
        output_dir: Output directory for visualizations
        max_images: Maximum images to process
        max_points: Maximum points to project per image

    Returns:
        Summary statistics
    """
    reconstruction_path = dataset_dir / "reconstruction.json"
    images_dir = dataset_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reconstruction...")
    recon = load_reconstruction(reconstruction_path)

    shots = recon.get("shots", {})
    points = recon.get("points", {})
    cameras = recon.get("cameras", {})

    # Get camera dimensions
    camera = next(iter(cameras.values())) if cameras else {}
    width = camera.get("width", 2048)
    height = camera.get("height", 1024)

    print(f"Camera: {width}x{height}")
    print(f"Shots: {len(shots)}, Points: {len(points)}")

    # Collect 3D points
    point_coords = []
    point_colors = []
    point_reproj_errors = []

    for point_id, point_data in points.items():
        coords = point_data.get("coordinates", [0, 0, 0])
        color = point_data.get("color", [128, 128, 128])
        reproj_error = point_data.get("reprojection_error", 0)

        point_coords.append(coords)
        point_colors.append(color)
        point_reproj_errors.append(reproj_error)

    point_coords = np.array(point_coords)
    point_colors = np.array(point_colors, dtype=np.uint8)
    point_reproj_errors = np.array(point_reproj_errors)

    # Sample points if too many
    if len(point_coords) > max_points:
        indices = np.random.choice(len(point_coords), max_points, replace=False)
        point_coords = point_coords[indices]
        point_colors = point_colors[indices]
        point_reproj_errors = point_reproj_errors[indices]

    # Select images to process
    shot_names = sorted(shots.keys())
    if len(shot_names) > max_images:
        indices = np.linspace(0, len(shot_names) - 1, max_images, dtype=int)
        shot_names = [shot_names[i] for i in indices]

    stats = {
        "images_processed": 0,
        "total_projections": 0,
    }

    html_entries = []

    for shot_name in tqdm(shot_names, desc="Projecting to images"):
        shot = shots[shot_name]
        image_path = images_dir / shot_name

        if not image_path.exists():
            continue

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        # Get camera pose
        rotation = shot.get("rotation", [0, 0, 0])
        translation = shot.get("translation", [0, 0, 0])
        R = rotation_from_angle_axis(rotation)
        t = np.array(translation)

        # Project all points
        n_projected = 0
        for i, (coords, color, reproj_err) in enumerate(zip(
            point_coords, point_colors, point_reproj_errors
        )):
            px, py, valid = project_point_to_equirect(
                coords, R, t, img_w, img_h
            )

            if not valid:
                continue

            # Check bounds
            if not (0 <= px < img_w and 0 <= py < img_h):
                continue

            # Choose color based on reprojection error
            if reproj_err < 1.0:
                circle_color = (0, 255, 0)  # Green - good
            elif reproj_err < 3.0:
                circle_color = (0, 255, 255)  # Yellow - ok
            else:
                circle_color = (0, 0, 255)  # Red - bad

            # Draw circle
            cv2.circle(img, (int(px), int(py)), 3, circle_color, -1)
            n_projected += 1

        stats["total_projections"] += n_projected

        # Add text overlay
        text = f"{shot_name}: {n_projected} points"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # Save image
        out_name = f"{Path(shot_name).stem}_reproj.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        stats["images_processed"] += 1

        html_entries.append({
            "image": out_name,
            "original": shot_name,
            "points": n_projected,
        })

    # Generate HTML summary
    html_content = generate_html_summary(html_entries, stats)
    (output_dir / "reprojection_summary.html").write_text(html_content)

    print(f"\nProcessed {stats['images_processed']} images")
    print(f"Total projections: {stats['total_projections']}")

    return stats


def generate_html_summary(entries: list[dict], stats: dict) -> str:
    """Generate HTML summary page."""
    rows = ""
    for entry in entries:
        rows += f"""
        <tr>
            <td><a href="{entry['image']}">{entry['original']}</a></td>
            <td>{entry['points']}</td>
            <td><img src="{entry['image']}" style="max-width: 800px; cursor: pointer;"
                     onclick="window.open('{entry['image']}', '_blank')"></td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>SfM Point Reprojection Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        img {{ border: 1px solid #ccc; }}
        .stats {{ background: #f0f0f0; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        .legend {{ display: flex; gap: 20px; margin-bottom: 10px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .green {{ background-color: #00ff00; }}
        .yellow {{ background-color: #ffff00; }}
        .red {{ background-color: #ff0000; }}
    </style>
</head>
<body>
    <h1>SfM Point Reprojection Visualization</h1>

    <div class="stats">
        <h2>Statistics</h2>
        <p><strong>Images processed:</strong> {stats['images_processed']}</p>
        <p><strong>Total projections:</strong> {stats['total_projections']}</p>

        <h3>Legend</h3>
        <div class="legend">
            <div class="legend-item"><div class="dot green"></div> Low error (&lt;1px)</div>
            <div class="legend-item"><div class="dot yellow"></div> Medium error (1-3px)</div>
            <div class="legend-item"><div class="dot red"></div> High error (&gt;3px)</div>
        </div>
    </div>

    <table>
        <tr>
            <th>Image</th>
            <th>Projected Points</th>
            <th>Preview</th>
        </tr>
        {rows}
    </table>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize SfM reprojection")
    parser.add_argument("dataset_dir", type=Path, help="OpenSfM dataset directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: dataset_dir/../debug/reprojection)")
    parser.add_argument("--max-images", type=int, default=20,
                        help="Maximum images to process")

    args = parser.parse_args()

    output_dir = args.output or args.dataset_dir.parent / "debug" / "reprojection"
    stats = visualize_reprojection(args.dataset_dir, output_dir, max_images=args.max_images)
