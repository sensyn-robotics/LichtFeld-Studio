#!/usr/bin/env python3
"""
360 Video to 3DGS One-Liner Pipeline

Converts 360 equirectangular video to 3D Gaussian Splatting using OpenSfM
and LichtFeld Studio.

Pipeline:
    360 Video → Frames (1fps) → OpenSfM → transforms.json → LichtFeld (--gut)

Usage:
    python scripts/360_to_3dgs.py /path/to/video.mp4 /path/to/output --iterations 10000

Requirements:
    - ffmpeg (for frame extraction)
    - OpenSfM (for SfM reconstruction)
    - LichtFeld Studio (for 3DGS training)
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


def check_dependencies() -> dict[str, bool]:
    """Check for required external dependencies."""
    deps = {}

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        deps["ffmpeg"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        deps["ffmpeg"] = False

    # Check OpenSfM
    try:
        subprocess.run(["opensfm", "--help"], capture_output=True, check=True)
        deps["opensfm"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        deps["opensfm"] = False

    return deps


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    start_time: Optional[float] = None,
    duration: Optional[float] = None
) -> int:
    """
    Extract frames from video using ffmpeg.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)

    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-vf", f"fps={fps}",
        "-qscale:v", "2",  # High quality JPEG
        str(output_dir / "%04d.jpg")
    ])

    print(f"Extracting frames at {fps} fps...")
    subprocess.run(cmd, check=True, capture_output=True)

    # Count extracted frames
    frame_count = len(list(output_dir.glob("*.jpg")))
    print(f"Extracted {frame_count} frames")
    return frame_count


def create_opensfm_config(
    dataset_dir: Path,
    width: int,
    height: int,
    matching_neighbors: int = 5
) -> Path:
    """
    Create OpenSfM configuration for equirectangular images.

    Args:
        dataset_dir: OpenSfM dataset directory
        width: Image width
        height: Image height
        matching_neighbors: Number of temporal neighbors for matching

    Returns:
        Path to config file
    """
    config_content = f"""# OpenSfM configuration for 360 equirectangular images

# Camera model override - force equirectangular for all images
camera_models_overrides:
  "*":
    projection_type: equirectangular
    width: {width}
    height: {height}

# Matching configuration - use temporal/sequential neighbors
matching_gps_neighbors: 0
matching_time_neighbors: {matching_neighbors}
matching_order_neighbors: {matching_neighbors}

# Feature detection
feature_type: SIFT
feature_root: true
feature_min_frames: 4000
feature_process_size: {max(width, height)}

# Reconstruction settings
retriangulation_ratio: 1.2
bundle_outlier_filtering_type: AUTO

# Depthmap settings
depthmap_min_patch_sd: 1.0
"""

    config_path = dataset_dir / "config.yaml"
    config_path.write_text(config_content)

    print(f"Created OpenSfM config: {config_path}")
    return config_path


def setup_opensfm_dataset(
    frames_dir: Path,
    dataset_dir: Path,
    width: int,
    height: int
) -> Path:
    """
    Set up OpenSfM dataset structure.

    Args:
        frames_dir: Directory containing extracted frames
        dataset_dir: OpenSfM dataset directory to create
        width: Image width
        height: Image height

    Returns:
        Path to dataset directory
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create images directory (symlink to frames)
    images_dir = dataset_dir / "images"
    if images_dir.exists():
        if images_dir.is_symlink():
            images_dir.unlink()
        else:
            shutil.rmtree(images_dir)

    # Copy frames to images directory (OpenSfM prefers actual files)
    shutil.copytree(frames_dir, images_dir)

    # Create config
    create_opensfm_config(dataset_dir, width, height)

    return dataset_dir


def run_opensfm(dataset_dir: Path) -> Path:
    """
    Run OpenSfM reconstruction pipeline.

    Args:
        dataset_dir: OpenSfM dataset directory

    Returns:
        Path to reconstruction.json
    """
    steps = [
        "extract_metadata",
        "detect_features",
        "match_features",
        "create_tracks",
        "reconstruct",
    ]

    for step in tqdm(steps, desc="OpenSfM"):
        print(f"\nRunning OpenSfM {step}...")
        cmd = ["opensfm", step, str(dataset_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error in {step}:")
            print(result.stderr)
            raise RuntimeError(f"OpenSfM {step} failed")

    reconstruction_path = dataset_dir / "reconstruction.json"
    if not reconstruction_path.exists():
        raise FileNotFoundError("OpenSfM reconstruction.json not found")

    return reconstruction_path


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


def convert_opensfm_to_transforms(
    reconstruction_path: Path,
    output_path: Path,
    images_dir: str = "images"
) -> dict:
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

    recon = reconstructions[0]
    cameras = recon.get("cameras", {})
    shots = recon.get("shots", {})

    if not shots:
        raise ValueError("No shots found in reconstruction")

    # Find equirectangular camera or use first
    equirect_camera = None
    for cam_id, cam in cameras.items():
        if cam.get("projection_type") == "equirectangular":
            equirect_camera = cam
            break

    if equirect_camera is None:
        equirect_camera = next(iter(cameras.values())) if cameras else {}
        print("Warning: No equirectangular camera found, using first available")

    width = equirect_camera.get("width", 3840)
    height = equirect_camera.get("height", 1920)

    transforms = {
        "camera_model": "EQUIRECTANGULAR",
        "w": width,
        "h": height,
        "frames": []
    }

    for shot_id, shot in sorted(shots.items()):
        rotation = shot.get("rotation", [0, 0, 0])
        translation = shot.get("translation", [0, 0, 0])

        R = rotation_from_angle_axis(rotation)
        t = np.array(translation)

        # Camera center in world coordinates
        C = -R.T @ t

        # Build 4x4 transform matrix (camera-to-world)
        transform = np.eye(4)
        transform[:3, :3] = R.T
        transform[:3, 3] = C

        frame = {
            "file_path": f"{images_dir}/{shot_id}",
            "transform_matrix": transform.tolist()
        }
        transforms["frames"].append(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)

    print(f"Converted {len(transforms['frames'])} frames to {output_path}")
    return transforms


def run_lichtfeld(
    dataset_dir: Path,
    output_dir: Path,
    iterations: int = 10000,
    lichtfeld_path: Optional[Path] = None,
    extra_args: Optional[list[str]] = None
) -> Path:
    """
    Run LichtFeld Studio training.

    Args:
        dataset_dir: Directory containing transforms.json and images
        output_dir: Output directory for splat files
        iterations: Number of training iterations
        lichtfeld_path: Path to LichtFeld-Studio executable
        extra_args: Additional command line arguments

    Returns:
        Path to output splat file
    """
    if lichtfeld_path is None:
        # Try to find LichtFeld-Studio in common locations
        script_dir = Path(__file__).parent.parent
        candidates = [
            script_dir / "build" / "LichtFeld-Studio",
            script_dir / "LichtFeld-Studio",
            Path("/usr/local/bin/LichtFeld-Studio"),
        ]
        for candidate in candidates:
            if candidate.exists():
                lichtfeld_path = candidate
                break

        if lichtfeld_path is None:
            raise FileNotFoundError(
                "LichtFeld-Studio executable not found. "
                "Please specify path with --lichtfeld-path"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(lichtfeld_path),
        "-d", str(dataset_dir),
        "-o", str(output_dir),
        "--gut",  # Enable GUT mode for equirectangular
        "--headless",
        "-i", str(iterations),
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\nRunning LichtFeld Studio...")
    print(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)

    # Find output splat file
    splat_files = list(output_dir.glob("splat_*.ply"))
    if splat_files:
        final_splat = max(splat_files, key=lambda p: int(p.stem.split('_')[1]))
        print(f"Output: {final_splat}")
        return final_splat

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert 360 video to 3D Gaussian Splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python 360_to_3dgs.py video.mp4 output/

    # With custom settings
    python 360_to_3dgs.py video.mp4 output/ --fps 2 --iterations 20000

    # Extract specific time range
    python 360_to_3dgs.py video.mp4 output/ --start 10 --duration 60
"""
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input 360 video file"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame extraction rate (default: 1.0)"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10000,
        help="LichtFeld training iterations (default: 10000)"
    )
    parser.add_argument(
        "--start",
        type=float,
        help="Start time in seconds"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Duration in seconds"
    )
    parser.add_argument(
        "--lichtfeld-path",
        type=Path,
        help="Path to LichtFeld-Studio executable"
    )
    parser.add_argument(
        "--skip-sfm",
        action="store_true",
        help="Skip SfM step (use existing reconstruction)"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip frame extraction (use existing frames)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip LichtFeld training (only run SfM)"
    )

    args = parser.parse_args()

    # Check input exists
    if not args.input.exists():
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    # Check dependencies
    deps = check_dependencies()
    missing = [name for name, available in deps.items() if not available]

    if "ffmpeg" in missing and not args.skip_extraction:
        print("Error: ffmpeg is required for frame extraction")
        print("Install with: sudo apt install ffmpeg")
        sys.exit(1)

    if "opensfm" in missing and not args.skip_sfm:
        print("Error: OpenSfM is required for SfM reconstruction")
        print("Install from: https://github.com/mapillary/OpenSfM")
        sys.exit(1)

    # Setup directories
    output_dir = args.output
    frames_dir = output_dir / "frames"
    opensfm_dir = output_dir / "opensfm"
    lichtfeld_output = output_dir / "output"

    # Step 1: Extract frames
    if not args.skip_extraction:
        extract_frames(
            args.input,
            frames_dir,
            fps=args.fps,
            start_time=args.start,
            duration=args.duration
        )

    # Get image dimensions from first frame
    if frames_dir.exists():
        import cv2
        first_frame = next(frames_dir.glob("*.jpg"))
        img = cv2.imread(str(first_frame))
        height, width = img.shape[:2]
        print(f"Frame dimensions: {width}x{height}")
    else:
        print("Error: No frames found")
        sys.exit(1)

    # Step 2: Run OpenSfM
    if not args.skip_sfm:
        setup_opensfm_dataset(frames_dir, opensfm_dir, width, height)
        reconstruction_path = run_opensfm(opensfm_dir)
    else:
        reconstruction_path = opensfm_dir / "reconstruction.json"

    # Step 3: Convert to transforms.json
    transforms_path = output_dir / "transforms.json"

    # Create images symlink for LichtFeld
    lichtfeld_images = output_dir / "images"
    if lichtfeld_images.exists():
        if lichtfeld_images.is_symlink():
            lichtfeld_images.unlink()
        else:
            shutil.rmtree(lichtfeld_images)
    lichtfeld_images.symlink_to(frames_dir.resolve())

    if reconstruction_path.exists():
        convert_opensfm_to_transforms(
            reconstruction_path,
            transforms_path,
            images_dir="images"
        )
    elif not args.skip_sfm:
        print("Error: OpenSfM reconstruction failed - no reconstruction.json found")
        sys.exit(1)
    else:
        print("Skipping transforms.json conversion (no reconstruction.json)")

    # Step 4: Run LichtFeld
    if not args.skip_training:
        run_lichtfeld(
            output_dir,
            lichtfeld_output,
            iterations=args.iterations,
            lichtfeld_path=args.lichtfeld_path
        )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Output directory: {output_dir}")
    print(f"Transforms: {transforms_path}")
    print(f"Splat output: {lichtfeld_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
