"""Visualize feature matches between image pairs."""

import gzip
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_features(features_dir: Path, image_name: str) -> dict | None:
    """Load features for an image."""
    # Try different naming patterns
    patterns = [
        features_dir / f"{image_name}.features.npz",
        features_dir / f"{Path(image_name).stem}.features.npz",
    ]

    for path in patterns:
        if path.exists():
            data = np.load(path)
            return {
                "points": data.get("points", data.get("keypoints", np.array([]))),
                "descriptors": data.get("descriptors", np.array([])),
            }

    return None


def load_matches_for_image(matches_dir: Path, img_name: str) -> dict:
    """
    Load all matches for a given image.

    Returns dict mapping other image names to Nx2 match arrays.
    """
    match_file = matches_dir / f"{img_name}_matches.pkl.gz"
    if not match_file.exists():
        return {}

    with gzip.open(match_file, 'rb') as f:
        return pickle.load(f)


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    matches: np.ndarray,
    max_matches: int = 100
) -> np.ndarray:
    """
    Draw matching lines between two images.

    Args:
        img1: First image
        img2: Second image
        pts1: Keypoints in first image (Nx2)
        pts2: Keypoints in second image (Nx2)
        matches: Match indices (Mx2, each row is [idx1, idx2])
        max_matches: Maximum number of matches to draw

    Returns:
        Combined image with match lines
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create combined image
    h_out = max(h1, h2)
    w_out = w1 + w2
    out = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    out[:h1, :w1] = img1
    out[:h2, w1:] = img2

    # Sample matches if too many
    if len(matches) > max_matches:
        indices = np.random.choice(len(matches), max_matches, replace=False)
        matches = matches[indices]

    # Draw matches with random colors
    for idx1, idx2 in matches:
        if idx1 >= len(pts1) or idx2 >= len(pts2):
            continue

        pt1 = tuple(pts1[int(idx1)][:2].astype(int))
        pt2 = (int(pts2[int(idx2)][0] + w1), int(pts2[int(idx2)][1]))

        color = tuple(np.random.randint(100, 255, 3).tolist())
        cv2.line(out, pt1, pt2, color, 1)
        cv2.circle(out, pt1, 3, color, -1)
        cv2.circle(out, pt2, 3, color, -1)

    return out


def visualize_matches(
    dataset_dir: Path,
    output_dir: Path,
    max_pairs: int = 20,
    max_matches_per_pair: int = 100
) -> dict:
    """
    Visualize feature matches for all image pairs.

    Args:
        dataset_dir: Path to OpenSfM dataset directory
        output_dir: Output directory for visualizations
        max_pairs: Maximum number of pairs to visualize
        max_matches_per_pair: Maximum matches to draw per pair

    Returns:
        Summary statistics
    """
    features_dir = dataset_dir / "features"
    matches_dir = dataset_dir / "matches"
    images_dir = dataset_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image list
    image_files = sorted(images_dir.glob("*.jpg"))
    image_names = [f.name for f in image_files]

    if not image_names:
        print(f"No images found in {images_dir}")
        return {"error": "no images"}

    # Collect all matches across all images
    all_pairs = {}  # (img1, img2) -> matches array
    total_match_count = 0

    print("Loading matches...")
    for img_name in tqdm(image_names, desc="Loading match files"):
        matches_dict = load_matches_for_image(matches_dir, img_name)
        for other_img, matches in matches_dict.items():
            if len(matches) == 0:
                continue
            # Use canonical ordering to avoid duplicates
            pair_key = tuple(sorted([img_name, other_img]))
            if pair_key not in all_pairs:
                all_pairs[pair_key] = {
                    "img1": pair_key[0],
                    "img2": pair_key[1],
                    "matches": matches,
                    "count": len(matches),
                }
                total_match_count += len(matches)

    print(f"Found {len(all_pairs)} image pairs with {total_match_count} total matches")

    # Statistics
    stats = {
        "total_pairs": len(all_pairs),
        "pairs_visualized": 0,
        "match_counts": [p["count"] for p in all_pairs.values()],
    }

    # Sort by match count and select top pairs
    sorted_pairs = sorted(all_pairs.values(), key=lambda x: -x["count"])

    # Sample evenly: some high-match pairs, some medium, some low
    n_to_viz = min(max_pairs, len(sorted_pairs))
    if n_to_viz < len(sorted_pairs):
        # Take top third, middle third, bottom third
        n_each = n_to_viz // 3
        selected_pairs = (
            sorted_pairs[:n_each] +
            sorted_pairs[len(sorted_pairs)//2 - n_each//2:len(sorted_pairs)//2 + n_each//2] +
            sorted_pairs[-n_each:]
        )
    else:
        selected_pairs = sorted_pairs[:n_to_viz]

    html_entries = []

    for pair_data in tqdm(selected_pairs, desc="Visualizing matches"):
        img1_name = pair_data["img1"]
        img2_name = pair_data["img2"]
        matches = pair_data["matches"]
        n_matches = pair_data["count"]

        # Load images
        img1_path = images_dir / img1_name
        img2_path = images_dir / img2_name

        if not img1_path.exists() or not img2_path.exists():
            continue

        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        if img1 is None or img2 is None:
            continue

        # Load features
        feat1 = load_features(features_dir, img1_name)
        feat2 = load_features(features_dir, img2_name)

        if feat1 is None or feat2 is None:
            continue

        # Draw matches
        out_img = draw_matches(
            img1, img2,
            feat1["points"], feat2["points"],
            matches,
            max_matches=max_matches_per_pair
        )

        # Add text overlay
        text = f"{img1_name} <-> {img2_name}: {n_matches} matches"
        cv2.putText(out_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # Save
        out_name = f"{Path(img1_name).stem}_{Path(img2_name).stem}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        stats["pairs_visualized"] += 1

        html_entries.append({
            "image": out_name,
            "img1": img1_name,
            "img2": img2_name,
            "matches": n_matches,
        })

    # Generate HTML summary
    html_content = generate_html_summary(html_entries, stats)
    (output_dir / "match_summary.html").write_text(html_content)

    # Print summary
    if stats["match_counts"]:
        counts = np.array(stats["match_counts"])
        print(f"\nMatch Statistics:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Pairs visualized: {stats['pairs_visualized']}")
        print(f"  Matches per pair: min={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.1f}, median={np.median(counts):.1f}")

    return stats


def generate_html_summary(entries: list[dict], stats: dict) -> str:
    """Generate HTML summary page."""
    rows = ""
    for entry in sorted(entries, key=lambda x: -x["matches"]):
        rows += f"""
        <tr>
            <td><a href="{entry['image']}">{entry['img1']} ↔ {entry['img2']}</a></td>
            <td>{entry['matches']}</td>
            <td><img src="{entry['image']}" style="max-width: 600px; cursor: pointer;"
                     onclick="window.open('{entry['image']}', '_blank')"></td>
        </tr>
        """

    match_counts = stats.get("match_counts", [])
    if match_counts:
        counts = np.array(match_counts)
        stats_html = f"""
        <p><strong>Total pairs:</strong> {stats['total_pairs']}</p>
        <p><strong>Pairs visualized:</strong> {stats['pairs_visualized']}</p>
        <p><strong>Matches per pair:</strong>
            min={counts.min()}, max={counts.max()},
            mean={counts.mean():.1f}, median={np.median(counts):.1f}</p>
        """
    else:
        stats_html = "<p>No match statistics available</p>"

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Feature Match Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        img {{ border: 1px solid #ccc; }}
        .stats {{ background: #f0f0f0; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Feature Match Visualization</h1>

    <div class="stats">
        <h2>Statistics</h2>
        {stats_html}
    </div>

    <table>
        <tr>
            <th>Image Pair</th>
            <th>Matches</th>
            <th>Preview</th>
        </tr>
        {rows}
    </table>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize feature matches")
    parser.add_argument("dataset_dir", type=Path, help="OpenSfM dataset directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: dataset_dir/../debug/matches)")
    parser.add_argument("--max-pairs", type=int, default=20,
                        help="Maximum pairs to visualize")

    args = parser.parse_args()

    output_dir = args.output or args.dataset_dir.parent / "debug" / "matches"
    visualize_matches(args.dataset_dir, output_dir, max_pairs=args.max_pairs)
