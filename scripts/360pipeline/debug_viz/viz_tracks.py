"""Analyze and visualize track consistency (multi-view coverage)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import load_reconstruction, load_tracks, write_ply, colormap_by_value


def analyze_tracks(
    dataset_dir: Path,
    output_dir: Path
) -> dict:
    """
    Analyze track consistency and multi-view coverage.

    Args:
        dataset_dir: Path to OpenSfM dataset directory
        output_dir: Output directory for visualizations

    Returns:
        Summary statistics
    """
    tracks_path = dataset_dir / "tracks.csv"
    reconstruction_path = dataset_dir / "reconstruction.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tracks
    print("Loading tracks...")
    tracks = load_tracks(tracks_path)

    # Count views per track
    track_lengths = [len(views) for views in tracks.values()]
    track_lengths = np.array(track_lengths)

    print(f"Total tracks: {len(tracks)}")
    print(f"Track lengths: min={track_lengths.min()}, max={track_lengths.max()}, "
          f"mean={track_lengths.mean():.1f}")

    # Generate histogram
    plt.figure(figsize=(10, 6))
    bins = np.arange(2, min(20, track_lengths.max() + 2))
    counts, edges, _ = plt.hist(track_lengths, bins=bins, edgecolor='black', alpha=0.7)

    plt.xlabel("Number of Views per Track")
    plt.ylabel("Number of Tracks")
    plt.title("Track Length Distribution (Multi-View Coverage)")

    # Add percentage labels
    total = len(track_lengths)
    for i, (count, edge) in enumerate(zip(counts, edges[:-1])):
        pct = count / total * 100
        if pct > 1:
            plt.text(edge + 0.4, count, f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)

    # Add statistics text
    stats_text = (f"Total tracks: {total}\n"
                  f"2-view tracks: {(track_lengths == 2).sum()} ({(track_lengths == 2).mean()*100:.1f}%)\n"
                  f"3+ view tracks: {(track_lengths >= 3).sum()} ({(track_lengths >= 3).mean()*100:.1f}%)\n"
                  f"5+ view tracks: {(track_lengths >= 5).sum()} ({(track_lengths >= 5).mean()*100:.1f}%)")
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "track_histogram.png", dpi=150)
    plt.close()

    # Load reconstruction to get 3D points with track info
    print("Loading reconstruction...")
    recon = load_reconstruction(reconstruction_path)
    points = recon.get("points", {})

    # Map track_id to number of views
    track_to_views = {tid: len(views) for tid, views in tracks.items()}

    # Collect point data
    positions = []
    view_counts = []

    for point_id, point_data in points.items():
        coords = point_data.get("coordinates", [0, 0, 0])
        positions.append(coords)

        # Get view count for this point (track)
        n_views = track_to_views.get(point_id, 2)
        view_counts.append(n_views)

    positions = np.array(positions)
    view_counts = np.array(view_counts)

    if len(positions) > 0:
        # Color: red=2 views, yellow=5 views, green=10+ views
        colors = colormap_by_value(view_counts, vmin=2, vmax=10, cmap="RdYlGn")

        # Export PLY colored by view count
        write_ply(output_dir / "sfm_points_by_views.ply", positions, colors)

        # Export PLY with only high-confidence points (5+ views)
        mask_5plus = view_counts >= 5
        if mask_5plus.sum() > 0:
            write_ply(
                output_dir / "sfm_points_5plus_views.ply",
                positions[mask_5plus],
                colors[mask_5plus]
            )
            print(f"Exported {mask_5plus.sum()} points with 5+ views")

        # Export PLY with only weak points (2 views)
        mask_2 = view_counts == 2
        if mask_2.sum() > 0:
            write_ply(
                output_dir / "sfm_points_2_views.ply",
                positions[mask_2],
                np.full((mask_2.sum(), 3), [255, 0, 0], dtype=np.uint8)
            )
            print(f"Exported {mask_2.sum()} points with only 2 views")

    # Summary statistics
    stats = {
        "total_tracks": len(tracks),
        "track_lengths": {
            "min": int(track_lengths.min()),
            "max": int(track_lengths.max()),
            "mean": float(track_lengths.mean()),
            "median": float(np.median(track_lengths)),
        },
        "view_distribution": {
            "2_views": int((track_lengths == 2).sum()),
            "3_plus_views": int((track_lengths >= 3).sum()),
            "5_plus_views": int((track_lengths >= 5).sum()),
            "10_plus_views": int((track_lengths >= 10).sum()),
        },
        "percentages": {
            "2_views_pct": float((track_lengths == 2).mean() * 100),
            "3_plus_pct": float((track_lengths >= 3).mean() * 100),
            "5_plus_pct": float((track_lengths >= 5).mean() * 100),
        }
    }

    # Print warnings
    if stats["percentages"]["2_views_pct"] > 50:
        print(f"\n⚠️  WARNING: {stats['percentages']['2_views_pct']:.1f}% of tracks have only 2 views")
        print("   This indicates weak triangulation and may cause poor 3DGS quality.")
        print("   Consider:")
        print("   - Increasing frame rate (--fps)")
        print("   - Using higher quality SfM settings (--quality high)")
        print("   - Filtering to use only 3+ view points for 3DGS initialization")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze track consistency")
    parser.add_argument("dataset_dir", type=Path, help="OpenSfM dataset directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: dataset_dir/../debug/tracks)")

    args = parser.parse_args()

    output_dir = args.output or args.dataset_dir.parent / "debug" / "tracks"
    stats = analyze_tracks(args.dataset_dir, output_dir)

    import json
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
