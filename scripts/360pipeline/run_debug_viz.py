#!/usr/bin/env python3
"""
Master script to run all debug visualizations for 360 video to 3DGS pipeline.

Usage:
    cd scripts/360pipeline
    uv run python run_debug_viz.py ../../test_360_full --output ../../test_360_full/output_v2/debug
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def find_latest_splat(output_dir: Path) -> Path | None:
    """Find the latest splat PLY file in output directory."""
    splat_files = list(output_dir.glob("**/splat_*.ply"))
    if not splat_files:
        return None
    return max(splat_files, key=lambda p: int(p.stem.split('_')[1]))


def run_all_visualizations(
    dataset_dir: Path,
    output_dir: Path,
    skip_comparison: bool = False,
) -> dict:
    """
    Run all debug visualizations.

    Args:
        dataset_dir: Root directory containing frames/, opensfm/, etc.
        output_dir: Output directory for debug visualizations
        skip_comparison: Skip SfM vs 3DGS comparison (if no 3DGS output)

    Returns:
        Combined statistics from all visualizations
    """
    # Validate inputs
    opensfm_dir = dataset_dir / "opensfm"
    if not opensfm_dir.exists():
        print(f"Error: OpenSfM directory not found: {opensfm_dir}")
        sys.exit(1)

    reconstruction_path = opensfm_dir / "reconstruction.json"
    if not reconstruction_path.exists():
        print(f"Error: reconstruction.json not found: {reconstruction_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {
        "timestamp": datetime.now().isoformat(),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
    }

    # Import visualization modules
    from debug_viz.viz_matches import visualize_matches
    from debug_viz.viz_tracks import analyze_tracks
    from debug_viz.viz_sfm_quality import analyze_sfm_quality
    from debug_viz.viz_cameras import analyze_cameras
    from debug_viz.viz_reprojection import visualize_reprojection

    # 1. Feature match visualization
    print("\n" + "=" * 60)
    print("1. Feature Match Visualization")
    print("=" * 60)
    try:
        match_stats = visualize_matches(
            opensfm_dir,
            output_dir / "matches",
            max_pairs=30,
        )
        all_stats["matches"] = match_stats
    except Exception as e:
        print(f"Warning: Match visualization failed: {e}")
        all_stats["matches"] = {"error": str(e)}

    # 2. Track consistency analysis
    print("\n" + "=" * 60)
    print("2. Track Consistency Analysis")
    print("=" * 60)
    try:
        track_stats = analyze_tracks(
            opensfm_dir,
            output_dir / "tracks",
        )
        all_stats["tracks"] = track_stats
    except Exception as e:
        print(f"Warning: Track analysis failed: {e}")
        all_stats["tracks"] = {"error": str(e)}

    # 3. SfM quality analysis
    print("\n" + "=" * 60)
    print("3. SfM Quality Analysis")
    print("=" * 60)
    try:
        sfm_stats = analyze_sfm_quality(
            opensfm_dir,
            output_dir / "sfm",
        )
        all_stats["sfm_quality"] = sfm_stats
    except Exception as e:
        print(f"Warning: SfM quality analysis failed: {e}")
        all_stats["sfm_quality"] = {"error": str(e)}

    # 4. Camera pose analysis
    print("\n" + "=" * 60)
    print("4. Camera Pose Analysis")
    print("=" * 60)
    try:
        camera_stats = analyze_cameras(
            opensfm_dir,
            output_dir / "cameras",
        )
        all_stats["cameras"] = camera_stats
    except Exception as e:
        print(f"Warning: Camera analysis failed: {e}")
        all_stats["cameras"] = {"error": str(e)}

    # 5. Per-image reprojection visualization
    print("\n" + "=" * 60)
    print("5. Per-Image Reprojection Visualization")
    print("=" * 60)
    try:
        reproj_stats = visualize_reprojection(
            opensfm_dir,
            output_dir / "reprojection",
            max_images=25,
        )
        all_stats["reprojection"] = reproj_stats
    except Exception as e:
        print(f"Warning: Reprojection visualization failed: {e}")
        all_stats["reprojection"] = {"error": str(e)}

    # 6. SfM vs 3DGS comparison (if 3DGS output exists)
    if not skip_comparison:
        sfm_ply = dataset_dir / "sfm_points.ply"
        splat_ply = find_latest_splat(dataset_dir / "output")

        # Also check output_v2
        if splat_ply is None:
            splat_ply = find_latest_splat(dataset_dir / "output_v2")

        if sfm_ply.exists() and splat_ply is not None:
            print("\n" + "=" * 60)
            print("6. SfM vs 3DGS Comparison")
            print("=" * 60)
            try:
                from debug_viz.viz_sfm_vs_3dgs import compare_sfm_vs_3dgs
                comparison_stats = compare_sfm_vs_3dgs(
                    sfm_ply,
                    splat_ply,
                    output_dir / "comparison",
                )
                all_stats["comparison"] = comparison_stats
            except Exception as e:
                print(f"Warning: SfM vs 3DGS comparison failed: {e}")
                all_stats["comparison"] = {"error": str(e)}
        else:
            print("\n[Skipping SfM vs 3DGS comparison - no splat file found]")
            all_stats["comparison"] = {"skipped": "no splat file found"}

    # Move large files/dirs to debug_data/ (sibling of debug/) to prevent browser freeze
    # Chrome recursively scans directories when opening local HTML files
    import shutil
    data_dir = output_dir.parent / "debug_data"
    data_dir.mkdir(exist_ok=True)

    # Move large directories (images, plotly visualizations)
    large_dirs = ["matches", "reprojection", "cameras"]
    for dirname in large_dirs:
        src = output_dir / dirname
        if src.exists():
            dest = data_dir / dirname
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))

    # Move all PLY files
    for ply_file in output_dir.glob("**/*.ply"):
        dest = data_dir / ply_file.name
        ply_file.rename(dest)

    print(f"Moved large files to {data_dir} (keeps debug/ fast to open)")

    # Generate summary report
    print("\n" + "=" * 60)
    print("Generating Summary Report")
    print("=" * 60)

    summary_html = generate_summary_report(all_stats, output_dir)
    (output_dir / "summary_report.html").write_text(summary_html)

    # Save raw stats as JSON
    (output_dir / "stats.json").write_text(json.dumps(all_stats, indent=2))

    # Create launcher script (avoids Chrome freeze with local files)
    launcher_script = output_dir / "open_report.sh"
    launcher_script.write_text("""#!/bin/bash
cd "$(dirname "$0")"
echo "Starting server at http://localhost:8000/summary_report.html"
xdg-open "http://localhost:8000/summary_report.html" &
python3 -m http.server 8000
""")
    launcher_script.chmod(0o755)

    print(f"\n✓ All visualizations complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  To view report: cd {output_dir} && ./open_report.sh")

    return all_stats


def generate_summary_report(stats: dict, output_dir: Path) -> str:
    """Generate an HTML summary report."""

    # Diagnosis section
    diagnosis = []
    recommendations = []

    # Check track quality
    track_stats = stats.get("tracks", {})
    if "percentages" in track_stats:
        two_view_pct = track_stats["percentages"].get("2_views_pct", 0)
        if two_view_pct > 50:
            diagnosis.append(f"<li>⚠️ <strong>Weak triangulation:</strong> {two_view_pct:.1f}% of tracks have only 2 views</li>")
            recommendations.append("<li>Increase frame rate (--fps) to get more views per track</li>")
            recommendations.append("<li>Filter SfM points to use only 3+ view points for 3DGS initialization</li>")
        elif two_view_pct > 30:
            diagnosis.append(f"<li>⚡ <strong>Moderate triangulation:</strong> {two_view_pct:.1f}% of tracks have only 2 views</li>")

    # Check camera baseline
    camera_stats = stats.get("cameras", {})
    if "baselines" in camera_stats:
        mean_baseline = camera_stats["baselines"].get("mean", 0)
        if mean_baseline < 0.01:
            diagnosis.append(f"<li>⚠️ <strong>Very small baseline:</strong> {mean_baseline*100:.1f}cm average</li>")
            recommendations.append("<li>Reduce frame rate to increase baseline</li>")
            recommendations.append("<li>Capture with more camera movement</li>")

    # Check loop closure
    if "is_circular" in camera_stats and camera_stats["is_circular"]:
        loop_dist = camera_stats.get("loop_closure_distance", 0)
        baseline = camera_stats.get("baselines", {}).get("mean", 0.01)
        if loop_dist > baseline * 3:
            diagnosis.append(f"<li>⚠️ <strong>Poor loop closure:</strong> {loop_dist:.2f}m gap (should be ~{baseline:.2f}m)</li>")
            recommendations.append("<li>Increase matching_time_neighbors in OpenSfM config</li>")

    # Check reprojection error
    sfm_stats = stats.get("sfm_quality", {})
    if "reprojection_error" in sfm_stats:
        mean_error = sfm_stats["reprojection_error"].get("mean", 0)
        if mean_error > 5:
            diagnosis.append(f"<li>⚠️ <strong>High reprojection error:</strong> {mean_error:.2f}px average</li>")
            recommendations.append("<li>Check for motion blur in input frames</li>")
            recommendations.append("<li>Consider using higher quality SfM settings</li>")

    # Check comparison
    comparison_stats = stats.get("comparison", {})
    if "dgs_to_sfm" in comparison_stats:
        mean_dist = comparison_stats["dgs_to_sfm"].get("mean", 0)
        if mean_dist > 0.1:
            diagnosis.append(f"<li>⚠️ <strong>3DGS diverged from SfM:</strong> {mean_dist:.3f}m average distance</li>")
            recommendations.append("<li>Use better SfM initialization (more points, higher quality)</li>")

    diagnosis_html = "<ul>" + "".join(diagnosis) + "</ul>" if diagnosis else "<p>✓ No major issues detected</p>"
    recommendations_html = "<ul>" + "".join(recommendations) + "</ul>" if recommendations else "<p>No specific recommendations</p>"

    # Build visualization links
    # Large files are in ../debug_data/, small files remain in debug/
    viz_links = []
    data_dir = output_dir.parent / "debug_data"

    # Check both debug/ and debug_data/ for visualization files
    viz_locations = [
        ("matches", data_dir / "matches", "../debug_data/matches"),
        ("tracks", output_dir / "tracks", "tracks"),
        ("sfm", output_dir / "sfm", "sfm"),
        ("cameras", data_dir / "cameras", "../debug_data/cameras"),
        ("reprojection", data_dir / "reprojection", "../debug_data/reprojection"),
        ("comparison", output_dir / "comparison", "comparison"),
    ]

    for name, path, href_prefix in viz_locations:
        if path.exists():
            html_files = list(path.glob("*.html"))
            png_files = list(path.glob("*.png"))
            if html_files:
                for html_file in html_files:
                    viz_links.append(f'<li><a href="{href_prefix}/{html_file.name}">{name}: {html_file.stem}</a></li>')
            elif png_files:
                viz_links.append(f'<li><a href="{href_prefix}/{png_files[0].name}">{name}: {png_files[0].stem}</a></li>')

    # Add link to PLY files location
    if data_dir.exists():
        ply_count = len(list(data_dir.glob("*.ply")))
        if ply_count > 0:
            viz_links.append(f'<li>PLY point clouds: {ply_count} files in <code>debug_data/</code></li>')

    viz_links_html = "<ul>" + "".join(viz_links) + "</ul>"

    # Create summarized stats for HTML (remove large arrays)
    def summarize_stats(obj):
        """Recursively summarize stats, replacing large arrays with summaries."""
        if isinstance(obj, dict):
            return {k: summarize_stats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if len(obj) > 20:
                return f"[{len(obj)} items, min={min(obj)}, max={max(obj)}, mean={sum(obj)/len(obj):.1f}]"
            return obj
        return obj

    summarized_stats = summarize_stats(stats)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>360 to 3DGS Debug Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .section {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .warning {{ background: #fff3cd; border-left-color: #ffc107; }}
        .error {{ background: #f8d7da; border-left-color: #dc3545; }}
        ul {{ margin: 10px 0; padding-left: 20px; }}
        li {{ margin: 5px 0; }}
        a {{ color: #007bff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
        .stat-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .stat-card h3 {{ margin-top: 0; color: #333; font-size: 14px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>360 Video to 3DGS Debug Report</h1>

    <p><strong>Generated:</strong> {stats.get('timestamp', 'N/A')}</p>
    <p><strong>Dataset:</strong> {stats.get('dataset_dir', 'N/A')}</p>

    <h2>Quick Stats</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <h3>SfM Points</h3>
            <div class="stat-value">{sfm_stats.get('total_points', 'N/A')}</div>
        </div>
        <div class="stat-card">
            <h3>Cameras</h3>
            <div class="stat-value">{camera_stats.get('n_cameras', 'N/A')}</div>
        </div>
        <div class="stat-card">
            <h3>Total Tracks</h3>
            <div class="stat-value">{track_stats.get('total_tracks', 'N/A')}</div>
        </div>
        <div class="stat-card">
            <h3>3+ View Tracks</h3>
            <div class="stat-value">{track_stats.get('percentages', {}).get('3_plus_pct', 0):.1f}%</div>
        </div>
    </div>

    <h2>Diagnosis</h2>
    <div class="section {'warning' if diagnosis else ''}">
        {diagnosis_html}
    </div>

    <h2>Recommendations</h2>
    <div class="section">
        {recommendations_html}
    </div>

    <h2>Visualizations</h2>
    <div class="section">
        {viz_links_html}
    </div>

    <h2>Detailed Statistics</h2>
    <div class="section">
        <details>
            <summary>Click to expand statistics summary</summary>
            <pre>{json.dumps(summarized_stats, indent=2)}</pre>
        </details>
        <p><small>Full statistics saved to <a href="stats.json">stats.json</a></small></p>
    </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run all debug visualizations for 360 to 3DGS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all visualizations
    uv run python run_debug_viz.py ../../test_360_full

    # Specify output directory
    uv run python run_debug_viz.py ../../test_360_full --output ../../test_360_full/debug

    # Skip SfM vs 3DGS comparison
    uv run python run_debug_viz.py ../../test_360_full --skip-comparison
"""
    )

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Root directory containing frames/, opensfm/, etc."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: dataset_dir/debug)"
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip SfM vs 3DGS comparison"
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output.resolve() if args.output else dataset_dir / "debug"

    run_all_visualizations(
        dataset_dir,
        output_dir,
        skip_comparison=args.skip_comparison,
    )


if __name__ == "__main__":
    main()
