"""
Consolidated Droplet Analysis Pipeline

This pipeline chains together multiple processing steps:
1. filter_droplets.py - Basic geometric and duration filtering
2. batch_focus_metrics_parallel.py - Compute focus quality metrics
3. edge_based_focus_filter.py - Apply edge-based focus filtering
4. trajectory_focus_analyzer.py - Trajectory-level quality assessment

Usage:
    python droplet_analysis_pipeline.py \
        --input raw_detections.parquet \
        --image-dir frames/ \
        --output-dir ./analysis_output \
        --pixel-to-micron 2.46

Author: Analysis pipeline for microfluidic droplet tracking
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
try:
    from filter_droplets import DropletFilter
except ImportError:
    print("ERROR: Could not import filter_droplets module")
    print("Make sure filter_droplets.py is in the same directory or Python path")
    sys.exit(1)

try:
    from batch_focus_metrics_parallel import ParallelBatchProcessor
except ImportError:
    print("ERROR: Could not import batch_focus_metrics_parallel module")
    print(
        "Make sure batch_focus_metrics_parallel.py is in the same directory or Python path"
    )
    sys.exit(1)

try:
    from edge_based_focus_filter import EdgeFocusFilter
except ImportError:
    print("ERROR: Could not import edge_based_focus_filter module")
    print(
        "Make sure edge_based_focus_filter.py is in the same directory or Python path"
    )
    sys.exit(1)

try:
    from trajectory_focus_analyzer import TrajectoryFocusAnalyzer
except ImportError:
    print("ERROR: Could not import trajectory_focus_analyzer module")
    print(
        "Make sure trajectory_focus_analyzer.py is in the same directory or Python path"
    )
    sys.exit(1)


class DropletAnalysisPipeline:
    """
    Consolidated pipeline for droplet analysis

    Stages:
    1. Geometric filtering (area, duration)
    2. Focus metric computation (edge gradients, etc.)
    3. Edge-based focus filtering
    4. Trajectory-level quality assessment
    """

    def __init__(
        self,
        input_parquet: str,
        image_dir: str,
        output_dir: str,
        pixel_to_micron: float = 2.46,
        # Stage 1: filter_droplets parameters
        min_area: float = 200,
        min_duration: int = 200,
        focus_percentile: float = 0.0,
        use_trajectory_area: bool = True,
        # Stage 2: focus metrics parameters
        n_workers: int = None,
        metrics_level: str = "comprehensive",
        # Stage 3: edge filter parameters
        edge_filter_mode: str = "moderate",
        # Stage 4: trajectory analyzer parameters
        trajectory_focus_threshold: float = 0.7,
    ):
        """
        Initialize pipeline

        Args:
            input_parquet: Path to input parquet from SAM2
            image_dir: Directory containing frame images
            output_dir: Directory for all outputs
            pixel_to_micron: Pixel to micron conversion factor

            # Stage 1 parameters
            min_area: Minimum droplet area (pixels²)
            min_duration: Minimum trajectory duration (frames)
            focus_percentile: Percentile for focus threshold (0 = disable)
            use_trajectory_area: Use mean trajectory area for filtering

            # Stage 2 parameters
            n_workers: Number of parallel workers (None = auto)
            metrics_level: Focus metrics level ('basic' or 'comprehensive')

            # Stage 3 parameters
            edge_filter_mode: Edge filter mode ('conservative', 'moderate', 'aggressive')

            # Stage 4 parameters
            trajectory_focus_threshold: Minimum fraction of in-focus frames
        """

        self.input_parquet = Path(input_parquet)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.pixel_to_micron = pixel_to_micron

        # Stage 1 parameters
        self.min_area = min_area
        self.min_duration = min_duration
        self.focus_percentile = focus_percentile
        self.use_trajectory_area = use_trajectory_area

        # Stage 2 parameters
        self.n_workers = n_workers
        self.metrics_level = metrics_level

        # Stage 3 parameters
        self.edge_filter_mode = edge_filter_mode

        # Stage 4 parameters
        self.trajectory_focus_threshold = trajectory_focus_threshold

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define stage output paths
        self.stage1_output = self.output_dir / "stage1_geometric_filtered.parquet"
        self.stage2_output = self.output_dir / "stage2_with_focus_metrics.parquet"
        self.stage3_output = self.output_dir / "stage3_edge_filtered.parquet"
        self.stage4_output = self.output_dir / "stage4_trajectory_quality.parquet"
        self.final_output = self.output_dir / "final_filtered_droplets.parquet"

        # Track timing
        self.stage_times = {}

    def run(self, skip_stages: list = None):
        """
        Run the complete pipeline

        Args:
            skip_stages: List of stage numbers to skip (e.g., [1, 2])
        """

        if skip_stages is None:
            skip_stages = []

        print("=" * 70)
        print("DROPLET ANALYSIS PIPELINE")
        print("=" * 70)
        print(f"Input: {self.input_parquet}")
        print(f"Image directory: {self.image_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Pixel to micron: {self.pixel_to_micron}")
        print()

        total_start = time.time()

        # Stage 1: Geometric filtering
        if 1 not in skip_stages:
            self.run_stage1()
        else:
            print("Skipping Stage 1 (Geometric filtering)")
            print()

        # Stage 2: Focus metrics computation
        if 2 not in skip_stages:
            self.run_stage2()
        else:
            print("Skipping Stage 2 (Focus metrics)")
            print()

        # Stage 3: Edge-based focus filtering
        if 3 not in skip_stages:
            self.run_stage3()
        else:
            print("Skipping Stage 3 (Edge filtering)")
            print()

        # Stage 4: Trajectory quality assessment
        if 4 not in skip_stages:
            self.run_stage4()
        else:
            print("Skipping Stage 4 (Trajectory quality)")
            print()

        total_time = time.time() - total_start

        # Final summary
        self.print_summary(total_time)

    def run_stage1(self):
        """Stage 1: Geometric filtering (area, duration)"""

        print("=" * 70)
        print("STAGE 1: GEOMETRIC FILTERING")
        print("=" * 70)
        print(f"  Min area: {self.min_area} px²")
        print(f"  Min duration: {self.min_duration} frames")
        print(f"  Focus percentile: {self.focus_percentile}")
        print(f"  Use trajectory area: {self.use_trajectory_area}")
        print()

        start_time = time.time()

        # Load data
        print(f"Loading: {self.input_parquet}")
        df = pd.read_parquet(self.input_parquet)

        # Handle trajectory_id column name
        if "trajectory_id" not in df.columns and "object_id" in df.columns:
            df["trajectory_id"] = df["object_id"]

        print(f"  Loaded {len(df)} detections")
        print(f"  Unique trajectories: {df['trajectory_id'].nunique()}")
        print()

        # Apply filters
        filter_obj = DropletFilter(df)
        df_filtered = filter_obj.apply_filters(
            min_area=self.min_area,
            min_duration=self.min_duration,
            min_detection_rate=None,  # Not using this filter
            focus_percentile=self.focus_percentile,
            use_trajectory_area=self.use_trajectory_area,
            combine_focus_metrics="or",
        )

        # Print statistics
        filter_obj.print_filter_stats(df_filtered)

        # Save with flags (for next stage)
        df_filtered.to_parquet(self.stage1_output)
        print(f"\n✓ Saved to: {self.stage1_output}")

        self.stage_times["stage1"] = time.time() - start_time
        print(f"  Time: {self.stage_times['stage1']:.1f}s")
        print()

    def run_stage2(self):
        """Stage 2: Focus metrics computation"""

        print("=" * 70)
        print("STAGE 2: FOCUS METRICS COMPUTATION")
        print("=" * 70)
        print(f"  Metrics level: {self.metrics_level}")
        print(f"  Workers: {self.n_workers if self.n_workers else 'auto'}")
        print()

        start_time = time.time()

        # Use stage 1 output as input
        input_file = self.stage1_output

        # Create processor
        processor = ParallelBatchProcessor(
            parquet_path=str(input_file),
            image_dir=str(self.image_dir),
            pixel_to_micron=self.pixel_to_micron,
            n_workers=self.n_workers,
            metrics_level=self.metrics_level,
        )

        # Process
        _ = processor.process_all(save_path=str(self.stage2_output))

        self.stage_times["stage2"] = time.time() - start_time
        print(f"\n✓ Time: {self.stage_times['stage2']:.1f}s")
        print()

    def run_stage3(self):
        """Stage 3: Edge-based focus filtering"""

        print("=" * 70)
        print("STAGE 3: EDGE-BASED FOCUS FILTERING")
        print("=" * 70)
        print(f"  Filter mode: {self.edge_filter_mode}")
        print()

        start_time = time.time()

        # Load data from stage 2
        df = pd.read_parquet(self.stage2_output)
        print(f"Loaded {len(df)} detections with focus metrics")
        print()

        # Apply edge-based filter
        filter_obj = EdgeFocusFilter(mode=self.edge_filter_mode)
        df["in_focus"] = filter_obj.filter_dataframe(df)

        # Print analysis
        filter_obj.print_analysis(df)

        # Save
        df.to_parquet(self.stage3_output)
        print(f"\n✓ Saved to: {self.stage3_output}")

        self.stage_times["stage3"] = time.time() - start_time
        print(f"  Time: {self.stage_times['stage3']:.1f}s")
        print()

    def run_stage4(self):
        """Stage 4: Trajectory-level quality assessment"""

        print("=" * 70)
        print("STAGE 4: TRAJECTORY QUALITY ASSESSMENT")
        print("=" * 70)
        print(f"  Focus threshold: {self.trajectory_focus_threshold}")
        print()

        start_time = time.time()

        # Load data from stage 3
        df = pd.read_parquet(self.stage3_output)
        print(f"Loaded {len(df)} detections with focus filtering")
        print()

        # Initialize analyzer
        analyzer = TrajectoryFocusAnalyzer(
            focus_threshold=self.trajectory_focus_threshold,
            edge_gradient_mean_thresh=114.0,
            edge_gradient_std_thresh=70.0,
            edge_laplacian_var_thresh=4435.0,
        )

        # Add trajectory quality to dataframe
        df_with_quality = analyzer.add_trajectory_quality_to_dataframe(df)

        # Print distribution
        print("Trajectory Quality Distribution:")
        print("-" * 70)
        for quality in ["excellent", "good", "marginal", "poor", "reject"]:
            n_traj = df_with_quality[df_with_quality["traj_quality_flag"] == quality][
                "trajectory_id"
            ].nunique()
            n_det = (df_with_quality["traj_quality_flag"] == quality).sum()
            pct = n_det / len(df_with_quality) * 100
            print(
                f"  {quality.capitalize():<12} {n_traj:>4} trajectories, "
                f"{n_det:>6} detections ({pct:>5.1f}%)"
            )
        print()

        # Save full data with quality flags
        df_with_quality.to_parquet(self.stage4_output)
        print(f"✓ Saved full data with quality flags to: {self.stage4_output}")

        # Also save separate trajectory quality summary
        traj_summary = analyzer.analyze_trajectories(df)
        summary_path = self.output_dir / "trajectory_quality_summary.csv"
        traj_summary.to_csv(summary_path, index=False)
        print(f"✓ Saved trajectory summary to: {summary_path}")

        # Create filtered versions
        print()
        print("Creating filtered versions:")
        print("-" * 70)

        # Conservative (excellent + good)
        conservative = df_with_quality[
            df_with_quality["traj_quality_flag"].isin(["excellent", "good"])
        ]
        conservative_path = self.output_dir / "filtered_conservative.parquet"
        conservative.to_parquet(conservative_path)
        print(
            f"  Conservative (excellent+good): {len(conservative)} detections "
            f"({len(conservative) / len(df_with_quality) * 100:.1f}%)"
        )
        print(f"    → {conservative_path}")

        # Balanced (excellent + good + marginal)
        balanced = df_with_quality[
            df_with_quality["traj_quality_flag"].isin(["excellent", "good", "marginal"])
        ]
        balanced_path = self.output_dir / "filtered_balanced.parquet"
        balanced.to_parquet(balanced_path)
        print(
            f"  Balanced (excellent+good+marginal): {len(balanced)} detections "
            f"({len(balanced) / len(df_with_quality) * 100:.1f}%)"
        )
        print(f"    → {balanced_path}")

        # Set final output to conservative by default
        conservative.to_parquet(self.final_output)
        print()
        print(f"✓ Final output (conservative): {self.final_output}")

        self.stage_times["stage4"] = time.time() - start_time
        print(f"\n  Time: {self.stage_times['stage4']:.1f}s")
        print()

    def print_summary(self, total_time: float):
        """Print pipeline summary"""

        print()
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print()

        # Timing summary
        print("Timing Summary:")
        print("-" * 70)
        for stage, duration in self.stage_times.items():
            print(f"  {stage}: {duration:.1f}s")
        print(f"  Total: {total_time:.1f}s")
        print()

        # Output files
        print("Output Files:")
        print("-" * 70)

        outputs = [
            ("Stage 1 (Geometric filter)", self.stage1_output),
            ("Stage 2 (With focus metrics)", self.stage2_output),
            ("Stage 3 (Edge filtered)", self.stage3_output),
            ("Stage 4 (With trajectory quality)", self.stage4_output),
            (
                "Trajectory summary (CSV)",
                self.output_dir / "trajectory_quality_summary.csv",
            ),
            (
                "Conservative filtered",
                self.output_dir / "filtered_conservative.parquet",
            ),
            ("Balanced filtered", self.output_dir / "filtered_balanced.parquet"),
            ("FINAL OUTPUT", self.final_output),
        ]

        for name, path in outputs:
            if path.exists():
                size_mb = path.stat().st_size / 1e6
                print(f"  {name}:")
                print(f"    {path}")
                print(f"    ({size_mb:.1f} MB)")

        print()
        print("=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print()
        print("1. Recommended for analysis:")
        print(f"   {self.final_output}")
        print("   (Conservative filtering: excellent + good trajectories only)")
        print()
        print("2. For testing with more data:")
        print(f"   {self.output_dir / 'filtered_balanced.parquet'}")
        print("   (Balanced filtering: includes marginal quality)")
        print()
        print("3. All data with quality flags:")
        print(f"   {self.stage4_output}")
        print("   (Use for custom filtering)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated droplet analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Basic usage with defaults
  python droplet_analysis_pipeline.py \\
      --input raw_detections.parquet \\
      --image-dir frames/ \\
      --output-dir ./output

  # Custom parameters
  python droplet_analysis_pipeline.py \\
      --input raw_detections.parquet \\
      --image-dir frames/ \\
      --output-dir ./output \\
      --pixel-to-micron 2.46 \\
      --min-area 300 \\
      --min-duration 150 \\
      --n-workers 16

  # Skip stages (if already run)
  python droplet_analysis_pipeline.py \\
      --input raw_detections.parquet \\
      --image-dir frames/ \\
      --output-dir ./output \\
      --skip-stages 1 2  # Skip stages 1 and 2
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input", required=True, help="Input parquet file from SAM2 detection"
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing frame images (JPG or PNG)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for all results"
    )

    # Optional: General parameters
    parser.add_argument(
        "--pixel-to-micron",
        type=float,
        default=2.46,
        help="Pixel to micron conversion factor (default: 2.46)",
    )

    # Optional: Stage 1 parameters
    parser.add_argument(
        "--min-area",
        type=float,
        default=200,
        help="Minimum droplet area in pixels² (default: 200)",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=200,
        help="Minimum trajectory duration in frames (default: 200)",
    )
    parser.add_argument(
        "--focus-percentile",
        type=float,
        default=0.0,
        help="Focus threshold percentile, 0=disable (default: 0.0)",
    )
    parser.add_argument(
        "--use-trajectory-area",
        action="store_true",
        default=True,
        help="Use mean trajectory area for filtering (default: True)",
    )

    # Optional: Stage 2 parameters
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)",
    )
    parser.add_argument(
        "--metrics",
        choices=["basic", "comprehensive"],
        default="comprehensive",
        help="Focus metrics level (default: comprehensive)",
    )

    # Optional: Stage 3 parameters
    parser.add_argument(
        "--edge-filter-mode",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
        help="Edge filter mode (default: moderate)",
    )

    # Optional: Stage 4 parameters
    parser.add_argument(
        "--trajectory-focus-threshold",
        type=float,
        default=0.7,
        help="Minimum fraction of in-focus frames (default: 0.7)",
    )

    # Optional: Pipeline control
    parser.add_argument(
        "--skip-stages",
        type=int,
        nargs="+",
        default=None,
        help="Stages to skip (e.g., --skip-stages 1 2)",
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = DropletAnalysisPipeline(
        input_parquet=args.input,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        pixel_to_micron=args.pixel_to_micron,
        # Stage 1
        min_area=args.min_area,
        min_duration=args.min_duration,
        focus_percentile=args.focus_percentile,
        use_trajectory_area=args.use_trajectory_area,
        # Stage 2
        n_workers=args.n_workers,
        metrics_level=args.metrics,
        # Stage 3
        edge_filter_mode=args.edge_filter_mode,
        # Stage 4
        trajectory_focus_threshold=args.trajectory_focus_threshold,
    )

    pipeline.run(skip_stages=args.skip_stages)


if __name__ == "__main__":
    main()

    # import os

    # indices = range(506, 519)
    # for i in indices:
    #     input_parquet = Path(
    #         f"/home/leviathan/cdrucker/ml/sam2/outputs/vid{i}_rolling1/blob_analysis/blob_features.parquet"
    #     )
    #     image_dir = Path(
    #         f"/home/leviathan/cdrucker/videos/jpegVideoFrames/{i}_0p8mlmin_800fps_lambda1_mu4_500prep_1000cent"
    #     )
    #     output_dir = Path(
    #         f"/home/leviathan/cdrucker/ml/sam2/outputs/vid{i}_rolling1/filtering"
    #     )
    #     if (
    #         os.path.exists(input_parquet)
    #         and os.path.exists(image_dir)
    #         and not os.path.exists(
    #             f"/home/leviathan/cdrucker/ml/sam2/outputs/vid{i}_rolling1/filtering/final_filtered_droplets.parquet"
    #         )
    #     ):
    #         os.makedirs(output_dir, exist_ok=True)
    #         # Create and run pipeline
    #         pipeline = DropletAnalysisPipeline(
    #             input_parquet=input_parquet,
    #             image_dir=image_dir,
    #             output_dir=output_dir,
    #             pixel_to_micron=2.46,
    #             # Stage 1
    #             min_area=200,
    #             min_duration=200,
    #             focus_percentile=0,
    #             use_trajectory_area=True,
    #             # Stage 2
    #             n_workers=None,
    #             metrics_level="comprehensive",
    #             # Stage 3
    #             edge_filter_mode="moderate",
    #             # Stage 4
    #             trajectory_focus_threshold=0.7,
    #         )

    #         pipeline.run(skip_stages=None)
    #     else:
    #         print(f"\nFailed to analyze index {i}")
    # continue
