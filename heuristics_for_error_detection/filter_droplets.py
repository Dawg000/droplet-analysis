"""
Practical Droplet Filtering Script
Combines minimum area filter with focus quality thresholds

Usage:
    python filter_droplets.py --parquet results.parquet --output filtered_results.parquet
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class DropletFilter:
    """Simple, interpretable droplet filtering with clear rationale"""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with SAM2 analysis results
        """
        self.df = df.copy()
        self.original_count = len(df)

    def compute_trajectory_stats(self) -> pd.DataFrame:
        """
        Compute trajectory-level statistics for filtering

        Returns:
            DataFrame with one row per trajectory
        """
        if "trajectory_id" not in self.df.columns:
            # If no trajectory tracking, treat each detection independently
            self.df["trajectory_id"] = self.df.index

        # Group by trajectory
        traj_stats = self.df.groupby("trajectory_id").agg(
            {
                "area": ["mean", "std", "min", "max", "count"],
                "frame": ["min", "max", "count"],
                # "laplacian_var": ["mean", "std", "min"],
                # "sobel_mean": ["mean", "std", "min"],
                # "brenner_gradient": ["mean", "std", "min"],
            }
        )

        # Flatten column names
        traj_stats.columns = [
            "_".join(col).strip() for col in traj_stats.columns.values
        ]
        traj_stats = traj_stats.reset_index()

        # Add duration
        traj_stats["duration"] = traj_stats["frame_max"] - traj_stats["frame_min"] + 1
        traj_stats["detection_rate"] = (
            traj_stats["frame_count"] / traj_stats["duration"]
        )

        return traj_stats

    def analyze_area_distribution(self):
        """Visualize area distribution to help choose threshold"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overall distribution
        ax = axes[0]
        ax.hist(
            self.df["area"],
            bins=200,
            alpha=0.7,
            edgecolor="black",
            range=(1, self.df["area"].quantile(0.98)),
        )
        ax.set_xlabel("Area (pixels²)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            "Area Distribution - All Detections", fontsize=14, fontweight="bold"
        )
        ax.axvline(
            self.df["area"].quantile(0.05),
            color="red",
            linestyle="--",
            linewidth=2,
            label="5th percentile",
        )
        ax.axvline(
            self.df["area"].quantile(0.10),
            color="orange",
            linestyle="--",
            linewidth=2,
            label="10th percentile",
        )
        ax.legend()
        ax.grid(alpha=0.3)

        # Trajectory average distribution
        ax = axes[1]
        traj_stats = self.compute_trajectory_stats()
        ax.hist(
            traj_stats["area_mean"],
            bins=200,
            alpha=0.7,
            edgecolor="black",
            color="green",
            range=(1, traj_stats["area_mean"].quantile(0.98)),
        )
        ax.set_xlabel("Mean Area per Trajectory (pixels²)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            "Area Distribution - Trajectory Averages", fontsize=14, fontweight="bold"
        )
        ax.axvline(
            traj_stats["area_mean"].quantile(0.05),
            color="red",
            linestyle="--",
            linewidth=2,
            label="5th percentile",
        )
        ax.axvline(
            traj_stats["area_mean"].quantile(0.10),
            color="orange",
            linestyle="--",
            linewidth=2,
            label="10th percentile",
        )
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Print statistics
        print("\nArea Distribution Statistics:")
        print("=" * 60)
        print("\nPer-detection area:")
        print(f"  5th percentile:  {self.df['area'].quantile(0.05):.1f} px²")
        print(f"  10th percentile: {self.df['area'].quantile(0.10):.1f} px²")
        print(f"  25th percentile: {self.df['area'].quantile(0.25):.1f} px²")
        print(f"  Median:          {self.df['area'].median():.1f} px²")
        print(f"  Mean:            {self.df['area'].mean():.1f} px²")

        print("\nPer-trajectory mean area:")
        print(f"  5th percentile:  {traj_stats['area_mean'].quantile(0.05):.1f} px²")
        print(f"  10th percentile: {traj_stats['area_mean'].quantile(0.10):.1f} px²")
        print(f"  25th percentile: {traj_stats['area_mean'].quantile(0.25):.1f} px²")
        print(f"  Median:          {traj_stats['area_mean'].median():.1f} px²")
        print(f"  Mean:            {traj_stats['area_mean'].mean():.1f} px²")

        return fig

    def apply_filters(
        self,
        min_area: float = None,
        min_duration: int = None,
        min_detection_rate: float = None,
        focus_percentile: float = 0.05,
        use_trajectory_area: bool = True,
        combine_focus_metrics: str = "or",  # 'or', 'and', 'vote'
    ) -> pd.DataFrame:
        """
        Apply filtering based on interpretable heuristics

        Args:
            min_area: Minimum area threshold (None = use 5th percentile)
            min_duration: Minimum trajectory duration in frames
            min_detection_rate: Minimum detection rate (detections/duration)
            focus_percentile: Percentile for focus thresholds (default 0.05 = bottom 5%)
            use_trajectory_area: If True, use mean trajectory area; if False, per-detection
            combine_focus_metrics: How to combine focus metrics ('or', 'and', 'vote')

        Returns:
            Filtered DataFrame with flags added
        """
        df_filtered = self.df.copy()

        # Compute trajectory stats
        traj_stats = self.compute_trajectory_stats()

        # Merge trajectory stats back
        df_filtered = df_filtered.merge(
            traj_stats[
                [
                    "trajectory_id",
                    "area_mean",
                    "duration",
                    "detection_rate",
                    # "laplacian_var_mean",
                    # "sobel_mean_mean",
                    # "brenner_gradient_mean",
                ]
            ],
            on="trajectory_id",
            how="left",
        )

        # Initialize flags
        df_filtered["filter_reason"] = "keep"
        df_filtered["flag_small_area"] = False
        df_filtered["flag_short_duration"] = False
        df_filtered["flag_low_detection_rate"] = False
        df_filtered["flag_out_of_focus"] = False

        # 1. Area filter
        if min_area is None:
            if use_trajectory_area:
                min_area = traj_stats["area_mean"].quantile(focus_percentile)
                print(
                    f"Using trajectory mean area threshold: {min_area:.1f} px² "
                    f"({focus_percentile * 100:.0f}th percentile)"
                )
            else:
                min_area = df_filtered["area"].quantile(focus_percentile)
                print(
                    f"Using per-detection area threshold: {min_area:.1f} px² "
                    f"({focus_percentile * 100:.0f}th percentile)"
                )

        if use_trajectory_area:
            df_filtered["flag_small_area"] = df_filtered["area_mean"] < min_area
        else:
            df_filtered["flag_small_area"] = df_filtered["area"] < min_area

        # 2. Duration filter
        if min_duration is not None:
            df_filtered["flag_short_duration"] = df_filtered["duration"] < min_duration
            print(f"Using minimum duration threshold: {min_duration} frames")

        # 3. Detection rate filter
        if min_detection_rate is not None:
            df_filtered["flag_low_detection_rate"] = (
                df_filtered["detection_rate"] < min_detection_rate
            )
            print(f"Using minimum detection rate threshold: {min_detection_rate:.2f}")

        # # 4. Focus filters
        # focus_thresholds = {
        #     "laplacian_var": df_filtered["laplacian_var"].quantile(focus_percentile),
        #     "sobel_mean": df_filtered["sobel_mean"].quantile(focus_percentile),
        #     "brenner_gradient": df_filtered["brenner_gradient"].quantile(
        #         focus_percentile
        #     ),
        # }

        # print(f"\nFocus metric thresholds ({focus_percentile * 100:.0f}th percentile):")
        # for metric, value in focus_thresholds.items():
        #     print(f"  {metric}: {value:.2f}")

        # # Create individual focus flags
        # df_filtered["flag_laplacian"] = (
        #     df_filtered["laplacian_var"] < focus_thresholds["laplacian_var"]
        # )
        # df_filtered["flag_sobel"] = (
        #     df_filtered["sobel_mean"] < focus_thresholds["sobel_mean"]
        # )
        # df_filtered["flag_brenner"] = (
        #     df_filtered["brenner_gradient"] < focus_thresholds["brenner_gradient"]
        # )

        # # Combine focus flags
        # if combine_focus_metrics == "or":
        #     # Flag if ANY metric is low (most aggressive)
        #     df_filtered["flag_out_of_focus"] = (
        #         df_filtered["flag_laplacian"]
        #         | df_filtered["flag_sobel"]
        #         | df_filtered["flag_brenner"]
        #     )
        # elif combine_focus_metrics == "and":
        #     # Flag if ALL metrics are low (most conservative)
        #     df_filtered["flag_out_of_focus"] = (
        #         df_filtered["flag_laplacian"]
        #         & df_filtered["flag_sobel"]
        #         & df_filtered["flag_brenner"]
        #     )
        # elif combine_focus_metrics == "vote":
        #     # Flag if 2 or more metrics agree (balanced)
        #     vote_count = (
        #         df_filtered["flag_laplacian"].astype(int)
        #         + df_filtered["flag_sobel"].astype(int)
        #         + df_filtered["flag_brenner"].astype(int)
        #     )
        #     df_filtered["flag_out_of_focus"] = vote_count >= 2

        # Combine all flags
        df_filtered["flagged"] = (
            df_filtered["flag_small_area"]
            | df_filtered["flag_short_duration"]
            | df_filtered["flag_low_detection_rate"]
            | df_filtered["flag_out_of_focus"]
        )

        # Assign filter reasons
        df_filtered.loc[df_filtered["flag_small_area"], "filter_reason"] = "small_area"
        df_filtered.loc[df_filtered["flag_short_duration"], "filter_reason"] = (
            "short_duration"
        )
        df_filtered.loc[df_filtered["flag_low_detection_rate"], "filter_reason"] = (
            "low_detection_rate"
        )
        df_filtered.loc[df_filtered["flag_out_of_focus"], "filter_reason"] = (
            "out_of_focus"
        )

        # If multiple flags, use 'multiple'
        multi_flag = (
            df_filtered["flag_small_area"].astype(int)
            + df_filtered["flag_short_duration"].astype(int)
            + df_filtered["flag_low_detection_rate"].astype(int)
            + df_filtered["flag_out_of_focus"].astype(int)
        ) > 1
        df_filtered.loc[multi_flag & df_filtered["flagged"], "filter_reason"] = (
            "multiple"
        )

        return df_filtered

    def print_filter_stats(self, df_filtered: pd.DataFrame):
        """Print detailed filtering statistics"""
        print("\n" + "=" * 60)
        print("FILTERING SUMMARY")
        print("=" * 60)

        total = len(df_filtered)
        flagged = df_filtered["flagged"].sum()
        kept = total - flagged

        print(f"\nTotal detections:    {total:6d}")
        print(f"Flagged for removal: {flagged:6d} ({flagged / total * 100:5.1f}%)")
        print(f"Kept (clean):        {kept:6d} ({kept / total * 100:5.1f}%)")

        print("\nBreakdown by filter reason:")
        for reason in df_filtered["filter_reason"].unique():
            if reason == "keep":
                continue
            count = (df_filtered["filter_reason"] == reason).sum()
            pct = count / total * 100
            print(f"  {reason:20s}: {count:6d} ({pct:5.1f}%)")

        print("\nIndividual flag counts (may overlap):")
        print(f"  Small area:           {df_filtered['flag_small_area'].sum():6d}")
        print(f"  Short duration:       {df_filtered['flag_short_duration'].sum():6d}")
        print(
            f"  Low detection rate:   {df_filtered['flag_low_detection_rate'].sum():6d}"
        )
        # print(f"  Out of focus:         {df_filtered['flag_out_of_focus'].sum():6d}")

        # # Focus metric breakdown
        # print("\n  Focus metric contributions:")
        # print(f"    Low laplacian_var:     {df_filtered['flag_laplacian'].sum():6d}")
        # print(f"    Low sobel_mean:        {df_filtered['flag_sobel'].sum():6d}")
        # print(f"    Low brenner_gradient:  {df_filtered['flag_brenner'].sum():6d}")

    def visualize_filtering_results(
        self, df_filtered: pd.DataFrame, save_path: str = None
    ):
        """Create visualization of filtering results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        kept = df_filtered[~df_filtered["flagged"]]
        flagged = df_filtered[df_filtered["flagged"]]

        # 1. Area comparison
        ax = axes[0, 0]
        ax.hist(
            [kept["area"], flagged["area"]],
            bins=30,
            label=["Kept", "Flagged"],
            alpha=0.7,
            color=["green", "red"],
        )
        ax.set_xlabel("Area (px²)")
        ax.set_ylabel("Count")
        ax.set_title("Area Distribution")
        ax.legend()
        ax.set_yscale("log")

        # # 2. Laplacian var comparison
        # ax = axes[0, 1]
        # ax.hist(
        #     [kept["laplacian_var"], flagged["laplacian_var"]],
        #     bins=30,
        #     label=["Kept", "Flagged"],
        #     alpha=0.7,
        #     color=["green", "red"],
        # )
        # ax.set_xlabel("Laplacian Variance")
        # ax.set_ylabel("Count")
        # ax.set_title("Laplacian Variance")
        # ax.legend()
        # ax.set_yscale("log")

        # # 3. Sobel mean comparison
        # ax = axes[0, 2]
        # ax.hist(
        #     [kept["sobel_mean"], flagged["sobel_mean"]],
        #     bins=30,
        #     label=["Kept", "Flagged"],
        #     alpha=0.7,
        #     color=["green", "red"],
        # )
        # ax.set_xlabel("Sobel Mean")
        # ax.set_ylabel("Count")
        # ax.set_title("Sobel Mean Edge Strength")
        # ax.legend()
        # ax.set_yscale("log")

        # 4. Spatial distribution
        ax = axes[1, 0]
        ax.scatter(
            kept["centroid_x"],
            kept["centroid_y"],
            s=5,
            alpha=0.3,
            c="green",
            label="Kept",
        )
        ax.scatter(
            flagged["centroid_x"],
            flagged["centroid_y"],
            s=5,
            alpha=0.3,
            c="red",
            label="Flagged",
        )
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Spatial Distribution")
        ax.legend()
        ax.set_aspect("equal")

        # 5. Area vs Focus
        ax = axes[1, 1]
        ax.scatter(
            kept["area"],
            kept["laplacian_var"],
            s=10,
            alpha=0.3,
            c="green",
            label="Kept",
        )
        ax.scatter(
            flagged["area"],
            flagged["laplacian_var"],
            s=10,
            alpha=0.3,
            c="red",
            label="Flagged",
        )
        ax.set_xlabel("Area (px²)")
        ax.set_ylabel("Laplacian Variance")
        ax.set_title("Area vs Focus Quality")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")

        # 6. Filter reason pie chart
        ax = axes[1, 2]
        reason_counts = df_filtered[df_filtered["flagged"]][
            "filter_reason"
        ].value_counts()
        ax.pie(reason_counts.values, labels=reason_counts.index, autopct="%1.1f%%")
        ax.set_title("Flagged Detection Reasons")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nVisualization saved to {save_path}")

        return fig


def main():
    parser = argparse.ArgumentParser(description="Filter droplet detections")
    parser.add_argument("--parquet", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument(
        "--min-area",
        type=float,
        default=None,
        help="Minimum area threshold (default: auto from 5th percentile)",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=5,
        help="Minimum trajectory duration (frames)",
    )
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.5,
        help="Minimum detection rate (0-1)",
    )
    parser.add_argument(
        "--focus-percentile",
        type=float,
        default=0.05,
        help="Focus threshold percentile (default: 0.05 = bottom 5%%)",
    )
    parser.add_argument(
        "--use-trajectory-area",
        action="store_true",
        help="Use mean trajectory area instead of per-detection",
    )
    parser.add_argument(
        "--focus-combination",
        choices=["or", "and", "vote"],
        default="vote",
        help="How to combine focus metrics",
    )
    parser.add_argument(
        "--viz-output", default=None, help="Path to save visualization (optional)"
    )
    parser.add_argument(
        "--keep-flags",
        action="store_true",
        help="Keep error flags in output (for trajectory-level filtering). "
        "If set, outputs ALL data with flags; if not set, outputs only clean data.",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    if "trajectory_id" not in df.columns and "object_id" in df.columns:
        df["trajectory_id"] = df["object_id"]
    print(f"Loaded {len(df)} detections")

    os.makedirs(Path(args.output).parent, exist_ok=True)

    # Initialize filter
    filter_obj = DropletFilter(df)

    # Show area distribution
    print("\nAnalyzing area distribution...")
    fig_area = filter_obj.analyze_area_distribution()
    area_plot_path = Path(args.output).parent / "area_distribution.png"
    fig_area.savefig(area_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved area distribution to {area_plot_path}")
    plt.close(fig_area)

    # Apply filters
    print("\nApplying filters...")
    df_filtered = filter_obj.apply_filters(
        min_area=args.min_area,
        min_duration=args.min_duration,
        min_detection_rate=args.min_detection_rate,
        focus_percentile=args.focus_percentile,
        use_trajectory_area=args.use_trajectory_area,
        combine_focus_metrics=args.focus_combination,
    )

    # Print statistics
    filter_obj.print_filter_stats(df_filtered)

    # Visualize results
    if args.viz_output:
        viz_path = args.viz_output
    else:
        viz_path = Path(args.output).parent / "filtering_results.png"

    # fig_results = filter_obj.visualize_filtering_results(df_filtered, viz_path)
    # plt.close(fig_results)

    # Save data based on --keep-flags option
    if args.keep_flags:
        # Keep ALL data with flags for trajectory-level filtering
        print("\n--keep-flags is set: Saving ALL data with error flags")

        df_filtered.to_parquet(args.output)
        print(f"Saved {len(df_filtered)} detections (flagged + clean) to {args.output}")
        print(f"  - Clean: {(~df_filtered['flagged']).sum()}")
        print(f"  - Flagged: {df_filtered['flagged'].sum()}")

        # Also save separate files for inspection
        flagged_path = str(args.output).replace(".parquet", "_flagged_only.parquet")
        df_filtered[df_filtered["flagged"]].to_parquet(flagged_path)
        print(f"Also saved flagged-only subset to {flagged_path} (for inspection)")

    else:
        # Original behavior: save only clean data
        print("\nSaving clean data only (no flags)")
        df_clean = df_filtered[~df_filtered["flagged"]].copy()

        # Drop temporary columns
        cols_to_drop = [
            c for c in df_clean.columns if c.startswith("flag_") or c == "filter_reason"
        ]
        df_clean = df_clean.drop(columns=cols_to_drop)

        df_clean.to_parquet(args.output)
        print(f"Saved {len(df_clean)} clean detections to {args.output}")

        # Also save flagged data for inspection
        flagged_path = str(args.output).replace(".parquet", "_flagged.parquet")
        df_filtered[df_filtered["flagged"]].to_parquet(flagged_path)
        print(
            f"Saved {df_filtered['flagged'].sum()} flagged detections to {flagged_path}"
        )


if __name__ == "__main__":
    main()
