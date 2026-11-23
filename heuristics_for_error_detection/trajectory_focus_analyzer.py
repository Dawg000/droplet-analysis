"""
Trajectory-Level Focus Quality Analysis

This module provides tools for:
1. Computing consolidated focus scores for entire trajectories
2. Identifying trajectory segments with poor focus
3. Detecting occlusion events (multiple droplets tracked as one)
4. Distinguishing occlusions from legitimate breakup events

Key insights from data analysis:
- Edge-based metrics (edge_gradient_mean, edge_gradient_std) are strongest
- Contrast metrics (contrast_ratio, modulation) are also very strong
- Temporal consistency helps identify occlusions and hallucinations

Usage:
    from trajectory_focus_analyzer import TrajectoryFocusAnalyzer

    analyzer = TrajectoryFocusAnalyzer()
    trajectory_scores = analyzer.analyze_trajectories(df)
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrajectoryQualityReport:
    """Report for a single trajectory"""

    trajectory_id: int
    n_frames: int

    # Focus quality metrics
    mean_focus_score: float
    median_focus_score: float
    min_focus_score: float
    fraction_in_focus: float  # Fraction of frames passing threshold

    # Temporal consistency metrics
    focus_score_std: float
    focus_score_trend: (
        float  # Linear trend (positive = improving, negative = degrading)
    )

    # Occlusion/breakup detection
    has_sudden_focus_change: bool
    focus_change_frames: List[int]
    suspected_occlusion: bool
    suspected_breakup: bool

    # Geometric consistency (for occlusion detection)
    area_cv: float  # Coefficient of variation in area
    solidity_mean: float  # Mean solidity (occlusions often have lower solidity)
    solidity_std: float

    # Recommendations
    quality_flag: str  # 'excellent', 'good', 'marginal', 'poor', 'reject'
    recommended_action: str


class TrajectoryFocusAnalyzer:
    """
    Analyze focus quality at the trajectory level

    Provides:
    1. Consolidated focus scores per trajectory
    2. Detection of poor-quality trajectory segments
    3. Occlusion detection via focus + geometry
    4. Breakup event detection
    """

    def __init__(
        self,
        focus_threshold: float = 0.7,  # Fraction of frames that must be in-focus
        edge_gradient_mean_thresh: float = 114.0,  # From data analysis
        edge_gradient_std_thresh: float = 70.0,
        edge_laplacian_var_thresh: float = 4435.0,
        sudden_change_threshold: float = 0.3,  # Relative change to flag as sudden
    ):
        """
        Initialize trajectory analyzer

        Args:
            focus_threshold: Minimum fraction of in-focus frames to accept trajectory
            edge_gradient_mean_thresh: Threshold for edge gradient mean
            edge_gradient_std_thresh: Threshold for edge gradient std
            edge_laplacian_var_thresh: Threshold for edge Laplacian variance
            sudden_change_threshold: Relative change threshold for detecting events
        """

        self.focus_threshold = focus_threshold
        self.edge_gradient_mean_thresh = edge_gradient_mean_thresh
        self.edge_gradient_std_thresh = edge_gradient_std_thresh
        self.edge_laplacian_var_thresh = edge_laplacian_var_thresh
        self.sudden_change_threshold = sudden_change_threshold

    def compute_frame_focus_score(
        self,
        edge_gradient_mean: float,
        edge_gradient_std: float,
        edge_laplacian_var: float,
    ) -> float:
        """
        Compute continuous focus score [0, 1] for a single frame

        Uses normalized edge metrics weighted equally
        """

        # Normalize relative to thresholds (2x threshold = perfect score)
        norm_grad_mean = np.clip(
            edge_gradient_mean / (2 * self.edge_gradient_mean_thresh), 0, 1
        )
        norm_grad_std = np.clip(
            edge_gradient_std / (2 * self.edge_gradient_std_thresh), 0, 1
        )
        norm_lap_var = np.clip(
            edge_laplacian_var / (2 * self.edge_laplacian_var_thresh), 0, 1
        )

        # Equal weighting
        score = (norm_grad_mean + norm_grad_std + norm_lap_var) / 3.0

        return score

    def analyze_single_trajectory(
        self, trajectory_df: pd.DataFrame
    ) -> TrajectoryQualityReport:
        """
        Analyze focus quality for a single trajectory

        Args:
            trajectory_df: DataFrame containing all frames for one trajectory
                          Must have: edge_gradient_mean, edge_gradient_std,
                          edge_laplacian_var, area, solidity

        Returns:
            TrajectoryQualityReport with comprehensive analysis
        """

        # Check required columns
        required = ["edge_gradient_mean", "edge_gradient_std", "edge_laplacian_var"]
        missing = [col for col in required if col not in trajectory_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        trajectory_id = (
            trajectory_df["trajectory_id"].iloc[0]
            if "trajectory_id" in trajectory_df.columns
            else -1
        )
        n_frames = len(trajectory_df)

        # Compute frame-by-frame focus scores
        focus_scores = np.array(
            [
                self.compute_frame_focus_score(
                    row["edge_gradient_mean"],
                    row["edge_gradient_std"],
                    row["edge_laplacian_var"],
                )
                for _, row in trajectory_df.iterrows()
            ]
        )

        # Binary in-focus flags
        in_focus = (
            (trajectory_df["edge_gradient_mean"] > self.edge_gradient_mean_thresh)
            & (trajectory_df["edge_gradient_std"] > self.edge_gradient_std_thresh)
            & (trajectory_df["edge_laplacian_var"] > self.edge_laplacian_var_thresh)
        ).values

        # Basic focus statistics
        mean_focus_score = np.mean(focus_scores)
        median_focus_score = np.median(focus_scores)
        min_focus_score = np.min(focus_scores)
        fraction_in_focus = np.mean(in_focus)
        focus_score_std = np.std(focus_scores)

        # Temporal trend (linear regression)
        if n_frames > 2:
            x = np.arange(n_frames)
            focus_score_trend = np.polyfit(x, focus_scores, 1)[0]  # Slope
        else:
            focus_score_trend = 0.0

        # Detect sudden focus changes (potential occlusion/separation events)
        focus_change_frames = []
        has_sudden_focus_change = False

        if n_frames > 1:
            # Look for large frame-to-frame changes
            focus_diffs = np.abs(np.diff(focus_scores))
            sudden_changes = focus_diffs > self.sudden_change_threshold

            if np.any(sudden_changes):
                has_sudden_focus_change = True
                focus_change_frames = np.where(sudden_changes)[0].tolist()

        # Geometric consistency (for occlusion detection)
        if "area" in trajectory_df.columns:
            areas = trajectory_df["area"].values
            area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0.0
        else:
            area_cv = 0.0

        if "solidity" in trajectory_df.columns:
            solidity_mean = np.mean(trajectory_df["solidity"].values)
            solidity_std = np.std(trajectory_df["solidity"].values)
        else:
            solidity_mean = 1.0
            solidity_std = 0.0

        # Occlusion detection heuristics
        suspected_occlusion = self._detect_occlusion(
            focus_scores, area_cv, solidity_mean, has_sudden_focus_change
        )

        # Breakup detection (rare but interesting)
        suspected_breakup = self._detect_breakup(
            focus_scores, area_cv, has_sudden_focus_change, focus_change_frames
        )

        # Overall quality assessment
        quality_flag, recommended_action = self._assess_quality(
            fraction_in_focus, mean_focus_score, suspected_occlusion, suspected_breakup
        )

        return TrajectoryQualityReport(
            trajectory_id=trajectory_id,
            n_frames=n_frames,
            mean_focus_score=mean_focus_score,
            median_focus_score=median_focus_score,
            min_focus_score=min_focus_score,
            fraction_in_focus=fraction_in_focus,
            focus_score_std=focus_score_std,
            focus_score_trend=focus_score_trend,
            has_sudden_focus_change=has_sudden_focus_change,
            focus_change_frames=focus_change_frames,
            suspected_occlusion=suspected_occlusion,
            suspected_breakup=suspected_breakup,
            area_cv=area_cv,
            solidity_mean=solidity_mean,
            solidity_std=solidity_std,
            quality_flag=quality_flag,
            recommended_action=recommended_action,
        )

    def _detect_occlusion(
        self,
        focus_scores: np.ndarray,
        area_cv: float,
        solidity_mean: float,
        has_sudden_focus_change: bool,
    ) -> bool:
        """
        Detect if trajectory likely represents occluded droplets

        Occlusions typically have:
        - Variable focus (parts in/out of plane)
        - High area variability (separation event)
        - Lower solidity (non-convex shape)
        - Sudden focus improvement when they separate
        """

        # Multiple indicators increase confidence
        indicators = 0

        # High area variability suggests separation/occlusion
        if area_cv > 0.3:
            indicators += 1

        # Low solidity suggests non-convex (occluded) shape
        if solidity_mean < 0.95:
            indicators += 1

        # Sudden focus improvement suggests separation
        if has_sudden_focus_change and np.mean(focus_scores[-5:]) > np.mean(
            focus_scores[:5]
        ):
            indicators += 1

        # Require at least 2 indicators
        return indicators >= 2

    def _detect_breakup(
        self,
        focus_scores: np.ndarray,
        area_cv: float,
        has_sudden_focus_change: bool,
        focus_change_frames: List[int],
    ) -> bool:
        """
        Detect potential droplet breakup event

        Breakup events typically have:
        - Sudden area decrease (loses mass)
        - Maintained focus quality (both pieces may stay in plane)
        - Single discrete event (not gradual)

        This is hard to distinguish from occlusion separation!
        """

        # High area variability
        if area_cv < 0.2:
            return False

        # Must have a discrete event
        if not has_sudden_focus_change:
            return False

        # Difference from occlusion: focus quality doesn't dramatically improve
        # (Both pieces likely stay reasonably in focus)
        if len(focus_change_frames) > 0:
            change_frame = focus_change_frames[0]
            if change_frame < len(focus_scores) - 5:
                focus_before = np.mean(focus_scores[:change_frame])
                focus_after = np.mean(focus_scores[change_frame:])

                # If focus dramatically improves, it's probably separation, not breakup
                if focus_after > focus_before + 0.2:
                    return False

        # Tentatively flag as possible breakup
        return True

    def _assess_quality(
        self,
        fraction_in_focus: float,
        mean_focus_score: float,
        suspected_occlusion: bool,
        suspected_breakup: bool,
    ) -> Tuple[str, str]:
        """
        Assess overall trajectory quality and recommend action

        Returns:
            (quality_flag, recommended_action)
        """

        # Occlusions should be flagged for review
        if suspected_occlusion:
            return ("marginal", "manual_review_occlusion")

        # Breakups are interesting - keep but flag
        if suspected_breakup:
            return ("good", "keep_flag_breakup")

        # Based on focus quality
        if fraction_in_focus >= 0.9 and mean_focus_score >= 0.7:
            return ("excellent", "keep")
        elif fraction_in_focus >= 0.7 and mean_focus_score >= 0.6:
            return ("good", "keep")
        elif fraction_in_focus >= 0.5 and mean_focus_score >= 0.5:
            return ("marginal", "keep_with_caution")
        elif fraction_in_focus >= 0.3:
            return ("poor", "trim_or_reject")
        else:
            return ("reject", "reject")

    def analyze_trajectories(
        self, df: pd.DataFrame, min_trajectory_length: int = 5
    ) -> pd.DataFrame:
        """
        Analyze all trajectories in a dataframe

        Args:
            df: DataFrame with trajectory_id column
            min_trajectory_length: Minimum frames to analyze

        Returns:
            DataFrame with one row per trajectory containing quality metrics
        """

        if "trajectory_id" not in df.columns:
            raise ValueError("DataFrame must have 'trajectory_id' column")

        results = []

        for traj_id, traj_df in df.groupby("trajectory_id"):
            if len(traj_df) < min_trajectory_length:
                continue

            report = self.analyze_single_trajectory(traj_df)

            # Convert to dict for DataFrame
            results.append(
                {
                    "trajectory_id": report.trajectory_id,
                    "n_frames": report.n_frames,
                    "mean_focus_score": report.mean_focus_score,
                    "median_focus_score": report.median_focus_score,
                    "min_focus_score": report.min_focus_score,
                    "fraction_in_focus": report.fraction_in_focus,
                    "focus_score_std": report.focus_score_std,
                    "focus_score_trend": report.focus_score_trend,
                    "has_sudden_focus_change": report.has_sudden_focus_change,
                    "n_focus_change_events": len(report.focus_change_frames),
                    "suspected_occlusion": report.suspected_occlusion,
                    "suspected_breakup": report.suspected_breakup,
                    "area_cv": report.area_cv,
                    "solidity_mean": report.solidity_mean,
                    "solidity_std": report.solidity_std,
                    "quality_flag": report.quality_flag,
                    "recommended_action": report.recommended_action,
                }
            )

        return pd.DataFrame(results)

    def add_trajectory_quality_to_dataframe(
        self, df: pd.DataFrame, min_trajectory_length: int = 5
    ) -> pd.DataFrame:
        """
        Add trajectory quality metrics to original dataframe

        This merges trajectory-level quality scores back into the detection-level
        dataframe, so each detection gets annotated with its trajectory's quality.

        Args:
            df: Original dataframe with trajectory_id
            min_trajectory_length: Minimum frames to analyze

        Returns:
            DataFrame with added columns:
                - traj_quality_flag: 'excellent', 'good', 'marginal', 'poor', 'reject'
                - traj_recommended_action: Action string
                - traj_mean_focus_score: Mean focus score for trajectory
                - traj_fraction_in_focus: Fraction of frames in focus
                - traj_suspected_occlusion: Boolean
                - traj_suspected_breakup: Boolean
        """

        # Get trajectory quality scores
        traj_quality = self.analyze_trajectories(df, min_trajectory_length)

        # Select columns to merge
        merge_cols = [
            "trajectory_id",
            "mean_focus_score",
            "median_focus_score",
            "fraction_in_focus",
            "suspected_occlusion",
            "suspected_breakup",
            "quality_flag",
            "recommended_action",
        ]

        # Rename to avoid confusion with per-frame metrics
        traj_quality_subset = traj_quality[merge_cols].copy()
        traj_quality_subset = traj_quality_subset.rename(
            columns={
                "mean_focus_score": "traj_mean_focus_score",
                "median_focus_score": "traj_median_focus_score",
                "fraction_in_focus": "traj_fraction_in_focus",
                "suspected_occlusion": "traj_suspected_occlusion",
                "suspected_breakup": "traj_suspected_breakup",
                "quality_flag": "traj_quality_flag",
                "recommended_action": "traj_recommended_action",
            }
        )

        # Merge into original dataframe
        df_with_quality = df.merge(traj_quality_subset, on="trajectory_id", how="left")

        return df_with_quality

    def filter_trajectories(
        self, df: pd.DataFrame, quality_flags: List[str] = ["excellent", "good"]
    ) -> pd.DataFrame:
        """
        Filter dataframe to keep only trajectories with specified quality

        Args:
            df: Original dataframe with trajectory_id
            quality_flags: List of quality flags to keep

        Returns:
            Filtered dataframe
        """

        # Get trajectory quality scores
        traj_quality = self.analyze_trajectories(df)

        # Filter to desired quality
        good_trajectories = traj_quality[
            traj_quality["quality_flag"].isin(quality_flags)
        ]["trajectory_id"].values

        return df[df["trajectory_id"].isin(good_trajectories)]

    def trim_trajectory_ends(
        self, trajectory_df: pd.DataFrame, min_consecutive_good: int = 3
    ) -> pd.DataFrame:
        """
        Trim poor-quality segments from trajectory start/end

        Useful for trajectories that start out-of-focus then come into focus,
        or vice versa

        Args:
            trajectory_df: Single trajectory dataframe
            min_consecutive_good: Minimum consecutive good frames to keep

        Returns:
            Trimmed trajectory dataframe
        """

        # Compute in-focus flags
        in_focus = (
            (trajectory_df["edge_gradient_mean"] > self.edge_gradient_mean_thresh)
            & (trajectory_df["edge_gradient_std"] > self.edge_gradient_std_thresh)
            & (trajectory_df["edge_laplacian_var"] > self.edge_laplacian_var_thresh)
        ).values

        if len(in_focus) < min_consecutive_good:
            return trajectory_df

        # Find first occurrence of min_consecutive_good True values
        start_idx = 0
        for i in range(len(in_focus) - min_consecutive_good + 1):
            if np.all(in_focus[i : i + min_consecutive_good]):
                start_idx = i
                break

        # Find last occurrence
        end_idx = len(in_focus)
        for i in range(len(in_focus) - 1, min_consecutive_good - 2, -1):
            if np.all(in_focus[i - min_consecutive_good + 1 : i + 1]):
                end_idx = i + 1
                break

        return trajectory_df.iloc[start_idx:end_idx]


def print_trajectory_summary(trajectory_report: TrajectoryQualityReport):
    """Pretty print a trajectory quality report"""

    print("=" * 70)
    print(f"TRAJECTORY {trajectory_report.trajectory_id} QUALITY REPORT")
    print("=" * 70)
    print(f"Length: {trajectory_report.n_frames} frames")
    print()
    print("Focus Quality:")
    print(f"  Mean score: {trajectory_report.mean_focus_score:.3f}")
    print(f"  Median score: {trajectory_report.median_focus_score:.3f}")
    print(f"  Min score: {trajectory_report.min_focus_score:.3f}")
    print(f"  Fraction in-focus: {trajectory_report.fraction_in_focus:.1%}")
    print(f"  Score variability (std): {trajectory_report.focus_score_std:.3f}")
    print(f"  Temporal trend: {trajectory_report.focus_score_trend:+.4f}")
    print()
    print("Geometric Consistency:")
    print(f"  Area CV: {trajectory_report.area_cv:.3f}")
    print(f"  Mean solidity: {trajectory_report.solidity_mean:.3f}")
    print(f"  Solidity std: {trajectory_report.solidity_std:.3f}")
    print()
    print("Event Detection:")
    print(
        f"  Sudden focus changes: {'Yes' if trajectory_report.has_sudden_focus_change else 'No'}"
    )
    if trajectory_report.focus_change_frames:
        print(f"  Change frames: {trajectory_report.focus_change_frames}")
    print(
        f"  Suspected occlusion: {'Yes' if trajectory_report.suspected_occlusion else 'No'}"
    )
    print(
        f"  Suspected breakup: {'Yes' if trajectory_report.suspected_breakup else 'No'}"
    )
    print()
    print(f"Overall Assessment: {trajectory_report.quality_flag.upper()}")
    print(f"Recommendation: {trajectory_report.recommended_action}")
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        parquet_path = sys.argv[1]

        print(f"Loading: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} detections")
        print()

        # Analyze all trajectories
        analyzer = TrajectoryFocusAnalyzer()

        # Option 1: Get separate trajectory quality table
        trajectory_scores = analyzer.analyze_trajectories(df)

        print("=" * 70)
        print("TRAJECTORY QUALITY SUMMARY")
        print("=" * 70)
        print(f"Total trajectories analyzed: {len(trajectory_scores)}")
        print()

        # Quality distribution
        for quality in ["excellent", "good", "marginal", "poor", "reject"]:
            count = (trajectory_scores["quality_flag"] == quality).sum()
            print(f"  {quality.capitalize()}: {count}")

        print()
        print(f"Suspected occlusions: {trajectory_scores['suspected_occlusion'].sum()}")
        print(f"Suspected breakups: {trajectory_scores['suspected_breakup'].sum()}")
        print()

        # Save trajectory-level results
        traj_output_path = parquet_path.replace(".parquet", "_trajectory_quality.csv")
        trajectory_scores.to_csv(traj_output_path, index=False)
        print(f"Saved trajectory quality scores to: {traj_output_path}")
        print()

        # Option 2: Merge quality flags back into original dataframe
        print("=" * 70)
        print("ADDING QUALITY FLAGS TO ORIGINAL DATAFRAME")
        print("=" * 70)

        df_with_quality = analyzer.add_trajectory_quality_to_dataframe(df)

        print(
            f"Added columns: {[col for col in df_with_quality.columns if col.startswith('traj_')]}"
        )
        print()

        # Show filtering examples
        print("FILTERING EXAMPLES:")
        print("-" * 70)

        # Filter by quality flag
        for quality in ["excellent", "good", "marginal", "poor", "reject"]:
            count = (df_with_quality["traj_quality_flag"] == quality).sum()
            n_traj = df_with_quality[df_with_quality["traj_quality_flag"] == quality][
                "trajectory_id"
            ].nunique()
            print(
                f"  {quality.capitalize()}: {count} detections in {n_traj} trajectories"
            )

        print()

        # Example: Keep only excellent and good
        excellent_good = df_with_quality[
            df_with_quality["traj_quality_flag"].isin(["excellent", "good"])
        ]
        print(
            f"Excellent + Good: {len(excellent_good)} detections ({len(excellent_good) / len(df_with_quality) * 100:.1f}%)"
        )

        # Example: Keep marginal or better (exclude poor and reject)
        marginal_or_better = df_with_quality[
            df_with_quality["traj_quality_flag"].isin(["excellent", "good", "marginal"])
        ]
        print(
            f"Marginal or better: {len(marginal_or_better)} detections ({len(marginal_or_better) / len(df_with_quality) * 100:.1f}%)"
        )

        # Example: Reject only 'reject' quality
        not_reject = df_with_quality[df_with_quality["traj_quality_flag"] != "reject"]
        print(
            f"Not reject: {len(not_reject)} detections ({len(not_reject) / len(df_with_quality) * 100:.1f}%)"
        )

        print()

        # Save merged dataframe
        merged_output_path = parquet_path.replace(
            ".parquet", "_with_trajectory_quality.parquet"
        )
        df_with_quality.to_parquet(merged_output_path)
        print(f"Saved merged dataframe to: {merged_output_path}")
        print()

        print("=" * 70)
        print("USAGE EXAMPLES FOR FILTERING:")
        print("=" * 70)
        print("""
# Keep only excellent and good trajectories:
df_clean = df_with_quality[
    df_with_quality['traj_quality_flag'].isin(['excellent', 'good'])
]

# Keep marginal or better (for testing):
df_test = df_with_quality[
    df_with_quality['traj_quality_flag'].isin(['excellent', 'good', 'marginal'])
]

# Reject only the worst:
df_filtered = df_with_quality[
    df_with_quality['traj_quality_flag'] != 'reject'
]

# Flag occlusions for manual review:
occlusions = df_with_quality[
    df_with_quality['traj_suspected_occlusion'] == True
]

# Get high-quality trajectories with good focus:
high_quality = df_with_quality[
    (df_with_quality['traj_quality_flag'].isin(['excellent', 'good'])) &
    (df_with_quality['traj_fraction_in_focus'] > 0.8)
]
        """)

    else:
        print("Usage: python trajectory_focus_analyzer.py <parquet_file>")
        print()
        print("This will:")
        print("  1. Analyze trajectory-level focus quality")
        print("  2. Save trajectory quality table as CSV")
        print("  3. Merge quality flags back into original parquet")
        print("  4. Show filtering examples")
