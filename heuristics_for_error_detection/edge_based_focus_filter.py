"""
Edge-Based Focus Quality Filter

Based on analysis of labeled good/bad trajectories, this filter uses edge-localized
metrics that strongly discriminate between in-focus and out-of-focus droplets.

Key findings from data analysis:
- edge_gradient_mean: Cohen's d = 2.13 (STRONG discriminator)
- edge_gradient_std: Cohen's d = 1.97 (STRONG discriminator)
- edge_laplacian_var: Cohen's d = 1.27 (STRONG discriminator)
- combined_focus_quality: Cohen's d = 0.57 (MODERATE - not as good)

The edge-based metrics are 2-4x more discriminative than the current combined score.

Usage:
    from edge_based_focus_filter import EdgeFocusFilter

    filter = EdgeFocusFilter(mode='moderate')  # or 'conservative', 'aggressive'
    df['in_focus'] = filter.filter_dataframe(df)
"""

from typing import Dict, Literal

import numpy as np
import pandas as pd


class EdgeFocusFilter:
    """
    Focus quality filter based on edge gradient metrics

    Three operating modes with different tradeoffs:
    - 'conservative': 99.6% bad removal, 64% good retention
    - 'moderate': 99.1% bad removal, 70% good retention
    - 'aggressive': 84.8% bad removal, 94% good retention

    Based on empirical analysis of labeled trajectories from your data.
    """

    # Thresholds determined from analysis of good/bad trajectories
    THRESHOLDS = {
        "conservative": {
            "edge_gradient_mean": 122.0,  # Good P25
            "edge_gradient_std": 82.0,  # Good P25
            "edge_laplacian_var": 4556.0,  # Good P25
            "description": "Strict filtering - minimal false positives (99.6% bad removed)",
            "expected_performance": {"good_retention": 0.64, "bad_rejection": 0.996},
        },
        "moderate": {
            "edge_gradient_mean": 114.0,  # Midpoint between good/bad medians
            "edge_gradient_std": 70.0,  # Midpoint
            "edge_laplacian_var": 4435.0,  # Midpoint
            "description": "Balanced filtering - good compromise (99.1% bad removed)",
            "expected_performance": {"good_retention": 0.70, "bad_rejection": 0.991},
        },
        "aggressive": {
            "edge_gradient_mean": 66.0,  # Good P5
            "edge_gradient_std": 41.0,  # Good P5
            "edge_laplacian_var": 1439.0,  # Good P5
            "description": "Permissive filtering - maximize data retention (84.8% bad removed)",
            "expected_performance": {"good_retention": 0.94, "bad_rejection": 0.848},
        },
    }

    def __init__(
        self,
        mode: Literal["conservative", "moderate", "aggressive"] = "moderate",
        custom_thresholds: Dict[str, float] = None,
    ):
        """
        Initialize focus filter

        Args:
            mode: Operating mode ('conservative', 'moderate', or 'aggressive')
            custom_thresholds: Optional dict with custom thresholds
                              {'edge_gradient_mean': X, 'edge_gradient_std': Y,
                               'edge_laplacian_var': Z}
        """

        if custom_thresholds is not None:
            self.thresholds = custom_thresholds
            self.mode = "custom"
        else:
            if mode not in self.THRESHOLDS:
                raise ValueError(f"Mode must be one of {list(self.THRESHOLDS.keys())}")
            self.mode = mode
            self.thresholds = self.THRESHOLDS[mode]

        # Extract threshold values
        self.edge_gradient_mean_thresh = self.thresholds["edge_gradient_mean"]
        self.edge_gradient_std_thresh = self.thresholds["edge_gradient_std"]
        self.edge_laplacian_var_thresh = self.thresholds["edge_laplacian_var"]

        print(f"EdgeFocusFilter initialized in '{self.mode}' mode")
        if self.mode != "custom":
            print(f"  {self.thresholds['description']}")
            print(
                f"  Expected: {self.thresholds['expected_performance']['good_retention'] * 100:.1f}% good retention, "
                f"{self.thresholds['expected_performance']['bad_rejection'] * 100:.1f}% bad rejection"
            )
        print("  Thresholds:")
        print(f"    edge_gradient_mean > {self.edge_gradient_mean_thresh:.1f}")
        print(f"    edge_gradient_std > {self.edge_gradient_std_thresh:.1f}")
        print(f"    edge_laplacian_var > {self.edge_laplacian_var_thresh:.1f}")

    def passes_filter(
        self,
        edge_gradient_mean: float,
        edge_gradient_std: float,
        edge_laplacian_var: float,
    ) -> bool:
        """
        Check if a single detection passes the focus filter

        Args:
            edge_gradient_mean: Mean gradient magnitude at droplet edge
            edge_gradient_std: Std of gradient magnitude at edge
            edge_laplacian_var: Variance of Laplacian at edge

        Returns:
            True if detection passes (in focus), False otherwise
        """

        return (
            edge_gradient_mean > self.edge_gradient_mean_thresh
            and edge_gradient_std > self.edge_gradient_std_thresh
            and edge_laplacian_var > self.edge_laplacian_var_thresh
        )

    def filter_dataframe(self, df: pd.DataFrame, inplace: bool = False) -> pd.Series:
        """
        Apply focus filter to entire dataframe

        Args:
            df: DataFrame with edge metric columns
            inplace: If True, add 'in_focus' column to df

        Returns:
            Boolean Series indicating which detections pass filter
        """

        # Check required columns
        required_cols = [
            "edge_gradient_mean",
            "edge_gradient_std",
            "edge_laplacian_var",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

        # Apply filter using vectorized operations
        passes = (
            (df["edge_gradient_mean"] > self.edge_gradient_mean_thresh)
            & (df["edge_gradient_std"] > self.edge_gradient_std_thresh)
            & (df["edge_laplacian_var"] > self.edge_laplacian_var_thresh)
        )

        if inplace:
            df["in_focus"] = passes

        return passes

    def get_focus_score(
        self,
        edge_gradient_mean: float,
        edge_gradient_std: float,
        edge_laplacian_var: float,
    ) -> float:
        """
        Get continuous focus quality score [0, 1]

        This is useful for soft weighting rather than hard filtering.
        Score is based on how far above threshold each metric is.

        Args:
            edge_gradient_mean: Mean gradient magnitude at droplet edge
            edge_gradient_std: Std of gradient magnitude at edge
            edge_laplacian_var: Variance of Laplacian at edge

        Returns:
            Continuous score in [0, 1] where higher = better focus
        """

        # Normalize each metric relative to threshold
        # Use aggressive threshold as "perfect" (1.0) and current threshold as "marginal" (0.5)

        # For edge_gradient_mean: aggressive=66, moderate=114, conservative=122
        # Linear interpolation: 0 -> 0, threshold -> 0.5, 2*threshold -> 1.0
        norm_grad_mean = np.clip(
            edge_gradient_mean / (2 * self.edge_gradient_mean_thresh), 0, 1
        )
        norm_grad_std = np.clip(
            edge_gradient_std / (2 * self.edge_gradient_std_thresh), 0, 1
        )
        norm_lap_var = np.clip(
            edge_laplacian_var / (2 * self.edge_laplacian_var_thresh), 0, 1
        )

        # Weighted average (equal weights)
        score = (norm_grad_mean + norm_grad_std + norm_lap_var) / 3.0

        return score

    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        """
        Analyze dataframe and report filtering statistics

        Args:
            df: DataFrame with edge metric columns

        Returns:
            Dictionary with statistics
        """

        passes = self.filter_dataframe(df)

        stats = {
            "total_detections": len(df),
            "passing_detections": passes.sum(),
            "failing_detections": (~passes).sum(),
            "pass_rate": passes.sum() / len(df) if len(df) > 0 else 0,
            "mode": self.mode,
            "thresholds": {
                "edge_gradient_mean": self.edge_gradient_mean_thresh,
                "edge_gradient_std": self.edge_gradient_std_thresh,
                "edge_laplacian_var": self.edge_laplacian_var_thresh,
            },
        }

        return stats

    def print_analysis(self, df: pd.DataFrame):
        """
        Print detailed analysis of filtering results

        Args:
            df: DataFrame with edge metric columns
        """

        stats = self.analyze_dataframe(df)

        print("=" * 70)
        print("FOCUS FILTER ANALYSIS")
        print("=" * 70)
        print(f"Mode: {stats['mode']}")
        print("Thresholds:")
        print(f"  edge_gradient_mean > {stats['thresholds']['edge_gradient_mean']:.1f}")
        print(f"  edge_gradient_std > {stats['thresholds']['edge_gradient_std']:.1f}")
        print(f"  edge_laplacian_var > {stats['thresholds']['edge_laplacian_var']:.1f}")
        print()
        print("Results:")
        print(f"  Total detections: {stats['total_detections']}")
        print(
            f"  Passing (in focus): {stats['passing_detections']} ({stats['pass_rate'] * 100:.1f}%)"
        )
        print(
            f"  Failing (out of focus): {stats['failing_detections']} ({(1 - stats['pass_rate']) * 100:.1f}%)"
        )
        print("=" * 70)


def compare_modes_on_dataframe(df: pd.DataFrame):
    """
    Compare all three modes on a dataframe

    Args:
        df: DataFrame with edge metric columns
    """

    print("=" * 70)
    print("COMPARISON OF FILTERING MODES")
    print("=" * 70)
    print()

    for mode in ["conservative", "moderate", "aggressive"]:
        filter = EdgeFocusFilter(mode=mode)
        stats = filter.analyze_dataframe(df)

        print(f"{mode.upper()}:")
        print(f"  {EdgeFocusFilter.THRESHOLDS[mode]['description']}")
        print(
            f"  Passing: {stats['passing_detections']}/{stats['total_detections']} "
            f"({stats['pass_rate'] * 100:.1f}%)"
        )
        print()


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Load parquet file and analyze
        import pandas as pd

        parquet_path = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else "moderate"

        print(f"Loading: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} detections")
        print()

        # Compare all modes
        compare_modes_on_dataframe(df)

        # Apply selected mode
        print()
        filter = EdgeFocusFilter(mode=mode)
        df["in_focus"] = filter.filter_dataframe(df)

        # Print analysis
        filter.print_analysis(df)

        # Save filtered data
        output_path = parquet_path.replace(".parquet", f"_filtered_{mode}.parquet")
        df.to_parquet(output_path)
        print(f"\nSaved filtered data to: {output_path}")

    else:
        print("Usage: python edge_based_focus_filter.py <parquet_file> [mode]")
        print("Modes: conservative, moderate (default), aggressive")
