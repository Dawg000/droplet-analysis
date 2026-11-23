import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class SAM2PostProcessor:
    """Advanced post-processing for SAM2 blob analysis results"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with results DataFrame from SAM2BlobAnalyzer

        Args:
            df: DataFrame from SAM2BlobAnalyzer.process_video_*
        """
        self.df = df.copy()
        self.feature_columns = [
            col
            for col in df.columns
            if col
            not in [
                "file",
                "frame",
                "object_id",
                "centroid_x",
                "centroid_y",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "is_in_focus",
                "is_redundant",
                "rle_counts_str",
            ]
        ]

    # 1. TEMPORAL TRACKING AND TRAJECTORY ANALYSIS
    def track_blob_trajectories(
        self, max_distance: float = 50.0, max_frame_gap: int = 3
    ) -> pd.DataFrame:
        """
        Track blob movements across frames to create trajectories

        Args:
            max_distance: Maximum distance for blob matching between frames
            max_frame_gap: Maximum frame gap to allow in trajectories

        Returns:
            DataFrame with trajectory_id added
        """
        df_tracked = self.df.copy()
        df_tracked["trajectory_id"] = -1
        trajectory_counter = 0

        # Sort by frame for sequential processing
        df_sorted = df_tracked.sort_values(["frame", "object_id"])

        # # This would be for when the object IDs are not retained across frames
        # for frame_num in df_sorted["frame"].unique():
        #     current_frame = df_sorted[df_sorted["frame"] == frame_num]

        #     if frame_num == df_sorted["frame"].min():
        #         # First frame - assign new trajectory IDs
        #         for idx in current_frame.index:
        #             df_tracked.loc[idx, "trajectory_id"] = trajectory_counter
        #             trajectory_counter += 1
        #     else:
        #         # Find previous frames within gap
        #         prev_frames = df_tracked[
        #             (df_tracked["frame"] < frame_num)
        #             & (df_tracked["frame"] >= frame_num - max_frame_gap)
        #             & (df_tracked["trajectory_id"] != -1)
        #         ]

        #         if not prev_frames.empty:
        #             # Calculate distances between current and previous blobs
        #             current_positions = current_frame[
        #                 ["centroid_x", "centroid_y"]
        #             ].values
        #             prev_positions = prev_frames[["centroid_x", "centroid_y"]].values

        #             if len(current_positions) > 0 and len(prev_positions) > 0:
        #                 distances = np.linalg.norm(
        #                     current_positions[:, np.newaxis]
        #                     - prev_positions[np.newaxis, :],
        #                     axis=2,
        #                 )

        #                 # Hungarian algorithm would be better, but simple greedy matching
        #                 used_prev = set()
        #                 for i, curr_idx in enumerate(current_frame.index):
        #                     min_dist_idx = np.argmin(distances[i])
        #                     min_dist = distances[i, min_dist_idx]

        #                     if (
        #                         min_dist <= max_distance
        #                         and min_dist_idx not in used_prev
        #                     ):
        #                         prev_idx = prev_frames.iloc[min_dist_idx].name
        #                         traj_id = df_tracked.loc[prev_idx, "trajectory_id"]
        #                         df_tracked.loc[curr_idx, "trajectory_id"] = traj_id
        #                         used_prev.add(min_dist_idx)
        #                     else:
        #                         # New trajectory
        #                         df_tracked.loc[curr_idx, "trajectory_id"] = (
        #                             trajectory_counter
        #                         )
        #                         trajectory_counter += 1

        # return df_tracked
        df_tracked["trajectory_id"] = df_tracked["object_id"]
        return df_tracked

    def compute_trajectory_features(self, df_tracked: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trajectory-based features

        Args:
            df_tracked: DataFrame with trajectory_id column

        Returns:
            DataFrame with trajectory statistics
        """
        trajectory_stats = []

        for traj_id in df_tracked["trajectory_id"].unique():
            if traj_id == -1:
                continue

            traj_data = df_tracked[df_tracked["trajectory_id"] == traj_id].sort_values(
                "frame"
            )

            if len(traj_data) < 2:
                continue

            # Trajectory metrics
            duration = traj_data["frame"].max() - traj_data["frame"].min() + 1
            actual_detections = len(traj_data)
            detection_rate = actual_detections / duration

            # Movement metrics
            positions = traj_data[["centroid_x", "centroid_y"]].values
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = np.sum(distances)
            mean_speed = np.mean(distances) if len(distances) > 0 else 0
            max_speed = np.max(distances) if len(distances) > 0 else 0

            # Directional consistency
            if len(distances) > 1:
                angles = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
                angle_changes = np.abs(np.diff(angles))
                angle_changes = np.minimum(angle_changes, 2 * np.pi - angle_changes)
                directional_consistency = 1 - np.mean(angle_changes) / np.pi
            else:
                directional_consistency = 1.0

            # Feature stability
            feature_stability = {}
            for feature in ["area", "circularity", "solidity", "focus_score"]:
                if feature in traj_data.columns:
                    cv = traj_data[feature].std() / (traj_data[feature].mean() + 1e-6)
                    feature_stability[f"{feature}_stability"] = 1 / (1 + cv)

            stats = {
                "trajectory_id": traj_id,
                "duration": duration,
                "detection_count": actual_detections,
                "detection_rate": detection_rate,
                "total_distance": total_distance,
                "mean_speed": mean_speed,
                "max_speed": max_speed,
                "directional_consistency": directional_consistency,
                "start_frame": traj_data["frame"].min(),
                "end_frame": traj_data["frame"].max(),
                "mean_area": traj_data["area"].mean(),
                "mean_focus_score": traj_data["focus_score"].mean(),
                **feature_stability,
            }
            trajectory_stats.append(stats)

        return pd.DataFrame(trajectory_stats)

    # 2. CLUSTERING AND PHENOTYPE CLASSIFICATION
    def cluster_blobs(
        self,
        features: List[str] = None,
        method: str = "kmeans",
        n_clusters: int = 5,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Cluster blobs based on morphological features

        Args:
            features: List of features to use for clustering
            method: 'kmeans', 'dbscan', or 'gaussian_mixture'
            n_clusters: Number of clusters (for kmeans/gmm)

        Returns:
            DataFrame with cluster labels added
        """
        if features is None:
            features = [
                "area",
                "circularity",
                "eccentricity",
                "solidity",
                "aspect_ratio",
                "major_axis",
                "minor_axis",
                "compactness",
                "focus_score",
            ]

        # Filter valid features
        available_features = [f for f in features if f in self.df.columns]

        # Prepare data
        X = self.df[available_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cluster
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
            cluster_labels = clusterer.fit_predict(X_scaled)
        elif method == "dbscan":
            clusterer = DBSCAN(**kwargs)
            cluster_labels = clusterer.fit_predict(X_scaled)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        df_clustered = self.df.copy()
        df_clustered["cluster_id"] = cluster_labels

        return df_clustered

    def analyze_cluster_characteristics(self, df_clustered: pd.DataFrame) -> Dict:
        """Analyze characteristics of each cluster"""
        cluster_analysis = {}

        for cluster_id in df_clustered["cluster_id"].unique():
            if cluster_id == -1:  # DBSCAN noise
                continue

            cluster_data = df_clustered[df_clustered["cluster_id"] == cluster_id]

            # Basic stats
            stats = {
                "count": len(cluster_data),
                "percentage": len(cluster_data) / len(df_clustered) * 100,
                "mean_features": {},
                "std_features": {},
            }

            # Feature statistics
            for feature in self.feature_columns:
                if feature in cluster_data.columns:
                    stats["mean_features"][feature] = cluster_data[feature].mean()
                    stats["std_features"][feature] = cluster_data[feature].std()

            # Quality metrics
            stats["in_focus_rate"] = cluster_data["is_in_focus"].mean()
            stats["redundant_rate"] = cluster_data["is_redundant"].mean()

            cluster_analysis[f"cluster_{cluster_id}"] = stats

        return cluster_analysis

    # 3. ANOMALY DETECTION
    def detect_anomalies(
        self, features: List[str] = None, contamination: float = 0.1
    ) -> pd.DataFrame:
        """
        Detect anomalous blobs using Isolation Forest

        Args:
            features: Features to use for anomaly detection
            contamination: Expected proportion of anomalies

        Returns:
            DataFrame with anomaly labels
        """
        if features is None:
            features = self.feature_columns

        available_features = [f for f in features if f in self.df.columns]
        X = self.df[available_features].fillna(0)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Detect anomalies
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)

        df_anomaly = self.df.copy()
        df_anomaly["is_anomaly"] = anomaly_labels == -1
        df_anomaly["anomaly_score"] = iso_forest.decision_function(X_scaled)

        return df_anomaly

    # 4. QUALITY FILTERING AND RANKING
    def compute_composite_quality_score(
        self, df: pd.DataFrame = None, weights: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Compute composite quality score from multiple metrics

        Args:
            weights: Dictionary of feature weights for scoring

        Returns:
            DataFrame with quality scores
        """
        if weights is None:
            weights = {
                "focus_score": 0.3,
                "circularity": 0.2,
                "solidity": 0.2,
                "edge_strength": 0.15,
                "compactness": -0.15,  # Lower is better
            }

        if df is None:
            df_scored = self.df.copy()
        else:
            df_scored = df

        # Normalize features to 0-1 scale
        composite_score = np.zeros(len(df_scored))

        for feature, weight in weights.items():
            if feature in df_scored.columns:
                values = df_scored[feature].values
                if weight > 0:
                    # Higher is better
                    normalized = (values - values.min()) / (
                        values.max() - values.min() + 1e-6
                    )
                else:
                    # Lower is better
                    normalized = 1 - (values - values.min()) / (
                        values.max() - values.min() + 1e-6
                    )
                    weight = abs(weight)

                composite_score += weight * normalized

        df_scored["quality_score"] = composite_score
        df_scored["quality_rank"] = df_scored["quality_score"].rank(ascending=False)

        return df_scored

    def filter_high_quality_blobs(
        self, df_scored: pd.DataFrame, percentile: float = 90
    ) -> pd.DataFrame:
        """Filter top percentile of blobs by quality score"""
        threshold = np.percentile(df_scored["quality_score"], percentile)
        return df_scored[df_scored["quality_score"] >= threshold]

    # 5. STATISTICAL ANALYSIS AND VISUALIZATION
    def generate_summary_statistics(self) -> Dict:
        """Generate comprehensive summary statistics"""
        summary = {
            "dataset_info": {
                "total_blobs": len(self.df),
                "unique_frames": self.df["frame"].nunique(),
                "frame_range": (self.df["frame"].min(), self.df["frame"].max()),
                "blobs_per_frame": len(self.df) / self.df["frame"].nunique(),
            },
            "quality_metrics": {
                "in_focus_percentage": self.df["is_in_focus"].mean() * 100,
                "redundant_percentage": self.df["is_redundant"].mean() * 100,
                "mean_focus_score": self.df["focus_score"].mean(),
                "mean_area": self.df["area"].mean(),
            },
            "morphology_stats": {},
            "feature_correlations": {},
        }

        # Feature statistics
        for feature in self.feature_columns:
            if feature in self.df.columns:
                summary["morphology_stats"][feature] = {
                    "mean": self.df[feature].mean(),
                    "std": self.df[feature].std(),
                    "min": self.df[feature].min(),
                    "max": self.df[feature].max(),
                    "median": self.df[feature].median(),
                }

        return summary

    def plot_feature_distributions(
        self, features: List[str] = None, figsize: Tuple[int, int] = (15, 10)
    ):
        """Plot distributions of key features"""
        if features is None:
            features = [
                "area",
                "circularity",
                "eccentricity",
                "solidity",
                "focus_score",
                "edge_strength",
            ]

        available_features = [f for f in features if f in self.df.columns]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for i, feature in enumerate(available_features[:6]):
            if i < len(axes):
                ax = axes[i]
                self.df[feature].hist(bins=50, ax=ax, alpha=0.7)
                ax.set_title(f"{feature.replace('_', ' ').title()} Distribution")
                ax.set_xlabel(feature)
                ax.set_ylabel("Frequency")

        plt.tight_layout()
        return fig

    def plot_quality_vs_features(
        self, quality_col: str = "focus_score", features: List[str] = None
    ):
        """Plot quality metric against morphological features"""
        if features is None:
            features = ["area", "circularity", "solidity", "eccentricity"]

        available_features = [f for f in features if f in self.df.columns]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, feature in enumerate(available_features[:4]):
            if i < len(axes):
                ax = axes[i]
                ax.scatter(self.df[feature], self.df[quality_col], alpha=0.5)
                ax.set_xlabel(feature.replace("_", " ").title())
                ax.set_ylabel(quality_col.replace("_", " ").title())

                # Add correlation coefficient
                corr = self.df[feature].corr(self.df[quality_col])
                ax.set_title(f"r = {corr:.3f}")

        plt.tight_layout()
        return fig

    # 6. EXPORT AND REPORTING
    def export_analysis_report(
        self,
        output_path: str,
        include_trajectories: bool = True,
        include_clusters: bool = True,
        include_anomalies: bool = True,
    ):
        """
        Export comprehensive analysis report

        Args:
            output_path: Base path for output files
            include_trajectories: Whether to include trajectory analysis
            include_clusters: Whether to include clustering analysis
            include_anomalies: Whether to include anomaly detection
        """
        base_path = output_path.replace(".parquet", "").replace(".csv", "")

        # 1. Enhanced main dataset
        df_enhanced = self.df.copy()

        # Add trajectory tracking
        if include_trajectories:
            df_tracked = self.track_blob_trajectories()
            df_enhanced = df_tracked

            # Export trajectory stats
            traj_stats = self.compute_trajectory_features(df_tracked)
            traj_stats.to_parquet(f"{base_path}_trajectories.parquet")

        # Add clustering
        if include_clusters:
            df_clustered = self.cluster_blobs()
            df_enhanced = df_clustered

            # Export cluster analysis
            cluster_analysis = self.analyze_cluster_characteristics(df_clustered)
            pd.DataFrame(cluster_analysis).T.to_csv(f"{base_path}_cluster_analysis.csv")

        # Add anomaly detection
        if include_anomalies:
            df_anomaly = self.detect_anomalies()
            df_enhanced["is_anomaly"] = df_anomaly["is_anomaly"]
            df_enhanced["anomaly_score"] = df_anomaly["anomaly_score"]

        # Add quality scoring
        df_enhanced = self.compute_composite_quality_score(df=df_enhanced)

        # Export enhanced dataset
        df_enhanced.to_parquet(f"{base_path}_enhanced.parquet")

        # Export summary statistics
        summary = self.generate_summary_statistics()
        with open(f"{base_path}_summary.txt", "w") as f:
            f.write("SAM2 Blob Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            for category, stats in summary.items():
                f.write(f"{category.upper()}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

        logging.info(f"Analysis report exported to {base_path}_*")


# Usage example
def example_post_processing():
    """Example of how to use the post-processing pipeline"""
    import os
    from pathlib import Path

    results_df_path = Path(
        "/home/leviathan/cdrucker/ml/sam2/outputs/vid64_rolling4/results.parquet"
    )
    results_df = pd.read_parquet(
        results_df_path,
        engine="pyarrow",
    )
    # Initialize post-processor
    processor = SAM2PostProcessor(results_df)

    # 1. Track trajectories
    df_tracked = processor.track_blob_trajectories(max_distance=30.0)
    trajectory_stats = processor.compute_trajectory_features(df_tracked)
    print(f"Found {len(trajectory_stats)} trajectories")

    # 2. Cluster blobs by morphology
    df_clustered = processor.cluster_blobs(method="kmeans", n_clusters=6)
    cluster_analysis = processor.analyze_cluster_characteristics(df_clustered)

    # 3. Detect anomalies
    df_anomalies = processor.detect_anomalies(contamination=0.05)
    anomaly_count = df_anomalies["is_anomaly"].sum()
    print(f"Detected {anomaly_count} anomalous blobs")

    # 4. Quality scoring and filtering
    df_scored = processor.compute_composite_quality_score()
    high_quality = processor.filter_high_quality_blobs(df_scored, percentile=95)
    print(f"Top 5% quality blobs: {len(high_quality)}")

    # 5. Generate visualizations
    fig1 = processor.plot_feature_distributions()
    # fig1.show()
    fig2 = processor.plot_quality_vs_features()
    # fig2.show()

    # 6. Export comprehensive report
    processor.export_analysis_report(
        os.path.join(results_df_path.parent, "analysis_results"),
        include_trajectories=True,
        include_clusters=True,
        include_anomalies=True,
    )

    pass


if __name__ == "__main__":
    example_post_processing()
