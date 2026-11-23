import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DropletDeformationFeatures:
    """Extended features for droplet deformation analysis"""

    # Basic geometric (inherited from BlobFeatures)
    area: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]

    # Deformation-specific features
    deformation_parameter: float  # (L-W)/(L+W)
    taylor_parameter: float  # Alternative deformation measure
    major_axis_length: float
    minor_axis_length: float
    aspect_ratio: float
    ellipticity: float

    # Shape evolution
    shape_factor: float  # 4π*Area/Perimeter²
    convexity: float
    roughness: float

    # Orientation and alignment
    principal_axis_angle: float  # Angle of major axis
    flow_alignment_angle: float  # Relative to flow direction

    # Breakup indicators
    neck_width: float  # Minimum width (potential breakup point)
    neck_position: float  # Location along major axis
    end_curvature: Tuple[float, float]  # Curvature at droplet ends
    breakup_probability: float

    # Temporal features (filled by trajectory analysis)
    deformation_rate: float
    rotation_rate: float
    area_change_rate: float


class DropletDeformationAnalyzer:
    """Specialized analyzer for droplet deformation in extensional flow"""

    def __init__(
        self,
        flow_direction: float = 0.0,
        pixel_size: float = 1.0,
        frame_rate: float = 1.0,
        expected_flow_rate: float = None,
    ):
        """
        Initialize droplet deformation analyzer

        Args:
            flow_direction: Flow direction in radians (0 = horizontal)
            pixel_size: Microns per pixel for scaling
            frame_rate: Frames per second
            expected_flow_rate: Expected strain rate (1/s) if known
        """
        self.flow_direction = flow_direction
        self.pixel_size = pixel_size
        self.frame_rate = frame_rate
        self.expected_flow_rate = expected_flow_rate

    def compute_deformation_features(
        self, mask: np.ndarray, original_area: float = None
    ) -> Dict:
        """
        Compute comprehensive deformation features from droplet mask

        Args:
            mask: Binary mask of droplet
            original_area: Undeformed droplet area for normalization

        Returns:
            Dictionary of deformation features
        """
        if np.sum(mask) == 0:
            return {}

        # Basic shape fitting
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return {}

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Fit ellipse to get major/minor axes
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center, (major_axis, minor_axis), angle = ellipse

            # Ensure major > minor
            if minor_axis > major_axis:
                major_axis, minor_axis = minor_axis, major_axis
                angle += 90

            # Convert angle to radians and normalize
            angle_rad = np.deg2rad(angle % 180)
        else:
            # Fallback for very small droplets
            major_axis = minor_axis = np.sqrt(4 * area / np.pi)
            angle_rad = 0
            center = (mask.shape[1] / 2, mask.shape[0] / 2)

        # Deformation parameters
        if major_axis > 0 and minor_axis > 0:
            deformation_parameter = (major_axis - minor_axis) / (
                major_axis + minor_axis
            )
            taylor_parameter = (major_axis**2 - minor_axis**2) / (
                major_axis**2 + minor_axis**2
            )
            aspect_ratio = major_axis / minor_axis
            ellipticity = 1 - (minor_axis / major_axis)
        else:
            deformation_parameter = taylor_parameter = ellipticity = 0
            aspect_ratio = 1

        # Shape factors
        shape_factor = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        # Roughness (perimeter deviation from smooth ellipse)
        expected_perimeter = np.pi * (
            3 * (major_axis + minor_axis)
            - np.sqrt((3 * major_axis + minor_axis) * (major_axis + 3 * minor_axis))
        )
        roughness = perimeter / expected_perimeter if expected_perimeter > 0 else 1

        # Flow alignment
        flow_alignment_angle = abs(angle_rad - self.flow_direction)
        flow_alignment_angle = min(flow_alignment_angle, np.pi - flow_alignment_angle)

        # Breakup indicators
        neck_features = self._analyze_neck_formation(
            contour, major_axis, minor_axis, angle_rad
        )

        features = {
            "area": area,
            "centroid": center,
            "deformation_parameter": deformation_parameter,
            "taylor_parameter": taylor_parameter,
            "major_axis_length": major_axis,
            "minor_axis_length": minor_axis,
            "aspect_ratio": aspect_ratio,
            "ellipticity": ellipticity,
            "shape_factor": shape_factor,
            "convexity": convexity,
            "roughness": roughness,
            "principal_axis_angle": angle_rad,
            "flow_alignment_angle": flow_alignment_angle,
            **neck_features,
        }

        return features

    def _analyze_neck_formation(
        self, contour: np.ndarray, major_axis: float, minor_axis: float, angle: float
    ) -> Dict:
        """
        Analyze neck formation and potential breakup points

        Args:
            contour: Droplet contour
            major_axis, minor_axis: Ellipse parameters
            angle: Principal axis angle

        Returns:
            Dictionary with neck analysis features
        """
        if len(contour) < 10:
            return {
                "neck_width": minor_axis,
                "neck_position": 0.5,
                "end_curvature": (0, 0),
                "breakup_probability": 0,
            }

        # Get contour points
        points = contour.reshape(-1, 2).astype(float)

        # Rotate to align with principal axis
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        rotated_points = centered_points @ rotation_matrix.T

        # Find width along major axis
        x_coords = rotated_points[:, 0]
        y_coords = rotated_points[:, 1]

        # Sample widths along the length
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        x_samples = np.linspace(x_min, x_max, 20)
        widths = []

        for x_sample in x_samples:
            # Find points near this x position
            mask = np.abs(x_coords - x_sample) < (x_max - x_min) / 40
            if np.any(mask):
                y_at_x = y_coords[mask]
                width = np.max(y_at_x) - np.min(y_at_x)
                widths.append(width)
            else:
                widths.append(0)

        widths = np.array(widths)

        # Find minimum width (neck)
        if len(widths) > 0:
            neck_idx = np.argmin(widths)
            neck_width = widths[neck_idx]
            neck_position = neck_idx / len(widths)  # Relative position along axis
        else:
            neck_width = minor_axis
            neck_position = 0.5

        # Estimate end curvatures (simplified)
        end_curvature_1 = self._estimate_curvature(rotated_points, "start")
        end_curvature_2 = self._estimate_curvature(rotated_points, "end")

        # Breakup probability (heuristic based on neck narrowing and aspect ratio)
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1
        neck_ratio = neck_width / minor_axis if minor_axis > 0 else 1

        # Higher probability for high aspect ratio and narrow neck
        breakup_probability = 0
        if aspect_ratio > 3:  # Significantly deformed
            breakup_probability = min(
                1.0, (aspect_ratio - 3) / 7
            )  # Scale 0-1 for AR 3-10
        if neck_ratio < 0.7:  # Neck forming
            breakup_probability += min(0.5, (0.7 - neck_ratio))

        breakup_probability = min(1.0, breakup_probability)

        return {
            "neck_width": neck_width,
            "neck_position": neck_position,
            "end_curvature": (end_curvature_1, end_curvature_2),
            "breakup_probability": breakup_probability,
        }

    def _estimate_curvature(self, points: np.ndarray, end: str) -> float:
        """Estimate curvature at droplet ends"""
        if len(points) < 5:
            return 0

        # Sort by x coordinate
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]

        if end == "start":
            end_points = sorted_points[:5]
        else:
            end_points = sorted_points[-5:]

        # Fit parabola to estimate curvature
        try:
            x = end_points[:, 0]
            y = end_points[:, 1]

            # Fit y = ax² + bx + c
            A = np.vstack([x**2, x, np.ones(len(x))]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a = coeffs[0]

            # Curvature = |2a| / (1 + (2ax + b)²)^(3/2)
            # Simplified to just |2a| for relative comparison
            curvature = abs(2 * a) if not np.isnan(a) else 0
            return curvature

        except:
            return 0

    def track_droplet_deformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Track deformation evolution over time for each droplet

        Args:
            df: DataFrame with trajectory_id and deformation features

        Returns:
            Enhanced DataFrame with temporal deformation features
        """
        df_enhanced = df.copy()

        # Initialize temporal columns with proper dtypes
        temporal_cols = [
            "deformation_rate",
            "rotation_rate",
            "area_change_rate",
            "deformation_acceleration",
        ]

        # Initialize float columns
        for col in temporal_cols:
            df_enhanced[col] = 0.0

        # Initialize boolean column with proper dtype
        df_enhanced["breakup_imminent"] = False
        df_enhanced["breakup_imminent"] = df_enhanced["breakup_imminent"].astype("bool")

        for traj_id in df_enhanced["trajectory_id"].unique():
            if traj_id == -1:
                continue

            traj_mask = df_enhanced["trajectory_id"] == traj_id
            traj_data = df_enhanced[traj_mask].sort_values("frame")

            if len(traj_data) < 2:
                continue

            # Time derivatives
            frames = traj_data["frame"].values
            dt = 1.0 / self.frame_rate

            # Deformation evolution
            deformation = traj_data["deformation_parameter"].values
            if len(deformation) > 1:
                deform_rate = np.gradient(deformation, dt)
                df_enhanced.loc[traj_data.index, "deformation_rate"] = deform_rate

                if len(deformation) > 2:
                    deform_accel = np.gradient(deform_rate, dt)
                    df_enhanced.loc[traj_data.index, "deformation_acceleration"] = (
                        deform_accel
                    )

            # Rotation rate
            if "moment_orientation" in traj_data.columns:
                angles = traj_data["moment_orientation"].values
                if len(angles) > 1:
                    # Handle angle wrapping
                    angle_diff = np.diff(angles)
                    angle_diff = np.where(
                        angle_diff > np.pi / 2, angle_diff - np.pi, angle_diff
                    )
                    angle_diff = np.where(
                        angle_diff < -np.pi / 2, angle_diff + np.pi, angle_diff
                    )

                    rotation_rates = np.concatenate([[0], angle_diff / dt])
                    df_enhanced.loc[traj_data.index, "rotation_rate"] = rotation_rates

            # Area change rate
            areas = traj_data["area"].values
            if len(areas) > 1:
                area_rates = np.gradient(areas, dt) / (
                    areas + 1e-8
                )  # Relative rate, avoid division by zero
                df_enhanced.loc[traj_data.index, "area_change_rate"] = area_rates

            # Breakup prediction (look for rapid deformation increase)
            if "breakup_probability" in traj_data.columns:
                breakup_scores = traj_data["breakup_probability"].values
                if len(breakup_scores) > 2:
                    try:
                        from scipy.signal import savgol_filter

                        # Smooth and find rapid increases
                        window_length = min(5, len(breakup_scores))
                        if window_length >= 3 and window_length % 2 == 0:
                            window_length -= 1  # Make it odd
                        if window_length >= 3:
                            smooth_scores = savgol_filter(
                                breakup_scores, window_length, 1
                            )
                            score_gradient = np.gradient(smooth_scores)

                            # Mark as imminent if high probability and increasing rapidly
                            imminent = (smooth_scores > 0.7) & (score_gradient > 0.1)

                            # Ensure we're assigning boolean values explicitly
                            imminent_bool = imminent.astype(bool)
                            df_enhanced.loc[traj_data.index, "breakup_imminent"] = (
                                imminent_bool
                            )
                        else:
                            # Not enough points for smoothing
                            df_enhanced.loc[traj_data.index, "breakup_imminent"] = False
                    except ImportError:
                        # Fallback if scipy not available
                        logging.warning(
                            "scipy not available for breakup prediction smoothing"
                        )
                        simple_imminent = breakup_scores > 0.8
                        df_enhanced.loc[traj_data.index, "breakup_imminent"] = (
                            simple_imminent.astype(bool)
                        )
                    except Exception as e:
                        logging.warning(
                            f"Error in breakup prediction for trajectory {traj_id}: {e}"
                        )
                        df_enhanced.loc[traj_data.index, "breakup_imminent"] = False

        # Ensure all columns have consistent dtypes
        df_enhanced["deformation_rate"] = df_enhanced["deformation_rate"].astype(
            "float64"
        )
        df_enhanced["rotation_rate"] = df_enhanced["rotation_rate"].astype("float64")
        df_enhanced["area_change_rate"] = df_enhanced["area_change_rate"].astype(
            "float64"
        )
        df_enhanced["deformation_acceleration"] = df_enhanced[
            "deformation_acceleration"
        ].astype("float64")
        df_enhanced["breakup_imminent"] = df_enhanced["breakup_imminent"].astype("bool")

        return df_enhanced

    def _ensure_consistent_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure consistent data types for all columns to avoid parquet conversion issues

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with consistent dtypes
        """
        df_clean = df.copy()

        # Define expected dtypes for known columns
        dtype_mapping = {
            # Temporal features
            "deformation_rate": "float64",
            "rotation_rate": "float64",
            "area_change_rate": "float64",
            "deformation_acceleration": "float64",
            "breakup_imminent": "bool",
            # Basic features
            "frame": "int64",
            "object_id": "int64",
            "trajectory_id": "int64",
            "area": "float64",
            "centroid_x": "float64",
            "centroid_y": "float64",
            # Shape features
            "major_axis": "float64",
            "minor_axis": "float64",
            "aspect_ratio": "float64",
            "deformation_parameter": "float64",
            "circularity": "float64",
            "solidity": "float64",
            "eccentricity": "float64",
            # Focus features
            "focus_score": "float64",
            "is_in_focus": "bool",
            # Quality indicators
            "is_anomaly": "bool",
            "is_redundant": "bool",
        }

        # Apply dtype conversions
        for col, target_dtype in dtype_mapping.items():
            if col in df_clean.columns:
                try:
                    if target_dtype == "bool":
                        # Handle boolean conversion carefully
                        if df_clean[col].dtype == "object":
                            # Convert object column to boolean
                            df_clean[col] = df_clean[col].astype("bool")
                        else:
                            # Convert numeric to boolean
                            df_clean[col] = (df_clean[col] != 0).astype("bool")
                    else:
                        df_clean[col] = df_clean[col].astype(target_dtype)
                except Exception as e:
                    logging.warning(
                        f"Could not convert column {col} to {target_dtype}: {e}"
                    )

        return df_clean

    def detect_breakup_events(self, df_tracked: pd.DataFrame) -> pd.DataFrame:
        """
        Detect actual breakup events by analyzing trajectory terminations

        Args:
            df_tracked: DataFrame with trajectory tracking

        Returns:
            DataFrame of detected breakup events
        """
        breakup_events = []

        for traj_id in df_tracked["trajectory_id"].unique():
            if traj_id == -1:
                continue

            traj_data = df_tracked[df_tracked["trajectory_id"] == traj_id].sort_values(
                "frame"
            )

            if len(traj_data) < 3:
                continue

            # Check if trajectory ends abruptly (potential breakup)
            last_frame = traj_data["frame"].iloc[-1]
            max_frame = df_tracked["frame"].max()

            # If trajectory ends before video ends, might be breakup
            if last_frame < max_frame - 2:
                last_detection = traj_data.iloc[-1]

                # Look for daughter droplets in subsequent frames
                subsequent_frames = range(
                    last_frame + 1, min(last_frame + 5, max_frame + 1)
                )
                daughter_candidates = []

                for frame in subsequent_frames:
                    frame_data = df_tracked[df_tracked["frame"] == frame]

                    # Find droplets near the last position
                    distances = np.sqrt(
                        (frame_data["centroid_x"] - last_detection["centroid_x"]) ** 2
                        + (frame_data["centroid_y"] - last_detection["centroid_y"]) ** 2
                    )

                    nearby = frame_data[distances < 100]  # Within 100 pixels
                    daughter_candidates.extend(nearby["trajectory_id"].unique())

                # Analyze potential breakup
                pre_breakup_deformation = (
                    traj_data["deformation_parameter"].iloc[-3:].mean()
                )
                pre_breakup_aspect_ratio = traj_data["aspect_ratio"].iloc[-3:].mean()
                pre_breakup_breakup_prob = (
                    traj_data["breakup_probability"].iloc[-3:].mean()
                )

                breakup_event = {
                    "parent_trajectory_id": traj_id,
                    "breakup_frame": last_frame,
                    "breakup_position_x": last_detection["centroid_x"],
                    "breakup_position_y": last_detection["centroid_y"],
                    "pre_breakup_deformation": pre_breakup_deformation,
                    "pre_breakup_aspect_ratio": pre_breakup_aspect_ratio,
                    "pre_breakup_probability": pre_breakup_breakup_prob,
                    "daughter_trajectory_count": len(set(daughter_candidates)),
                    "daughter_trajectory_ids": list(set(daughter_candidates)),
                    "parent_final_area": last_detection["area"],
                    "parent_final_major_axis": last_detection[
                        "major_axis"
                    ],  # changed to major_axis from major_axis_length
                    "parent_final_minor_axis": last_detection["minor_axis"],
                }

                breakup_events.append(breakup_event)

        return pd.DataFrame(breakup_events)

    def analyze_flow_field_effects(
        self, df_tracked: pd.DataFrame, flow_field_func: callable = None
    ) -> pd.DataFrame:
        """
        Analyze how local flow field affects droplet deformation

        Args:
            df_tracked: DataFrame with tracked droplets
            flow_field_func: Function that returns (strain_rate, vorticity) for (x, y, t)

        Returns:
            Enhanced DataFrame with flow field analysis
        """
        df_flow = df_tracked.copy()

        # Initialize flow-related columns
        flow_cols = [
            "local_strain_rate",
            "local_vorticity",
            "capillary_number",
            "predicted_deformation",
            "deformation_deviation",
        ]
        for col in flow_cols:
            df_flow[col] = 0.0

        if flow_field_func is None:
            # Use simple uniform extensional flow assumption
            flow_field_func = lambda x, y, t: (self.expected_flow_rate or 1.0, 0.0)

        for idx, row in df_flow.iterrows():
            x, y = row["centroid_x"], row["centroid_y"]
            t = row["frame"] / self.frame_rate

            strain_rate, vorticity = flow_field_func(x, y, t)

            df_flow.loc[idx, "local_strain_rate"] = strain_rate
            df_flow.loc[idx, "local_vorticity"] = vorticity

            # Estimate capillary number (needs material properties)
            # This is a placeholder - you'd need actual viscosities and surface tension
            radius = (
                np.sqrt(row["area"] / np.pi) * self.pixel_size
            )  # Convert to physical units
            capillary_number = 0.1 * strain_rate * radius  # Placeholder calculation
            df_flow.loc[idx, "capillary_number"] = capillary_number

            # Predict deformation based on theory (simple linear approximation)
            predicted_deformation = min(
                0.9, 0.1 * capillary_number
            )  # Placeholder model
            df_flow.loc[idx, "predicted_deformation"] = predicted_deformation
            df_flow.loc[idx, "deformation_deviation"] = (
                row["deformation_parameter"] - predicted_deformation
            )

        return df_flow

    def classify_deformation_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify droplets into deformation regimes

        Args:
            df: DataFrame with deformation features

        Returns:
            DataFrame with regime classifications
        """
        df_classified = df.copy()

        # Define regime boundaries (these are example values - adjust based on your system)
        regimes = []

        for idx, row in df.iterrows():
            deformation = row["deformation_parameter"]
            aspect_ratio = row["aspect_ratio"]
            breakup_prob = row["breakup_probability"]

            if deformation < 0.1:
                regime = "nearly_spherical"
            elif deformation < 0.3:
                regime = "slightly_deformed"
            elif deformation < 0.6:
                regime = "moderately_deformed"
            elif breakup_prob > 0.8:
                regime = "critical_breakup"
            else:
                regime = "highly_deformed"

            regimes.append(regime)

        df_classified["deformation_regime"] = regimes

        return df_classified

    def export_deformation_analysis(
        self, df_enhanced: pd.DataFrame, breakup_events: pd.DataFrame, output_base: str
    ):
        """
        Export comprehensive deformation analysis results

        Args:
            df_enhanced: Enhanced droplet DataFrame
            breakup_events: Breakup events DataFrame
            output_base: Base path for output files
        """

        # Ensure consistent dtypes before saving
        df_enhanced_clean = self._ensure_consistent_dtypes(df_enhanced)

        # 1. Main enhanced dataset
        df_enhanced_clean.to_parquet(f"{output_base}_droplet_analysis.parquet")

        # 2. Breakup events
        if not breakup_events.empty:
            breakup_events.to_parquet(f"{output_base}_breakup_events.parquet")

        # 3. Summary statistics by regime
        if "deformation_regime" in df_enhanced_clean.columns:
            regime_summary = (
                df_enhanced_clean.groupby("deformation_regime")
                .agg(
                    {
                        "deformation_parameter": ["mean", "std", "min", "max"],
                        "aspect_ratio": ["mean", "std"],
                        "breakup_probability": ["mean", "std"]
                        if "breakup_probability" in df_enhanced_clean.columns
                        else ["count"],
                        "area": ["mean", "std"],
                        "trajectory_id": "nunique",
                    }
                )
                .round(4)
            )

            regime_summary.to_csv(f"{output_base}_regime_summary.csv")

        # 4. Temporal analysis for trajectories
        temporal_stats = []
        for traj_id in df_enhanced_clean["trajectory_id"].unique():
            if traj_id == -1:
                continue

            traj_data = df_enhanced_clean[
                df_enhanced_clean["trajectory_id"] == traj_id
            ].sort_values("frame")

            if len(traj_data) < 3:
                continue

            stats = {
                "trajectory_id": traj_id,
                "duration": len(traj_data),
                "max_deformation": traj_data["deformation_parameter"].max(),
                "deformation_range": traj_data["deformation_parameter"].max()
                - traj_data["deformation_parameter"].min(),
                "mean_deformation_rate": traj_data["deformation_rate"].mean()
                if "deformation_rate" in traj_data.columns
                else 0,
                "max_aspect_ratio": traj_data["aspect_ratio"].max(),
                "final_breakup_probability": traj_data["breakup_probability"].iloc[-1]
                if "breakup_probability" in traj_data.columns
                else 0,
                "regime_transitions": len(traj_data["deformation_regime"].unique())
                if "deformation_regime" in traj_data.columns
                else 1,
                "broke_up": traj_id in breakup_events["parent_trajectory_id"].values
                if not breakup_events.empty
                and "parent_trajectory_id" in breakup_events.columns
                else False,
            }
            temporal_stats.append(stats)

        if temporal_stats:
            pd.DataFrame(temporal_stats).to_parquet(
                f"{output_base}_trajectory_analysis.parquet"
            )

        # 5. Create summary plots
        self._create_deformation_plots(df_enhanced_clean, breakup_events, output_base)

        logging.info(f"Deformation analysis exported to {output_base}_*")

    def _create_deformation_plots(
        self, df: pd.DataFrame, breakup_events: pd.DataFrame, output_base: str
    ):
        """Create summary visualization plots"""

        # Plot 1: Deformation vs Time for sample trajectories
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Sample a few long trajectories
        trajectory_counts = df["trajectory_id"].value_counts()
        long_trajectories = trajectory_counts[trajectory_counts >= 10].head(5).index

        ax = axes[0, 0]
        for traj_id in long_trajectories:
            if traj_id == -1:
                continue
            traj_data = df[df["trajectory_id"] == traj_id].sort_values("frame")
            ax.plot(
                traj_data["frame"],
                traj_data["deformation_parameter"],
                alpha=0.7,
                label=f"Traj {traj_id}",
            )

        ax.set_xlabel("Frame")
        ax.set_ylabel("Deformation Parameter")
        ax.set_title("Deformation Evolution Over Time")
        ax.legend()

        # Plot 2: Aspect ratio vs Deformation parameter
        ax = axes[0, 1]
        scatter = ax.scatter(
            df["deformation_parameter"],
            df["aspect_ratio"],
            c=df["breakup_probability"],
            cmap="viridis",
            alpha=0.6,
        )
        ax.set_xlabel("Deformation Parameter")
        ax.set_ylabel("Aspect Ratio")
        ax.set_title("Deformation vs Aspect Ratio (colored by breakup probability)")
        plt.colorbar(scatter, ax=ax)

        # Plot 3: Deformation regime distribution
        ax = axes[1, 0]
        regime_counts = df["deformation_regime"].value_counts()
        ax.bar(regime_counts.index, regime_counts.values)
        ax.set_xlabel("Deformation Regime")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Deformation Regimes")
        plt.xticks(rotation=45)

        # Plot 4: Breakup analysis
        ax = axes[1, 1]
        if not breakup_events.empty:
            ax.hist(breakup_events["pre_breakup_deformation"], bins=20, alpha=0.7)
            ax.set_xlabel("Pre-breakup Deformation Parameter")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Deformation at Breakup")
        else:
            ax.text(
                0.5,
                0.5,
                "No breakup events detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Breakup Events")

        plt.tight_layout()
        plt.savefig(
            f"{output_base}_deformation_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


# Integration with existing SAM2PostProcessor
class IntegratedDropletAnalyzer:
    """Combines SAM2 blob analysis with droplet deformation analysis"""

    def __init__(
        self,
        sam2_df: pd.DataFrame,
        flow_direction: float = 0.0,
        pixel_size: float = 1.0,
        frame_rate: float = 1.0,
        expected_flow_rate: float = 1.0,
    ):
        """
        Initialize integrated analyzer

        Args:
            sam2_df: DataFrame from SAM2BlobAnalyzer
            flow_direction: Flow direction in radians
            pixel_size: Microns per pixel
            frame_rate: Frames per second
        """
        self.sam2_df = sam2_df
        self.droplet_analyzer = DropletDeformationAnalyzer(
            flow_direction=flow_direction,
            pixel_size=pixel_size,
            frame_rate=frame_rate,
            expected_flow_rate=expected_flow_rate,
        )

    def run_complete_analysis(self, output_base: str) -> Dict[str, pd.DataFrame]:
        """
        Run complete droplet deformation analysis pipeline

        Args:
            output_base: Base path for output files

        Returns:
            Dictionary of analysis results
        """
        logging.info("Starting complete droplet deformation analysis...")

        # 1. Add deformation features to existing SAM2 features
        df_enhanced = self.sam2_df.copy()

        # Note: This would need mask reconstruction from RLE data
        # For now, using existing geometric features to approximate deformation features

        # Approximate deformation features from existing SAM2 data
        df_enhanced["deformation_parameter"] = (
            df_enhanced["major_axis"] - df_enhanced["minor_axis"]
        ) / (df_enhanced["major_axis"] + df_enhanced["minor_axis"])
        df_enhanced["taylor_parameter"] = (
            df_enhanced["major_axis"] ** 2 - df_enhanced["minor_axis"] ** 2
        ) / (df_enhanced["major_axis"] ** 2 + df_enhanced["minor_axis"] ** 2)
        df_enhanced["ellipticity"] = 1 - (
            df_enhanced["minor_axis"] / df_enhanced["major_axis"]
        )

        # Add flow alignment (assuming flow is horizontal)
        df_enhanced["flow_alignment_angle"] = np.abs(
            np.deg2rad(df_enhanced["orientation"])
            - self.droplet_analyzer.flow_direction
        )
        df_enhanced["flow_alignment_angle"] = np.minimum(
            df_enhanced["flow_alignment_angle"],
            np.pi - df_enhanced["flow_alignment_angle"],
        )

        # Estimate breakup probability from aspect ratio and shape factors
        aspect_ratios = df_enhanced["aspect_ratio"].values
        solidities = df_enhanced["solidity"].values
        df_enhanced["breakup_probability"] = np.clip(
            (aspect_ratios - 3) / 7 + (1 - solidities) * 0.5, 0, 1
        )

        # 2. Track trajectories with SAM2PostProcessor methods
        from sam2_blob_analysis_postprocessing_claude import SAM2PostProcessor

        post_processor = SAM2PostProcessor(df_enhanced)
        df_tracked = post_processor.track_blob_trajectories(
            max_distance=30.0, max_frame_gap=3
        )

        # 3. Add temporal deformation analysis
        df_deformation = self.droplet_analyzer.track_droplet_deformation(df_tracked)

        # 4. Detect breakup events
        breakup_events = self.droplet_analyzer.detect_breakup_events(df_deformation)

        # 5. Classify deformation regimes
        df_classified = self.droplet_analyzer.classify_deformation_regimes(
            df_deformation
        )

        # 6. Flow field analysis (if flow function provided)
        df_flow = self.droplet_analyzer.analyze_flow_field_effects(df_classified)

        # 7. Export results
        self.droplet_analyzer.export_deformation_analysis(
            df_flow, breakup_events, output_base
        )

        results = {
            "enhanced_data": df_flow,
            "breakup_events": breakup_events,
            "trajectory_stats": post_processor.compute_trajectory_features(df_tracked),
            "regime_analysis": self._analyze_regime_transitions(df_flow),
        }

        logging.info("Complete droplet analysis finished!")
        return results

    def _analyze_regime_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze transitions between deformation regimes"""
        transitions = []

        for traj_id in df["trajectory_id"].unique():
            if traj_id == -1:
                continue

            traj_data = df[df["trajectory_id"] == traj_id].sort_values("frame")

            if len(traj_data) < 2:
                continue

            regimes = traj_data["deformation_regime"].values
            regime_changes = []

            current_regime = regimes[0]
            change_frame = traj_data["frame"].iloc[0]

            for i, regime in enumerate(regimes[1:], 1):
                if regime != current_regime:
                    regime_changes.append(
                        {
                            "from_regime": current_regime,
                            "to_regime": regime,
                            "transition_frame": traj_data["frame"].iloc[i],
                            "duration_in_previous": traj_data["frame"].iloc[i]
                            - change_frame,
                        }
                    )
                    current_regime = regime
                    change_frame = traj_data["frame"].iloc[i]

            for change in regime_changes:
                change["trajectory_id"] = traj_id
                transitions.append(change)

        return pd.DataFrame(transitions)


# Advanced Analysis Functions for Fluid Mechanics Insights
class FluidMechanicsAnalyzer:
    """Specialized analysis for fluid mechanics insights"""

    def __init__(self, material_properties: Dict = None):
        """
        Initialize with material properties

        Args:
            material_properties: Dict with viscosities, surface tension, etc.
                Expected keys: 'mu_matrix', 'mu_droplet', 'sigma', 'characteristic_length'
        """
        self.props = material_properties or {
            "mu_matrix": 1.0,  # Matrix viscosity (Pa⋅s)
            "mu_droplet": 1.0,  # Droplet viscosity (Pa⋅s)
            "sigma": 0.01,  # Surface tension (N/m)
            "characteristic_length": 1e-4,  # Characteristic length (m)
        }

    def compute_dimensionless_groups(
        self, df: pd.DataFrame, strain_rate_col: str = "local_strain_rate"
    ) -> pd.DataFrame:
        """
        Compute relevant dimensionless groups for droplet deformation

        Args:
            df: DataFrame with droplet data
            strain_rate_col: Column name for strain rate data

        Returns:
            DataFrame with dimensionless groups added
        """
        df_enhanced = df.copy()

        # Extract radius from area (assuming circular cross-section)
        radii = np.sqrt(df["area"] / np.pi)  # in pixels
        radii_physical = radii * 1e-6  # Convert to meters (assuming 1 pixel = 1 μm)

        # Capillary number: Ca = μ * γ̇ * R / σ
        if strain_rate_col in df.columns:
            strain_rates = df[strain_rate_col].values
        else:
            strain_rates = np.ones(len(df))  # Default strain rate

        capillary_numbers = (
            self.props["mu_matrix"]
            * strain_rates
            * radii_physical
            / self.props["sigma"]
        )

        # Viscosity ratio: λ = μ_droplet / μ_matrix
        viscosity_ratio = self.props["mu_droplet"] / self.props["mu_matrix"]

        # Reynolds number: Re = ρ * γ̇ * R² / μ (assuming low Re)
        # Typically neglected in Stokes flow, but included for completeness
        density = 1000  # kg/m³ (water-like)
        reynolds_numbers = (
            density * strain_rates * radii_physical**2 / self.props["mu_matrix"]
        )

        df_enhanced["capillary_number"] = capillary_numbers
        df_enhanced["viscosity_ratio"] = viscosity_ratio
        df_enhanced["reynolds_number"] = reynolds_numbers
        df_enhanced["physical_radius"] = radii_physical

        return df_enhanced

    def fit_deformation_models(self, df: pd.DataFrame) -> Dict:
        """
        Fit theoretical deformation models to experimental data

        Args:
            df: DataFrame with capillary numbers and deformation data

        Returns:
            Dictionary with fitted model parameters and goodness of fit
        """
        results = {}

        # Filter valid data
        valid_mask = (df["capillary_number"] > 0) & (
            df["deformation_parameter"].notna()
        )
        ca = df[valid_mask]["capillary_number"].values
        deformation = df[valid_mask]["deformation_parameter"].values

        if len(ca) < 10:
            logging.warning("Insufficient data for model fitting")
            return results

        # Model 1: Linear small deformation theory
        # D = (19λ + 16)/(16λ + 16) * Ca  (for small Ca)
        try:
            lambda_visc = df["viscosity_ratio"].iloc[0]
            theoretical_coeff = (19 * lambda_visc + 16) / (16 * lambda_visc + 16)

            # Fit D = α * Ca where α should ≈ theoretical_coeff
            slope, intercept = np.polyfit(ca, deformation, 1)

            # Goodness of fit
            predicted = slope * ca + intercept
            r_squared = 1 - np.sum((deformation - predicted) ** 2) / np.sum(
                (deformation - np.mean(deformation)) ** 2
            )

            results["linear_model"] = {
                "fitted_slope": slope,
                "theoretical_slope": theoretical_coeff,
                "slope_ratio": slope / theoretical_coeff,
                "intercept": intercept,
                "r_squared": r_squared,
                "valid_for_ca": "Ca < 0.3",
            }
        except Exception as e:
            logging.warning(f"Linear model fitting failed: {e}")

        # Model 2: Taylor's series expansion
        # D = α₁*Ca + α₂*Ca² + α₃*Ca³
        try:
            if len(ca) > 20:  # Need more data for polynomial fitting
                poly_coeffs = np.polyfit(ca, deformation, 3)
                predicted = np.polyval(poly_coeffs, ca)
                r_squared = 1 - np.sum((deformation - predicted) ** 2) / np.sum(
                    (deformation - np.mean(deformation)) ** 2
                )

                results["polynomial_model"] = {
                    "coefficients": poly_coeffs.tolist(),
                    "r_squared": r_squared,
                    "valid_for_ca": "General",
                }
        except Exception as e:
            logging.warning(f"Polynomial model fitting failed: {e}")

        # Model 3: Breakup criterion
        # Critical Ca for breakup
        breakup_mask = df[valid_mask]["breakup_probability"] > 0.8
        if np.any(breakup_mask):
            breakup_ca = ca[breakup_mask]
            if len(breakup_ca) > 5:
                critical_ca = np.mean(breakup_ca)
                ca_std = np.std(breakup_ca)

                results["breakup_model"] = {
                    "critical_capillary_number": critical_ca,
                    "critical_ca_std": ca_std,
                    "breakup_count": len(breakup_ca),
                    "total_count": len(ca),
                }

        return results

    def analyze_breakup_mechanisms(
        self, breakup_events: pd.DataFrame, df_full: pd.DataFrame
    ) -> Dict:
        """
        Analyze breakup mechanisms and classify breakup types

        Args:
            breakup_events: DataFrame of detected breakup events
            df_full: Full droplet DataFrame for context

        Returns:
            Dictionary with breakup mechanism analysis
        """
        if breakup_events.empty:
            return {"message": "No breakup events detected"}

        analysis = {}

        # 1. Breakup threshold analysis
        critical_deformations = breakup_events["pre_breakup_deformation"].values
        critical_aspect_ratios = breakup_events["pre_breakup_aspect_ratio"].values

        analysis["threshold_analysis"] = {
            "mean_critical_deformation": np.mean(critical_deformations),
            "std_critical_deformation": np.std(critical_deformations),
            "mean_critical_aspect_ratio": np.mean(critical_aspect_ratios),
            "std_critical_aspect_ratio": np.std(critical_aspect_ratios),
            "breakup_count": len(breakup_events),
        }

        # 2. Classify breakup mechanisms based on aspect ratio and daughter count
        mechanisms = []
        for _, event in breakup_events.iterrows():
            aspect_ratio = event["pre_breakup_aspect_ratio"]
            daughter_count = event["daughter_trajectory_count"]

            if aspect_ratio < 3:
                mechanism = "capillary_instability"  # Low AR, surface tension dominated
            elif aspect_ratio > 10:
                if daughter_count <= 2:
                    mechanism = "end_pinching"  # Very elongated, end pinch-off
                else:
                    mechanism = "capillary_pearling"  # Multiple breakup points
            else:
                mechanism = "viscous_breakup"  # Moderate AR, viscous forces

            mechanisms.append(mechanism)

        breakup_events_enhanced = breakup_events.copy()
        breakup_events_enhanced["breakup_mechanism"] = mechanisms

        mechanism_counts = pd.Series(mechanisms).value_counts()
        analysis["mechanism_distribution"] = mechanism_counts.to_dict()

        # 3. Size distribution analysis of daughter droplets
        if "daughter_trajectory_ids" in breakup_events.columns:
            daughter_sizes = []
            for daughters in breakup_events["daughter_trajectory_ids"]:
                if daughters:  # If list is not empty
                    for daughter_id in daughters:
                        if daughter_id in df_full["trajectory_id"].values:
                            # Get first detection of daughter droplet
                            daughter_data = df_full[
                                df_full["trajectory_id"] == daughter_id
                            ]
                            if not daughter_data.empty:
                                first_detection = daughter_data.sort_values(
                                    "frame"
                                ).iloc[0]
                                daughter_sizes.append(first_detection["area"])

            if daughter_sizes:
                analysis["daughter_size_distribution"] = {
                    "mean_daughter_size": np.mean(daughter_sizes),
                    "std_daughter_size": np.std(daughter_sizes),
                    "size_ratio_to_parent": np.mean(daughter_sizes)
                    / np.mean(breakup_events["parent_final_area"]),
                    "daughter_count": len(daughter_sizes),
                }

        return analysis, breakup_events_enhanced

    def compute_flow_characterization(self, df: pd.DataFrame) -> Dict:
        """
        Characterize the flow field based on droplet behavior

        Args:
            df: DataFrame with tracked droplets and deformation data

        Returns:
            Dictionary with flow field characterization
        """
        analysis = {}

        # 1. Spatial strain rate mapping
        # Bin droplets by position and analyze average deformation
        x_bins = np.linspace(df["centroid_x"].min(), df["centroid_x"].max(), 10)
        y_bins = np.linspace(df["centroid_y"].min(), df["centroid_y"].max(), 10)

        strain_rate_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
        count_map = np.zeros_like(strain_rate_map)

        for i, row in df.iterrows():
            x_idx = np.digitize(row["centroid_x"], x_bins) - 1
            y_idx = np.digitize(row["centroid_y"], y_bins) - 1

            if (
                0 <= x_idx < strain_rate_map.shape[1]
                and 0 <= y_idx < strain_rate_map.shape[0]
            ):
                if "deformation_rate" in df.columns:
                    strain_rate_map[y_idx, x_idx] += abs(row["deformation_rate"])
                else:
                    strain_rate_map[y_idx, x_idx] += row["deformation_parameter"]
                count_map[y_idx, x_idx] += 1

        # Average strain rates in each bin
        nonzero_mask = count_map > 0
        strain_rate_map[nonzero_mask] /= count_map[nonzero_mask]

        analysis["spatial_strain_mapping"] = {
            "strain_rate_map": strain_rate_map.tolist(),
            "x_bin_centers": ((x_bins[:-1] + x_bins[1:]) / 2).tolist(),
            "y_bin_centers": ((y_bins[:-1] + y_bins[1:]) / 2).tolist(),
            "mean_strain_rate": np.mean(strain_rate_map[nonzero_mask])
            if np.any(nonzero_mask)
            else 0,
            "strain_rate_std": np.std(strain_rate_map[nonzero_mask])
            if np.any(nonzero_mask)
            else 0,
        }

        # 2. Flow uniformity analysis
        deformation_std = df["deformation_parameter"].std()
        deformation_mean = df["deformation_parameter"].mean()
        uniformity_coefficient = (
            1 - (deformation_std / deformation_mean) if deformation_mean > 0 else 0
        )

        analysis["flow_uniformity"] = {
            "uniformity_coefficient": uniformity_coefficient,
            "deformation_cv": deformation_std / deformation_mean
            if deformation_mean > 0
            else float("inf"),
            "spatial_variance": np.var(strain_rate_map[nonzero_mask])
            if np.any(nonzero_mask)
            else 0,
        }

        # 3. Temporal stability analysis
        if "frame" in df.columns:
            frame_groups = df.groupby("frame")["deformation_parameter"].mean()
            temporal_stability = (
                1 - (frame_groups.std() / frame_groups.mean())
                if frame_groups.mean() > 0
                else 0
            )

            analysis["temporal_stability"] = {
                "stability_coefficient": temporal_stability,
                "temporal_cv": frame_groups.std() / frame_groups.mean()
                if frame_groups.mean() > 0
                else float("inf"),
                "frame_to_frame_variation": np.mean(
                    np.abs(np.diff(frame_groups.values))
                ),
            }

        return analysis


# Complete workflow integration
def run_complete_droplet_analysis(
    sam2_results_path: str,
    output_base: str,
    flow_direction: float = 0.0,
    pixel_size: float = 1.0,
    frame_rate: float = 1.0,
    expected_flow_rate: float = 1.0,
    material_properties: Dict = None,
) -> Dict:
    """
    Run complete droplet deformation analysis workflow

    Args:
        sam2_results_path: Path to SAM2BlobAnalyzer results parquet file
        output_base: Base path for output files
        flow_direction: Flow direction in radians (0 = horizontal)
        pixel_size: Microns per pixel for scaling
        frame_rate: Frames per second
        material_properties: Dict with fluid properties

    Returns:
        Dictionary with all analysis results
    """

    # Load SAM2 results
    logging.info(f"Loading SAM2 results from {sam2_results_path}")
    sam2_df = pd.read_parquet(sam2_results_path)

    # Initialize integrated analyzer
    analyzer = IntegratedDropletAnalyzer(
        sam2_df=sam2_df,
        flow_direction=flow_direction,
        pixel_size=pixel_size,
        frame_rate=frame_rate,
        expected_flow_rate=expected_flow_rate,
    )

    # Run complete analysis
    results = analyzer.run_complete_analysis(output_base)

    # Add fluid mechanics analysis
    fluid_analyzer = FluidMechanicsAnalyzer(material_properties)

    # Compute dimensionless groups
    df_with_dimensionless = fluid_analyzer.compute_dimensionless_groups(
        results["enhanced_data"]
    )

    # Fit deformation models
    model_fits = fluid_analyzer.fit_deformation_models(df_with_dimensionless)

    # Analyze breakup mechanisms
    breakup_analysis, breakup_enhanced = fluid_analyzer.analyze_breakup_mechanisms(
        results["breakup_events"], df_with_dimensionless
    )

    # Flow field characterization
    flow_characterization = fluid_analyzer.compute_flow_characterization(
        df_with_dimensionless
    )

    # Save enhanced results
    df_with_dimensionless.to_parquet(f"{output_base}_complete_analysis.parquet")
    breakup_enhanced.to_parquet(f"{output_base}_breakup_mechanisms.parquet")

    # Save analysis summaries
    import json

    with open(f"{output_base}_model_fits.json", "w") as f:
        json.dump(model_fits, f, indent=2, default=str)

    with open(f"{output_base}_breakup_analysis.json", "w") as f:
        json.dump(breakup_analysis, f, indent=2, default=str)

    with open(f"{output_base}_flow_characterization.json", "w") as f:
        json.dump(flow_characterization, f, indent=2, default=str)

    # Combine all results
    complete_results = {
        **results,
        "dimensionless_data": df_with_dimensionless,
        "model_fits": model_fits,
        "breakup_analysis": breakup_analysis,
        "breakup_mechanisms": breakup_enhanced,
        "flow_characterization": flow_characterization,
    }

    logging.info("Complete droplet deformation analysis finished!")
    logging.info(f"Results saved to {output_base}_*")

    return complete_results


# Example usage
if __name__ == "__main__":
    # Example material properties for water-oil system
    material_props = {
        "mu_matrix": 20.71e-3,  # Water viscosity (Pa⋅s)
        "mu_droplet": 20.71e-3,  # Oil viscosity (Pa⋅s)
        "sigma": 2e-3,  # Surface tension (N/m)
        "characteristic_length": 1e-4,  # 100 μm droplets
    }

    # Run complete analysis
    results = run_complete_droplet_analysis(
        sam2_results_path="/home/leviathan/cdrucker/ml/sam2/outputs/vid64_rolling4/results.parquet",
        output_base="/home/leviathan/cdrucker/ml/sam2/outputs/vid64_rolling4/droplet_analysis_results",
        flow_direction=0.0,  # Horizontal flow
        pixel_size=0.4,  # 1 μm/pixel
        frame_rate=400.0,  # 30 fps
        expected_flow_rate=900,  # strain rate
        material_properties=material_props,
    )

    pass
