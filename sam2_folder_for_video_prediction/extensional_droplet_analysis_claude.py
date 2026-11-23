from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import spatial, stats


@dataclass
class ExtensionalFlowMetrics:
    """Metrics specific to extensional flow analysis"""

    # Flow direction analysis
    flow_alignment: float  # Alignment with extensional axis (vertical in your case)
    cross_flow_position: float  # Position across flow (horizontal)
    extensional_strain_rate: float  # Local strain rate estimate

    # Crowding effects
    local_packing_density: float  # Number density in local neighborhood
    nearest_neighbor_distance: float  # Distance to closest droplet
    coordination_number: int  # Number of neighbors within interaction range
    confinement_ratio: float  # Droplet size relative to available space

    # Faceting analysis
    facet_count: int  # Number of detected facets/edges
    facet_angle_variance: float  # Variance in facet angles
    roundness_deficiency: float  # Deviation from circular due to faceting
    contact_area_estimate: float  # Estimated area in contact with neighbors

    # 3D sampling effects
    apparent_size_rank: int  # Size rank in frame (proxy for z-position)
    focus_depth_estimate: float  # Estimated z-position from focus metrics
    occlusion_probability: float  # Likelihood of being partially occluded


@dataclass
class CrowdingContext:
    """Context information about local crowding environment"""

    frame_packing_fraction: float  # Overall packing in frame
    local_packing_gradient: float  # Spatial gradient in packing
    flow_heterogeneity: float  # Variance in local flow conditions
    visible_droplet_fraction: float  # Estimate of visible vs total droplets


class ConcentratedExtensionalAnalyzer:
    """Specialized analyzer for concentrated extensional flow systems"""

    def __init__(
        self,
        df: pd.DataFrame,
        pixel_to_micron: float = 1.0,
        frame_rate: float = 1000.0,
        channel_width_microns: float = 100.0,
        extensional_axis: str = "vertical",  # 'vertical' or 'horizontal'
        fluid_properties: Dict[str, float] = None,
        visible_droplet_fraction: float = 0.005,
    ):
        """
        Initialize analyzer for concentrated extensional flows

        Args:
            df: Enhanced DataFrame from SAM2PostProcessor
            pixel_to_micron: Conversion factor
            frame_rate: Video frame rate in fps
            channel_width_microns: Width of microfluidic channel
            extensional_axis: Direction of extension ('vertical' for your setup)
            fluid_properties: Fluid properties dictionary
        """
        self._validate_input_dataframe(df)

        self.df = df.copy()
        self.pixel_to_micron = pixel_to_micron
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
        self.channel_width = (
            channel_width_microns / pixel_to_micron
        )  # Convert to pixels
        self.extensional_axis = extensional_axis

        # Fluid properties with defaults suitable for concentrated emulsions
        self.fluid_props = fluid_properties or {
            "surface_tension": 0.02,  # Lower due to surfactants
            "viscosity_d": 0.01,  # Higher due to confinement
            "viscosity_c": 0.05,  # Effective viscosity in crowded medium
            "density_d": 900,
            "density_c": 1000,
            "volume_fraction": 0.7,  # High packing fraction
        }

        # Geometric parameters for your setup
        self.flow_direction = "vertical"  # Inlets top/bottom
        self.gradient_direction = "horizontal"  # Outlets left/right
        self.visible_droplet_fraction = visible_droplet_fraction

        # Initialize spatial analysis
        self._setup_spatial_analysis()

    def _setup_spatial_analysis(self):
        """Setup spatial analysis parameters"""
        # Estimate image dimensions from data
        self.image_width = self.df["centroid_x"].max()
        self.image_height = self.df["centroid_y"].max()

        # Define flow regions
        self.inlet_region = self.image_height * 0.1  # Top 10%
        self.outlet_region = self.image_height * 0.9  # Bottom 90%
        self.central_region = (self.image_height * 0.3, self.image_height * 0.7)

        # Characteristic interaction distance (for crowding analysis)
        median_size = np.sqrt(self.df["area"].median() / np.pi) * 2  # Diameter
        self.interaction_distance = median_size * 2.5  # ~2.5 diameters

    def _validate_input_dataframe(self, df: pd.DataFrame):
        """Validate that DataFrame has required columns"""
        print("🔍 Validating input DataFrame...")

        # Required columns
        required_columns = ["frame", "object_id", "centroid_x", "centroid_y", "area"]

        # Preferred columns (will create defaults if missing)
        preferred_columns = [
            "trajectory_id",
            "major_axis",
            "minor_axis",
            "orientation",
            "circularity",
            "solidity",
            "perimeter",
            "focus_score",
            "is_in_focus",
            "deformation_parameter",
        ]

        # Check required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            raise ValueError(f"❌ Missing required columns: {missing_required}")
        print("✅ All required columns present")

        # Check preferred columns and create defaults if missing
        for col in preferred_columns:
            if col not in df.columns:
                print(f"⚠️  Missing preferred column '{col}' - creating default values")

                if col == "trajectory_id":
                    df[col] = df["object_id"]
                elif col in ["major_axis", "minor_axis"]:
                    radius = np.sqrt(df["area"] / np.pi)
                    df[col] = radius * 2
                elif col == "orientation":
                    df[col] = 0
                elif col in ["circularity", "solidity"]:
                    df[col] = 1.0
                elif col == "perimeter":
                    df[col] = 2 * np.pi * np.sqrt(df["area"] / np.pi)
                elif col == "focus_score":
                    df[col] = 100.0
                elif col == "is_in_focus":
                    df[col] = True
                elif col == "deformation_parameter":
                    print(
                        "   📏 Computing deformation parameter from major/minor axes..."
                    )
                    df[col] = self._compute_deformation_parameter(df)

        print("✅ DataFrame validation complete")

    def _compute_deformation_parameter(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute deformation parameter: (L-W)/(L+W) where L=major axis, W=minor axis
        This is the standard deformation parameter used in droplet literature:
        - D = 0: perfectly spherical droplet
        - D > 0: elongated droplet (higher values = more deformed)
        - D → 1: extremely elongated (like a needle)
        """
        # Ensure we have major and minor axes
        if "major_axis" not in df.columns or "minor_axis" not in df.columns:
            print("   ⚠️  Major/minor axes missing - estimating from area")
            # For circular droplets, major_axis = minor_axis = diameter
            radius = np.sqrt(df["area"] / np.pi)
            diameter = radius * 2
            major_axis = diameter
            minor_axis = diameter
        else:
            major_axis = df["major_axis"]
            minor_axis = df["minor_axis"]

        # Calculate deformation parameter
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        deformation = (major_axis - minor_axis) / (major_axis + minor_axis + epsilon)

        # Ensure deformation is non-negative (major should be >= minor)
        deformation = np.maximum(deformation, 0)

        print(
            f"   📊 Deformation parameter computed - range: {deformation.min():.3f} to {deformation.max():.3f}"
        )
        print(f"   📊 Mean deformation: {deformation.mean():.3f}")

        return deformation

    # =================== EXTENSIONAL FLOW ANALYSIS ===================

    def compute_extensional_metrics(self) -> pd.DataFrame:
        """Compute metrics specific to extensional flow geometry"""
        df_ext = self.df.copy()

        # Flow alignment analysis
        df_ext["flow_alignment"] = self._compute_flow_alignment(df_ext)

        # Cross-flow position (normalized across channel width)
        df_ext["cross_flow_position"] = df_ext["centroid_x"] / self.image_width

        # Local strain rate estimation
        df_ext["extensional_strain_rate"] = self._estimate_strain_rate(df_ext)

        # Flow region classification
        df_ext["flow_region"] = self._classify_flow_regions(df_ext)

        return df_ext

    def _compute_flow_alignment(self, df: pd.DataFrame) -> pd.Series:
        """Compute alignment with extensional flow direction"""
        # For vertical flow, alignment is based on major axis orientation
        # Perfect alignment = 0° (vertical), perpendicular = 90°

        orientation_rad = np.deg2rad(df["orientation"])

        if self.extensional_axis == "vertical":
            # Vertical flow: 0° and 180° are aligned, 90° and 270° are perpendicular
            alignment = np.abs(
                np.sin(orientation_rad)
            )  # 0 = aligned, 1 = perpendicular
        else:
            # Horizontal flow
            alignment = np.abs(np.cos(orientation_rad))

        return (
            1 - alignment
        )  # Convert to alignment score (1 = aligned, 0 = perpendicular)

    def _estimate_strain_rate(self, df: pd.DataFrame) -> pd.Series:
        """Estimate local extensional strain rate"""
        strain_rates = []

        for traj_id in df["trajectory_id"].unique():
            if traj_id == -1:
                continue

            traj_data = df[df["trajectory_id"] == traj_id].sort_values("frame")

            if len(traj_data) < 3:
                strain_rates.extend([0] * len(traj_data))
                continue

            # Compute velocity gradient in flow direction
            positions_y = traj_data["centroid_y"].values
            frames = traj_data["frame"].values

            # Velocity in flow direction
            velocities_y = np.gradient(positions_y, frames * self.dt)

            # Strain rate approximation: dv/dy
            if len(velocities_y) > 1:
                local_strain_rates = np.gradient(velocities_y, positions_y)
                local_strain_rates = np.abs(local_strain_rates)  # Magnitude
            else:
                local_strain_rates = np.zeros_like(velocities_y)

            strain_rates.extend(local_strain_rates.tolist())

        return pd.Series(strain_rates, index=df.index)

    def _classify_flow_regions(self, df: pd.DataFrame) -> pd.Series:
        """Classify droplets by flow region"""
        conditions = [
            df["centroid_y"] < self.inlet_region,
            (df["centroid_y"] >= self.central_region[0])
            & (df["centroid_y"] <= self.central_region[1]),
            df["centroid_y"] > self.outlet_region,
        ]

        choices = ["inlet", "central", "outlet"]

        return pd.Series(
            np.select(conditions, choices, default="transition"), index=df.index
        )

    # =================== CROWDING ANALYSIS ===================

    def compute_crowding_metrics(
        self, df_input: pd.DataFrame = None, neighborhood_radius: float = None
    ) -> pd.DataFrame:
        """Compute crowding and neighbor interaction metrics"""
        if neighborhood_radius is None:
            neighborhood_radius = self.interaction_distance

        # Use input dataframe if provided, otherwise use self.df
        if df_input is not None:
            df_crowd = df_input.copy()
        else:
            df_crowd = self.df.copy()

        # Frame-by-frame analysis for computational efficiency
        crowding_results = []

        for frame_num in df_crowd["frame"].unique():
            frame_data = df_crowd[df_crowd["frame"] == frame_num]
            frame_crowding = self._analyze_frame_crowding(
                frame_data, neighborhood_radius
            )
            crowding_results.append(frame_crowding)

        # Combine results
        crowding_df = pd.concat(crowding_results, ignore_index=True)

        # Merge with original data
        df_crowd = df_crowd.merge(
            crowding_df[
                [
                    "frame",
                    "object_id",
                    "local_packing_density",
                    "nearest_neighbor_distance",
                    "coordination_number",
                    "confinement_ratio",
                    "apparent_size_rank",
                ]
            ],
            on=["frame", "object_id"],
            how="left",
        )

        return df_crowd

    def _analyze_frame_crowding(
        self, frame_data: pd.DataFrame, radius: float
    ) -> pd.DataFrame:
        """Analyze crowding metrics for a single frame"""
        if len(frame_data) < 2:
            # Handle single droplet case
            return frame_data.assign(
                local_packing_density=0,
                nearest_neighbor_distance=np.inf,
                coordination_number=0,
                confinement_ratio=0,
                apparent_size_rank=1,
            )

        positions = frame_data[["centroid_x", "centroid_y"]].values
        areas = frame_data["area"].values
        radii = np.sqrt(areas / np.pi)

        # Build spatial tree for efficient neighbor finding
        tree = spatial.cKDTree(positions)

        results = []

        for i, (idx, row) in enumerate(frame_data.iterrows()):
            # Find neighbors within interaction distance
            neighbor_indices = tree.query_ball_point(positions[i], radius)
            neighbor_indices.remove(i)  # Remove self

            # Local packing density (neighbors per unit area)
            local_area = np.pi * radius**2
            local_packing_density = len(neighbor_indices) / local_area

            # Nearest neighbor distance
            if neighbor_indices:
                neighbor_distances = [
                    np.linalg.norm(positions[i] - positions[j])
                    for j in neighbor_indices
                ]
                nearest_neighbor_distance = min(neighbor_distances)
                coordination_number = len(neighbor_indices)
            else:
                nearest_neighbor_distance = np.inf
                coordination_number = 0

            # Confinement ratio (droplet size vs available space)
            if nearest_neighbor_distance < np.inf:
                available_space = nearest_neighbor_distance - radii[i]
                confinement_ratio = radii[i] / (available_space + radii[i])
            else:
                confinement_ratio = 0

            # Size rank (proxy for z-position in crowded system)
            size_rank = stats.rankdata(areas, method="ordinal")[i]

            results.append(
                {
                    "frame": row["frame"],
                    "object_id": row["object_id"],
                    "local_packing_density": local_packing_density,
                    "nearest_neighbor_distance": nearest_neighbor_distance,
                    "coordination_number": coordination_number,
                    "confinement_ratio": confinement_ratio,
                    "apparent_size_rank": size_rank,
                }
            )

        return pd.DataFrame(results)

    # =================== FACETING ANALYSIS ===================

    def analyze_faceting(self, df_crowd: pd.DataFrame = None) -> pd.DataFrame:
        """Analyze droplet faceting due to crowding"""
        # Use input dataframe if provided, otherwise use self.df
        if df_crowd is not None:
            df_facet = df_crowd.copy()
        else:
            df_facet = self.df.copy()

        # Faceting indicators
        df_facet["roundness_deficiency"] = 1 - df_facet["circularity"]

        # Shape complexity as faceting proxy
        if "coordination_number" in df_facet.columns:
            df_facet["faceting_index"] = (
                df_facet["convexity_defect_count"]
                * df_facet["roundness_deficiency"]
                * (
                    1 + df_facet["coordination_number"] / 10
                )  # More neighbors = more faceting
            )
        else:
            df_facet["faceting_index"] = (
                df_facet["convexity_defect_count"] * df_facet["roundness_deficiency"]
            )

        # Contact area estimation (rough approximation)
        df_facet["contact_area_estimate"] = self._estimate_contact_areas(df_facet)

        # Facet angle analysis (simplified)
        df_facet["facet_angle_variance"] = self._analyze_facet_angles(df_facet)

        return df_facet

    def _estimate_contact_areas(self, df: pd.DataFrame) -> pd.Series:
        """Estimate contact area with neighboring droplets"""
        # Simplified model: contact area increases with coordination number
        # and confinement ratio

        if "coordination_number" in df.columns and "confinement_ratio" in df.columns:
            contact_areas = (
                df["coordination_number"]
                * df["confinement_ratio"]
                * df["perimeter"]
                * 0.1  # Rough scaling factor
            )
        else:
            # Fallback if crowding metrics not available
            contact_areas = df["perimeter"] * 0.05

        return contact_areas

    def _analyze_facet_angles(self, df: pd.DataFrame) -> pd.Series:
        """Analyze variance in facet angles (simplified approach)"""
        # Use shape irregularity metrics as proxy for facet angle variance
        if "coordination_number" in df.columns:
            angle_variance = (
                df["roughness"]
                * df["convexity_defect_count"]
                * (1 + df["coordination_number"])
            )
        else:
            angle_variance = df["roughness"] * df["convexity_defect_count"]

        return angle_variance

    # =================== 3D SAMPLING EFFECTS ===================

    def analyze_3d_sampling_effects(
        self, df_facet: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Analyze effects of 3D sampling on observed droplet population"""
        # Use input dataframe if provided, otherwise use self.df
        if df_facet is not None:
            df_3d = df_facet.copy()
        else:
            df_3d = self.df.copy()

        # Focus-based depth estimation
        df_3d["focus_depth_estimate"] = self._estimate_depth_from_focus(df_3d)

        # Occlusion probability
        df_3d["occlusion_probability"] = self._estimate_occlusion_probability(df_3d)

        # Size-based visibility bias
        df_3d["visibility_bias"] = self._compute_visibility_bias(df_3d)

        # Frame-level crowding context
        frame_context = self._compute_frame_crowding_context(df_3d)
        df_3d = df_3d.merge(frame_context, on="frame", how="left")

        return df_3d

    def _estimate_depth_from_focus(self, df: pd.DataFrame) -> pd.Series:
        """Estimate relative z-position from focus quality"""
        # Normalize focus scores to estimate relative depth
        # Higher focus = closer to focal plane (z=0)

        focus_normalized = (df["focus_score"] - df["focus_score"].min()) / (
            df["focus_score"].max() - df["focus_score"].min() + 1e-6
        )

        # Convert to depth estimate (0 = focal plane, 1 = far from focal plane)
        depth_estimate = 1 - focus_normalized

        return depth_estimate

    def _estimate_occlusion_probability(self, df: pd.DataFrame) -> pd.Series:
        """Estimate probability that droplet is partially occluded"""
        # Based on local crowding and size relative to neighbors

        if "local_packing_density" in df.columns and "confinement_ratio" in df.columns:
            occlusion_prob = (
                df["local_packing_density"]
                * df["confinement_ratio"]
                * (
                    1
                    - df["apparent_size_rank"]
                    / df.groupby("frame")["apparent_size_rank"].transform("max")
                )
            )
        else:
            # Fallback based on focus and size
            size_rank = df.groupby("frame")["area"].rank(ascending=False)
            max_rank = df.groupby("frame")["area"].transform("count")
            occlusion_prob = (1 - df["focus_score"] / df["focus_score"].max()) * (
                size_rank / max_rank
            )

        return np.clip(occlusion_prob, 0, 1)

    def _compute_visibility_bias(self, df: pd.DataFrame) -> pd.Series:
        """Compute bias toward larger, more in-focus droplets"""
        # Larger droplets are more likely to be visible in crowded systems

        size_bias = df["area"] / df.groupby("frame")["area"].transform("mean")
        focus_bias = df["focus_score"] / df.groupby("frame")["focus_score"].transform(
            "mean"
        )

        visibility_bias = size_bias * focus_bias

        return visibility_bias

    def _compute_frame_crowding_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute frame-level crowding context"""
        frame_stats = []

        for frame_num in df["frame"].unique():
            frame_data = df[df["frame"] == frame_num]

            # Overall packing fraction estimate
            total_droplet_area = frame_data["area"].sum()
            frame_area = self.image_width * self.image_height
            apparent_packing_fraction = total_droplet_area / frame_area

            # Account for 3D nature - visible droplets are subset
            # Use the actual visible droplet fraction from initialization
            estimated_total_packing = (
                apparent_packing_fraction / self.visible_droplet_fraction
            )

            # Flow heterogeneity
            velocity_variance = (
                frame_data["velocity_magnitude"].var()
                if "velocity_magnitude" in frame_data.columns
                else 0
            )

            # Spatial heterogeneity in packing
            packing_gradient = (
                frame_data["local_packing_density"].std()
                if "local_packing_density" in frame_data.columns
                else 0
            )

            frame_stats.append(
                {
                    "frame": frame_num,
                    "frame_packing_fraction": apparent_packing_fraction,
                    "estimated_total_packing": estimated_total_packing,
                    "flow_heterogeneity": velocity_variance,
                    "local_packing_gradient": packing_gradient,
                    "visible_droplet_count": len(frame_data),
                    "estimated_total_droplet_count": len(frame_data)
                    / self.visible_droplet_fraction,  # Use actual fraction
                    "visible_fraction_used": self.visible_droplet_fraction,  # Track what fraction was used
                }
            )

        return pd.DataFrame(frame_stats)

    # =================== CONCENTRATED SYSTEM BREAKUP ANALYSIS ===================

    def detect_crowded_breakup_events(self, df_3d: pd.DataFrame = None) -> List[Dict]:
        """Detect breakup events specific to crowded systems"""
        if df_3d is None:
            df_3d = self.analyze_3d_sampling_effects()

        # Modified breakup detection for crowded systems
        breakup_events = []

        # Look for sudden appearance of multiple small droplets
        # (may indicate breakup event just became visible)

        for frame_num in range(df_3d["frame"].min() + 5, df_3d["frame"].max() - 5):
            current_frame = df_3d[df_3d["frame"] == frame_num]
            prev_frames = df_3d[
                (df_3d["frame"] >= frame_num - 5) & (df_3d["frame"] < frame_num)
            ]

            # Look for clusters of new small droplets
            new_small_droplets = current_frame[
                (current_frame["area"] < current_frame["area"].median())
                & (current_frame["apparent_size_rank"] <= 3)  # Among smallest
            ]

            if len(new_small_droplets) >= 2:
                # Check if these are clustered spatially
                positions = new_small_droplets[["centroid_x", "centroid_y"]].values

                if len(positions) > 1:
                    # Check clustering
                    distances = spatial.distance.pdist(positions)
                    if np.mean(distances) < self.interaction_distance:
                        # Potential breakup event
                        breakup_events.append(
                            {
                                "frame": frame_num,
                                "event_type": "crowded_breakup",
                                "fragment_count": len(new_small_droplets),
                                "fragment_positions": positions.tolist(),
                                "local_packing_density": new_small_droplets[
                                    "local_packing_density"
                                ].mean(),
                                "mean_strain_rate": new_small_droplets[
                                    "extensional_strain_rate"
                                ].mean()
                                if "extensional_strain_rate"
                                in new_small_droplets.columns
                                else 0,
                            }
                        )

        return breakup_events

    # =================== STATISTICAL SAMPLING ANALYSIS ===================

    def analyze_sampling_representativeness(self, df_complete: pd.DataFrame) -> Dict:
        """Analyze how representative the visible droplet sample is"""

        # Size distribution analysis
        size_stats = self._analyze_size_distribution_bias(df_complete)

        # Spatial sampling bias
        spatial_bias = self._analyze_spatial_sampling_bias(df_complete)

        # Temporal consistency
        temporal_consistency = self._analyze_temporal_sampling_consistency(df_complete)

        # Flow region representation
        flow_region_stats = self._analyze_flow_region_representation(df_complete)

        return {
            "size_distribution_bias": size_stats,
            "spatial_sampling_bias": spatial_bias,
            "temporal_consistency": temporal_consistency,
            "flow_region_representation": flow_region_stats,
            "overall_assessment": self._assess_overall_representativeness(
                size_stats, spatial_bias, temporal_consistency, flow_region_stats
            ),
        }

    def _analyze_size_distribution_bias(self, df: pd.DataFrame) -> Dict:
        """Analyze bias in size distribution due to visibility"""

        # Compare size distributions across focus levels
        high_focus = df[df["focus_score"] > df["focus_score"].quantile(0.8)]
        low_focus = df[df["focus_score"] < df["focus_score"].quantile(0.2)]

        size_bias_ratio = high_focus["area"].mean() / (low_focus["area"].mean() + 1e-6)

        return {
            "high_focus_mean_size": high_focus["area"].mean(),
            "low_focus_mean_size": low_focus["area"].mean(),
            "size_bias_ratio": size_bias_ratio,
            "size_distribution_skewness": stats.skew(df["area"]),
        }

    def _analyze_spatial_sampling_bias(self, df: pd.DataFrame) -> Dict:
        """Analyze spatial bias in droplet sampling"""

        # Analyze droplet density across frame regions
        df["x_region"] = pd.cut(
            df["centroid_x"],
            bins=5,
            labels=["left", "center_left", "center", "center_right", "right"],
        )
        df["y_region"] = pd.cut(
            df["centroid_y"],
            bins=5,
            labels=["top", "upper", "middle", "lower", "bottom"],
        )

        # Fix the FutureWarning by explicitly setting observed=False
        density_by_region = df.groupby(["x_region", "y_region"], observed=False).size()
        density_cv = density_by_region.std() / (density_by_region.mean() + 1e-6)

        # Convert tuple keys to string keys for JSON serialization
        region_counts_dict = {}
        for key, value in density_by_region.items():
            # Convert tuple key to string
            if isinstance(key, tuple):
                string_key = f"{key[0]}_{key[1]}"
            else:
                string_key = str(key)
            region_counts_dict[string_key] = int(value)  # Convert to int for JSON

        # Calculate center bias with proper key handling
        center_middle_key = ("center", "middle")
        center_bias = 1.0  # Default value

        if center_middle_key in density_by_region.index:
            center_count = density_by_region[center_middle_key]
            mean_count = density_by_region.mean()
            center_bias = float(center_count / mean_count) if mean_count > 0 else 1.0

        return {
            "spatial_density_cv": float(density_cv),
            "region_counts": region_counts_dict,
            "center_bias": center_bias,
        }

    def _analyze_temporal_sampling_consistency(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal consistency of sampling"""

        frame_counts = df.groupby("frame").size()
        temporal_cv = frame_counts.std() / (frame_counts.mean() + 1e-6)

        # Autocorrelation in droplet counts
        autocorr = np.corrcoef(frame_counts[:-1], frame_counts[1:])[0, 1]

        return {
            "temporal_count_cv": temporal_cv,
            "temporal_autocorrelation": autocorr,
            "mean_droplets_per_frame": frame_counts.mean(),
        }

    def _analyze_flow_region_representation(self, df: pd.DataFrame) -> Dict:
        """Analyze representation across flow regions"""
        if "flow_region" not in df.columns:
            return {"error": "Flow region classification not available"}

        # Get the columns that exist in the dataframe
        agg_dict = {"area": ["count", "mean", "std"]}

        if "deformation_parameter" in df.columns:
            agg_dict["deformation_parameter"] = ["mean", "std"]

        if "local_packing_density" in df.columns:
            agg_dict["local_packing_density"] = ["mean", "std"]

        region_stats = df.groupby("flow_region", observed=False).agg(agg_dict).round(3)

        # Convert to JSON-serializable format
        result_dict = {}
        for level1_key in region_stats.columns.levels[0]:
            result_dict[level1_key] = {}
            for level2_key in region_stats.columns.levels[1]:
                if (level1_key, level2_key) in region_stats.columns:
                    series_data = region_stats[(level1_key, level2_key)]
                    # Convert to regular dict with string keys
                    result_dict[level1_key][level2_key] = {
                        str(k): float(v) if pd.notna(v) else 0.0
                        for k, v in series_data.items()
                    }

        return result_dict

    def _assess_overall_representativeness(
        self,
        size_stats: Dict,
        spatial_bias: Dict,
        temporal_consistency: Dict,
        flow_region_stats: Dict,
    ) -> Dict:
        """Provide overall assessment of sampling representativeness"""

        # Simple scoring system
        scores = {}

        # Size bias score (lower is better)
        size_score = 1 / (1 + abs(size_stats["size_bias_ratio"] - 1))
        scores["size_representativeness"] = size_score

        # Spatial uniformity score
        spatial_score = 1 / (1 + spatial_bias["spatial_density_cv"])
        scores["spatial_representativeness"] = spatial_score

        # Temporal consistency score
        temporal_score = 1 / (1 + temporal_consistency["temporal_count_cv"])
        scores["temporal_representativeness"] = temporal_score

        # Overall score
        overall_score = np.mean(list(scores.values()))

        return {
            "component_scores": scores,
            "overall_representativeness_score": overall_score,
            "assessment": "good"
            if overall_score > 0.7
            else "moderate"
            if overall_score > 0.5
            else "poor",
            "recommendations": self._generate_sampling_recommendations(scores),
        }

    def _generate_sampling_recommendations(self, scores: Dict) -> List[str]:
        """Generate recommendations for improving sampling"""
        recommendations = []

        if scores["size_representativeness"] < 0.6:
            recommendations.append(
                "Consider correcting for size bias in statistical analyses"
            )

        if scores["spatial_representativeness"] < 0.6:
            recommendations.append(
                "Spatial sampling is uneven - consider regional subsampling"
            )

        if scores["temporal_representativeness"] < 0.6:
            recommendations.append(
                "Temporal sampling varies significantly - use weighted statistics"
            )

        if not recommendations:
            recommendations.append(
                "Sampling appears representative for population statistics"
            )

        return recommendations

    def _make_json_serializable(self, obj):
        """
        Convert pandas objects and complex data structures to JSON-serializable format

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {
                str(k) if not isinstance(k, str) else k: self._make_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return {
                str(k) if not isinstance(k, str) else k: self._make_json_serializable(v)
                for k, v in obj.to_dict().items()
            }
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj


# Example usage function for concentrated extensional flows
def example_concentrated_flow_analysis():
    """Example workflow for concentrated extensional flow analysis"""

    # Load your SAM2 results (use the enhanced version with trajectory tracking)
    results_df = pd.read_parquet(
        "/home/leviathan/cdrucker/ml/sam2/outputs/vid64_rolling4/results.parquet"
    )
    results_df["trajectory_id"] = results_df.object_id.values

    # Initialize specialized analyzer with your calibration
    analyzer = ConcentratedExtensionalAnalyzer(
        df=results_df,
        pixel_to_micron=0.41,  # Your calibration
        frame_rate=400,  # Your frame rate
        channel_width_microns=1000,  # Your channel width
        visible_droplet_fraction=0.001,  # 0.1% of droplets are visible (adjustable)
        extensional_axis="vertical",
        fluid_properties={
            "surface_tension": 0.002,  # Lower due to surfactants
            "viscosity_d": 20.71e-3,  # Effective viscosity
            "viscosity_c": 20.71e-3,  # Higher due to crowding
            "density_d": 776,
            "density_c": 1123,
            "volume_fraction": 0.95,  # High packing
        },
    )

    # Run comprehensive analysis
    print("1. Computing extensional flow metrics...")
    df_ext = analyzer.compute_extensional_metrics()

    print("2. Analyzing crowding effects...")
    df_crowd = analyzer.compute_crowding_metrics(df_ext)

    print("3. Analyzing faceting...")
    df_facet = analyzer.analyze_faceting(df_crowd)

    print("4. Analyzing 3D sampling effects...")
    df_3d = analyzer.analyze_3d_sampling_effects(df_facet)

    print("5. Detecting crowded breakup events...")
    breakup_events = analyzer.detect_crowded_breakup_events(df_3d)

    print("6. Analyzing sampling representativeness...")
    sampling_analysis = analyzer.analyze_sampling_representativeness(df_3d)

    # Export results
    df_3d.to_parquet(
        "/home/leviathan/cdrucker/ml/sam2/outputs/vid64_rolling4/concentrated_flow_analysis.parquet"
    )

    print("\nAnalysis complete!")
    print(f"Detected {len(breakup_events)} potential breakup events")
    assessment = sampling_analysis["overall_assessment"]["assessment"]
    print(f"Sampling representativeness: {assessment}")

    return df_3d, breakup_events, sampling_analysis


if __name__ == "__main__":
    df_3d, breakup_events, sampling_analysis = example_concentrated_flow_analysis()
    print(df_3d)
    print(breakup_events)
    print(sampling_analysis)
    print("Done!")
