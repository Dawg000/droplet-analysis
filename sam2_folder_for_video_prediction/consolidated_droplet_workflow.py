#!/home/leviathan/anaconda3/envs/cd_ml/bin/python
"""
Consolidated Droplet Analysis Workflow Script

This script runs the complete pipeline from SAM2 detection through comprehensive
droplet analysis, producing all output files needed for dashboard visualization.

Workflow:
1. SAM2 rolling detection (run_rolling_sam2.py)
2. Blob feature extraction (sam2_blob_analysis_claude.py)
3. Post-processing and trajectory tracking (sam2_blob_analysis_postprocessing_claude.py)
4. Droplet deformation analysis (droplet_analysis_claude.py)
5. Concentrated extensional flow analysis (extensional_droplet_analysis_claude.py)

Usage:
    python consolidated_droplet_workflow.py --config config.yaml
    python consolidated_droplet_workflow.py --video_key vid64_rolling4
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import yaml

# Add the necessary module paths (adjust as needed for your setup)
sys.path.append("/home/leviathan/cdrucker/ml/sam2")

# Import workflow components
from mask_video_generator import write_video_from_masks_perframe
from sam2_memory_reporting import (
    start_record_memory_history,
    stop_record_memory_history,
)
from sam2_video_predictor_init import initialize_sam2_video_predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConsolidatedDropletWorkflow:
    """Main workflow coordinator for droplet analysis pipeline"""

    def __init__(self, config: Dict):
        """Initialize workflow with configuration"""
        self.config = config
        self.results = {}

        # Setup paths
        self.setup_paths()

        # Validate configuration
        self.validate_config()

    def setup_paths(self):
        """Setup and validate all required paths"""
        paths = self.config.get("paths", {})

        # Required input paths
        self.video_dir = Path(paths["video_dir"])
        if self.video_dir.is_file():
            self.video_dir = (
                Path("/home/leviathan/cdrucker/videos/jpegVideoFrames")
                / self.video_dir.stem
            )
        self.pkl_dir = Path(paths["pkl_dir"])
        self.output_dir = Path(paths["output_dir"])

        # SAM2 model paths
        self.sam2_checkpoint = paths.get(
            "sam2_checkpoint",
            "/home/leviathan/cdrucker/ml/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        )
        self.model_cfg = paths.get("model_cfg", "configs/sam2.1/sam2.1_hiera_t.yaml")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup subdirectories for different analysis stages
        self.sam2_output_dir = self.output_dir / "sam2_masks"
        self.blob_analysis_dir = self.output_dir / "blob_analysis"
        self.postprocessing_dir = self.output_dir / "postprocessing"
        self.droplet_analysis_dir = self.output_dir / "droplet_analysis"
        self.flow_analysis_dir = self.output_dir / "flow_analysis"

        for dir_path in [
            self.sam2_output_dir,
            self.blob_analysis_dir,
            # self.postprocessing_dir,
            # self.droplet_analysis_dir,
            # self.flow_analysis_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

    def validate_config(self):
        """Validate configuration parameters"""
        required_paths = ["video_dir", "pkl_dir", "output_dir"]
        paths = self.config.get("paths", {})

        for path_key in required_paths:
            if path_key not in paths:
                raise ValueError(f"Missing required path: {path_key}")

        # Validate input directories exist
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")
        if not self.pkl_dir.exists():
            raise FileNotFoundError(f"PKL directory not found: {self.pkl_dir}")

        logger.info("Configuration validated successfully")

    def run_sam2_detection(self) -> str:
        """
        Run SAM2 rolling detection
        Returns path to output masks directory
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Running SAM2 Rolling Detection")
        logger.info("=" * 60)

        from run_rolling_sam2 import collect_prompts_from_masks, run_rolling_propagation

        try:
            # Start memory monitoring
            start_record_memory_history()

            # Get SAM2 parameters from config
            sam2_params = self.config.get("sam2_parameters", {})

            # Initialize SAM2 predictor
            logger.info("Initializing SAM2 predictor...")
            predictor, inference_state = initialize_sam2_video_predictor(
                video_dir=str(self.video_dir),
                sam2_checkpoint=self.sam2_checkpoint,
                model_cfg=self.model_cfg,
                async_loading_frames=sam2_params.get("async_loading_frames", True),
                offload_state_to_cpu=sam2_params.get("offload_state_to_cpu", True),
                offload_video_to_cpu=sam2_params.get("offload_video_to_cpu", True),
            )

            # Collect prompts from masks
            logger.info("Collecting prompts from masks...")
            prompts_by_frame = collect_prompts_from_masks(
                predictor=predictor,
                inference_state=inference_state,
                pkl_folder=str(self.pkl_dir),
                initial_frame=sam2_params.get("initial_frame", 0),
                max_frame_num_to_track=sam2_params.get("max_frame_num_to_track", 2000),
                n_frames_for_new_masks=sam2_params.get("n_frames_for_new_masks", 50),
                verbose=True,
                min_area=sam2_params.get("min_object_area", 25),
                max_area=sam2_params.get("max_object_area", 10000),
                border_proportion=sam2_params.get("border_proportion", 0.5),
                n_points_per_mask=sam2_params.get("n_points_per_mask", 0),
            )

            # Run rolling propagation
            logger.info("Running rolling propagation...")
            masks_output_dir = self.sam2_output_dir / "per_frame_video_segments"

            video_segments = run_rolling_propagation(
                predictor=predictor,
                inference_state=inference_state,
                prompt_dict=prompts_by_frame,
                chunk_size=sam2_params.get("chunk_size", 50),
                max_prompts=sam2_params.get("max_prompts", 75),
                iou_threshold=sam2_params.get("iou_threshold", 0.8),
                report_memory=sam2_params.get("report_memory", True),
                max_frames_for_memory=sam2_params.get("max_frames_for_memory", 50),
                history_window=sam2_params.get("history_window", 10),
                output_dir=str(masks_output_dir),
            )

            # Cleanup
            predictor.reset_state(inference_state)
            del video_segments
            torch.cuda.empty_cache()
            gc.collect()

            # Export memory snapshot
            # export_memory_snapshot()
            stop_record_memory_history()

            logger.info(f"SAM2 detection completed. Masks saved to: {masks_output_dir}")
            self.results["sam2_masks_dir"] = str(masks_output_dir)

            return str(masks_output_dir)

        except Exception as e:
            logger.error(f"SAM2 detection failed: {e}")
            raise

    def run_blob_analysis(self, masks_dir: str) -> str:
        """
        Run comprehensive blob feature extraction
        Returns path to blob analysis results
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Running Blob Feature Analysis")
        logger.info("=" * 60)

        from sam2_blob_analysis_claude import SAM2BlobAnalyzer, VideoProcessor

        try:
            # Get blob analysis parameters
            blob_params = self.config.get("blob_analysis_parameters", {})

            # Initialize analyzer
            analyzer = SAM2BlobAnalyzer(
                focus_threshold=blob_params.get("focus_threshold", 100.0),
                min_area=blob_params.get("min_area", 50),
                max_area=blob_params.get("max_area", 10000),
                redundancy_iou_threshold=blob_params.get(
                    "redundancy_iou_threshold", 0.7
                ),
                use_gpu=blob_params.get("use_gpu", True),
                gpu_backend=blob_params.get("gpu_backend", "torch"),
                n_workers=blob_params.get("n_workers", 30),
            )

            # Setup processor
            processor = VideoProcessor(analyzer)

            # Define output path
            results_path = self.blob_analysis_dir / "blob_features.parquet"

            # Run analysis
            logger.info("Processing video sequence for blob features...")
            start_time = time.time()

            results_df = processor.process_video_sequence(
                detection_dir=masks_dir,
                output_path=str(results_path),
                image_dir=str(self.video_dir),
                image_pattern="*.jpg",
                chunk_size=blob_params.get("chunk_size", 2000),
                save_intermediate=True,
            )

            processing_time = time.time() - start_time

            # Analyze results quality
            quality_metrics = analyzer.analyze_video_quality(results_df)

            # Save quality metrics
            quality_path = self.blob_analysis_dir / "quality_metrics.json"
            with open(quality_path, "w") as f:
                json.dump(quality_metrics, f, indent=2, default=str)

            logger.info(f"Blob analysis completed in {processing_time:.1f} seconds")
            logger.info(f"Total detections: {quality_metrics['total_detections']}")
            logger.info(
                f"Quality detections: {quality_metrics['quality_detections']} "
                f"({quality_metrics['quality_percentage']:.1f}%)"
            )

            self.results["blob_features_path"] = str(results_path)
            self.results["blob_quality_metrics"] = quality_metrics

            return str(results_path)

        except Exception as e:
            logger.error(f"Blob analysis failed: {e}")
            raise

    def run_post_processing(self, blob_features_path: str) -> str:
        """
        Run post-processing including trajectory tracking and clustering
        Returns path to enhanced results
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Running Post-Processing Analysis")
        logger.info("=" * 60)

        from sam2_blob_analysis_postprocessing_claude import SAM2PostProcessor

        try:
            # Load blob analysis results
            logger.info(f"Loading blob features from: {blob_features_path}")
            blob_df = pd.read_parquet(blob_features_path)

            # Initialize post-processor
            processor = SAM2PostProcessor(blob_df)

            # Get post-processing parameters
            post_params = self.config.get("post_processing_parameters", {})

            # Run trajectory tracking
            logger.info("Tracking blob trajectories...")
            df_tracked = processor.track_blob_trajectories(
                max_distance=post_params.get("max_distance", 50.0),
                max_frame_gap=post_params.get("max_frame_gap", 3),
            )

            # Compute trajectory features
            logger.info("Computing trajectory features...")
            trajectory_stats = processor.compute_trajectory_features(df_tracked)

            # Run clustering analysis
            logger.info("Performing blob clustering...")
            df_clustered = processor.cluster_blobs(
                method=post_params.get("clustering_method", "kmeans"),
                n_clusters=post_params.get("n_clusters", 5),
            )

            # Analyze cluster characteristics
            cluster_analysis = processor.analyze_cluster_characteristics(df_clustered)

            # Detect anomalies
            logger.info("Detecting anomalous blobs...")
            df_anomalies = processor.detect_anomalies(
                contamination=post_params.get("contamination", 0.1)
            )

            # Compute quality scores
            logger.info("Computing quality scores...")
            df_scored = processor.compute_composite_quality_score()

            # Combine all enhancements
            df_enhanced = df_tracked.copy()
            if "cluster_id" in df_clustered.columns:
                df_enhanced["cluster_id"] = df_clustered["cluster_id"]
            if "is_anomaly" in df_anomalies.columns:
                df_enhanced["is_anomaly"] = df_anomalies["is_anomaly"]
                df_enhanced["anomaly_score"] = df_anomalies["anomaly_score"]
            if "quality_score" in df_scored.columns:
                df_enhanced["quality_score"] = df_scored["quality_score"]
                df_enhanced["quality_rank"] = df_scored["quality_rank"]

            # Export comprehensive results
            output_base = str(self.postprocessing_dir / "enhanced_analysis")

            processor.export_analysis_report(
                output_path=output_base,
                include_trajectories=True,
                include_clusters=True,
                include_anomalies=True,
            )

            # Save main enhanced dataset
            enhanced_path = self.postprocessing_dir / "enhanced_blob_analysis.parquet"
            df_enhanced.to_parquet(enhanced_path)

            # Save trajectory statistics
            trajectory_path = self.postprocessing_dir / "trajectory_statistics.parquet"
            trajectory_stats.to_parquet(trajectory_path)

            # Save cluster analysis
            cluster_path = self.postprocessing_dir / "cluster_analysis.json"
            with open(cluster_path, "w") as f:
                json.dump(cluster_analysis, f, indent=2, default=str)

            logger.info(
                f"Post-processing completed. Enhanced data saved to: {enhanced_path}"
            )
            logger.info(f"Found {len(trajectory_stats)} trajectories")
            logger.info(
                f"Identified {len(df_enhanced['cluster_id'].unique())} clusters"
            )

            self.results["enhanced_features_path"] = str(enhanced_path)
            self.results["trajectory_stats_path"] = str(trajectory_path)

            return str(enhanced_path)

        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            raise

    def run_droplet_deformation_analysis(self, enhanced_features_path: str) -> str:
        """
        Run droplet deformation analysis
        Returns path to deformation analysis results
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Running Droplet Deformation Analysis")
        logger.info("=" * 60)

        from droplet_analysis_claude import run_complete_droplet_analysis

        try:
            # Get deformation analysis parameters
            deformation_params = self.config.get("deformation_analysis_parameters", {})

            # Prepare material properties
            material_properties = deformation_params.get(
                "material_properties",
                {
                    "mu_matrix": 20.71e-3,  # Water viscosity (Pa⋅s)
                    "mu_droplet": 20.71e-3,  # Oil viscosity (Pa⋅s)
                    "sigma": 2e-3,  # Surface tension (N/m)
                    "characteristic_length": 1e-4,  # 100 μm droplets
                },
            )

            # Define output base path
            output_base = str(self.droplet_analysis_dir / "deformation_analysis")

            # Run complete deformation analysis
            logger.info("Running complete droplet deformation analysis...")
            results = run_complete_droplet_analysis(
                sam2_results_path=enhanced_features_path,
                output_base=output_base,
                flow_direction=deformation_params.get("flow_direction", 0.0),
                pixel_size=deformation_params.get("pixel_size", 0.41),
                frame_rate=deformation_params.get("frame_rate", 400.0),
                expected_flow_rate=deformation_params.get("expected_flow_rate", 900),
                material_properties=material_properties,
            )

            # Extract key results paths
            deformation_data_path = f"{output_base}_complete_analysis.parquet"
            breakup_events_path = f"{output_base}_breakup_mechanisms.parquet"

            logger.info("Droplet deformation analysis completed")
            logger.info(f"Main results saved to: {deformation_data_path}")
            if os.path.exists(breakup_events_path):
                breakup_df = pd.read_parquet(breakup_events_path)
                logger.info(f"Detected {len(breakup_df)} breakup events")

            self.results["deformation_analysis_path"] = deformation_data_path
            self.results["breakup_events_path"] = breakup_events_path

            return deformation_data_path

        except Exception as e:
            logger.error(f"Droplet deformation analysis failed: {e}")
            raise

    def run_extensional_flow_analysis(self, deformation_data_path: str) -> str:
        """
        Run concentrated extensional flow analysis
        Returns path to final comprehensive results
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Running Extensional Flow Analysis")
        logger.info("=" * 60)

        from extensional_droplet_analysis_claude import ConcentratedExtensionalAnalyzer

        try:
            # Load deformation analysis results
            logger.info(f"Loading deformation analysis from: {deformation_data_path}")
            deformation_df = pd.read_parquet(deformation_data_path)

            # Get extensional flow parameters
            flow_params = self.config.get("extensional_flow_parameters", {})

            # Initialize analyzer
            analyzer = ConcentratedExtensionalAnalyzer(
                df=deformation_df,
                pixel_to_micron=flow_params.get("pixel_to_micron", 0.41),
                frame_rate=flow_params.get("frame_rate", 400.0),
                channel_width_microns=flow_params.get("channel_width_microns", 1000.0),
                visible_droplet_fraction=flow_params.get(
                    "visible_droplet_fraction", 0.01
                ),
                extensional_axis=flow_params.get("extensional_axis", "vertical"),
                fluid_properties=flow_params.get(
                    "fluid_properties",
                    {
                        "surface_tension": 0.002,
                        "viscosity_d": 20.71e-3,
                        "viscosity_c": 20.71e-3,
                        "density_d": 776,
                        "density_c": 1123,
                        "volume_fraction": 0.95,
                    },
                ),
            )

            # Run comprehensive extensional flow analysis
            logger.info("1. Computing extensional flow metrics...")
            df_ext = analyzer.compute_extensional_metrics()

            logger.info("2. Analyzing crowding effects...")
            df_crowd = analyzer.compute_crowding_metrics(df_ext)

            logger.info("3. Analyzing faceting...")
            df_facet = analyzer.analyze_faceting(df_crowd)

            logger.info("4. Analyzing 3D sampling effects...")
            df_3d = analyzer.analyze_3d_sampling_effects(df_facet)

            logger.info("5. Detecting crowded breakup events...")
            breakup_events = analyzer.detect_crowded_breakup_events(df_3d)

            logger.info("6. Analyzing sampling representativeness...")
            sampling_analysis = analyzer.analyze_sampling_representativeness(df_3d)

            # Save comprehensive results
            final_results_path = (
                self.flow_analysis_dir / "concentrated_flow_analysis.parquet"
            )
            df_3d.to_parquet(final_results_path)

            # Save breakup events
            breakup_path = self.flow_analysis_dir / "crowded_breakup_events.json"
            with open(breakup_path, "w") as f:
                json.dump(breakup_events, f, indent=2, default=str)

            # Save sampling analysis
            sampling_path = self.flow_analysis_dir / "sampling_analysis.json"
            with open(sampling_path, "w") as f:
                json.dump(sampling_analysis, f, indent=2, default=str)

            logger.info("Extensional flow analysis completed")
            logger.info(f"Final results saved to: {final_results_path}")
            logger.info(f"Detected {len(breakup_events)} potential breakup events")

            assessment = sampling_analysis["overall_assessment"]["assessment"]
            logger.info(f"Sampling representativeness: {assessment}")

            self.results["final_results_path"] = str(final_results_path)
            self.results["breakup_events"] = breakup_events
            self.results["sampling_analysis"] = sampling_analysis

            return str(final_results_path)

        except Exception as e:
            logger.error(f"Extensional flow analysis failed: {e}")
            raise

    def generate_summary_report(self):
        """Generate comprehensive workflow summary report"""
        logger.info("=" * 60)
        logger.info("GENERATING WORKFLOW SUMMARY REPORT")
        logger.info("=" * 60)

        try:
            # Load final results for statistics
            final_path = self.results.get("final_results_path")
            if final_path and os.path.exists(final_path):
                final_df = pd.read_parquet(final_path)

                # Generate summary statistics
                summary = {
                    "workflow_info": {
                        "config_file": getattr(
                            self, "config_file", "Direct configuration"
                        ),
                        "total_processing_time": getattr(self, "total_time", "Unknown"),
                        "output_directory": str(self.output_dir),
                    },
                    "data_summary": {
                        "total_detections": len(final_df),
                        "unique_frames": final_df["frame"].nunique(),
                        "frame_range": (
                            int(final_df["frame"].min()),
                            int(final_df["frame"].max()),
                        ),
                        "unique_trajectories": final_df["trajectory_id"].nunique()
                        if "trajectory_id" in final_df.columns
                        else "Unknown",
                        "mean_detections_per_frame": len(final_df)
                        / final_df["frame"].nunique(),
                    },
                    "quality_metrics": {
                        "in_focus_percentage": final_df["is_in_focus"].mean() * 100
                        if "is_in_focus" in final_df.columns
                        else "Unknown",
                        "mean_deformation": final_df["deformation_parameter"].mean()
                        if "deformation_parameter" in final_df.columns
                        else "Unknown",
                        "mean_area": final_df["area"].mean(),
                        "breakup_events_detected": len(
                            self.results.get("breakup_events", [])
                        ),
                    },
                    "files_generated": {
                        "sam2_masks_directory": self.results.get("sam2_masks_dir"),
                        "blob_features": self.results.get("blob_features_path"),
                        "enhanced_features": self.results.get("enhanced_features_path"),
                        "deformation_analysis": self.results.get(
                            "deformation_analysis_path"
                        ),
                        "final_results": self.results.get("final_results_path"),
                    },
                    "sampling_assessment": self.results.get(
                        "sampling_analysis", {}
                    ).get("overall_assessment", {}),
                }

                # Save summary report
                summary_path = self.output_dir / "workflow_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)

                # Print key metrics
                logger.info(
                    f"📊 Total detections processed: {summary['data_summary']['total_detections']:,}"
                )
                logger.info(
                    f"🎬 Frames analyzed: {summary['data_summary']['unique_frames']:,}"
                )
                logger.info(
                    f"🔬 Mean area: {summary['quality_metrics']['mean_area']:.1f} px²"
                )

                if summary["quality_metrics"]["breakup_events_detected"] != "Unknown":
                    logger.info(
                        f"💥 Breakup events detected: {summary['quality_metrics']['breakup_events_detected']}"
                    )

                logger.info(f"📁 Summary report saved to: {summary_path}")

            else:
                logger.warning(
                    "Could not generate detailed summary - final results not found"
                )

        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")

    def write_animation(self, masks_dir):
        # mask_dir = Path(os.path.join(self.output_dir, "sam2_masks", "per_frame_video_segments"))
        output_dir = Path(masks_dir).parent
        output_path = output_dir / f"{Path(self.output_dir).stem}_animation_100fps.avi"
        write_video_from_masks_perframe(
            mask_dir=masks_dir,
            output_path=output_path,
            image_folder=self.video_dir,
            fps=100,
            alpha=0.5,
            shape=(800, 1280),
            use_torch=True,
            device="cuda",
            target_obj_ids=None,
        )
        print(f"Saved animation to {output_path}")
        return output_path

    def write_enhanced_animation_from_parquet(self, parquet_file):
        from mask_video_generator_enhanced import (
            MaskDataLoader,
            VisualizationConfig,
            write_video_from_masks,
        )

        frame_dir = self.video_dir
        mask_dict, metadata_df = MaskDataLoader.from_parquet(
            parquet_path=parquet_file,
            metadata_columns=[],
            obj_id_column="object_id",
        )

        config = VisualizationConfig(
            alpha=0.5,
            fps=100,
            show_frame_number=True,
            show_scale_bar=True,
            scale_bar_length_um=100.0,
            pixels_per_um=0.41,
        )

        # Should save to output_dir/output_dir_animation_100fps.mp4
        output_path = (
            Path(self.output_dir) / f"{Path(self.output_dir).stem}_animation_100fps.mp4"
        )
        write_video_from_masks(
            mask_dict=mask_dict,
            output_path=output_path,
            image_folder=frame_dir,
            config=config,
            use_gpu=True,
        )
        return output_path

    def run_complete_workflow(self) -> str:
        """
        Run the complete analysis workflow
        Returns path to final results
        """
        workflow_start_time = time.time()

        try:
            logger.info("🚀 Starting Complete Droplet Analysis Workflow")
            logger.info(f"📁 Output directory: {self.output_dir}")

            # Step 1: SAM2 Detection
            masks_dir = self.run_sam2_detection()

            # Step 2: Blob Analysis
            blob_features_path = self.run_blob_analysis(masks_dir)

            # # Step 3: Post-processing
            # enhanced_features_path = self.run_post_processing(blob_features_path)

            # # Step 4: Droplet Deformation Analysis
            # deformation_data_path = self.run_droplet_deformation_analysis(
            #     enhanced_features_path
            # )

            # # Step 5: Extensional Flow Analysis
            # final_results_path = self.run_extensional_flow_analysis(
            #     deformation_data_path
            # )

            # Step 6: Save an animation
            # animation_path = self.write_animation(masks_dir)
            animation_path = self.write_enhanced_animation_from_parquet(
                blob_features_path
            )
            logger.info(f"Saved animation to {animation_path}")

            # Calculate total processing time
            self.total_time = time.time() - workflow_start_time

            # Generate summary report
            self.generate_summary_report()

            logger.info("=" * 60)
            logger.info("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"⏱️  Total processing time: {self.total_time:.1f} seconds")
            # logger.info(f"📊 Final results: {final_results_path}")
            logger.info(f"Animation saved to {animation_path}")
            logger.info(f"📁 All outputs in: {self.output_dir}")
            logger.info("=" * 60)

            # return final_results_path
            return

        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            raise


def load_config(
    config_path: Optional[str] = None, video_key: Optional[str] = None
) -> Dict:
    """Load configuration from file or create default config"""

    if config_path and os.path.exists(config_path):
        # Load from YAML file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from: {config_path}")
        return config

    elif video_key:
        # Use directory_dict approach (if available)
        try:
            from directory_dict import (
                directory_dict,
                get_mask_and_frame_and_parent_dirs,
            )

            input_dir, image_dir, parent_dir = get_mask_and_frame_and_parent_dirs(
                directory_dict, video_key
            )

            config = create_default_config(
                video_dir=image_dir,
                pkl_dir=input_dir,
                output_dir=os.path.join(parent_dir, "consolidated_analysis"),
            )
            logger.info(f"Configuration created for video key: {video_key}")
            return config

        except ImportError:
            logger.warning("directory_dict not available, using manual configuration")

    # Create default configuration
    logger.info("Using default configuration - please update paths")
    return create_default_config()


def create_default_config(
    video_dir: str = "/path/to/video/frames",
    pkl_dir: str = "/path/to/pkl/masks",
    output_dir: str = "/path/to/output",
) -> Dict:
    """Create default configuration dictionary"""

    return {
        "paths": {
            "video_dir": video_dir,
            "pkl_dir": pkl_dir,
            "output_dir": output_dir,
            "sam2_checkpoint": "/home/leviathan/cdrucker/ml/sam2/checkpoints/sam2.1_hiera_tiny.pt",
            "model_cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
        },
        "sam2_parameters": {
            "initial_frame": 0,
            "max_frame_num_to_track": 2000,
            "n_frames_for_new_masks": 50,
            "min_object_area": 25,
            "max_object_area": 10000,
            "border_proportion": 0.5,
            "chunk_size": 50,
            "max_prompts": 75,
            "iou_threshold": 0.8,
            "max_frames_for_memory": 50,
            "history_window": 10,
            "report_memory": True,
            "async_loading_frames": True,
            "offload_state_to_cpu": True,
            "offload_video_to_cpu": True,
        },
        "blob_analysis_parameters": {
            "focus_threshold": 100.0,
            "min_area": 50,
            "max_area": 10000,
            "redundancy_iou_threshold": 0.7,
            "use_gpu": True,
            "gpu_backend": "torch",
            "n_workers": 30,
            "chunk_size": 2000,
        },
        "post_processing_parameters": {
            "max_distance": 50.0,
            "max_frame_gap": 3,
            "clustering_method": "kmeans",
            "n_clusters": 5,
            "contamination": 0.1,
        },
        "deformation_analysis_parameters": {
            "flow_direction": 0.0,
            "pixel_size": 0.41,
            "frame_rate": 400.0,
            "expected_flow_rate": 900,
            "material_properties": {
                "mu_matrix": 20.71e-3,
                "mu_droplet": 20.71e-3,
                "sigma": 2e-3,
                "characteristic_length": 1e-4,
            },
        },
        "extensional_flow_parameters": {
            "pixel_to_micron": 0.41,
            "frame_rate": 400.0,
            "channel_width_microns": 1000.0,
            "visible_droplet_fraction": 0.01,
            "extensional_axis": "vertical",
            "fluid_properties": {
                "surface_tension": 0.002,
                "viscosity_d": 20.71e-3,
                "viscosity_c": 20.71e-3,
                "density_d": 776,
                "density_c": 1123,
                "volume_fraction": 0.95,
            },
        },
    }


def save_config_template(output_path: str):
    """Save a configuration template file for user customization"""
    config = create_default_config()

    # Add comments to the configuration
    config_with_comments = {
        "_comments": {
            "paths": "Update these paths to match your data location",
            "sam2_parameters": "SAM2 detection parameters - adjust based on your video properties",
            "blob_analysis_parameters": "Blob feature extraction parameters",
            "post_processing_parameters": "Trajectory tracking and clustering parameters",
            "deformation_analysis_parameters": "Droplet deformation analysis parameters",
            "extensional_flow_parameters": "Concentrated extensional flow analysis parameters",
        },
        **config,
    }

    with open(output_path, "w") as f:
        yaml.dump(config_with_comments, f, default_flow_style=False, indent=2)

    logger.info(f"Configuration template saved to: {output_path}")


def main():
    """Main entry point for the consolidated workflow"""
    parser = argparse.ArgumentParser(
        description="Consolidated Droplet Analysis Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using configuration file
    python consolidated_droplet_workflow.py --config my_config.yaml
    
    # Using video key (requires directory_dict.py)
    python consolidated_droplet_workflow.py --video_key vid64_rolling4
    
    # Generate configuration template
    python consolidated_droplet_workflow.py --save_config_template config_template.yaml
    
    # Manual configuration (update paths in script)
    python consolidated_droplet_workflow.py --manual
        """,
    )

    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    parser.add_argument(
        "--video_key",
        type=str,
        help="Video key for directory_dict lookup (e.g., vid64_rolling4)",
    )

    parser.add_argument(
        "--save_config_template",
        type=str,
        help="Save configuration template to specified path and exit",
    )

    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual configuration (update paths in create_default_config)",
    )

    parser.add_argument(
        "--skip_sam2",
        action="store_true",
        help="Skip SAM2 detection (use existing masks directory)",
    )

    parser.add_argument(
        "--start_from",
        type=str,
        choices=[
            "sam2",
            "blob_analysis",
            "post_processing",
            "deformation",
            "flow_analysis",
        ],
        default="sam2",
        help="Start workflow from specific step",
    )

    args = parser.parse_args()

    # Handle configuration template generation
    if args.save_config_template:
        save_config_template(args.save_config_template)
        return

    try:
        # Load configuration
        if args.config:
            config = load_config(config_path=args.config)
        elif args.video_key:
            config = load_config(video_key=args.video_key)
        elif args.manual:
            config = load_config()
        else:
            logger.error(
                "Must specify --config, --video_key, --manual, or --save_config_template"
            )
            parser.print_help()
            return

        # Initialize workflow
        workflow = ConsolidatedDropletWorkflow(config)
        workflow.config_file = args.config or f"video_key:{args.video_key}" or "manual"

        # Handle partial workflow execution
        if args.start_from != "sam2" or args.skip_sam2:
            logger.info(f"Starting workflow from: {args.start_from}")

            if args.start_from == "blob_analysis" or args.skip_sam2:
                # Start from blob analysis - need existing masks
                masks_dir = workflow.results.get("sam2_masks_dir") or input(
                    "Enter path to SAM2 masks directory: "
                )
                blob_features_path = workflow.run_blob_analysis(masks_dir)
                enhanced_features_path = workflow.run_post_processing(
                    blob_features_path
                )
                deformation_data_path = workflow.run_droplet_deformation_analysis(
                    enhanced_features_path
                )
                final_results_path = workflow.run_extensional_flow_analysis(
                    deformation_data_path
                )

            elif args.start_from == "post_processing":
                # Start from post-processing - need existing blob features
                blob_features_path = input("Enter path to blob features parquet file: ")
                enhanced_features_path = workflow.run_post_processing(
                    blob_features_path
                )
                deformation_data_path = workflow.run_droplet_deformation_analysis(
                    enhanced_features_path
                )
                final_results_path = workflow.run_extensional_flow_analysis(
                    deformation_data_path
                )

            elif args.start_from == "deformation":
                # Start from deformation analysis
                enhanced_features_path = input(
                    "Enter path to enhanced features parquet file: "
                )
                deformation_data_path = workflow.run_droplet_deformation_analysis(
                    enhanced_features_path
                )
                final_results_path = workflow.run_extensional_flow_analysis(
                    deformation_data_path
                )

            elif args.start_from == "flow_analysis":
                # Start from flow analysis only
                deformation_data_path = input(
                    "Enter path to deformation analysis parquet file: "
                )
                final_results_path = workflow.run_extensional_flow_analysis(
                    deformation_data_path
                )

            workflow.generate_summary_report()

        else:
            # Run complete workflow
            final_results_path = workflow.run_complete_workflow()

        # Print final instructions for dashboard
        logger.info("=" * 60)
        logger.info("🎯 READY FOR DASHBOARD")
        logger.info("=" * 60)
        logger.info("To launch the interactive dashboard, run:")
        logger.info("")
        logger.info("from droplet_dashboard import launch_dashboard")
        logger.info("launch_dashboard(")
        logger.info(f"    parquet_file='{final_results_path}',")
        logger.info(f"    image_folder='{workflow.video_dir}'")
        logger.info(")")
        logger.info("")
        logger.info("Or use the dashboard script directly:")
        logger.info(
            f"python droplet_dashboard.py --parquet '{final_results_path}' --images '{workflow.video_dir}'"
        )
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
    except Exception as e:
        logger.error(f"Workflow failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


# Additional utility functions for workflow management


def check_workflow_prerequisites():
    """Check if all required dependencies and files are available"""
    logger.info("Checking workflow prerequisites...")

    required_modules = [
        "torch",
        "cv2",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "plotly",
        "dash",
        "pycocotools",
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        logger.error("Install missing modules with: pip install <module_name>")
        return False

    logger.info("✅ All required modules available")
    return True


def estimate_processing_time(video_dir: str, pkl_dir: str) -> Dict[str, float]:
    """Estimate processing time based on data size"""
    try:
        # Count video frames
        video_path = Path(video_dir)
        frame_count = len(list(video_path.glob("*.jpg"))) + len(
            list(video_path.glob("*.png"))
        )

        # Count mask files
        pkl_path = Path(pkl_dir)
        mask_count = len(list(pkl_path.glob("*.pkl.gz"))) + len(
            list(pkl_path.glob("*.pkl"))
        )

        # Rough time estimates (adjust based on your hardware)
        estimates = {
            "sam2_detection": frame_count * 0.5,  # ~0.5 sec per frame
            "blob_analysis": mask_count * 0.1,  # ~0.1 sec per mask file
            "post_processing": frame_count * 0.05,  # ~0.05 sec per frame
            "deformation_analysis": frame_count * 0.02,  # ~0.02 sec per frame
            "flow_analysis": frame_count * 0.01,  # ~0.01 sec per frame
        }

        estimates["total_estimated"] = sum(estimates.values())

        logger.info("Processing time estimates:")
        logger.info(f"  📹 Frames to process: {frame_count}")
        logger.info(f"  🎭 Mask files: {mask_count}")
        logger.info(
            f"  ⏱️  Estimated total time: {estimates['total_estimated']:.1f} seconds"
        )

        return estimates

    except Exception as e:
        logger.warning(f"Could not estimate processing time: {e}")
        return {}


def create_workflow_script(config_path: str, output_script: str):
    """Create a standalone script for this specific workflow"""
    script_content = f'''#!/home/leviathan/anaconda3/envs/cd_ml/bin/python
"""
Auto-generated workflow script for droplet analysis
Generated from configuration: {config_path}
"""

import sys
from pathlib import Path

# Add the consolidated workflow to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from consolidated_droplet_workflow import ConsolidatedDropletWorkflow, load_config

def main():
    config = load_config(config_path="{config_path}")
    workflow = ConsolidatedDropletWorkflow(config)
    final_results_path = workflow.run_complete_workflow()
    
    print("🎉 Workflow completed!")
    print(f"📊 Results: {{final_results_path}}")

if __name__ == "__main__":
    main()
'''

    with open(output_script, "w") as f:
        f.write(script_content)

    # Make executable
    os.chmod(output_script, 0o755)

    logger.info(f"Standalone workflow script created: {output_script}")


if __name__ == "__main__":
    # Check prerequisites before running
    if not check_workflow_prerequisites():
        sys.exit(1)

    main()

# # Example Bash for loop:
# for i in {485..504}; do
#     config="/home/leviathan/cdrucker/ml/sam2/outputs/vid${i}_rolling1/config.yaml"

#     if [ ! -f "$config" ]; then
#         echo "Warning: '$config' not found, skipping." >&2
#         continue
#     fi
#     python3 consolidated_droplet_workflow.py --config "$config"
# done
