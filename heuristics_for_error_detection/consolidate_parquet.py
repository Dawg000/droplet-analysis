"""
Module for consolidating multiple parquet files with trajectory quality filtering.

This module efficiently merges parquet files while filtering based on
traj_quality_flag values, using chunked processing for memory efficiency.
Includes automatic schema harmonization to handle type mismatches (e.g., float32 vs float64).
"""

import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def harmonize_schemas(schemas: list[pa.Schema]) -> pa.Schema:
    """
    Harmonize multiple schemas by finding compatible types for each field.

    This handles common type mismatches like float32 vs float64, int32 vs int64, etc.

    Parameters
    ----------
    schemas : List[pa.Schema]
        List of PyArrow schemas to harmonize

    Returns
    -------
    pa.Schema
        Unified schema with compatible types

    Raises
    ------
    ValueError
        If schemas have incompatible fields (different names or irreconcilable types)
    """
    if not schemas:
        raise ValueError("No schemas provided")

    if len(schemas) == 1:
        return schemas[0]

    # Get all field names from first schema as reference
    reference_schema = schemas[0]
    field_names = reference_schema.names

    # Check all schemas have the same fields
    for i, schema in enumerate(schemas[1:], 1):
        if schema.names != field_names:
            missing_in_current = set(field_names) - set(schema.names)
            extra_in_current = set(schema.names) - set(field_names)
            msg = f"Schema mismatch in file {i}:\n"
            if missing_in_current:
                msg += f"  Missing columns: {missing_in_current}\n"
            if extra_in_current:
                msg += f"  Extra columns: {extra_in_current}"
            raise ValueError(msg)

    # Build unified schema field by field
    unified_fields = []

    for field_name in field_names:
        # Collect all types for this field
        types = [schema.field(field_name).type for schema in schemas]

        # If all types are identical, use it
        if all(t == types[0] for t in types):
            unified_fields.append(reference_schema.field(field_name))
            continue

        # Handle numeric type promotions
        unified_type = _unify_numeric_types(types, field_name)

        if unified_type is None:
            # Could not unify - report the conflict
            type_strs = [f"  Schema {i}: {t}" for i, t in enumerate(types)]
            raise ValueError(
                f"Cannot unify incompatible types for field '{field_name}':\n"
                + "\n".join(type_strs)
            )

        # Create field with unified type, preserving nullable from reference
        unified_fields.append(
            pa.field(
                field_name,
                unified_type,
                nullable=reference_schema.field(field_name).nullable,
            )
        )

    return pa.schema(unified_fields)


def _unify_numeric_types(
    types: list[pa.DataType], field_name: str
) -> pa.DataType | None:
    """
    Unify numeric types by promoting to the widest/most precise type.

    Promotion rules:
    - int8/16/32/64 -> widest int type
    - uint8/16/32/64 -> widest uint type
    - float/double (float32/float64) -> double (float64)
    - int + float -> float
    - int + double -> double
    """
    # Type categories
    int_types = {pa.int8(), pa.int16(), pa.int32(), pa.int64()}
    uint_types = {pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()}
    float_types = {pa.float32(), pa.float64()}

    types_set = set(types)

    # All integers - promote to widest
    if types_set.issubset(int_types):
        if pa.int64() in types_set:
            return pa.int64()
        elif pa.int32() in types_set:
            return pa.int32()
        elif pa.int16() in types_set:
            return pa.int16()
        else:
            return pa.int8()

    # All unsigned integers - promote to widest
    if types_set.issubset(uint_types):
        if pa.uint64() in types_set:
            return pa.uint64()
        elif pa.uint32() in types_set:
            return pa.uint32()
        elif pa.uint16() in types_set:
            return pa.uint16()
        else:
            return pa.uint8()

    # All floats - promote to float64 (double)
    if types_set.issubset(float_types):
        logger.info(f"Field '{field_name}': Promoting float32/float64 mix to float64")
        return pa.float64()

    # Mix of int and float - promote to appropriate float
    all_numeric = types_set.issubset(int_types | uint_types | float_types)
    if all_numeric:
        if pa.float64() in types_set or any(
            t in types_set for t in int_types | uint_types
        ):
            logger.info(f"Field '{field_name}': Promoting int/float mix to float64")
            return pa.float64()
        else:
            logger.info(f"Field '{field_name}': Promoting to float32")
            return pa.float32()

    # Cannot unify
    return None


def cast_table_to_schema(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    """
    Cast a table to match a target schema.

    Parameters
    ----------
    table : pa.Table
        Table to cast
    target_schema : pa.Schema
        Target schema

    Returns
    -------
    pa.Table
        Table with columns cast to target schema types
    """
    if table.schema == target_schema:
        return table

    # Cast each column to target type
    new_columns = []
    for i, field in enumerate(target_schema):
        col = table.column(i)
        if col.type != field.type:
            # Cast to target type
            new_col = pc.cast(col, field.type)
            new_columns.append(new_col)
        else:
            new_columns.append(col)

    return pa.Table.from_arrays(new_columns, schema=target_schema)


def extract_video_number(file_path: str | Path) -> int | None:
    """
    Extract video number from file path.

    Looks for pattern 'vidXXX' or 'vidXXXX' in the path where X is a digit.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file

    Returns
    -------
    Optional[int]
        Video number if found, None otherwise

    Examples
    --------
    >>> extract_video_number('/path/to/vid403_rolling1/file.parquet')
    403
    >>> extract_video_number('/path/to/vid1234_data/file.parquet')
    1234
    """
    path_str = str(file_path)

    # Pattern: 'vid' followed by 3-4 digits
    match = re.search(r"vid(\d{3,4})", path_str)

    if match:
        return int(match.group(1))

    return None


def create_prefixed_trajectory_id(traj_id: int, video_number: int) -> int:
    """
    Create a prefixed trajectory ID by combining video number and trajectory ID.

    The video number is used as a prefix, effectively multiplying it by a large
    power of 10 to ensure separation between videos.

    Parameters
    ----------
    traj_id : int
        Original trajectory ID
    video_number : int
        Video number to use as prefix

    Returns
    -------
    int
        Prefixed trajectory ID

    Examples
    --------
    >>> create_prefixed_trajectory_id(12345678, 403)
    40312345678  # 403 * 100000000 + 12345678
    >>> create_prefixed_trajectory_id(87654321, 403)
    40387654321  # 403 * 100000000 + 87654321
    """
    # Use 100000000 (1e8) as multiplier to allow for up to 99,999,999 trajectory IDs per video
    # This accommodates typical trajectory IDs that are ~8 digits
    return video_number * 100000000 + traj_id


class QualityLevel:
    """Quality levels in descending order of desirability."""

    EXCELLENT = "excellent"
    GOOD = "good"
    MARGINAL = "marginal"
    POOR = "poor"
    REJECT = "reject"

    # Ordered list for filtering
    ORDER = [EXCELLENT, GOOD, MARGINAL, POOR, REJECT]

    @classmethod
    def get_filter_set(cls, min_quality: str) -> set:
        """
        Get set of quality levels at or above the specified minimum.

        Parameters
        ----------
        min_quality : str
            Minimum quality level to include (e.g., 'good' includes 'excellent' and 'good')

        Returns
        -------
        set
            Set of quality levels to keep

        Raises
        ------
        ValueError
            If min_quality is not a valid quality level
        """
        if min_quality not in cls.ORDER:
            raise ValueError(
                f"Invalid quality level: {min_quality}. Must be one of {cls.ORDER}"
            )

        cutoff_idx = cls.ORDER.index(min_quality)
        return set(cls.ORDER[: cutoff_idx + 1])


def consolidate_parquet_files(
    input_files: list[str | Path],
    output_file: str | Path,
    min_quality: str = "marginal",
    chunk_size: int = 100000,
    compression: str = "snappy",
    quality_column: str = "traj_quality_flag",
    trajectory_id_column: str = "trajectory_id",
    renumber_trajectories: bool = True,
    harmonize_schema: bool = True,
    min_trajectory_duration: int | None = None,
) -> dict:
    """
    Consolidate multiple parquet files into a single file with quality filtering.

    This function reads parquet files in chunks to minimize memory usage,
    filters rows based on trajectory quality, harmonizes schemas across files,
    and writes to a single output file with renumbered trajectory IDs.

    Parameters
    ----------
    input_files : List[Union[str, Path]]
        List of paths to input parquet files
    output_file : Union[str, Path]
        Path for the consolidated output parquet file
    min_quality : str, optional
        Minimum quality level to include. Options (in descending order):
        'excellent', 'good', 'marginal', 'poor', 'reject'
        Default is 'marginal' (keeps excellent, good, and marginal)
    chunk_size : int, optional
        Number of rows to process at a time. Default is 100000
    compression : str, optional
        Compression codec for output file. Default is 'snappy'
    quality_column : str, optional
        Name of the quality flag column. Default is 'traj_quality_flag'
    trajectory_id_column : str, optional
        Name of the trajectory ID column. Default is 'trajectory_id'
    renumber_trajectories : bool, optional
        If True, renumber trajectory IDs using video number as prefix.
        Default is True. Video number is extracted from 'vidXXX' pattern in
        file path and used to create unique IDs: video_num * 100000000 + traj_id.
        For example, trajectory 12345678 from vid403 becomes 40312345678.
        This ensures no overlaps between videos while maintaining traceability.
    harmonize_schema : bool, optional
        If True, automatically harmonize schemas across files to handle type
        mismatches (e.g., float32 vs float64). Default is True.
    min_trajectory_duration : int, optional
        Minimum trajectory duration (number of frames) to keep. Trajectories
        shorter than this will be filtered out. If None, no duration filtering
        is applied. Note: This requires the data to have trajectory duration
        information available (computed as count of rows per trajectory_id).

    Returns
    -------
    dict
        Statistics about the consolidation:
        - total_rows_read: Total rows across all input files
        - rows_kept: Rows meeting quality criteria
        - rows_filtered: Rows filtered out
        - files_processed: Number of input files processed
        - quality_counts: Count of each quality level in kept rows

    Raises
    ------
    FileNotFoundError
        If any input file doesn't exist
    ValueError
        If min_quality is invalid or quality_column not found

    Examples
    --------
    >>> # Keep only excellent and good trajectories
    >>> stats = consolidate_parquet_files(
    ...     input_files=['file1.parquet', 'file2.parquet'],
    ...     output_file='consolidated.parquet',
    ...     min_quality='good'
    ... )

    >>> # Keep excellent, good, and marginal
    >>> stats = consolidate_parquet_files(
    ...     input_files=['file1.parquet', 'file2.parquet'],
    ...     output_file='consolidated.parquet',
    ...     min_quality='marginal'
    ... )
    """
    # Validate inputs
    input_files = [Path(f) for f in input_files]
    output_file = Path(output_file)

    for f in input_files:
        if not f.exists():
            raise FileNotFoundError(f"Input file not found: {f}")

    # Get quality filter set
    quality_filter = QualityLevel.get_filter_set(min_quality)
    logger.info(f"Filtering to keep quality levels: {sorted(quality_filter)}")

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # First pass: collect all schemas if harmonization is enabled
    unified_schema = None
    if harmonize_schema:
        logger.info("Collecting schemas from all files for harmonization...")
        schemas = []
        for input_file in input_files:
            pf = pq.ParquetFile(input_file)
            schemas.append(pf.schema_arrow)

        # Harmonize schemas
        try:
            unified_schema = harmonize_schemas(schemas)
            logger.info("Schema harmonization successful")

            # Check for type promotions
            ref_schema = schemas[0]
            for field_name in unified_schema.names:
                unified_type = unified_schema.field(field_name).type
                ref_type = ref_schema.field(field_name).type
                if unified_type != ref_type:
                    logger.info(f"  {field_name}: {ref_type} -> {unified_type}")
        except ValueError as e:
            logger.error(f"Schema harmonization failed: {e}")
            raise

    # Statistics tracking
    stats = {
        "total_rows_read": 0,
        "rows_kept": 0,
        "rows_filtered": 0,
        "rows_filtered_by_quality": 0,
        "rows_filtered_by_duration": 0,
        "files_processed": 0,
        "quality_counts": {q: 0 for q in QualityLevel.ORDER},
        "video_numbers": [],
        "trajectory_id_ranges": {},
        "trajectories_filtered_by_duration": 0,
        "trajectories_kept": 0,
    }

    # Writer handle (will be initialized on first write)
    writer = None
    schema = None

    try:
        # First pass: if duration filtering is requested, we need to collect trajectory lengths
        # from all files first to know which trajectories to keep
        trajectories_to_keep = None

        if min_trajectory_duration is not None:
            logger.info(
                f"First pass: Computing trajectory durations (min duration: {min_trajectory_duration} frames)..."
            )
            trajectory_lengths = {}

            for input_file in input_files:
                parquet_file = pq.ParquetFile(input_file)

                # Get video number for this file
                video_number = None
                if renumber_trajectories:
                    video_number = extract_video_number(input_file)
                    if video_number is None:
                        raise ValueError(
                            f"Could not extract video number from path: {input_file}\n"
                            f"Expected pattern 'vidXXX' or 'vidXXXX' in path"
                        )

                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    table = pa.Table.from_batches([batch])

                    # Cast to unified schema if harmonizing
                    if (
                        harmonize_schema
                        and unified_schema is not None
                        and table.schema != unified_schema
                    ):
                        table = cast_table_to_schema(table, unified_schema)

                    # Filter by quality first
                    quality_col = table.column(quality_column)
                    mask = pc.is_in(
                        quality_col, value_set=pa.array(list(quality_filter))
                    )
                    filtered_table = table.filter(mask)

                    if len(filtered_table) == 0:
                        continue

                    # Count trajectory lengths
                    traj_ids = filtered_table.column(trajectory_id_column).to_pylist()

                    for traj_id in traj_ids:
                        # Create the key that matches what will be used after renumbering
                        if renumber_trajectories:
                            key = (video_number, traj_id)
                        else:
                            key = traj_id

                        trajectory_lengths[key] = trajectory_lengths.get(key, 0) + 1

            # Filter trajectories by duration
            trajectories_to_keep = {
                key
                for key, length in trajectory_lengths.items()
                if length >= min_trajectory_duration
            }

            total_trajectories = len(trajectory_lengths)
            kept_trajectories = len(trajectories_to_keep)
            filtered_trajectories = total_trajectories - kept_trajectories

            stats["trajectories_filtered_by_duration"] = filtered_trajectories
            stats["trajectories_kept"] = kept_trajectories

            logger.info(
                f"  Total trajectories: {total_trajectories:,}, "
                f"Kept: {kept_trajectories:,} ({100 * kept_trajectories / max(total_trajectories, 1):.1f}%), "
                f"Filtered: {filtered_trajectories:,} ({100 * filtered_trajectories / max(total_trajectories, 1):.1f}%)"
            )

        # Second pass: process and consolidate files
        for file_idx, input_file in enumerate(input_files):
            logger.info(
                f"Processing file {file_idx + 1}/{len(input_files)}: {input_file.name}"
            )

            # Open parquet file for reading
            parquet_file = pq.ParquetFile(input_file)

            # Verify quality column exists
            if schema is None:
                schema = (
                    unified_schema
                    if unified_schema is not None
                    else parquet_file.schema_arrow
                )
                if quality_column not in schema.names:
                    raise ValueError(
                        f"Quality column '{quality_column}' not found in {input_file}. "
                        f"Available columns: {schema.names}"
                    )
                if renumber_trajectories and trajectory_id_column not in schema.names:
                    raise ValueError(
                        f"Trajectory ID column '{trajectory_id_column}' not found in {input_file}. "
                        f"Available columns: {schema.names}"
                    )

            # Extract video number for trajectory ID prefixing
            video_number = None
            if renumber_trajectories:
                video_number = extract_video_number(input_file)
                if video_number is None:
                    raise ValueError(
                        f"Could not extract video number from path: {input_file}\n"
                        f"Expected pattern 'vidXXX' or 'vidXXXX' in path"
                    )
                logger.info(f"  Video number: {video_number}")
                stats["video_numbers"].append(video_number)

            # Track trajectory ID range for this video
            file_min_traj_id = float("inf")
            file_max_traj_id = float("-inf")

            # Process file in chunks
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                table = pa.Table.from_batches([batch])

                # Cast to unified schema if harmonizing
                if harmonize_schema and table.schema != schema:
                    table = cast_table_to_schema(table, schema)

                stats["total_rows_read"] += len(table)

                # Filter based on quality
                quality_col = table.column(quality_column)
                mask = pc.is_in(quality_col, value_set=pa.array(list(quality_filter)))
                filtered_table = table.filter(mask)

                rows_in_chunk = len(filtered_table)
                rows_after_quality = rows_in_chunk
                stats["rows_filtered_by_quality"] += len(table) - rows_after_quality

                # Apply duration filter if requested
                if min_trajectory_duration is not None and rows_in_chunk > 0:
                    traj_ids = filtered_table.column(trajectory_id_column)

                    # Create mask for trajectories that meet duration requirement
                    keep_mask = []
                    for traj_id in traj_ids.to_pylist():
                        if renumber_trajectories:
                            key = (video_number, traj_id)
                        else:
                            key = traj_id
                        keep_mask.append(key in trajectories_to_keep)

                    duration_mask = pa.array(keep_mask)
                    filtered_table = filtered_table.filter(duration_mask)

                    rows_in_chunk = len(filtered_table)
                    stats["rows_filtered_by_duration"] += (
                        rows_after_quality - rows_in_chunk
                    )

                stats["rows_kept"] += rows_in_chunk

                # Count quality levels in this chunk
                if rows_in_chunk > 0:
                    quality_values = filtered_table.column(quality_column).to_pylist()
                    for q in quality_values:
                        if q in stats["quality_counts"]:
                            stats["quality_counts"][q] += 1

                # Renumber trajectory IDs with video prefix if requested
                if rows_in_chunk > 0 and renumber_trajectories:
                    # Get trajectory IDs
                    traj_ids = filtered_table.column(trajectory_id_column)

                    # Track min/max original IDs
                    chunk_min = pc.min(traj_ids).as_py()
                    chunk_max = pc.max(traj_ids).as_py()
                    if chunk_min is not None:
                        file_min_traj_id = min(file_min_traj_id, chunk_min)
                    if chunk_max is not None:
                        file_max_traj_id = max(file_max_traj_id, chunk_max)

                    # Apply video number prefix: video_num * 100000000 + traj_id
                    # This allows up to 99,999,999 trajectory IDs per video (8 digits)
                    prefix_multiplier = pa.scalar(video_number * 100000000)
                    new_traj_ids = pc.add(traj_ids, prefix_multiplier)

                    # Replace trajectory ID column in table
                    traj_id_idx = filtered_table.schema.get_field_index(
                        trajectory_id_column
                    )
                    columns = [
                        filtered_table.column(i) if i != traj_id_idx else new_traj_ids
                        for i in range(filtered_table.num_columns)
                    ]
                    filtered_table = pa.Table.from_arrays(
                        columns, schema=filtered_table.schema
                    )

                # Write filtered data
                if rows_in_chunk > 0:
                    if writer is None:
                        # Initialize writer on first write
                        writer = pq.ParquetWriter(
                            output_file,
                            schema=filtered_table.schema,
                            compression=compression,
                        )

                    writer.write_table(filtered_table)

            # Record trajectory ID range for this video
            if renumber_trajectories and file_min_traj_id != float("inf"):
                original_range = (file_min_traj_id, file_max_traj_id)
                new_min = video_number * 100000000 + file_min_traj_id
                new_max = video_number * 100000000 + file_max_traj_id
                new_range = (new_min, new_max)
                stats["trajectory_id_ranges"][video_number] = {
                    "original": original_range,
                    "new": new_range,
                }
                logger.info(
                    f"  Trajectory IDs: {file_min_traj_id}-{file_max_traj_id} → "
                    f"{new_min}-{new_max}"
                )

            stats["files_processed"] += 1

        # Close writer
        if writer is not None:
            writer.close()
            logger.info(f"Successfully wrote consolidated file: {output_file}")
        else:
            logger.warning("No data met quality criteria. Output file not created.")

        # Calculate filtered rows
        stats["rows_filtered"] = stats["total_rows_read"] - stats["rows_kept"]

        # Log summary
        logger.info("Consolidation complete:")
        logger.info(f"  Total rows read: {stats['total_rows_read']:,}")
        logger.info(
            f"  Rows kept: {stats['rows_kept']:,} ({100 * stats['rows_kept'] / max(stats['total_rows_read'], 1):.1f}%)"
        )
        logger.info(f"  Rows filtered: {stats['rows_filtered']:,}")

        if min_trajectory_duration is not None:
            logger.info(
                f"    Filtered by quality: {stats['rows_filtered_by_quality']:,}"
            )
            logger.info(
                f"    Filtered by duration (<{min_trajectory_duration} frames): {stats['rows_filtered_by_duration']:,}"
            )
            logger.info(f"  Trajectories kept: {stats['trajectories_kept']:,}")
            logger.info(
                f"  Trajectories filtered: {stats['trajectories_filtered_by_duration']:,}"
            )

        logger.info(f"  Files processed: {stats['files_processed']}")

        if renumber_trajectories and stats["video_numbers"]:
            logger.info(f"  Video numbers processed: {sorted(stats['video_numbers'])}")
            logger.info("  Trajectory ID format: vidXXX * 100000000 + original_traj_id")

        logger.info("  Quality distribution:")
        for quality in QualityLevel.ORDER:
            count = stats["quality_counts"][quality]
            if count > 0:
                pct = 100 * count / stats["rows_kept"]
                logger.info(f"    {quality}: {count:,} ({pct:.1f}%)")

        return stats

    except Exception:
        # Clean up partial output on error
        if output_file.exists():
            output_file.unlink()
        raise


def consolidate_from_directory(
    input_dir: str | Path,
    output_file: str | Path,
    min_quality: str = "marginal",
    pattern: str = "*.parquet",
    **kwargs,
) -> dict:
    """
    Convenience function to consolidate all parquet files in a directory.

    Parameters
    ----------
    input_dir : Union[str, Path]
        Directory containing parquet files
    output_file : Union[str, Path]
        Path for consolidated output file
    min_quality : str, optional
        Minimum quality level to include
    pattern : str, optional
        Glob pattern for matching files. Default is '*.parquet'
    **kwargs
        Additional arguments passed to consolidate_parquet_files

    Returns
    -------
    dict
        Consolidation statistics

    Examples
    --------
    >>> stats = consolidate_from_directory(
    ...     input_dir='./trajectory_data',
    ...     output_file='./all_trajectories.parquet',
    ...     min_quality='good'
    ... )
    """
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    # Find all matching parquet files
    input_files = sorted(input_dir.glob(pattern))

    if not input_files:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in {input_dir}"
        )

    logger.info(f"Found {len(input_files)} files matching pattern '{pattern}'")

    return consolidate_parquet_files(
        input_files=input_files,
        output_file=output_file,
        min_quality=min_quality,
        **kwargs,
    )


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="Consolidate parquet files with quality filtering"
    # )
    # parser.add_argument(
    #     "input_files",
    #     nargs="+",
    #     help="Input parquet files to consolidate"
    # )
    # parser.add_argument(
    #     "-o", "--output",
    #     required=True,
    #     help="Output parquet file path"
    # )
    # parser.add_argument(
    #     "-q", "--min-quality",
    #     default="marginal",
    #     choices=QualityLevel.ORDER,
    #     help="Minimum quality level to keep (default: marginal)"
    # )
    # parser.add_argument(
    #     "-c", "--chunk-size",
    #     type=int,
    #     default=100000,
    #     help="Rows per chunk for processing (default: 100000)"
    # )
    # parser.add_argument(
    #     "--compression",
    #     default="snappy",
    #     choices=["snappy", "gzip", "brotli", "lz4", "zstd"],
    #     help="Compression codec (default: snappy)"
    # )
    # parser.add_argument(
    #     "--no-renumber",
    #     action="store_true",
    #     help="Disable trajectory ID renumbering (keep original IDs)"
    # )
    # parser.add_argument(
    #     "--no-harmonize",
    #     action="store_true",
    #     help="Disable schema harmonization (fail on type mismatches)"
    # )
    # parser.add_argument(
    #     "-d", "--min-duration",
    #     type=int,
    #     default=None,
    #     help="Minimum trajectory duration in frames (default: no filtering)"
    # )

    # args = parser.parse_args()

    # consolidate_parquet_files(
    #     input_files=args.input_files,
    #     output_file=args.output,
    #     min_quality=args.min_quality,
    #     chunk_size=args.chunk_size,
    #     compression=args.compression,
    #     renumber_trajectories=not args.no_renumber,
    #     harmonize_schema=not args.no_harmonize,
    #     min_trajectory_duration=args.min_duration
    # )

    # input_files = [
    #     f"/home/leviathan/cdrucker/ml/sam2/outputs/vid{i}_rolling1/filtering/final_filtered_droplets.parquet"
    #     for i in range(381, 404)
    # ]
    # output_file = f"/home/leviathan/cdrucker/ml/sam2/outputs/consolidated_381_403/consolidated_{min_quality}.parquet"

    input_files = [
        f"/home/leviathan/cdrucker/ml/sam2/outputs/vid{i}_rolling2/filtering/final_filtered_droplets.parquet"
        for i in range(364, 381)
    ]
    min_quality = "good"
    output_file = f"/home/leviathan/cdrucker/ml/sam2/outputs/consolidated_363_380_2_lambda5_0p8mlmin_800fps/consolidated_{min_quality}_2.parquet"

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 100000
    compression = "snappy"
    min_trajectory_duration = 200

    consolidate_parquet_files(
        input_files=input_files,
        output_file=output_file,
        min_quality=min_quality,
        chunk_size=chunk_size,
        compression=compression,
        min_trajectory_duration=min_trajectory_duration,
    )
