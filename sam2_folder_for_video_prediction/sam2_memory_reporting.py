import logging
from datetime import datetime
from time import perf_counter

import psutil
import torch

# https://pytorch.org/blog/understanding-gpu-memory-1/
logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    # host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    # file_prefix = f"{host_name}_{timestamp}"
    file_prefix = f"/home/leviathan/cdrucker/temp/gpuUsage_{timestamp}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


def report_memory_usage(tprev):
    if (perf_counter() > tprev + 1.0) and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        maximum_memory_allocated = torch.cuda.max_memory_allocated()
        maximum_memory_reserved = torch.cuda.max_memory_reserved()
        # CPU
        process = psutil.Process()
        ram_usage = process.memory_info().rss
        # print("VRAM:", (total_bytes - free_bytes) // 1_000_000, "MB")
        print(
            f"Total VRAM Used: {(total_bytes - free_bytes) / (1024**3):.2f} GB, Allocated: {allocated_memory / (1024**3):.2f} GB, Reserved: {reserved_memory / (1024**3):.2f}, RAM Used: {ram_usage / (1024**3):.2f} GB"
        )
        print(
            f"Max VRAM Allocated: {maximum_memory_allocated / (1024**3):.2f} GB, Reserved: {maximum_memory_reserved / (1024**3):.2f} GB"
        )
        tprev = perf_counter()
    return tprev
