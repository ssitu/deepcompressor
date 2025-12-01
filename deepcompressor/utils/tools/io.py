import os
import shutil
from collections import namedtuple

BenchmarkPaths = namedtuple("BenchmarkPaths", ["dirname", "benchmark_type", "dirpath", "dataset_name"])

def handle_symlink(src: str, dst: str, logger=None):
    """Handle symlink creation and account for Windows, existing files, and permission issues."""
    if os.path.exists(dst):
        os.remove(dst)
    
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError) as e:
        print(f"Symlink creation failed: {e}.")
        try:
            os.link(src, dst)
            if logger:
                logger.info(f"Created hard link from {src} to {dst} as a fallback.")
        except OSError as e:
            if logger:
                logger.error(f"Failed to create hard link from {src} to {dst}: {e}.")
            shutil.copy2(src, dst)
            if logger:
                logger.info(f"Copied file from {src} to {dst} as a last resort.")

def get_benchmark_paths(dataset, root: str, benchmark: str, max_dataset_size: int) -> BenchmarkPaths:
    """
    Get the names and paths for benchmark datasets.
    """
    if benchmark.endswith(".yaml") or benchmark.endswith(".yml"):
        dataset_name = os.path.splitext(os.path.basename(benchmark))[0]
        dirname = f"{dataset_name}-{max_dataset_size}"
        benchmark_type = "YAML"
    elif benchmark.endswith(".json"):
        dataset_name = os.path.splitext(os.path.basename(benchmark))[0]
        dirname = f"{dataset_name}-{max_dataset_size}"
        benchmark_type = "JSON"
    else:
        dataset_name = dataset.config_name
        dirname = f"{dataset.config_name}-{max_dataset_size}"
        benchmark_type = benchmark
    dirpath = os.path.join(root, "samples", benchmark_type, dirname)
    return BenchmarkPaths(dirname, benchmark_type, dirpath, dataset_name)
