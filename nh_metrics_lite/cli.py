from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from pathlib import Path
import threading
from typing import Iterable

import pandas as pd
import xarray as xr

from .nh_metrics_lite import calculate_metrics


METRIC_NAMES = [
    "nse",
    "mse",
    "rmse",
    "kge",
    "alpha-nse",
    "beta-kge",
    "beta-nse",
    "pearson-r",
    "fhv",
    "fms",
    "flv",
    "peak-timing",
    "missed-peaks",
    "peak-mape",
]

SITE_COLUMN_CANDIDATES = [
    "usgs_site_code",
    "site_no",
    "site_id",
    "gage_id",
    "station_id",
    "location_id",
    "feature_id",
    "comid",
    "site",
    "id",
]

SIM_FILE_PATTERN = re.compile(r"sim_(\d+)\.parquet$")
RUN_DIR_SITE_PATTERN = re.compile(r"_(\d{8,10})_output$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute NeuralHydrology-style metrics for every sim_<iteration>.parquet file "
            "under a batch of model output directories."
        )
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Directory containing one subdirectory per site/model combination.",
    )
    parser.add_argument(
        "observed_parquet",
        type=Path,
        help="Parquet file containing stacked observed flow time series.",
    )
    parser.add_argument(
        "resolution",
        nargs="?",
        default="1H",
        help="Time resolution for peak metrics. Allowed values: 1H (default) or 1D.",
    )
    parser.add_argument(
        "--site-column",
        help="Column in the observed parquet identifying the site for each row.",
    )
    parser.add_argument(
        "--site-id-regex",
        help=(
            "Optional regex with one capture group used to extract the site id from each run directory name. "
            "If omitted, the script matches observed site ids as substrings of the directory name."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="nh_metrics.parquet",
        help="Filename to write into each run subdirectory under --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Root directory where output files are written. "
            "A subdirectory named after each run folder will be created there, "
            "containing the output metrics file."
        ),
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=1,
        help="Number of CPU cores (worker threads) to use per run folder. Default is 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    observed_parquet = args.observed_parquet.expanduser().resolve()

    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    if observed_parquet.suffix != ".parquet" or not observed_parquet.is_file():
        raise FileNotFoundError(f"Observed parquet does not exist: {observed_parquet}")

    resolution = _resolve_resolution(args.resolution)
    n_cores = _resolve_n_cores(args.n_cores)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    observed = pd.read_parquet(observed_parquet)
    observed["value_time"] = pd.to_datetime(observed["value_time"], utc=False)
    site_column = _resolve_site_column(observed, args.site_column)
    observed_by_site = _prepare_observed_by_site(observed, site_column)

    run_directories = list(_iter_run_directories(root_dir))
    if not run_directories:
        raise RuntimeError(f"No run directories with model_outputs were found under {root_dir}")
    total_folders = len(run_directories)
    completed_folders = 0
    failed_log_path = output_dir / "failed_directories.txt"
    failed_log_lock = threading.Lock()

    site_id_pattern = re.compile(args.site_id_regex) if args.site_id_regex else None
    if n_cores == 1 or len(run_directories) == 1:
        for run_directory in run_directories:
            try:
                _, was_skipped = _process_run_directory(
                    run_directory=run_directory,
                    observed_by_site=observed_by_site,
                    site_id_pattern=site_id_pattern,
                    resolution=resolution,
                    output_dir=output_dir,
                    output_name=args.output_name,
                    failed_log_path=failed_log_path,
                    failed_log_lock=failed_log_lock,
                )
            except Exception as err:
                _append_problem_entry(
                    failed_log_path,
                    failed_log_lock,
                    f"DIRECTORY\t{run_directory.name}\t{type(err).__name__}: {err}",
                )
                print(f"Failed {run_directory.name}: {err}")
                completed_folders += 1
                print(f"Completed {run_directory.name} ({completed_folders}/{total_folders} folders)")
                continue
            if was_skipped:
                print(f"Skipped {run_directory.name} because metrics were already calculated")
            completed_folders += 1
            print(f"Completed {run_directory.name} ({completed_folders}/{total_folders} folders)")
    else:
        max_workers = min(n_cores, len(run_directories))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_run_directory,
                    run_directory=run_directory,
                    observed_by_site=observed_by_site,
                    site_id_pattern=site_id_pattern,
                    resolution=resolution,
                    output_dir=output_dir,
                    output_name=args.output_name,
                    failed_log_path=failed_log_path,
                    failed_log_lock=failed_log_lock,
                ): run_directory
                for run_directory in run_directories
            }
            for future in as_completed(futures):
                run_directory = futures[future]
                try:
                    _, was_skipped = future.result()
                except Exception as err:
                    _append_problem_entry(
                        failed_log_path,
                        failed_log_lock,
                        f"DIRECTORY\t{run_directory.name}\t{type(err).__name__}: {err}",
                    )
                    print(f"Failed {run_directory.name}: {err}")
                    completed_folders += 1
                    print(f"Completed {run_directory.name} ({completed_folders}/{total_folders} folders)")
                    continue
                if was_skipped:
                    print(f"Skipped {run_directory.name} because metrics were already calculated")
                completed_folders += 1
                print(f"Completed {run_directory.name} ({completed_folders}/{total_folders} folders)")

    print(
        f"Finished. Metrics files are written under: "
        f"{output_dir}/<run_folder>/{args.output_name}"
    )

    return 0


def _iter_run_directories(root_dir: Path) -> Iterable[Path]:
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and (child / "model_outputs").is_dir():
            yield child


def _resolve_site_column(observed: pd.DataFrame, requested_column: str | None) -> str:
    required_columns = {"value_time", "value"}
    missing_columns = required_columns - set(observed.columns)
    if missing_columns:
        raise ValueError(f"Observed parquet is missing required columns: {sorted(missing_columns)}")

    if requested_column:
        if requested_column not in observed.columns:
            raise ValueError(f"Observed parquet does not contain site column {requested_column!r}")
        return requested_column

    present_candidates = [column for column in SITE_COLUMN_CANDIDATES if column in observed.columns]
    if len(present_candidates) == 1:
        return present_candidates[0]

    remaining_columns = [column for column in observed.columns if column not in required_columns]
    if len(remaining_columns) == 1:
        return remaining_columns[0]

    raise ValueError(
        "Could not infer the observed site column. Pass --site-column explicitly. "
        f"Available columns: {list(observed.columns)}"
    )


def _resolve_site_id(run_directory_name: str, observed_site_ids: Iterable[str], site_id_pattern: re.Pattern[str] | None) -> str:
    observed_site_ids = list(observed_site_ids)

    if site_id_pattern is not None:
        match = site_id_pattern.search(run_directory_name)
        if match is None or match.lastindex != 1:
            raise ValueError(
                "--site-id-regex must match each run directory name and contain exactly one capture group. "
                f"Directory: {run_directory_name}"
            )
        site_id = match.group(1)
        if site_id not in observed_site_ids:
            raise ValueError(f"Extracted site id {site_id!r} from {run_directory_name} was not found in observed data")
        return site_id

    # Common ngen run-directory format: <model>_<usgs_site_code>_output
    default_match = RUN_DIR_SITE_PATTERN.search(run_directory_name)
    if default_match is not None:
        site_id = default_match.group(1)
        if site_id in observed_site_ids:
            return site_id

    matches = [site_id for site_id in observed_site_ids if site_id and site_id in run_directory_name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "Could not match a site id from the observed data to run directory "
            f"{run_directory_name!r}. Pass --site-id-regex if the id is not a direct substring."
        )
    raise ValueError(
        f"Matched multiple observed site ids {matches} for run directory {run_directory_name!r}. "
        "Pass --site-id-regex to disambiguate."
    )


def _resolve_resolution(resolution: str) -> str:
    canonical = resolution.strip().upper()
    if canonical in {"H", "1H"}:
        return "1h"
    if canonical in {"D", "1D"}:
        return "1D"
    raise ValueError(f"Unsupported resolution {resolution!r}. Allowed values are 1H and 1D.")


def _resolve_n_cores(n_cores: int) -> int:
    if n_cores < 1:
        raise ValueError("--n-cores must be >= 1")
    max_cores = os.cpu_count() or 1
    return min(n_cores, max_cores)


def _prepare_observed_by_site(observed: pd.DataFrame, site_column: str) -> dict[str, pd.DataFrame]:
    observed_by_site: dict[str, pd.DataFrame] = {}
    for site_id, site_frame in observed.groupby(site_column, dropna=False):
        cleaned = site_frame[["value_time", "value"]].dropna(subset=["value_time", "value"])
        cleaned = cleaned.sort_values("value_time").drop_duplicates(subset=["value_time"], keep="last")
        observed_by_site[str(site_id)] = cleaned
    return observed_by_site


def _process_run_directory(
    run_directory: Path,
    observed_by_site: dict[str, pd.DataFrame],
    site_id_pattern: re.Pattern[str] | None,
    resolution: str,
    output_dir: Path,
    output_name: str,
    failed_log_path: Path,
    failed_log_lock: threading.Lock,
) -> tuple[Path, bool]:
    output_path = output_dir / run_directory.name / output_name
    if output_path.exists():
        return output_path, True

    model_output_dir = run_directory / "model_outputs"
    site_id = _resolve_site_id(run_directory.name, observed_by_site.keys(), site_id_pattern)
    observed_site = observed_by_site[site_id]
    metrics_frame = _build_metrics_frame(
        model_output_dir,
        observed_site,
        resolution=resolution,
        run_directory_name=run_directory.name,
        failed_log_path=failed_log_path,
        failed_log_lock=failed_log_lock,
    )
    run_output_dir = output_dir / run_directory.name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_parquet(output_path, index=False)
    return output_path, False


def _build_metrics_frame(
    model_output_dir: Path,
    observed_site: pd.DataFrame,
    resolution: str,
    run_directory_name: str,
    failed_log_path: Path,
    failed_log_lock: threading.Lock,
) -> pd.DataFrame:
    sim_files = []
    for sim_file in model_output_dir.glob("sim_*.parquet"):
        match = SIM_FILE_PATTERN.search(sim_file.name)
        if match is None:
            continue
        sim_files.append((int(match.group(1)), sim_file))

    sim_files.sort(key=lambda pair: pair[0])

    if not sim_files:
        raise RuntimeError(f"No sim_<iteration>.parquet files found in {model_output_dir}")

    metric_values_by_iteration: dict[int, dict[str, float]] = {}
    for iteration, sim_file in sim_files:
        metric_values = _calculate_iteration_metrics(
            sim_file=sim_file,
            observed_site=observed_site,
            resolution=resolution,
            run_directory_name=run_directory_name,
            failed_log_path=failed_log_path,
            failed_log_lock=failed_log_lock,
        )
        if metric_values is None:
            continue
        metric_values_by_iteration[iteration] = metric_values

    if not metric_values_by_iteration:
        raise RuntimeError(f"No usable sim_<iteration>.parquet files were found in {model_output_dir}")

    ordered_metric_values = {
        iteration: metric_values_by_iteration[iteration]
        for iteration, _ in sim_files
        if iteration in metric_values_by_iteration
    }

    metrics_frame = pd.DataFrame(ordered_metric_values)
    metrics_frame["__index_level_0__"] = metrics_frame.index
    return metrics_frame.reset_index(drop=True)


def _calculate_iteration_metrics(
    sim_file: Path,
    observed_site: pd.DataFrame,
    resolution: str,
    run_directory_name: str,
    failed_log_path: Path,
    failed_log_lock: threading.Lock,
) -> dict[str, float] | None:
    if not sim_file.exists() or sim_file.stat().st_size == 0:
        print(f"WARNING: File {sim_file} is empty or missing. Skipping this iteration.")
        _append_problem_entry(
            failed_log_path,
            failed_log_lock,
            f"ITERATION\t{run_directory_name}\t{sim_file.name}\tFile is empty or missing",
        )
        return None

    sim_frame = pd.read_parquet(sim_file)

    if sim_frame.empty:
        print(f"WARNING: File {sim_file} has no rows. Skipping this iteration.")
        _append_problem_entry(
            failed_log_path,
            failed_log_lock,
            f"ITERATION\t{run_directory_name}\t{sim_file.name}\tFile has no rows",
        )
        return None

    if "value_time" not in sim_frame.columns:
        if sim_frame.index.name == "value_time" or isinstance(sim_frame.index, pd.DatetimeIndex):
            sim_frame = sim_frame.reset_index()
            first_column = sim_frame.columns[0]
            if first_column != "value_time":
                sim_frame = sim_frame.rename(columns={first_column: "value_time"})

    required_sim_columns = {"value_time", "sim_flow"}
    missing_sim_columns = required_sim_columns - set(sim_frame.columns)
    if missing_sim_columns:
        raise ValueError(
            f"Simulation file {sim_file} is missing required columns: {sorted(missing_sim_columns)}. "
            f"Available columns: {list(sim_frame.columns)}"
        )

    sim_frame["value_time"] = pd.to_datetime(sim_frame["value_time"], utc=False)
    sim_frame = sim_frame[["value_time", "sim_flow"]].dropna(subset=["value_time", "sim_flow"])
    sim_frame = sim_frame.sort_values("value_time").drop_duplicates(subset=["value_time"], keep="last")

    merged = observed_site.merge(sim_frame, on="value_time", how="inner")
    if merged.empty:
        raise RuntimeError(f"No overlapping timestamps between observed data and {sim_file}")

    obs_da = xr.DataArray(
        merged["value"].to_numpy(),
        coords={"value_time": merged["value_time"].to_numpy()},
        dims=["value_time"],
    )
    sim_da = xr.DataArray(
        merged["sim_flow"].to_numpy(),
        coords={"value_time": merged["value_time"].to_numpy()},
        dims=["value_time"],
    )
    return calculate_metrics(
        obs_da,
        sim_da,
        metrics=METRIC_NAMES,
        resolution=resolution,
        datetime_coord="value_time",
    )


def _append_problem_entry(log_path: Path, lock: threading.Lock, entry: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{entry}\n")


if __name__ == "__main__":
    raise SystemExit(main())