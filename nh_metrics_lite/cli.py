from __future__ import annotations

import argparse
import re
from pathlib import Path
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
        help="Filename to write into each run directory.",
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

    observed = pd.read_parquet(observed_parquet)
    observed["value_time"] = pd.to_datetime(observed["value_time"], utc=False)
    site_column = _resolve_site_column(observed, args.site_column)
    observed_by_site = {
        str(site_id): site_frame.copy()
        for site_id, site_frame in observed.groupby(site_column, dropna=False)
    }

    run_directories = list(_iter_run_directories(root_dir))
    if not run_directories:
        raise RuntimeError(f"No run directories with model_outputs were found under {root_dir}")

    site_id_pattern = re.compile(args.site_id_regex) if args.site_id_regex else None
    for run_directory in run_directories:
        model_output_dir = run_directory / "model_outputs"
        site_id = _resolve_site_id(run_directory.name, observed_by_site.keys(), site_id_pattern)
        observed_site = observed_by_site[site_id]
        metrics_frame = _build_metrics_frame(model_output_dir, observed_site, resolution=resolution)
        output_path = model_output_dir / args.output_name
        metrics_frame.to_parquet(output_path, index=False)
        print(f"Wrote {output_path}")

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


def _build_metrics_frame(model_output_dir: Path, observed_site: pd.DataFrame, resolution: str) -> pd.DataFrame:
    sim_files = []
    for sim_file in model_output_dir.glob("sim_*.parquet"):
        match = SIM_FILE_PATTERN.search(sim_file.name)
        if match is None:
            continue
        sim_files.append((int(match.group(1)), sim_file))

    sim_files.sort(key=lambda pair: pair[0])

    if not sim_files:
        raise RuntimeError(f"No sim_<iteration>.parquet files found in {model_output_dir}")

    observed_site = observed_site[["value_time", "value"]].dropna(subset=["value_time", "value"])
    observed_site = observed_site.sort_values("value_time").drop_duplicates(subset=["value_time"], keep="last")

    metric_values_by_iteration: dict[int, dict[str, float]] = {}
    for iteration, sim_file in sim_files:
        sim_frame = pd.read_parquet(sim_file)

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
        metric_values_by_iteration[iteration] = calculate_metrics(
            obs_da,
            sim_da,
            metrics=METRIC_NAMES,
            resolution=resolution,
            datetime_coord="value_time",
        )

    metrics_frame = pd.DataFrame(metric_values_by_iteration)
    metrics_frame["__index_level_0__"] = metrics_frame.index
    return metrics_frame.reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())