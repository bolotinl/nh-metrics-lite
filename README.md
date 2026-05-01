# nh-metrics-lite
A lightweight version of the evaluation/metrics module from [NeuralHydrology](https://github.com/neuralhydrology), without the machine learning dependencies of the rest of that library

## Installation

Requires Python 3.12 or higher.

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/bolotinl/nh-metrics-lite.git
```

To add it as a dependency in another project, include it in your `requirements.txt`:

```
nh-metrics-lite @ git+https://github.com/bolotinl/nh-metrics-lite.git@v0.1.0
```

Or in `pyproject.toml`:

```toml
[project]
dependencies = [
    "nh-metrics-lite @ git+https://github.com/bolotinl/nh-metrics-lite.git@v0.1.0",
]
```

## CLI usage

This package includes a command-line tool for batch metric calculation:

```bash
nh-metrics-lite <root_dir> <observed_parquet> [resolution] [--n-cores N] [--site-column COL] [--site-id-regex REGEX] [--output-name NAME]
```

### Positional arguments

- `root_dir`: directory containing many run folders.
- `observed_parquet`: parquet file with stacked observed streamflow rows.
- `resolution` (optional): temporal resolution used by peak-based metrics.
    - Allowed values: `1H` (default) or `1D`.

### Optional arguments

- `--n-cores`: number of worker threads for folder-level parallelization. Default: `1`.
- `--site-column`: column name in observed parquet identifying site id.
- `--site-id-regex`: regex with exactly one capture group to extract site id from run folder names.
- `--output-name`: output filename written to each run folder's `model_outputs` directory. Default: `nh_metrics.parquet`.

## Expected directory layout

The CLI expects this structure under `root_dir`:

```text
root_dir/
    run_folder_1/
        model_outputs/
            sim_0.parquet
            sim_1.parquet
            ...
    run_folder_2/
        model_outputs/
            sim_0.parquet
            sim_1.parquet
            ...
```

Run folder names are used to determine site id.

- Default matching supports names like `<model>_<site_id>_output`, where `<site_id>` is 8-10 digits.
- If your naming differs, pass `--site-id-regex`.

## Input parquet requirements

### Observed parquet

Must include:

- `value_time`
- `value`
- one site id column (for example `usgs_site_code`, `site_no`, `site_id`), or pass `--site-column`.

### Simulation parquet (`sim_<iteration>.parquet`)

Must include:

- `sim_flow`
- `value_time` either as a column or as the parquet index.

## Output

For each run folder, the tool writes:

```text
<root_dir>/<run_folder>/model_outputs/nh_metrics.parquet
```

Output format:

- one column per iteration number (`0`, `1`, `2`, ... in numeric ascending order)
- final column `__index_level_0__` containing metric names

## Included metrics

The CLI computes all metrics currently implemented in `nh_metrics_lite.py`:

- NSE
- MSE
- RMSE
- KGE
- Alpha-NSE
- Beta-KGE
- Beta-NSE
- Pearson-r
- FHV
- FMS
- FLV
- Peak-Timing
- Missed-Peaks
- Peak-MAPE

## Examples

Default usage (hourly resolution):

```bash
nh-metrics-lite /path/to/root_dir /path/to/observed.parquet
```

Daily resolution:

```bash
nh-metrics-lite /path/to/root_dir /path/to/observed.parquet 1D
```

Parallel by folder using 8 cores:

```bash
nh-metrics-lite /path/to/root_dir /path/to/observed.parquet 1H --n-cores 8
```

Explicit site column and custom run-folder site extraction:

```bash
nh-metrics-lite /path/to/root_dir /path/to/observed.parquet 1H \
    --site-column usgs_site_code \
    --site-id-regex '.*_(\d{8,10})_output$'
```

## Runtime logging

The CLI reports progress as folders complete:

```text
Completed X/Y folders
```

At the end, it prints a single generic output location pattern.



## Attribution

Functions in this package are adapted from [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology),
a Python library for training neural networks with a focus on hydrological applications.

> Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology — A Python library
> for Deep Learning research in hydrology. *Journal of Open Source Software*, 7(71), 4050.
> https://doi.org/10.21105/joss.04050

This package and it's documentation were written with the help of GitHub Copilot, reviewed by the author @bolotinl
