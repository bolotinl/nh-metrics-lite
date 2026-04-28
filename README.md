# nh_metrics_light
A lightweight version of the evaluation/metrics module from [NeuralHydrology](https://github.com/neuralhydrology), without the machine learning dependencies of the rest of that library

## Installation

Requires Python 3.12 or higher.

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/bolotinl/nh_metrics_light.git
```

To add it as a dependency in another project, include it in your `requirements.txt`:

```
nh_metrics_light @ git+https://github.com/bolotinl/nh_metrics_light.git@v0.1.0
```

Or in `pyproject.toml`:

```toml
[project]
dependencies = [
    "nh_metrics_light @ git+https://github.com/bolotinl/nh_metrics_light.git@v0.1.0",
]
```



## Attribution

Functions in this package are adapted from [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology),
a Python library for training neural networks with a focus on hydrological applications.

> Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology — A Python library
> for Deep Learning research in hydrology. *Journal of Open Source Software*, 7(71), 4050.
> https://doi.org/10.21105/joss.04050
