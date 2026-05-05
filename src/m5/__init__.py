"""M5 Forecasting Accuracy — reproducible Nixtla + LightGBM solution."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("m5")
except PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
