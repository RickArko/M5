"""M5 Forecasting Accuracy — reproducible Nixtla + LightGBM solution."""

import warnings
from importlib.metadata import PackageNotFoundError, version

# statsforecast 2.x has raw-string escape issues in a few docstrings; the warnings
# fire at import time and are pure noise. Filter once, here, before any submodule
# triggers a statsforecast import.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"statsforecast.*")

try:
    __version__ = version("m5")
except PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
