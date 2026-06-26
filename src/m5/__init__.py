"""M5 Forecasting Accuracy — reproducible Nixtla + LightGBM solution."""

import warnings
from importlib.metadata import PackageNotFoundError, version

# statsforecast 2.x has raw-string escape issues in a few docstrings; the warnings
# fire at import time and are pure noise. Register the filter before m5.backend
# pulls in statsforecast.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"statsforecast.*")

# Expose the backend at the package root for convenience.
from m5.backend import B, Backend, nw  # noqa: E402

try:
    __version__ = version("m5")
except PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.0.0+local"

__all__ = ["B", "Backend", "__version__", "nw"]
