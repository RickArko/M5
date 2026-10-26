"""M5 Forecasting Accuracy — reproducible Nixtla + LightGBM solution."""

import warnings
from importlib.metadata import PackageNotFoundError, version

# statsforecast 2.x has raw-string escape issues in a few docstrings; the warnings
# fire at import/compile time and are pure noise. Match on the full file path
# (``…/statsforecast/models.py``) — ``module=r"statsforecast.*"`` does not,
# because SyntaxWarning's "module" field is the absolute path, not the dotted name.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r".*statsforecast.*")

# Expose the backend at the package root for convenience.
from m5.backend import B, Backend, nw  # noqa: E402

try:
    __version__ = version("m5")
except PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.0.0+local"

__all__ = ["B", "Backend", "__version__", "nw"]
