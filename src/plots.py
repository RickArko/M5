import functools
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.ticker import FuncFormatter
from utilsforecast.plotting import plot_series

plt.style.use('ggplot')

# Set Plot Parameters
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.titlesize': 18})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


def thousands_formatter(x, pos):
    return "{:,.0f}".format(x)


def format_yaxis_thousands(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


def percentage_formatter(x, pos):
    return "{:.0f}%".format(x * 100)  # Assuming x is in decimal form (e.g., 0.25 for 25%)


def format_yaxis_percentage(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))


def save_figure(filepath):
    """Decorator to save the figure to a specified filepath."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = func(*args, **kwargs)
            fig.savefig(filepath, bbox_inches='tight')
            logger.info(f"Saving Output Figure at: {filepath}")
            return fig
        return wrapper
    return decorator