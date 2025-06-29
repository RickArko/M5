from matplotlib.ticker import FuncFormatter


def thousands_formatter(x, pos):
    return "{:,.0f}".format(x)


def format_yaxis_thousands(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


def percentage_formatter(x, pos):
    return "{:.0f}%".format(x * 100)  # Assuming x is in decimal form (e.g., 0.25 for 25%)


def format_yaxis_percentage(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
