import pandas as pd
import enum
from matplotlib import pyplot as plt

# Map for the weekday number and the name of the weekday
weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

# Plotting vars
PLOT_GMAX = 12
PLOT_GMIN = 3

# Utility function that returns True if a weekday number refers to a weekend
def is_weekend(x):
    return True if x % 7 > 4 else False


# Define devices that are currently implemented
class Devices(enum.Enum):
    abbott = "abbott"  # FreeStyle Libre


class Units(enum.Enum):
    mmolL = "mmol/L"
    mgdL = "mg/dL"
    gL = "g/L" # TODO handle more units


# Will be used to convert glucose to mmol/L (Glucose in mmol/L = units_to_mmolL_factor[input unit] * Glucose in input unit)
units_to_mmolL_factor = {
    Units.mmolL.value: 1,
    Units.mgdL.value: 1/18.0182
    }


def find_nearest(df, pivot, col, n_iter=100):
    """Finds nearest value to a pivot in a dataframe column
    Returns None if no value is found. Returns the column value otherwise.


    df: dataframe to search in
    pivot: timestamp to search for
    col: column of the dataframe to search in
    n_iter: number of iterations before saying there is nothing
    """
    # TODO add prioritise smaller or larger value
    items = list(df.index)
    n = items.copy()
    for i in range(n_iter):
        m = min(n, key=lambda x: abs(x - pivot))
        q = df.loc[m][col]
        if type(q) == pd.Series:
            q = q[0]
        if pd.isna(q):
            n.remove(m)
        else:
            return m


"""Plotting Utils
"""


def init_plot(l=8, w=6, gmin=PLOT_GMIN, gmax=PLOT_GMAX):
    """Initialize plot

    :param l: length, defaults to 8
    :param w: width, defaults to 6
    :return:
    """
    plt.figure(num=None, figsize=(l, w), dpi=120, facecolor="w", edgecolor="k")
    plt.ylim(gmin, gmax)


def end_plot(r=45, legend=True, save_to: str = None, show=True):
    """End plot by rotating xticks, adding legend and showing the plot

    :param r:
    :return:
    """
    plt.xticks(rotation=r)
    if legend:
        plt.legend()
    if save_to:
        plt.savefig(save_to)
    if show:
        plt.show()


def autoplot(
    func, l=8, w=6, r=45, gmin=PLOT_GMIN, gmax=PLOT_GMAX, legend=True, save_to=None
):
    def wrapper(*args, **kwargs):
        init_plot(l, w, gmin, gmax)
        func(*args, **kwargs)
        end_plot(r, legend, save_to)

    return wrapper
