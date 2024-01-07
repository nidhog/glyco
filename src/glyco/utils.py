"""Defines general utility functions and variables
"""
import pandas as pd
import enum
from matplotlib import pyplot as plt
from typing import Callable

# Map for the weekday number and the name of the weekday
weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

# Plotting vars
PLOT_GMAX = 12
PLOT_GMIN = 3

# Utility function that returns True if a weekday number refers to a weekend
def is_weekend(day_number: int):
    """Map to check if the day is a weekend based on the day number.

    Args:
        day_number (int): Number of the day to check.

    Returns:
        bool: True if weekend, False if not a weekend day
    """
    return True if day_number % 7 > 4 else False


# Define devices that are currently implemented
class Devices(enum.Enum):
    """Enum for supported devices.
    Other devices are also supported but may need more manual changes.
    """
    abbott = "abbott"  # FreeStyle Libre


class Units(enum.Enum):
    """Enum for supported glucose units
    Glyco mainly uses mmol/L but performs conversion if the unit is different.
    """
    mmolL = "mmol/L"
    mgdL = "mg/dL"
    gL = "g/L"


# Will be used to convert glucose to mmol/L (Glucose in mmol/L = units_to_mmolL_factor[input unit] * Glucose in input unit)
units_to_mmolL_factor = {
    Units.mmolL.value: 1,
    Units.mgdL.value: 1/18.0182,
    Units.gL.value: 100/18.0182
    }

def find_nearest(df: pd.DataFrame, pivot: pd.Timestamp, col: str, n_iter: int = 100):
    """Finds nearest value to a pivot in a dataframe column
    Returns None if no value is found. Returns the column value otherwise.

    Args:
        df (pd.DataFrame): dataframe to search in
        pivot (pd.Timestamp): timestamp to search for
        col (str): column of the dataframe to search in
        n_iter (int, optional): number of iterations before saying there is nothing. Defaults to 100.
    """
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

    Args:
        l (int, optional): lenght of the plot. Defaults to 8.
        w (int, optional): widht of the plot. Defaults to 6.
        gmin (_type_, optional): the minimum Y-axis value to show. Defaults to PLOT_GMIN.
        gmax (_type_, optional): the maximum Y-axis value to show. Defaults to PLOT_GMAX.
    """
    plt.figure(num=None, figsize=(l, w), dpi=120, facecolor="w", edgecolor="k")
    plt.ylim(gmin, gmax)


def end_plot(r=45, legend=True, save_to: str = None, show=True):
    """End plot by rotating xticks, adding legend and showing the plot.

    Args:
        r (int, optional): rotation angle of the xticks. Defaults to 45.
        legend (bool, optional): whether or not to add legend. Defaults to True.
        save_to (str, optional): file path to save plot to. If None, not saved. Defaults to None.
        show (bool, optional): whether or not to show the plot. Defaults to True.
    """
    plt.xticks(rotation=r)
    if legend:
        plt.legend()
    if save_to:
        plt.savefig(save_to)
    if show:
        plt.show()


def autoplot(
    func: Callable, l: int=8, w: int=6, r: int=45, gmin: float=PLOT_GMIN, gmax: float=PLOT_GMAX, legend: bool=True, save_to: str=None
):
    """Decorator that automatically plots the decorated function.

    Args:
        func (Callable): function to plot.
        l (int, optional): lenght of the plot. Defaults to 8.
        w (int, optional): widht of the plot. Defaults to 6.
        r (int, optional): rotation angle of the xticks of the plot. Defaults to 45.
        gmin (float, optional): the minimum glucose value to plot (Y-axis). Defaults to PLOT_GMIN.
        gmax (float, optional): the maximum glucose value to plot (Y-axis). Defaults to PLOT_GMAX.
        legend (bool, optional): whether or not to show legend when plotting. Defaults to True.
        save_to (str, optional): file to save the plot to (if None, does not save). Defaults to None.
    """
    def wrapper(*args, **kwargs):
        init_plot(l, w, gmin, gmax)
        func(*args, **kwargs)
        end_plot(r, legend, save_to)

    return wrapper
