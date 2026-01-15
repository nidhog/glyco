"""Defines general utility functions and variables"""

import pandas as pd
import enum
from matplotlib import pyplot as plt
from typing import Callable, Optional
import functools

# Map for the weekday number and the name of the weekday
WEEKDAY_MAP = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

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
    other = "other"  # Other device not defined above


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
    Units.mgdL.value: 1 / 18.0182,
    Units.gL.value: 100 / 18.0182,
}


def find_nearest(df: pd.DataFrame, pivot: pd.Timestamp, col: str):
    """Finds nearest value to a pivot in a dataframe column
    Returns None if no value is found. Returns the column value otherwise.
    **Assumes time is in the index of the dataframe.**

    Args:
        df (pd.DataFrame): dataframe to search in
        pivot (pd.Timestamp): timestamp to search for
        col (str): column of the dataframe to search in
    """
    s = df[col].dropna()
    if s.empty:
        return None
    return s.index[s.index.get_indexer([pivot], method="nearest")[0]]


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
    func: Optional[Callable] = None,
    *,
    l: int = 8,
    w: int = 6,
    r: int = 45,
    gmin: float = PLOT_GMIN,
    gmax: float = PLOT_GMAX,
    show_legend: bool = True,
    save_to: Optional[str] = None,
):
    """Decorator that automatically plots the decorated function.

    This decorator can be used with or without arguments to wrap a plotting
    function, handling plot initialization and finalization.

    Examples:
        To use with default plotting parameters:
        ```python
        @autoplot
        def my_plot_function():
            plt.plot([1, 2, 3], [4, 5, 6])
        ```

        To customize plotting parameters:
        ```python
        @autoplot(l=10, w=5)
        def my_custom_plot():
            plt.scatter([1, 2, 3], [4, 5, 6])
        ```

        You can disable the autoplot behavior for a specific call by passing
        `autoplot=False` to the decorated function:
        ```python
        my_plot_function(autoplot=False)
        ```

    Args:
        func (Callable, optional): The function to be decorated. This is handled automatically.
        l (int, optional): length of the plot. Defaults to 8.
        w (int, optional): width of the plot. Defaults to 6.
        r (int, optional): rotation angle of the xticks of the plot. Defaults to 45.
        gmin (float, optional): the minimum glucose value to plot (Y-axis). Defaults to PLOT_GMIN.
        gmax (float, optional): the maximum glucose value to plot (Y-axis). Defaults to PLOT_GMAX.
        show_legend (bool, optional): whether or not to show legend when plotting. Defaults to True.
        save_to (str, optional): file to save the plot to (if None, does not save). Defaults to None.
    """

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if kwargs.pop("autoplot", True):
                init_plot(l, w, gmin, gmax)
                fn(*args, **kwargs)
                end_plot(r=r, legend=kwargs.pop("show_legend", show_legend), save_to=save_to)
            else:
                fn(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)
