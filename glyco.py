from calendar import week
from fileinput import filename
import pandas as pd

import numpy as np
from utils import Units, Devices
from matplotlib import pyplot as plt

"""Warning and Error Messages
"""
warn_meals_conflict = "MEALS CONFLICT: Meals that were consumed close in time may affect each other's glucose response."
warn_exercise_meal_conflict = "EXERCISE & MEAL CONFLICT: Meals & Activities that happened close in time may affect each other's glucose response."
warn_missing_values = "MISSING VALUES: Missing values for glucose may hide meals, activities or other glucose events."
error_not_implemented_method = "NOT IMPLEMENTED: Method not yet supported in this version."

"""Default Values
"""
# Default values for column names as found in Freestyle Libre data
TIMESTAMP_COL_DEFAULT = 'Device Timestamp'
GLUCOSE_MMOL_COL_DEFAULT = 'Historic Glucose mmol/L'

# Default formats
TIMESTAMP_FMT_DEFAULT = '%d-%m-%Y %H:%M'

# Dafault value for glucose limit used as a threshold in the #TODO: properties, features, what name?
GLUCOSE_LIMIT_DEFAULT = 6

# Values for column names glyco generates in a dataframe
# Note: All generated column variables start with an underscor '_'
_GLUCOSE_COL = 'G'  # Generated glucose TODO: explain how it is generated
_TIMESTAMP_COL = 't'  # Generated time TODO: handle different tmz, UTC etc.
_DT_COL = 'dt'
_DG_COL = 'dg'
_DGDT_COL = 'dg_dt'
_AUC_COL = 'auc_mean'
_AUCLIM_COL = 'auc_lim'
_AUCMIN_MIN = 'auc_min'
# time derivatives
_DATE_COL = 'date'  # TODO
_HOUR_COL = 'hour'
_DAYOFWEEK_COL = 'dayofweek'
_WEEKDAY_COL = 'weekday'
_ISWEEKEND_COL = 'is_weekend'

# List of implemented devices and units
device_names = list(map(lambda x: x.name, Devices))
unit_names = list(map(lambda x: x.name, Units))

"""Utility functions
Processing timestamps, strings etc.
"""
weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed',
               3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
def is_weekend(x): return 1 if x % 7 > 4 else 0

def is_valid_entry(device: str, unit: str):
    if device in device_names and unit in unit_names:
        return True
    raise NotImplementedError(f"Device '{device}' or unit {unit} are not yet supported.")


def set_labels_by_device_unit(device: str, unit: str):
    device_unit_map = {
        Devices.abbott.value: {
            Units.mmol: { # TODO implement
            }
        }
    }
    raise NotImplementedError(error_not_implemented_method)
    return device_unit_map

def convert_unit(g, from_unit, to_unit):
    # TODO all conversions
    raise NotImplementedError(error_not_implemented_method)
    if from_unit == to_unit:
        return g
    if from_unit == Units.MMOL:
        pass
    pass

def get_percentile_dist(df: pd.DataFrame, percentiles: list, stat_col: str, group_by_col: str):
    grouped = df.groupby([group_by_col])[stat_col]
    mean = grouped.mean()
    med = grouped.median()
    dev = grouped.std()
    perc_l = [grouped.quantile(q) for q in percentiles]
    perc_h = [grouped.quantile(1-q) for q in percentiles]
    return mean, dev, med, perc_l, perc_h
    

"""File reading
"""

# TODO reformat Python
def read_csv(file_path: str, delimiter: str=',', skiprows: int=0,
            t_col: str = TIMESTAMP_COL_DEFAULT, t_fmt: str = TIMESTAMP_FMT_DEFAULT, glucose_col: str = GLUCOSE_MMOL_COL_DEFAULT,
            device: str = Devices.abbott.value, set_labels_by_device: bool=False, unit: str=Units.mmol.value,
            only_read_as_is=False,
            glbl=_GLUCOSE_COL,
            tlbl=_TIMESTAMP_COL,
            glim=GLUCOSE_LIMIT_DEFAULT,  # static glucose used for auc
            dlbl=_DATE_COL,
            # TODO take in those inputs as map
            interpolate: bool=True, interp_met: str='polynomial', interp_ord: int=2, rolling_avg: int=3):
    # TODO implement setting labels by device and unit and conversion
    if set_labels_by_device:
        is_valid_entry(device=device, unit=unit)
        # TODO log warning about labels being set
        set_labels_by_device_unit(device=device, unit=unit)
    df = pd.read_csv(filepath_or_buffer=file_path, delimiter=delimiter, skiprows=skiprows)
    if not only_read_as_is:
        prepare_glucose(df, glucose_col=glucose_col, tsp_lbl=t_col, tsp_fmt=t_fmt, unit=unit, glbl=glbl, tlbl=tlbl, dlbl=dlbl, interpolate=interpolate, interp_met=interp_met, interp_ord=interp_ord, rolling_avg=rolling_avg)
        get_properties(df, glbl=glbl, tlbl=tlbl, glim=glim)
    return df

def prepare_glucose(df: pd.DataFrame,
                   glucose_col: str,
                   tsp_lbl: str,
                   tsp_fmt: str, 
                   unit: str=Units.mmol.value,
                   glbl: str=_GLUCOSE_COL,
                   tlbl: str=_TIMESTAMP_COL,
                   dlbl: str=_DATE_COL,
                   interpolate: bool=True, interp_met: str='polynomial', interp_ord: int=2, rolling_avg: int=3):

    # get datetime, date, hour etc. from timestamp
    df[tlbl] = pd.to_datetime(df[tsp_lbl], format=tsp_fmt)
    df[dlbl] = df[tlbl].dt.date
    df[_HOUR_COL] = df[tlbl].dt.hour
    df[_DAYOFWEEK_COL] = df[tlbl].dt.weekday
    df[_WEEKDAY_COL] = df[_DAYOFWEEK_COL].map(weekday_map)
    df[_ISWEEKEND_COL] = df[_DAYOFWEEK_COL].map(is_weekend)

    # convert to mmol/L
    df[glbl] = df[glucose_col] if unit == Units.mmol.value \
        else convert_unit(df[glucose_col], from_unit=unit, to_unit=Units.mmol.value)

    # index by time and keep time column
    df = df.set_index(tlbl)
    df[tlbl] = df.index
    df.sort_index(inplace=True)

    # interpolate and smoothen glucose
    if interpolate:
        # TODO handle Nan
        # TODO FIX missing values should still be nan otherwise averaged
        df[glbl] = df[glbl].rolling(window=rolling_avg).mean()
        # TODO smoothen methods
        df[glbl] = df[glbl].interpolate(method=interp_met, order=interp_ord)
        df = df[df[glbl].map(lambda g: g>0 and g<30)]
    return df

def get_properties(df: pd.DataFrame, glbl: str=_GLUCOSE_COL,
                   tlbl: str=_TIMESTAMP_COL,
                   glim: float=GLUCOSE_LIMIT_DEFAULT):
    """Requires prepare_glucose
    """
    # Generated values: derivative, integral, stats
    # set derivative and area under the curve properties
    df = set_derivative(df, glbl, tlbl)
    df = set_auc(df, glbl, tlbl, glim)
    pass

"""Properties and Stats
"""
def set_derivative(df: pd.DataFrame, glucose_col: str, timestamp_col: str) -> pd.DataFrame:
    """Sets the glucose time derivative (dG/dt).
    """
    df[_DG_COL], df[_DT_COL], df[_DGDT_COL] = compute_derivative(df, glucose_col, timestamp_col)
    return df

def compute_derivative(df: pd.DataFrame, glucose_col: str, timestamp_col: str):
    dG = df[glucose_col].diff()
    dT = df[timestamp_col].diff().dt.total_seconds()
    return dG, dT, dG/dT

def set_auc(df: pd.DataFrame, glucose_col: str, timestamp_col: str, glucose_auc_lim: float) -> pd.DataFrame:
    """
    Sets Area Under the Curve (integral).
    Requires the derivative, will set automatically if not done.
    
    """
    if _DGDT_COL not in df.columns:
        df = set_derivative(df, glucose_col, timestamp_col)
    mean_g = df[glucose_col].mean()
    min_g = df[glucose_col].min()
    g_above_mean = df[glucose_col].map(lambda x: mean_g if x < mean_g else x)
    g_above_lim = df[glucose_col].map(lambda x: glucose_auc_lim if x < glucose_auc_lim else x)
    g_above_min = df[glucose_col].map(lambda x: min_g if x < min_g else x)
    df[_AUC_COL] = (g_above_mean-mean_g) * df[_DT_COL]
    df[_AUCLIM_COL] = (g_above_lim-glucose_auc_lim) * df[_DT_COL]
    df[_AUCMIN_MIN] = (g_above_min-min_g) * df[_DT_COL]
    return df

"""Plotting | Utils
"""
def init_plot(l=8, w=6):
    """Initialize plot

    :param l: length, defaults to 8
    :param w: width, defaults to 6
    :return:
    """
    plt.figure(num=None, figsize=(l, w), dpi=120, facecolor='w', edgecolor='k')

def end_plot(r=45, legend=True):
    """End plot by rotating xticks, adding legend and showing the plot

    :param r:
    :return:
    """
    plt.xticks(rotation=r)
    if legend:
        plt.legend()
    plt.show()

def autoplot(func, l=8, w=6, r=45, legend=True):
    def wrapper(*args, **kwargs):
        init_plot(l, w)
        func(*args, **kwargs)
        end_plot(r, legend)
    return wrapper

"""Plotting | General
"""
@autoplot
def plot(df: pd.DataFrame, glbl: str = _GLUCOSE_COL, tlbl: str = _TIMESTAMP_COL):
    # TODO make as similar to pyplot as possible
    if glbl not in df.keys():
        raise KeyError(f"Glucose Column {glbl} does not seem to be in the DataFrame.")
    plt.plot(df[tlbl], df[glbl])

def plot_summaries(df):
    """Plots all summaries of the dataframe glucose
    """
    pass

@autoplot
def plot_daily_trend(df: pd.DataFrame, glbl: str = _GLUCOSE_COL):
    """Plots daily trend

    :param df: Dataframe containing glucose
    :param glbl: Name of the glucose column
    :return:
    """
    plot_percentiles(df, stat_col=glbl, percentiles=[0.01, 0.05])

@autoplot
def plot_hourly_trend(df: pd.DataFrame, hlbl: str = _HOUR_COL):
    """Plots hourly trend

    :param df: Dataframe containing glucose
    :param hlbl: Name of the hour column
    :return:
    """
    plot_percentiles(df, stat_col=hlbl, percentiles=[0.01, 0.05])


def plot_percentiles(df, stat_col, percentiles, group_by_col=_HOUR_COL, color='green'):
    """By default, groups by column and plots percentiles of glucose
    """
    _, _, med, perc_l, perc_h = get_percentile_dist(
        df, percentiles, stat_col, group_by_col)
    plt.plot(med, label='Median')
    for i in range(len(percentiles)):
        plt.fill_between(
            # TODO enable changing alpha and label
            med.index, perc_l[i], perc_h[i], color=color, alpha=0.2, label=f"{100*(1-percentiles[i])}th")
    plt.title('Trend of {} for the percentiles: {} as well as {}'.format(stat_col,
                                                                         ', '.join(
                                                                             [str(int(i*100)) for i in percentiles]),
                                                                         ', '.join([str(int((1-i)*100)) for i in percentiles])))
    plt.xlabel(group_by_col)
    plt.ylabel(stat_col)

def barplot():
    # similar to pyplot
    pass

def violinplot():
    pass

def plot_integral_above(threshold):
    pass

def plot_event(timestamp):
    pass

"""Day features | General
"""

"""Meal detection

TODO: Generalise meal to event
"""
def get_meals_from_folder():
    pass

def get_meals_timestamps():
    pass

def get_meals_autodetect():
    pass

def get_meals_recordtype():
    pass

"""Meal features
"""

"""Day features | Meals
"""

"""Plotting | Meals
"""

