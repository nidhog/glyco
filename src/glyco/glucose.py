from typing import Dict, Optional, Union
import pandas as pd

from datetime import datetime as dt, timedelta as tdel, date as date_type
from matplotlib import pyplot as plt

from .utils import Units, Devices, find_nearest, weekday_map, is_weekend, autoplot, units_to_mmolL_factor

"""Warning and Error Messages
"""
error_not_implemented_method = (
    "NOT IMPLEMENTED: Method not yet supported in this version."
)

"""Default Values
"""
# Default values for column names as found in Freestyle Libre data
TIMESTAMP_COL_DEFAULT = "Device Timestamp"
GLUCOSE_MMOL_COL_DEFAULT = "Historic Glucose mmol/L"

# Default formats
TIMESTAMP_FMT_DEFAULT = "%d-%m-%Y %H:%M"
DATE_FMT_DEFAULT = "%d-%m-%Y (%A)"

# Dafault value for glucose limit used as a threshold in the #TODO: properties, features, what name?
GLUCOSE_LIMIT_DEFAULT = 6

# Values for column names glyco generates in a dataframe
# Note: All generated column variables start with an underscor '_'
_GLUCOSE_COL = "G"  # Generated glucose TODO: explain how it is generated
_TIMESTAMP_COL = "t"  # Generated time TODO: handle different tmz, UTC etc.
_DT_COL = "dt"
_DG_COL = "dg"
_DGDT_COL = "dg_dt"
_AUC_COL = "auc_mean"
_AUCLIM_COL = "auc_lim"
_AUCMIN_MIN = "auc_min"
# time derivatives
_DATE_COL = "date"  # TODO
_HOUR_COL = "hour"
_DAYOFWEEK_COL = "weekday_number"
_WEEKDAY_COL = "weekday_name"
_ISWEEKEND_COL = "is_weekend"
# Used for smoothening the glucose curve
# FIXME: _default_glucose_prep_kwargs to GlucoseTransform dataclass
_default_glucose_prep_kwargs = {
    'interpolate': True,
    'interp_met':'polynomial',
    'interp_ord':1,
    'rolling_avg':3,
}

# Meals
_meal_note_col = "Notes"
_meal_ref_col = "Reference"
_freestyle_rec_type_col = "Record Type"
_freestyle_notes_rec_type = 6
_freestyle_glucose_rec_type = 0
_optional_cols = [_meal_note_col, _meal_ref_col]
meal_default_cols = [_TIMESTAMP_COL, _meal_ref_col, _meal_ref_col]

general_date_type = Union[str, pd.Timestamp, date_type]

"""File reading
"""


def read_csv(
    file_path: str,
    timestamp_col: str = TIMESTAMP_COL_DEFAULT,
    timestamp_fmt: str = TIMESTAMP_FMT_DEFAULT,
    glucose_col: str = GLUCOSE_MMOL_COL_DEFAULT,
    glucose_unit: str = Units.mmolL.value,
    calculate_glucose_properties: bool=True,
    glucose_lim: int=GLUCOSE_LIMIT_DEFAULT,  # predefined glucose limit used for AUC calculation
    filter_glucose_rows=True,
    delimiter: str = ",",
    skiprows: int = 0,
    generated_glucose_col: str = _GLUCOSE_COL,
    generated_date_col: str = _DATE_COL,
    generated_timestamp_col: str =_TIMESTAMP_COL,
    glucose_prep_kwargs: Optional[Dict] = _default_glucose_prep_kwargs
) -> pd.DataFrame:
    """Reads a glucose CSV file.
    The file needs to have at least: one column for glucose, one timestamp column.


    Args:
        file_path (str): _description_
        timestamp_col (str, optional): _description_. Defaults to TIMESTAMP_COL_DEFAULT.
        timestamp_fmt (str, optional): _description_. Defaults to TIMESTAMP_FMT_DEFAULT.
        glucose_col (str, optional): _description_. Defaults to GLUCOSE_MMOL_COL_DEFAULT.
        glucose_unit (str, optional): _description_. Defaults to Units.mmolL.value.
        device (str, optional): _description_. Defaults to Devices.abbott.value.
        calculate_glucose_properties (bool, optional): _description_. Defaults to True.
        glucose_lim (int, optional): _description_. Defaults to GLUCOSE_LIMIT_DEFAULT.
        delimiter (str, optional): _description_. Defaults to ",".
        skiprows (int, optional): _description_. Defaults to 0.
        generated_glucose_col (str, optional): _description_. Defaults to _GLUCOSE_COL.
        generated_date_col (str, optional): _description_. Defaults to _DATE_COL.
        generated_timestamp_col (_type_, optional): _description_. Defaults to _TIMESTAMP_COL.
        glucose_curve_kwargs (Dict, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: Glucose Unified Dataframe
    """
    df = pd.read_csv(
        filepath_or_buffer=file_path,
        delimiter=delimiter,
        skiprows=skiprows
    )

    df = read_df(
        df,
        timestamp_col,
        timestamp_fmt,
        glucose_col,
        glucose_unit,
        calculate_glucose_properties,
        glucose_lim,
        filter_glucose_rows,
        generated_glucose_col,
        generated_date_col,
        generated_timestamp_col,
        glucose_prep_kwargs,
    )

    return df

def read_df(df: pd.Dataframe,
    timestamp_col: str = TIMESTAMP_COL_DEFAULT,
    timestamp_fmt: str = TIMESTAMP_FMT_DEFAULT,
    glucose_col: str = GLUCOSE_MMOL_COL_DEFAULT,
    glucose_unit: str = Units.mmolL.value,
    calculate_glucose_properties: bool = True,
    glucose_lim: int=GLUCOSE_LIMIT_DEFAULT,  # predefined glucose limit used for AUC calculation
    filter_glucose_rows=True,
    generated_glucose_col: str = _GLUCOSE_COL,
    generated_date_col: str = _DATE_COL,
    generated_timestamp_col: str =_TIMESTAMP_COL,
    glucose_prep_kwargs: Optional[Dict] = _default_glucose_prep_kwargs
):
    df = (
        df
        if not (filter_glucose_rows)
        else filter_glucose_by_column_val(
            df,
            filter_col=_freestyle_rec_type_col,
            filter_val=_freestyle_glucose_rec_type,
        )
    )
    if calculate_glucose_properties:
        if glucose_prep_kwargs is None:
            glucose_prep_kwargs = _default_glucose_prep_kwargs
        df = prepare_glucose(
            df,
            glucose_col=glucose_col,
            tsp_lbl=timestamp_col,
            tsp_fmt=timestamp_fmt,
            unit=glucose_unit,
            glbl=generated_glucose_col,
            tlbl=generated_timestamp_col,
            dlbl=generated_date_col,
            **glucose_prep_kwargs
        )
        df = get_properties(
            df,
            glbl=generated_glucose_col,
            tlbl=generated_timestamp_col,
            glim=glucose_lim,
        )
    return df

# Verify the file
# List of implemented devices and units
implemented_devices = list(map(lambda x: x.name, Devices))
implemented_units = list(map(lambda x: x.name, Units))

def is_valid_entry(device: str, unit: str, fail_on_invalid: bool = True) -> bool:
    """Verifies the device and unit are implemented.

    Args:
        device (str): name of the device used, e.g.: abbott
        unit (str): unit used, e.g.: mg/dL, mmol/L
        fail_on_invalid (bool): defaults to True. 
            If True raise an exception on an invalid entry.


    Raises:
        NotImplementedError: if fail_on_invalid is set to True and entry is invalid.

    Returns:
        bool: True if the entry is valid. 
        If the entry is invalid, an exception is raised if fail_on_invalid is True
        Otherwise False is returned.
    """
    if device.lower() in implemented_devices and unit.lower() in implemented_units:
        return True
    elif fail_on_invalid:
        raise NotImplementedError(
            f"Device '{device}' or unit {unit} are not yet supported.\n\
        We currently only support:\n- Devices: {implemented_devices}.\n- Units: {implemented_units}."
        )
    return False


def set_columns_by_device_unit():
    """Sets the column names of the glucose file
    based on the device and units used.
    WARNING: Not implemented yet

    Raises:
        NotImplementedError: _description_
    """
    raise NotImplementedError(error_not_implemented_method)


def filter_glucose_by_column_val(
    df, filter_col=_freestyle_rec_type_col, filter_val=_freestyle_glucose_rec_type
):
    return df[df[filter_col] == filter_val]


# TODO turn into function
# get datetime, date, hour etc. from timestamp
def get_time_values(df, tlbl, dlbl, tsp_lbl, tsp_fmt, weekday_map=weekday_map):
    df[tlbl] = pd.to_datetime(df[tsp_lbl], format=tsp_fmt)
    df[dlbl] = df[tlbl].dt.date
    df[f"{dlbl}_str"] = df[dlbl].map(lambda x: x.strftime(DATE_FMT_DEFAULT))
    df[_HOUR_COL] = df[tlbl].dt.hour
    df[_DAYOFWEEK_COL] = df[tlbl].dt.weekday
    df[_WEEKDAY_COL] = df[_DAYOFWEEK_COL].map(weekday_map)
    df[_ISWEEKEND_COL] = df[_DAYOFWEEK_COL].map(is_weekend)
    return df

def prepare_glucose(
    glucose_dataframe: pd.DataFrame,
    glucose_col: str,
    tsp_lbl: str,
    tsp_fmt: str,
    unit: str = Units.mmolL.value,
    glbl: str = _GLUCOSE_COL,
    tlbl: str = _TIMESTAMP_COL,
    dlbl: str = _DATE_COL,
    interpolate: bool = True,
    interp_met: str = "polynomial",
    interp_ord: int = 1,
    rolling_avg: int = 3,
    extra_shift_in_time: int = 7,
):
    """Creates extra columns for hours, days, etc.
    Sorts the dataframe by time.
    Creates columns for shifted time if needed (used for certain computations).
    Converts units if needed.
    Adds interpolated glucose measures to fill in the gaps.

    Args:
        df (pd.DataFrame): _description_
        glucose_col (str): _description_
        tsp_lbl (str): _description_
        tsp_fmt (str): _description_
        unit (str, optional): _description_. Defaults to Units.mmol.value.
        glbl (str, optional): _description_. Defaults to _GLUCOSE_COL.
        tlbl (str, optional): _description_. Defaults to _TIMESTAMP_COL.
        dlbl (str, optional): _description_. Defaults to _DATE_COL.
        interpolate (bool, optional): _description_. Defaults to True.
        interp_met (str, optional): _description_. Defaults to "polynomial".
        interp_ord (int, optional): _description_. Defaults to 1.
        rolling_avg (int, optional): _description_. Defaults to 3.
        extra_shift_in_time (int, optional): _description_. Defaults to 7.

    Returns:
        _type_: _description_
    """
    df = get_time_values(glucose_dataframe, tlbl, dlbl=dlbl, tsp_lbl=tsp_lbl, tsp_fmt=tsp_fmt, weekday_map=weekday_map)

    if extra_shift_in_time:
        df = add_shifted_time(df, tlbl, dlbl, extra_shift_in_time)

    # convert to mmol/L
    df[glbl] = (
        df[glucose_col]
        if unit == Units.mmolL.value
        else convert_unit(df[glucose_col], from_unit=unit, to_unit=Units.mmolL.value)
    )

    # index by time and keep time column
    df = df.set_index(tlbl)
    df[tlbl] = df.index
    df = df.sort_index()

    # interpolate and smoothen glucose
    if interpolate:
        # TODO handle Nan
        # TODO FIX missing values should still be nan otherwise averaged
        df[glbl] = df[glbl].rolling(window=rolling_avg).mean()
        # TODO smoothen methods
        df[glbl] = df[glbl].interpolate(method=interp_met, order=interp_ord)
        df = df[df[glbl].map(lambda g: g > 0 and g < 30)]
    return df


def add_shifted_time(df: pd.DataFrame, tlbl: str, dlbl: str, extra_shift_in_time: int):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        tlbl (str): _description_
        dlbl (str): _description_
        extra_shift_in_time (int): _description_
    """
    shift_tlbl = f"shifted_{tlbl}"
    shift_dlbl = f"shifted_{dlbl}"

    df[shift_tlbl] = df[tlbl].map(lambda x: x - tdel(hours=extra_shift_in_time))
    df[shift_dlbl] = df[shift_tlbl].dt.date
    df[f"{shift_dlbl}_str"] = df[shift_dlbl].map(lambda x: x.strftime(DATE_FMT_DEFAULT))
    df[f"shifted_{_HOUR_COL}"] = df[shift_tlbl].dt.hour
    df[f"shifted_{_DAYOFWEEK_COL}"] = df[shift_tlbl].dt.weekday
    df[f"shifted_{_WEEKDAY_COL}"] = df[f"shifted_{_DAYOFWEEK_COL}"].map(weekday_map)
    df[f"shifted_{_ISWEEKEND_COL}"] = df[f"shifted_{_DAYOFWEEK_COL}"].map(is_weekend)
    return df




"""Properties and Stats
"""
def set_derivative(
    df: pd.DataFrame, glucose_col: str, timestamp_col: str
) -> pd.DataFrame:
    """Sets the glucose time derivative (dG/dt)

    Args:
        df (pd.DataFrame): _description_
        glucose_col (str): _description_
        timestamp_col (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df[_DG_COL], df[_DT_COL], df[_DGDT_COL] = compute_derivative(
        df, glucose_col, timestamp_col
    )
    return df


def compute_derivative(df: pd.DataFrame, glucose_col: str, timestamp_col: str):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        glucose_col (str): _description_
        timestamp_col (str): _description_

    Returns:
        _type_: _description_
    """
    dG = df[glucose_col].diff()
    dT = df[timestamp_col].diff().dt.total_seconds()
    return dG, dT, dG / dT


def set_auc(
    df: pd.DataFrame, glucose_col: str, timestamp_col: str, glucose_auc_lim: float
) -> pd.DataFrame:
    """
    Sets Area Under the Curve (integral).
    Requires the derivative, will set automatically if not done.

    Args:
        df (pd.DataFrame): _description_
        glucose_col (str): _description_
        timestamp_col (str): _description_
        glucose_auc_lim (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if _DGDT_COL not in df.columns:
        df = set_derivative(df, glucose_col, timestamp_col)
    mean_g = df[glucose_col].mean()
    min_g = df[glucose_col].min()
    g_above_mean = df[glucose_col].map(lambda x: mean_g if x < mean_g else x)
    g_above_lim = df[glucose_col].map(
        lambda x: glucose_auc_lim if x < glucose_auc_lim else x
    )
    g_above_min = df[glucose_col].map(lambda x: min_g if x < min_g else x)
    df[_AUC_COL] = (g_above_mean - mean_g) * df[_DT_COL]
    df[_AUCLIM_COL] = (g_above_lim - glucose_auc_lim) * df[_DT_COL]
    df[_AUCMIN_MIN] = (g_above_min - min_g) * df[_DT_COL]
    return df


def get_properties(
    df: pd.DataFrame,
    glbl: str = _GLUCOSE_COL,
    tlbl: str = _TIMESTAMP_COL,
    glim: float = GLUCOSE_LIMIT_DEFAULT,
):
    """Apply prepare_glucose first

    Args:
        df (pd.DataFrame): _description_
        glbl (str, optional): _description_. Defaults to _GLUCOSE_COL.
        tlbl (str, optional): _description_. Defaults to _TIMESTAMP_COL.
        glim (float, optional): _description_. Defaults to GLUCOSE_LIMIT_DEFAULT.

    Returns:
        _type_: _description_
    """
    # Generated values: derivative, integral, stats
    # set derivative and area under the curve properties
    df = set_derivative(df, glbl, tlbl)
    df = set_auc(df, glbl, tlbl, glim)
    return df


def convert_unit(g: float, from_unit: str, to_unit: str) -> float:
    """"""
    if to_unit == Units.mmolL:
        if from_unit in implemented_units:
            return g * units_to_mmolL_factor[from_unit] # TODO fix inconsistency between implemented units and dict keys
    raise NotImplementedError(error_not_implemented_method)


"""Plotting
"""


@autoplot
def plot_glucose(
    df: pd.DataFrame,
    glbl: str = _GLUCOSE_COL,
    tlbl: str = _TIMESTAMP_COL,
    from_time: Optional[general_date_type] = None,
    to_time: Optional[general_date_type] = None,
):
    """Plots the glucose curve for a given dataframe and time frame

    Args:
        df (pd.DataFrame): _description_
        glbl (str, optional): _description_. Defaults to _GLUCOSE_COL.
        tlbl (str, optional): _description_. Defaults to _TIMESTAMP_COL.
        from_time (_type_, optional): _description_. Defaults to None.
        to_time (_type_, optional): _description_. Defaults to None.

    Raises:
        KeyError: _description_
    """
    plot_df = df[from_time:to_time]
    # TODO make as similar to pyplot as possible
    for d in plot_df.date.unique():
        plt.axvline(d, color="brown", linestyle="--", alpha=0.5)
    plt.axhline(GLUCOSE_LIMIT_DEFAULT)
    plt.axhline(GLUCOSE_LIMIT_DEFAULT - 1)
    plt.axhline(GLUCOSE_LIMIT_DEFAULT + 1)
    plt.axhline(plot_df[glbl].median())

    if glbl not in plot_df.keys():
        raise KeyError(f"Glucose Column {glbl} does not seem to be in the DataFrame.")
    plt.plot(plot_df[tlbl], plot_df[glbl])

@autoplot
def plot_daily_trend(df: pd.DataFrame, glbl: str = _GLUCOSE_COL):
    plot_percentiles(df, stat_col=glbl, percentiles=[0.01, 0.05])

def plot_percentiles(df, stat_col, percentiles, group_by_col=_HOUR_COL, color='green'):
    """By default, groups by column and plots percentiles of glucose
    """
    _, _, med, perc_l, perc_h = get_percentiles_and_stats(
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

def get_percentiles_and_stats(df: pd.DataFrame, percentiles: list, stat_col: str, group_by_col: str):
    grouped = df.groupby([group_by_col])[stat_col]
    mean = grouped.mean()
    med = grouped.median()
    dev = grouped.std()
    perc_l = [grouped.quantile(q) for q in percentiles]
    perc_h = [grouped.quantile(1-q) for q in percentiles]
    return mean, dev, med, perc_l, perc_h # FIXME: use dataclass, do we need mean, med?

def plot_comparison(df, glbl=_GLUCOSE_COL, compare_by=_WEEKDAY_COL, outliers=False, label_map=None, method='box', sort_vals = False):
    """

    :param df: dataframe containing the values to be compared and the comparison field
    :param glbl: label of the box plot values in the dataframe (Y-axis, defaults to the glucose label _GLUCOSE_COL)
    :param compare_by: field to compare by (X-axis, defaults to weekend label WEEKENDLBL)
    :param outliers: boolean to show or not show outliers, defaults to False
    :param label_map: lambda function to map the unique values of the compare_by field to some labels,
    defaults to None (showing original)
    :return:
    """
    all_vals = df[compare_by].unique()
    if sort_vals: all_vals.sort()
    if method == 'box':
        plt.boxplot([df[df[compare_by] == i][glbl].dropna() for i in all_vals],
                    labels=all_vals if label_map is None else [
                        label_map(i) for i in all_vals],
                    showfliers=outliers)
    else:
        raise NotImplementedError(f"Method {method} not implemented for comparison please use one of: 'box'")
    plt.title('Comparing {} by {}. Outliers are {}.'.
                format(glbl, compare_by, 'shown' if outliers else 'not shown'))


def get_response_bounds(df: pd.DataFrame, event_time: pd.Timestamp, pre_pad_min: int = 20, post_pad_min: int = 0, resp_time_min: int = 120, glbl: str = _GLUCOSE_COL, t_lbl: str = _TIMESTAMP_COL):
    # TODO improve inputs
    """ Assumes indexing by time
    """
    df_time = find_nearest(df, event_time, glbl, t_lbl, n_iter=100)
    start = df_time - tdel(minutes=pre_pad_min)
    end = df_time + tdel(minutes=resp_time_min) + tdel(minutes=post_pad_min)
    return start, end, df_time


def plot_response_to_event(df: pd.DataFrame, event_time: pd.Timestamp, event_title: Optional[str] = None, pre_pad_min: int = 20, post_pad_min: int = 0, resp_time_min: int = 120, glbl: str = _GLUCOSE_COL, t_lbl: str = _TIMESTAMP_COL, auc_lim=GLUCOSE_LIMIT_DEFAULT, show_auc=True, use_local_min=False):
    # TODO: clean inputs AUC/pre-pad, have multi-options large, medium, small
    """Plots the response to a specific event given by its event time.
    """
    s, e, t = get_response_bounds(df, event_time, pre_pad_min, post_pad_min, resp_time_min, glbl=glbl, t_lbl=t_lbl)
    plot_df = df.loc[s:e][glbl]
    plt.plot(plot_df)
    if show_auc:
        alim = auc_lim if not(use_local_min) else plot_df.mean()
        lim_df = plot_df.map(lambda x: x if x>alim else alim)
        plt.gca()
        plt.axhline(alim, color='red', label='limit', linestyle='--', alpha=0.3)
        plt.fill_between(lim_df.index,  lim_df, [alim for a in lim_df.index], color='green', alpha=0.1, label=f"Estimated glucose quantity consumed")

    plt.axvline(t, color='black', label='Event time', linestyle='--', alpha=0.1)
    if event_title:
        plt.title(event_title)

