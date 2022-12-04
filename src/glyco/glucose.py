from typing import Dict
import pandas as pd

from datetime import datetime as dt, timedelta as tdel
from matplotlib import pyplot as plt

from .utils import Units, Devices, find_nearest, weekday_map, is_weekend, autoplot

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

"""File reading
"""


def read_csv(
    file_path: str,
    timestamp_col: str = TIMESTAMP_COL_DEFAULT,
    timestamp_fmt: str = TIMESTAMP_FMT_DEFAULT,
    glucose_col: str = GLUCOSE_MMOL_COL_DEFAULT,
    glucose_unit: str = Units.mmol.value,
    calculate_glucose_properties: bool=True,
    glucose_lim: int=GLUCOSE_LIMIT_DEFAULT,  # predefined glucose limit used for AUC calculation
    filter_glucose_rows=True,
    delimiter: str = ",",
    skiprows: int = 0,
    generated_glucose_col: str = _GLUCOSE_COL,
    generated_date_col: str = _DATE_COL,
    generated_timestamp_col=_TIMESTAMP_COL,
    glucose_prep_kwargs: Dict = None
) -> pd.DataFrame:
    """Reads a glucose CSV file.
    The file needs to have at least: one column for glucose, one timestamp column.


    Args:
        file_path (str): _description_
        timestamp_col (str, optional): _description_. Defaults to TIMESTAMP_COL_DEFAULT.
        timestamp_fmt (str, optional): _description_. Defaults to TIMESTAMP_FMT_DEFAULT.
        glucose_col (str, optional): _description_. Defaults to GLUCOSE_MMOL_COL_DEFAULT.
        glucose_unit (str, optional): _description_. Defaults to Units.mmol.value.
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

def is_valid_entry(device: str, unit: str) -> bool:
    """Verifies the device and unit are implemented.

    Args:
        device (str): _description_
        unit (str): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        bool: _description_
    """
    if device in implemented_devices and unit in implemented_units:
        return True
    raise NotImplementedError(
        f"Device '{device}' or unit {unit} are not yet supported.\n\
    We currently only support:\n- Devices: {implemented_devices}.\n- Units: {implemented_units}."
    )


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
    unit: str = Units.mmol.value,
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
        if unit == Units.mmol.value
        else convert_unit(df[glucose_col], from_unit=unit, to_unit=Units.mmol.value)
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
    raise NotImplementedError(error_not_implemented_method)


"""Plotting
"""


@autoplot
def plot_glucose(
    df: pd.DataFrame,
    glbl: str = _GLUCOSE_COL,
    tlbl: str = _TIMESTAMP_COL,
    from_time=None,
    to_time=None,
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
