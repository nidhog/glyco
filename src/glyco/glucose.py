"""
Glucose analysis and visualization.

This module provides functions for:
- Reading and validating continuous glucose monitoring (CGM) data
- Preparing and cleaning glucose time-series data
- Computing descriptive statistics and daily/hourly metrics
- Plotting trends, percentiles, and summary statistics
- Masking private CGM data

It is designed to work with pandas DataFrames and supports
privacy-preserving workflows, unit normalization, and
robust statistical summaries for exploratory and analytical use.
"""
# pylint: disable=too-many-lines
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import date as date_type, datetime as dt, timedelta as tdel
from typing import Callable, Iterable, Optional, Tuple, Union, Any, Dict

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table

from .privacy import mask_private_information
from .utils import (
    Devices,
    Units,
    end_plot,
    find_nearest,
    is_weekend,
    units_to_mmolL_factor,
    WEEKDAY_MAP,
    autoplot,
)

logger = logging.getLogger(__name__)
"""Warning and Error Messages
"""
ERR_NOT_IMPLEMENTED = "NOT IMPLEMENTED: Method not yet supported in this version."

"""Default Values
"""
TIMESTAMP_COL = "tsp"
GLUCOSE_COL = "glucose"
# Default values for column names as found in Freestyle Libre data
DEFAULT_INPUT_TSP_COL = "Device Timestamp"
DEFAULT_INPUT_GLUC_COL = "Historic Glucose mmol/L"

# Default formats
DEFAULT_INPUT_TSP_FMT = "%d-%m-%Y %H:%M"
DEFAULT_OUT_DATE_FMT = "%d-%m-%Y (%A)"

DEFAULT_GLUC_LIMIT = 5  # glucose threshold used for calculating Area Under the Curve
DEFAULT_CSV_DELIMITER = ","

# Values for column names glyco generates in a dataframe
# Note: All generated column variables start with an underscore '_'
_DT_COL = "dt"
_DG_COL = "dg"
_DGDT_COL = "dg_dt"
_AUC_COL = "auc_mean"
_AUCLIM_COL = "auc_lim"
_AUCMIN_MIN = "auc_min"
# time derivatives
_DATE_COL = "date"
_HOUR_COL = "hour"
_DAYOFWEEK_COL = "weekday_number"
_WEEKDAY_COL = "weekday_name"
_ISWEEKEND_COL = "is_weekend"

# Meals
_MEAL_NOTE_COL = "Notes"
# _MEAL_REF_COL = "Reference" TODO remove unused
_FREESTYLE_REC_TYPE_COL = (
    "Record Type"
)
_FREESTYLE_SERIALNUM_COL = "Serial Number"
# _FREESTYLE_NOTE_REC_TYPE = 6 TODO: remove unused
_FREESTYLE_GLUCOSE_REC_TYPE = 0
# _OPTIONAL_COLS = [_MEAL_NOTE_COL, _MEAL_REF_COL]
# MEAL_DEFAULT_COLS = [TIMESTAMP_COL, _MEAL_REF_COL, _MEAL_REF_COL]
_DEFAULT_SHIFT_HOURS = 7


def _default_hash_func(x):
    """Default hash function for masking private info."""
    return hashlib.sha256(str(x).encode()).hexdigest()


@dataclass(frozen=True)
class GlucosePrepKwargs:
    """Keyword Arguments used for preparing and cleaning glucose data."""
    interpolate: bool = True
    interp_met: str = "polynomial"
    interp_ord: int = 1
    rolling_avg: int = 3

    def to_kwargs(self) -> dict[str, Any]:
        """as kwargs dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PrivateInfoKwargs:
    """Keyword Arguments used for masking private information."""
    set_start_date: str = "01-01-2023 00:00"
    remove_columns: list[str] = field(default_factory=lambda: [_FREESTYLE_SERIALNUM_COL])
    replace_columns: list[str] = field(default_factory=lambda: [_MEAL_NOTE_COL])
    replace_func: Callable = _default_hash_func
    noise_std: float = 0.2

    def to_kwargs(self) -> dict[str, Any]:
        """as kwargs dictionary."""
        return asdict(self)



GeneralDateType = Union[str, pd.Timestamp, date_type]

"""File reading
"""
def read_csv( # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    file_path: str,
    timestamp_col: str = DEFAULT_INPUT_TSP_COL,
    timestamp_fmt: str = DEFAULT_INPUT_TSP_FMT,
    glucose_col: str = DEFAULT_INPUT_GLUC_COL,
    glucose_unit: str = Units.mmolL.value,
    unit_autodetect: bool = False,
    calculate_glucose_properties: bool = True,
    glucose_lim: int = DEFAULT_GLUC_LIMIT,  # predefined glucose limit used for AUC calculation
    filter_glucose_rows=False,
    delimiter: str = DEFAULT_CSV_DELIMITER,
    skiprows: int = 0,
    generated_glucose_col: str = GLUCOSE_COL,
    generated_date_col: str = _DATE_COL,
    generated_timestamp_col: str = TIMESTAMP_COL,
    glucose_prep_kwargs: Optional[GlucosePrepKwargs] = None,
    mask_private_info: Optional[bool] = False,
    private_info_kwargs: Optional[PrivateInfoKwargs] = None,
) -> pd.DataFrame:
    """Reads a CSV file with glucose data and generates a Glucose DataFrame.
    - The file needs to have at least: one column for glucose, one timestamp column.

    Args:
        file_path (str): the file path to the glucose CSV file (for example: 'data/sample_glucose.csv')
        timestamp_col (str, optional): the name of the timestamp column in the CSV file.
            Defaults to the value of DEFAULT_INPUT_TSP_COL.
        timestamp_fmt (str, optional): the format of the timestamps in the CSV file.
            Must follow ISO 8601 format, for example: 'YYYY-MM-DDTHH:MM:SS' .
            This will be used to convert the timestamp column values to a 'datetime'.
            Defaults to DEFAULT_INPUT_TSP_FMT.
        glucose_col (str, optional): the name of the glucose column in the CSV file.
            Defaults to DEFAULT_INPUT_GLUC_COL.
        glucose_unit (str, optional): the unit of the glucose values in the CSV file.
            These will be converted to the mmol/L unit. See the units documentation.
            Defaults to Units.mmolL.value.
        unit_autodetect (bool, optional): if 'true' you do not need to define the glucose unit.
            If true, the unit will be automatically inferred from the values.
            Defaults to False.
        calculate_glucose_properties (bool, optional): if true the Generated Glucose Properties
            will be calculated and added to the resulting dataframe.
            See the Generated Glucose Properties section of the Glucose documentation.
            Defaults to True.
        glucose_lim (int, optional): a lower limit/threshold in the value of glucose that will be used
            by some of the Generated Glucose Properties.
            See the Generated Glucose Properties section of the Glucose documentation.
            Defaults to DEFAULT_GLUC_LIMIT.
        delimiter (str, optional): the delimiter that separates column values in the CSV file.
            For example "," or ";".
            Defaults to DEFAULT_CSV_DELIMITER.
        skiprows (int, optional): number of rows to skip in the CSV file.
            Defaults to 0.
        generated_glucose_col (str, optional): the name of the generated glucose
            column in the resulting Glucose Dataframe.
            Defaults to GLUCOSE_COL.
        generated_date_col (str, optional): the name of the generated date column
            in the resulting Glucose Dataframe.
            Defaults to _DATE_COL.
        generated_timestamp_col (str, optional): the name of the generated timestamp
            column in the resulting Glucose Dataframe.
            Defaults to TIMESTAMP_COL.
        glucose_prep_kwargs (Optional[GlucosePrepConfig], optional): Configuration for glucose data preparation.
            If None, default values are used.
            See GlucosePrepConfig for details.
            Defaults to None.
        mask_private_info (bool, optional): choose to mask or not to mask private information.
            This uses the 'mask_private_information' function.
            See the Privacy documentation for more on how this works.
            Defaults to false.
        private_info_kwargs (Optional[PrivateInfoConfig], optional): Configuration for masking private information.
            If None, default values are used.
            See PrivateInfoConfig for details.
            Defaults to None.

    Returns:
        pd.DataFrame: The resulting Glucose Dataframe that contains the file data,
            along with the Generated Glucose Properties
    """
    df = pd.read_csv(filepath_or_buffer=file_path, delimiter=delimiter, skiprows=skiprows)
    df = read_df(
        df=df,
        timestamp_col=timestamp_col,
        timestamp_fmt=timestamp_fmt,
        glucose_col=glucose_col,
        unit_autodetect=unit_autodetect,
        glucose_unit=glucose_unit,
        calculate_glucose_properties=calculate_glucose_properties,
        glucose_lim=glucose_lim,
        filter_glucose_rows=filter_glucose_rows,
        generated_glucose_col=generated_glucose_col,
        generated_date_col=generated_date_col,
        generated_timestamp_col=generated_timestamp_col,
        glucose_prep_kwargs=glucose_prep_kwargs,
        mask_private_info=mask_private_info,
        private_info_kwargs=private_info_kwargs,
    )
    return df


def validate_glucose_columns(df: pd.DataFrame, glucose_col: str, timestamp_col: str):
    """Validates the glucose and timestamp columns in the dataframe.
    Currently, only checks their existence.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        glucose_col (str): the glucose column name.
        timestamp_col (str): the timestamp column name.

    Raises:
        ValueError: raised if the glucose column does not exist in the dataframe.
        ValueError: raised if the timestamp column does not exist in the dataframe.
    """
    if glucose_col not in df.columns:
        raise ValueError(
            f"The Glucose column '{glucose_col}' is not in the input columns."
            "Please provide 'glucose_col' as input."
        )
    if timestamp_col not in df.columns:
        raise ValueError(
            f"The Timestamp column '{timestamp_col}' is not in the input columns."
            "Please provide 'timestamp_col' as input."
        )
    coerced = pd.to_numeric(df[glucose_col], errors="coerce")
    if coerced.isna().all():
        raise ValueError(
            f"The Glucose column '{glucose_col}' does not seem to have any numeric values."
        )
    if coerced.isna().any():
        logger.warning("The Glucose column '%s' contains some non-numeric values.", glucose_col)


def read_df( # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    df: pd.DataFrame,
    timestamp_col: str = DEFAULT_INPUT_TSP_COL,
    timestamp_fmt: str = DEFAULT_INPUT_TSP_FMT,
    glucose_col: str = DEFAULT_INPUT_GLUC_COL,
    glucose_unit: str = Units.mmolL.value,
    unit_autodetect: bool = False,
    calculate_glucose_properties: bool = True,
    glucose_lim: int = DEFAULT_GLUC_LIMIT,
    filter_glucose_rows=False,
    generated_glucose_col: str = GLUCOSE_COL,
    generated_date_col: str = _DATE_COL,
    generated_timestamp_col: str = TIMESTAMP_COL,
    glucose_prep_kwargs: Optional[GlucosePrepKwargs] = None,
    mask_private_info: Optional[bool] = False,
    private_info_kwargs: Optional[PrivateInfoKwargs] = None,
):
    """Reads a pandas Dataframe with glucose data and generates a Glucose Dataframe.
    - The Dataframe needs to have at least: one column for glucose, one timestamp column.

    Args:
        df (pd.DataFrame): the pandas Dataframe with glucose data.
        timestamp_col (str, optional): the name of the timestamp column in the CSV file.
            Defaults to the value of DEFAULT_INPUT_TSP_COL.
        timestamp_fmt (str, optional): the format of the timestamps in the CSV file.
            Must follow ISO 8601 format, for example: 'YYYY-MM-DDTHH:MM:SS' .
            This will be used to convert the timestamp column values to a 'datetime'.
            Defaults to DEFAULT_INPUT_TSP_FMT.
        glucose_col (str, optional): the name of the glucose column in the CSV file.
            Defaults to DEFAULT_INPUT_GLUC_COL.
        glucose_unit (str, optional): the unit of the glucose values in the CSV file.
            These will be converted to the mmol/L unit. See the units documentation.
            Defaults to Units.mmolL.value.
        unit_autodetect (bool, optional): if 'true' you do not need to define the glucose unit.
            If true, the unit will be automatically inferred from the values.
            Defaults to False.
        calculate_glucose_properties (bool, optional): if true the Generated Glucose Properties
            will be calculated and added to the resulting dataframe.
            See the Generated Glucose Properties section of the Glucose documentation.
            Defaults to True.
        glucose_lim (int, optional): a lower limit/threshold in the value of glucose that will be used
            by some of the Generated Glucose Properties.
            See the Generated Glucose Properties section of the Glucose documentation.
            Defaults to DEFAULT_GLUC_LIMIT.
        filter_glucose_rows: (bool, optional): if set to true it will filter specific columns and column values.
            Defaults to False.
        generated_glucose_col (str, optional): the name of the generated glucose
            column in the resulting Glucose Dataframe.
            Defaults to GLUCOSE_COL.
        generated_date_col (str, optional): the name of the generated date column
            in the resulting Glucose Dataframe.
            Defaults to _DATE_COL.
        generated_timestamp_col (str, optional): the name of the generated timestamp
            column in the resulting Glucose Dataframe.
            Defaults to TIMESTAMP_COL.
        glucose_prep_kwargs (Optional[GlucosePrepConfig], optional): Configuration for glucose data preparation.
            If None, default values are used.
            See GlucosePrepConfig for details.
            Defaults to None.
        mask_private_info (bool, optional): choose to mask or not to mask private information.
            This uses the 'mask_private_information' function.
            See the Privacy documentation for more on how this works.
            Defaults to false.
        private_info_kwargs (Optional[PrivateInfoConfig], optional): Configuration for masking private information.
            If None, default values are used.
            See PrivateInfoConfig for details.
            Defaults to None.

    Returns:
        pd.DataFrame: The resulting Glucose Dataframe that contains the file data,
            along with the Generated Glucose Properties.
    """
    if glucose_prep_kwargs is None:
        glucose_prep_kwargs = GlucosePrepKwargs()
    if private_info_kwargs is None:
        private_info_kwargs = PrivateInfoKwargs()

    validate_glucose_columns(df=df, glucose_col=glucose_col, timestamp_col=timestamp_col)
    if unit_autodetect:
        glucose_unit = autodetect_unit(df[glucose_col])
    logger.info("Using the glucose unit (%s)", glucose_unit)
    # filter rows based on the values of a column (filter_val)
    if filter_glucose_rows:
        df = filter_glucose_by_column_val(
            df, filter_col=_FREESTYLE_REC_TYPE_COL, filter_val=_FREESTYLE_GLUCOSE_REC_TYPE
        )
    # mask private information
    if mask_private_info:
        df, _, _ = mask_private_information(
            gdf=df,
            glucose_col=glucose_col,
            tsp_col=timestamp_col,
            tsp_fmt=timestamp_fmt,
            **private_info_kwargs.to_kwargs(),
        )
    # add calculated glucose properties
    if calculate_glucose_properties:
        df = prepare_glucose(
            glucose_df=df,
            glucose_col=glucose_col,
            tsp_lbl=timestamp_col,
            timestamp_fmt=timestamp_fmt,
            timestamp_is_formatted=False,
            unit=glucose_unit,
            glbl=generated_glucose_col,
            tlbl=generated_timestamp_col,
            dlbl=generated_date_col,
            **glucose_prep_kwargs.to_kwargs(),
        ).pipe(
            get_properties,
            glbl=generated_glucose_col,
            tlbl=generated_timestamp_col,
            glim=glucose_lim,
        )
    return df


# Verify the file
# List of implemented devices and units
implemented_devices = list(map(lambda x: x.value, Devices))
implemented_units = list(map(lambda x: x.value, Units))


def is_valid_entry(unit: str, device: str, fail_on_invalid: bool = True) -> bool:
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
    if fail_on_invalid:
        raise NotImplementedError(
            f"Device '{device}' or unit {unit} are not yet supported.\n\
        We currently only support:\n- Devices: {implemented_devices}.\n- Units: {implemented_units}."
        )
    return False


def set_columns_by_device_unit():
    """(WARNING: Not implemented yet)
    Sets the column names of the glucose file
    based on the device and units used.

    Raises:
        NotImplementedError: this method is not yet implemented
    """
    raise NotImplementedError(ERR_NOT_IMPLEMENTED)


def filter_glucose_by_column_val(
    df: pd.DataFrame,
    filter_col: str = _FREESTYLE_REC_TYPE_COL,
    filter_val=_FREESTYLE_GLUCOSE_REC_TYPE,
):
    """Selects only columns with a specific value.
    - By default, filters glucose columns based on Freestyle Libre data.

    Args:
        df (pd.DataFrame): The glucose dataframe to filter.
        filter_col (str, optional): Filter column.
            Defaults to _freestyle_rec_type_col which is
            the 'Record Type' column in freestyle libre.
        filter_val (_type_, optional): The value to select for the filter column.
            Defaults to _freestyle_glucose_rec_type which is
            the record type of glucose in the freestyle libre data.

    Returns:
        pd.DataFrame: dataframe where only specific columns are selected.
    """
    logger.info("Selecting only columns with a %s with the value %s", filter_col, filter_val)
    return df[df[filter_col] == filter_val]


def add_time_values( # pylint: disable=too-many-arguments,too-many-positional-arguments
    df,
    tlbl: str = TIMESTAMP_COL,
    tsp_lbl: str = DEFAULT_INPUT_TSP_COL,
    timestamp_fmt: str = DEFAULT_INPUT_TSP_FMT,
    dlbl: str = _DATE_COL,
    weekday_map=None,
    timestamp_is_formatted: bool = False,
):
    """Adds generated time-values to the dataframe based on the timestamp.
    These include:
        Date, Date string, Hour, Weekday number, Weekday name,
        Is or is not a weekend day.

    Args:
        df (_type_): the glucose dataframe.
        tlbl (str, optional): the name of the new timestamp column generated with datetime values.
            Defaults to the value of TIMESTAMP_COL.
        tsp_lbl (str, optional): the name of the input timestamp column used. Defaults to DEFAULT_INPUT_TSP_COL.
        timestamp_fmt (str, optional): the format of the values used in the input timestamp column used.
            Not needed if the timestamp is already formatted, see argument 'timestamp_is_formatted'.
            Defaults to DEFAULT_INPUT_TSP_FMT.
        dlbl (str, optional): label used for the date. Defaults to _DATE_COL.
        weekday_map (_type_, optional): label used for the weekday. Defaults to weekday_map.
        timestamp_is_formatted (bool, optional): whether or not the timestamp in the input is already formatted.
            If it is, this will not be converted using 'timestamp_fmt'. Defaults to False.

    Raises:
        ValueError: raised if the timestamp conversion using 'timestamp_fmt' fails.

    Returns:
        pd.DataFrame: the glucose dataframe with the generated time values.
            See the documentation on generated time values for the naming of columns.
    """
    ndf = df.copy()
    # if timestamp is not a string but already a pd.Timestamp type
    if timestamp_is_formatted:
        ndf[tlbl] = ndf[tsp_lbl]
    # else convert timestamp using timestamp_fmt
    else:
        ndf = convert_tsp(ndf, tsp_lbl, tlbl, timestamp_fmt)
    ndf[dlbl] = ndf[tlbl].dt.date
    ndf[f"{dlbl}_str"] = ndf[dlbl].map(
        lambda x: x.strftime(DEFAULT_OUT_DATE_FMT) if isinstance(x, (dt, pd.Timestamp, date_type)) else str(x)
    )
    ndf[_HOUR_COL] = ndf[tlbl].dt.hour
    ndf[_DAYOFWEEK_COL] = ndf[tlbl].dt.weekday
    ndf = ndf.assign()
    if weekday_map is None:
        weekday_map = WEEKDAY_MAP
    ndf[_WEEKDAY_COL] = ndf[_DAYOFWEEK_COL].map(weekday_map)
    ndf[_ISWEEKEND_COL] = ndf[_DAYOFWEEK_COL].map(is_weekend)
    return ndf


def convert_tsp(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    timestamp_fmt: str,
    *,
    errors: str = "raise",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Convert a timestamp column to pandas datetime using a specified format.

    Args:
        df: Input DataFrame.
        source_col: Name of the column containing timestamps to parse.
        target_col: Name of the column where parsed datetimes will be stored.
        timestamp_fmt: Datetime format string (strftime-compatible).
        errors: How to handle parsing errors: {"raise", "coerce"}. Defaults to "raise".
        copy: Whether to operate on a copy of the DataFrame.

    Returns:
        DataFrame with the converted timestamp column added.

    Raises:
        KeyError: If `source_col` does not exist.
        ValueError: If parsing fails and `errors="raise"`.
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' does not exist in DataFrame")

    out = df.copy() if copy else df

    try:
        out[target_col] = pd.to_datetime(
            out[source_col],
            format=timestamp_fmt,
            errors=errors,
        )
    except ValueError as e:
        raise ValueError(
            f"Failed to convert column '{source_col}' to datetime "
            f"using format '{timestamp_fmt}'."
            " Verify that you are using the correct 'timestamp_fmt' as input"
        ) from e

    return out


def prepare_glucose( # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    glucose_df: pd.DataFrame,
    glucose_col: str,
    tsp_lbl: str,
    timestamp_fmt: str,
    unit: str = Units.mmolL.value,
    glbl: str = GLUCOSE_COL,
    tlbl: str = TIMESTAMP_COL,
    dlbl: str = _DATE_COL,
    timestamp_is_formatted: bool = True,
    interpolate: bool = True,
    interp_met: str = "polynomial",
    interp_ord: int = 1,
    rolling_avg: int = 3,
    extra_shift_in_time: int = _DEFAULT_SHIFT_HOURS,
):
    """Parses the glucose data.
    - Creates extra columns for hours, days, etc.
    - Sorts the dataframe by time.
    - Creates columns for shifted time if needed (used for certain computations).
    - Converts units if needed.
    - Adds interpolated glucose measures to fill in the gaps.
    - Filters out unrealistic glucose values.

    Args:
        df (pd.DataFrame): the glucose dataframe
        glucose_col (str): the name of the original glucose column
        tsp_lbl (str): the name of the original timestamp column
        tsp_fmt (str): the format of timestamps in the original timestamp column
        timestamp_is_formatted (bool): wether the timestamp is already a datetime (no need for formatting)
        unit (str, optional): the unit of glucose values in the glucose column.
            Defaults to Units.mmol.value.
        glbl (str, optional): the name of the glucose column to be created.
            Defaults to GLUCOSE_COL.
        tlbl (str, optional): the name of the timestamp column to be created.
            Defaults to TIMESTAMP_COL.
        dlbl (str, optional): the name of the date column to be created.
            Defaults to _DATE_COL.
        interpolate (bool, optional): whether or not to use interpolation
            to fill and smoothen glucose values. Defaults to True.
        interp_met (str, optional): the method to be used for interpolation.
            Defaults to "polynomial".
        interp_ord (int, optional): the order to be used for interpolation.
            Defaults to 1.
        rolling_avg (int, optional): the number used as a rolling average
            for glucose. Defaults to 3.
        extra_shift_in_time (int, optional): adds shifted time values. Defaults to 7.

    Returns:
        _type_: _description_
    """
    df = add_time_values(
        glucose_df,
        tlbl=tlbl,
        dlbl=dlbl,
        tsp_lbl=tsp_lbl,
        timestamp_fmt=timestamp_fmt,
        weekday_map=WEEKDAY_MAP,
        timestamp_is_formatted=timestamp_is_formatted,
    )
    if extra_shift_in_time:
        df = add_shifted_time(df, tlbl, dlbl, extra_shift_in_time)
    # convert to mmol/L
    col_dtype = df[glucose_col].dtype
    if not pd.api.types.is_numeric_dtype(col_dtype):
        if pd.api.types.is_string_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype):
            df[glucose_col] = pd.to_numeric(df[glucose_col].str.replace(",", "."), errors="coerce")
        else:
            raise TypeError(
                f"Unsupported dtype '{col_dtype}' for column '{glucose_col}'. Please ensure it is a string or numeric."
            )
    df[glbl] = (
        df[glucose_col]
        if unit == Units.mmolL.value
        else convert_to_mmoll(df[glucose_col], from_unit=unit)
    )

    # index by time and keep time column
    df["idx"] = df[tlbl]
    df = df.set_index("idx").sort_index()
    # interpolate and smoothen glucose
    if interpolate:
        df[glbl] = df[glbl].rolling(window=rolling_avg).mean()
        df[glbl] = df[glbl].ffill().bfill()
        df[glbl] = df[glbl].interpolate(method=interp_met, order=interp_ord, limit_direction="both")
        df = df[df[glbl].between(0, 30)]
    return df


def add_shifted_time(df: pd.DataFrame, tlbl: str, dlbl: str, shift_hours_back: int):
    """Adds shifted time values.
    These are used by certain utility functions to make calculations faster,
    and to include nighttime glucose in certain calculations.
    See the shifted time values chapter in the glucose documentation for more.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        tlbl (str): the timestamp column name.
        dlbl (str): the date column name.
        shift_hours_back (int): how many hours for the shifted time values.
            This value will be substracted (if negative, shift will happen forward).
    """
    shift_tlbl = f"shifted_{tlbl}"
    shift_dlbl = f"shifted_{dlbl}"

    df[shift_tlbl] = df[tlbl].map(lambda x: x - tdel(hours=shift_hours_back))
    df[shift_dlbl] = df[shift_tlbl].dt.date
    df[f"{shift_dlbl}_str"] = df[shift_dlbl].map(
        lambda x: x.strftime(DEFAULT_OUT_DATE_FMT) if pd.notna(x) else ""
    )
    df[f"shifted_{_HOUR_COL}"] = df[shift_tlbl].dt.hour
    df[f"shifted_{_DAYOFWEEK_COL}"] = df[shift_tlbl].dt.weekday
    df[f"shifted_{_WEEKDAY_COL}"] = df[f"shifted_{_DAYOFWEEK_COL}"].map(WEEKDAY_MAP)
    df[f"shifted_{_ISWEEKEND_COL}"] = df[f"shifted_{_DAYOFWEEK_COL}"].map(is_weekend)
    return df


# Properties and Stats
def set_derivative(
    df: pd.DataFrame,
    glucose_col: str,
    timestamp_col: str,
    *,
    cols: Optional[DerivativeCols] = None,
    time_unit: str = "s",
    sort_by_time: bool = True,
    copy: bool = False,
) -> pd.DataFrame:
    """
    Add derivative columns to df:
      - dG: glucose difference
      - dt: time delta in seconds/minutes
      - dG/dt: rate of change

    By default mutates df (copy=False). Set copy=True to return a new DataFrame.
    Args:
        df: Input Glucose DataFrame.
        glucose_col: Glucose column name.
        timestamp_col: Timestamp column name.
        cols: Optional names for the derivative columns.
        time_unit: Time unit for dt and dg/dt: "s" (seconds) or "min" (minutes).
        sort_by_time: If True, sort by timestamp before computing derivative.
        copy: If True, return a new DataFrame; else mutate df in place.
    Returns:
        DataFrame with derivative columns added.
    """
    out = df.copy() if copy else df
    dg, dt, dgdt = compute_derivative(
        out,
        glucose_col,
        timestamp_col,
        time_unit=time_unit,
        sort_by_time=sort_by_time,
    )
    
    if cols is None:
        dg_col = globals().get("_DG_COL", "dG")
        dt_col = globals().get("_DT_COL", f"dt_{time_unit}")
        dgdt_col = globals().get("_DGDT_COL", f"dGdt_per_{time_unit}")
    else:
        dg_col, dt_col, dgdt_col = cols.dg, cols.dt, cols.dgdt

    # Ensure columns exist
    for c in (dg_col, dt_col, dgdt_col):
        if c not in out.columns:
            out[c] = np.nan
    # If compute_derivative sorted, indices differ, align back onto out by timestamp order
    if sort_by_time:
        ts = out[timestamp_col]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, errors="raise")
        order = ts.argsort(kind="mergesort") # safe with duplicates
    else:
        order = np.arange(len(out))

    out.iloc[order, out.columns.get_loc(dg_col)] = dg.to_numpy()
    out.iloc[order, out.columns.get_loc(dt_col)] = dt.to_numpy()
    out.iloc[order, out.columns.get_loc(dgdt_col)] = dgdt.to_numpy()
    return out


@dataclass(frozen=True)
class DerivativeCols:
    dg: str
    dt: str
    dgdt: str


def compute_derivative(
    df: pd.DataFrame,
    glucose_col: str,
    timestamp_col: str,
    *,
    time_unit: str = "s",
    sort_by_time: bool = True,
    require_monotonic: bool = False,
    max_dt: Optional[float] = None,          # e.g. 10*60 for 10 minutes if time_unit="s"
    gap_factor: float = 3.0,                 # used only when max_dt is None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # column checks
    missing = [c for c in (glucose_col, timestamp_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    work = df[[timestamp_col, glucose_col]].copy()

    if not pd.api.types.is_datetime64_any_dtype(work[timestamp_col]):
        work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="raise")

    if sort_by_time:
        work = work.sort_values(timestamp_col, kind="mergesort")

    if require_monotonic and not work[timestamp_col].is_monotonic_increasing:
        raise ValueError(f"'{timestamp_col}' must be monotonic increasing.")

    dg = work[glucose_col].astype("float64").diff()

    dt = work[timestamp_col].diff()
    dt_s = dt.dt.total_seconds()

    if time_unit not in {"s", "min"}:
        raise ValueError("time_unit must be 's' or 'min'")

    dt_out = dt_s if time_unit == "s" else (dt_s / 60.0)

    # ---- GAP HANDLING ----
    # Decide a gap threshold. If not provided, infer it from typical sampling interval.
    if max_dt is None:
        typical = dt_out[(dt_out > 0) & dt_out.notna()].median()
        # fallback if median can't be computed
        if pd.isna(typical):
            typical = 0
        max_dt = typical * gap_factor if typical > 0 else None

    invalid = (dt_out <= 0) | dt_out.isna()
    if max_dt is not None:
        invalid = invalid | (dt_out > max_dt)

    with np.errstate(divide="ignore", invalid="ignore"):
        dgdt = dg / dt_out

    # Break derivatives on invalid or gap points
    dgdt = dgdt.mask(invalid)
    dt_out = dt_out.mask(invalid)

    # Optional: also break dG at gap points so plots/logic don't treat it as a jump
    dg = dg.mask(invalid)

    return dg.rename("dG"), dt_out.rename(f"dt_{time_unit}"), dgdt.rename(f"dGdt_per_{time_unit}")


def set_auc(
    df: pd.DataFrame,
    glucose_col: str,
    timestamp_col: str,
    glucose_auc_lim: float,
    *,
    copy: bool = False,
    sort_by_time: bool = True,
) -> pd.DataFrame:
    """
    Add AUC-like (area above baseline) columns.

    Computes per-interval "excess area" using the rectangle rule:
        area_i = max(g_i - baseline, 0) * dt_i

    Baselines:
      - mean baseline: baseline = mean(glucose)
      - limit baseline: baseline = glucose_auc_lim
      - min baseline: baseline = min(glucose)

    Notes:
      - Requires dt (time delta). If missing, will compute derivative (which sets dt).
      - dt is assumed to be in seconds unless your _DT_COL is configured otherwise.

    Args:
        df: Input DataFrame.
        glucose_col: Glucose column name.
        timestamp_col: Timestamp column name.
        glucose_auc_lim: Baseline limit for the "limit" AUC column.
        copy: If True, return a copy; otherwise mutate df.
        sort_by_time: Ensure time ordering when computing dt/derivative if needed.

    Returns:
        DataFrame with added columns:
          - _AUC_COL: area above mean baseline
          - _AUCLIM_COL: area above glucose_auc_lim baseline
          - _AUCMIN_MIN: area above min baseline
    """
    out = df.copy() if copy else df

    # ensure we have dt available
    if _DT_COL not in out.columns:
        out = set_derivative(
            out,
            glucose_col=glucose_col,
            timestamp_col=timestamp_col,
            sort_by_time=sort_by_time,
            copy=False,
        )

    # validate required inputs
    if glucose_col not in out.columns:
        raise KeyError(f"Missing column '{glucose_col}'")
    if _DT_COL not in out.columns:
        raise KeyError(f"Missing required dt column '{_DT_COL}' (expected set by set_derivative)")

    g = out[glucose_col].astype("float64")
    dt = out[_DT_COL].astype("float64")

    # avoid negative/zero dt contributions (duplicates / out-of-order)
    dt = dt.where(dt > 0)

    mean_g = float(g.mean(skipna=True))
    min_g = float(g.min(skipna=True))

    # Excess above baseline (clipped at 0); NaNs propagate naturally
    excess_mean = (g - mean_g).clip(lower=0.0)
    excess_lim = (g - float(glucose_auc_lim)).clip(lower=0.0)
    excess_min = (g - min_g).clip(lower=0.0)

    out[_AUC_COL] = excess_mean * dt
    out[_AUCLIM_COL] = excess_lim * dt
    out[_AUCMIN_MIN] = excess_min * dt

    return out


def get_area_under_curve(
    df: pd.DataFrame,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    timestamp_col: Optional[str] = None,
    auc_cols: Optional[Iterable[str]] = None,
    require_auc: bool = True,
    normalize_by_time: bool = False,
) -> Dict[str, float]:
    """
    Compute regional AUC metrics by summing precomputed AUC columns.

    This function does NOT recompute AUC per interval.
    It only aggregates existing AUC columns (e.g. from set_auc).

    Args:
        df: DataFrame containing AUC columns.
        start: Optional start timestamp (inclusive).
        end: Optional end timestamp (exclusive).
        timestamp_col: Required if start/end are provided.
        auc_cols: AUC columns to aggregate. Defaults to known AUC columns.
        require_auc: If True, raises if AUC columns are missing.
        normalize_by_time: If True, divide AUC by region duration (seconds).

    Returns:
        Dict[str, float]: aggregated AUC metrics.
    """

    if auc_cols is None:
        auc_cols = [
            c for c in (_AUC_COL, _AUCLIM_COL, _AUCMIN_MIN)
            if c in df.columns
        ]

    if require_auc:
        missing = [c for c in auc_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing AUC columns {missing}. "
                f"Run set_auc() first."
            )

    view = df

    # Time slicing
    if start is not None or end is not None:
        if timestamp_col is None:
            raise ValueError("timestamp_col is required when using start/end")

        ts = view[timestamp_col]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, errors="raise")

        mask = pd.Series(True, index=view.index)
        if start is not None:
            mask &= ts >= start
        if end is not None:
            mask &= ts < end

        view = view.loc[mask]

    # Aggregate
    out = {
        col: float(view[col].sum(skipna=True))
        for col in auc_cols
    }

    if normalize_by_time:
        if _DT_COL not in view.columns:
            raise KeyError("dt column required for normalization")

        duration = view[_DT_COL].sum(skipna=True)
        if duration > 0:
            out = {k: v / duration for k, v in out.items()}
        else:
            out = {k: np.nan for k in out}

    return out


def get_properties(
    df: pd.DataFrame,
    glbl: str = GLUCOSE_COL,
    tlbl: str = TIMESTAMP_COL,
    glim: float = DEFAULT_GLUC_LIMIT,
    *,
    copy: bool = False,
    sort_by_time: bool = True,
) -> pd.DataFrame:
    """Adds the derivative columns and area under the curve columns to the dataframe.
    Assumes timestamps are parseable as datetimes and data is prepared (e.g. by using prepare_glucose).

    Args:
        df (pd.DataFrame): _description_
        glbl (str, optional): _description_. Defaults to GLUCOSE_COL.
        tlbl (str, optional): _description_. Defaults to TIMESTAMP_COL.
        glim (float, optional): _description_. Defaults to GLUCOSE_LIMIT_DEFAULT.
        copy (bool): If True, operate on and return a copy of the dataframe.
            If False, mutate the dataframe in place. Defaults to False.
        sort_by_time (bool): If True, ensure data is sorted by timestamp before
            computing derivatives and AUC. Defaults to True.

    Returns:
        pd.DataFrame: The dataframe with derivative and AUC columns added or updated.

    Logs warnings if derivative or AUC columns already exist and are recomputed.
    """
    out = df.copy() if copy else df

    # Derivative columns
    derivative_cols = {_DG_COL, _DT_COL, _DGDT_COL}
    if derivative_cols.intersection(out.columns):
        logger.warning(
            "Derivative columns already exist (%s). Recomputing them.",
            derivative_cols.intersection(out.columns),
        )

    if not derivative_cols.issubset(out.columns):
        out = set_derivative(
            out,
            glucose_col=glbl,
            timestamp_col=tlbl,
            sort_by_time=sort_by_time,
            copy=False,
        )

    # AUC columns
    auc_cols = {_AUC_COL, _AUCLIM_COL, _AUCMIN_MIN}
    if auc_cols.intersection(out.columns):
        logger.warning(
            "AUC columns already exist (%s). Recomputing them.",
            auc_cols.intersection(out.columns),
        )

    if not auc_cols.issubset(out.columns):
        out = set_auc(
            out,
            glucose_col=glbl,
            timestamp_col=tlbl,
            glucose_auc_lim=glim,
            sort_by_time=sort_by_time,
            copy=False,
        )

    return out


def convert_to_mmoll(g: float, from_unit: str) -> float:
    """Converts a glucose value to mmol/L

    Args:
        g (float): glucose value in original unit
        from_unit (str): unit to convert from

    Raises:
        NotImplementedError: if the unit is not implemented.
            Implemented units are found in the Units Enum under utils.py

    Returns:
        float: converted glucose value
    """
    if from_unit in implemented_units:
        return g * units_to_mmolL_factor[from_unit]
    raise NotImplementedError(ERR_NOT_IMPLEMENTED)


def autodetect_unit(glucose_values: pd.Series) -> str:
    """Autodetects the Glucose unit as one of:
    - mmol/L
    - mg/dL
    - g/L
    To do so it selects a sample of 100 without replacement
    if the input contains more than 100. Otherwise it selects
    a sample of the input size.
    Warning: this may result in unexpected behavior if the autodetected unit is wrong.

    Args:
        glucose_values (pd.Series): glucose values to detect unit from

    Returns:
        str: the detected glucose unit.
    """
    logger.warning("Using unit autodetection. This may result in unexpected behavior.")
    cast_glucose_sample = pd.to_numeric(
        (glucose_values.sample(n=min(100, len(glucose_values)), replace=False)), errors="coerce"
    )
    m = cast_glucose_sample.mean()
    if m > 33:
        return Units.mgdL.value
    if m > 6:
        return Units.mmolL.value
    if cast_glucose_sample.std() < 0.4:
        return Units.gL.value
    return Units.mmolL.value


# Plotting
@autoplot
def plot_glucose( # pylint: disable=too-many-arguments,too-many-positional-arguments
    df: pd.DataFrame,
    glbl: str = GLUCOSE_COL,
    tlbl: str = TIMESTAMP_COL,
    from_time: Optional[GeneralDateType] = None,
    to_time: Optional[GeneralDateType] = None,
    title: Optional[str] = None,
    label: str = "Glucose in mmol/L",
    show_full_day: bool = False,
    **kwargs, # pylint: disable=unused-argument
):
    """Plots the glucose curve for a given dataframe, and optional time frame

    This function uses `@autoplot`. To prevent it from automatically showing
    the plot, call it with `autoplot=False`.

    Args:
        df (pd.DataFrame): The glucose dataframe.
        glbl (str, optional): The glucose column name. Defaults to GLUCOSE_COL.
        tlbl (str, optional): The timestamp column name. Defaults to TIMESTAMP_COL.
        from_time (Union[str, pd.Timestamp, date_type], optional): time or date to start plotting from.
            Could be a string or timestamp or date.
            Defaults to None.
        to_time (Union[str, pd.Timestamp, date_type], optional): time or date to stop plotting at.
            Could be a string or timestamp or date.
            Defaults to None.
        title (Optional[str], optional): title of the plot.
            If None, it generates automatically showing dates.
            Defaults to None.
        label (str, optional): label of the lineplot.
            Defaults to Glucose in mmol/.
        show_full_day (bool, optional): if True, shows full day boundaries even if from_time and to_time are within the same day.
            Defaults to False.

    Keyword arguments
        autoplot (bool, optional): if True, this automatically shows the plot and makes it more readable.
            This can be disabled for example to use this function along with other plots.
            Defaults to True.
        show_legend (bool, optional): if True, shows the legend on the plot.
            Defaults to True.

    Raises:
        KeyError: if the glucose column is not in the glucose dataframe
    """
    if glbl not in df.columns:
        raise KeyError(f"Glucose column '{glbl}' is not in the DataFrame.")
    if tlbl not in df.columns:
        raise KeyError(f"Timestamp column '{tlbl}' is not in the DataFrame.")

    # build datetime timestamp series (don’t mutate caller df)
    ts = df[tlbl]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce")
    if ts.isna().all():
        raise ValueError(f"Column '{tlbl}' could not be parsed as datetime.")

    # filter by time using the timestamp column (not the index)
    from_ts = pd.to_datetime(from_time) if from_time is not None else None
    to_ts = pd.to_datetime(to_time) if to_time is not None else None

    mask = ts.notna()
    if from_ts is not None and to_ts is not None and from_ts == to_ts:
        # same date or same timestamp
        day_start = from_ts.normalize()
        day_end = day_start + pd.Timedelta(days=1)
        mask &= (ts >= day_start) & (ts < day_end)
    else:
        if from_ts is not None:
            mask &= ts >= from_ts
        if to_ts is not None:
            mask &= ts <= to_ts
    
    plot_df = df.loc[mask].copy()
    ts_plot = ts.loc[mask]

    if plot_df.empty:
        raise ValueError("No data to plot in the requested time range.")

    # sort by time (important for line plots / fill_between)
    order = ts_plot.argsort(kind="mergesort")
    plot_df = plot_df.iloc[order]
    ts_plot = ts_plot.iloc[order]

    ax = plt.gca()

    x_left = from_ts if from_ts is not None else ts_plot.iloc[0]
    x_right = to_ts if to_ts is not None else ts_plot.iloc[-1]

    if show_full_day:
        day_start = pd.to_datetime(x_left).normalize()
        day_end = day_start + pd.Timedelta(days=1)
        x_left, x_right = day_start, day_end
    else:
        # small padding so the line doesn't touch the plot border
        span = (pd.to_datetime(x_right) - pd.to_datetime(x_left))
        pad = max(pd.Timedelta(minutes=5), span * 0.02)
        x_left, x_right = pd.to_datetime(x_left) - pad, pd.to_datetime(x_right) + pad


    # day boundaries: prefer existing date col if present, otherwise derive
    if "date" in plot_df.columns and pd.api.types.is_datetime64_any_dtype(plot_df["date"]):
        days = plot_df["date"].dropna().unique()
    else:
        days = ts_plot.dt.normalize().dropna().unique()

    for d in days:
        if x_left <= d <= x_right:
            plt.axvline(d, color="brown", linestyle="--", alpha=0.5)

    medval = plot_df[glbl].median()
    minval = plot_df[glbl].min()
    maxval = plot_df[glbl].max()

    plt.axhline(medval, color="red", linestyle="--", alpha=0.5,
                label=f"Glucose Median value: ({medval:.2f} mmol/L)")
    plt.axhline(minval, color="orange", linestyle="--", alpha=0.5,
                label=f"Glucose Minimum value: ({minval:.2f} mmol/L)")
    plt.axhline(maxval, color="orange", linestyle="--", alpha=0.5,
                label=f"Glucose Maximum value: ({maxval:.2f} mmol/L)")

    plt.plot(ts_plot, plot_df[glbl], label=label)

    plt.fill_between(
        ts_plot,
        plot_df[glbl],
        medval,
        where=(plot_df[glbl] > medval),
        alpha=0.2,
        interpolate=True,
        label="Glucose above median",
    )
    ax.set_xlim(x_left, x_right)
    plt.xlabel("Time")
    plt.ylabel("Glucose")

    if title:
        plt.title(title)
    else:
        start = ts_plot.iloc[0].date()
        end = ts_plot.iloc[-1].date()
        plt.title(f"Glucose variation from {start} to {end}")


def plot_trend_by_hour(df: pd.DataFrame,
                       glbl: str = GLUCOSE_COL): # pylint: disable=unused-argument
    """Plots the glucose hourly trend as an averaged curve for each hour
    with percentile distributions.

    This function uses `@autoplot`. To prevent it from automatically showing
    the plot, call it with `autoplot=False`.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        glbl (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
    """
    plot_percentiles(df, stat_col=glbl, group_by_col=_HOUR_COL, percentiles=[0.01, 0.05])


def plot_trend_by_weekday(df: pd.DataFrame, glbl=GLUCOSE_COL): # pylint: disable=unused-argument
    """Plots the glucose trend for each weekday (Monday to Sunday) as
    a box plot for each weekday.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        glbl (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
    """
    plot_compare_by(
        df=df,
        glbl=glbl,
        compare_by=_WEEKDAY_COL,
        outliers=False,
        label_map=None,
        method="box",
        sort_vals=False,
        show_legend=False
    )


def plot_trend_by_day(df: pd.DataFrame, glbl=GLUCOSE_COL): # pylint: disable=unused-argument
    """Plots the glucose trend for each weekday (Monday to Sunday) as
    a box plot for each day.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        glbl (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
    """
    plot_compare_by(
        df=df,
        glbl=glbl,
        compare_by=_DATE_COL,
        outliers=False,
        label_map=None,
        method="box",
        sort_vals=False,
        show_legend=False
    )


@autoplot
def plot_percentiles( # pylint: disable=too-many-arguments,too-many-positional-arguments
    df: pd.DataFrame,
    stat_col: str,
    percentiles: list[float],
    group_by_col: str = _HOUR_COL,
    color: str = "green",
    title: str = None,
    **kwargs, # pylint: disable=unused-argument
):
    """Groups glucose by a column column and plots percentiles of glucose.
    Percentiles are plotted using an area color between the main curve and each percentile.

    This function uses `@autoplot`. To prevent it from automatically showing
    the plot, call it with `autoplot=False`.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        stat_col (str): the glucose column name or column for which to get stats (Y-axis).
        percentiles (list[float]): a list of percentiles to plot (each value between 0 and 1)
        group_by_col (str, optional): the name of the column to group values by (X-axis).
            Defaults to _HOUR_COL.
        color (str, optional): the name of the color to use for the percentiles area.
            Defaults to 'green'.
        title (str, optional): the title of the plot.
            Defaults to None.
    """
    stats_df = df.pipe(
        get_stats, stats_cols=[stat_col], group_by_col=group_by_col, percentiles=percentiles
    )
    med = stats_df[(stat_col, "50%")]
    med.plot(label="50%")
    perc_l = [stats_df[(stat_col, f"{int(p*100)}%")] for p in percentiles]
    perc_h = [stats_df[(stat_col, f"{int((1-p)*100)}%")] for p in percentiles]

    for i, p in enumerate(percentiles):
        plt.fill_between(
            med.index,
            perc_l[i],
            perc_h[i],
            color=color,
            alpha=0.2,
            label=f"{int(p*100)}-{int((1-p)*100)}th percentile",
        )
    if not title:
        title = (
            f"Trend of {stat_col} for the percentiles: {', '.join([str(int(i*100)) for i in percentiles])}"
            f" as well as {', '.join([str(int((1-i)*100)) for i in percentiles])}"
        )
    plt.title(title)
    plt.xlabel(group_by_col)
    plt.ylabel(stat_col)


def plot_sleep_trends(
    df: pd.DataFrame,
    glbl: str = GLUCOSE_COL,
    sleep_time_filter_col: str = f"shifted_{_HOUR_COL}",
    sleep_time_hour: int = 24 - (_DEFAULT_SHIFT_HOURS + 2),
):
    """Plots sleep

    Args:
        df (pd.DataFrame): _description_
        glbl (str, optional): _description_. Defaults to GLUCOSE_COL.
        sleep_time_filter_col (str, optional): Column to use for filtering sleep time.
        Defaults to f'shifted_{_HOUR_COL}'.
        sleep_time_hour (int, optional): Total sleep hours to consider.
        Defaults to 24-(_default_shift_hours + 2).
    """
    # filter sleep glucose data
    gdf = df[df[sleep_time_filter_col] >= sleep_time_hour]
    # make new plotting time
    gdf.loc[:, "sleep_hours"] = gdf[sleep_time_filter_col] - gdf[sleep_time_filter_col].min()
    plot_percentiles(
        df=gdf,
        stat_col=glbl,
        group_by_col="sleep_hours",
        percentiles=[0.01, 0.05],
        label="Hourly trend of Glucose during Sleep",
        autoplot=False,
    )
    plt.ylabel("Glucose during sleep")
    plt.xlabel("Hours of sleep (from 0-8)")
    end_plot()
    plot_compare_by(
        df=gdf,
        glbl=glbl,
        compare_by=f"shifted_{_DATE_COL}_str",
        outliers=False,
        method="box",
        sort_vals=False,
        title="Daily trend of Glucose during Sleep",
    )
    plt.ylabel("Glucose during sleep")
    plt.xlabel("Day (sleep from evening of this day)")
    end_plot()


# TODO not looking great
@autoplot
def plot_day_curve(
    df: pd.DataFrame,
    day_str: str,
    glbl: str = GLUCOSE_COL,
    tlbl: str = TIMESTAMP_COL,
    extended=False,
):
    """Plots a glucose curve for a specific day.
    Can be extended to show the following night as well.

    Args:
        df (pd.DataFrame): Dataframe containing glucose
        day_str (str): Day string TODO: verify that this can be other than str
        glbl (str, optional): Name of the glucose column. Defaults to GLUCOSE_COL.
        tlbl (str, optional): Name of the timestamp column. Defaults to TIMESTAMP_COL.
        extended (bool, optional): If true, shows the extended day with the
            nighttime (sleeping period). Defaults to False.
    """
    plot_df = df.loc[day_str]
    # plt.axvline(d, color='brown', linestyle='--', alpha=0.5)
    plt.axhline(plot_df[glbl].mean(), color="red", linestyle="--", alpha=0.5, label="Day Average")
    plt.axhline(
        df[glbl].mean(), color="brown", linestyle="--", alpha=0.2, label="Your general Average"
    )
    plt.axhline(
        df[glbl].mean() + 2, color="green", linestyle="--", alpha=0.5, label="Recommended range"
    )
    plt.axhline(df[glbl].mean() - 1, color="green", linestyle="--", alpha=0.5)
    plt.plot(plot_df[tlbl], plot_df[glbl])

    if extended:
        xt = df.loc[
            day_str : (dt.strptime(day_str, "%Y-%m-%d") + tdel(days=1, hours=7)).strftime(
                "%Y-%m-%d %H"
            )
        ]
        plt.plot(xt[tlbl], xt[glbl], color="brown", alpha=0.5, label="sleep")


def _normalize_percentiles(percentiles: list[float]) -> list[float]:
    """
    Ensure percentile symmetry: for each p include (1 - p),
    remove duplicates, keep values in [0, 1], and sort.
    """
    if not percentiles:
        return percentiles

    pset = set()

    for p in percentiles:
        if not 0 <= p <= 1:
            raise ValueError(f"Percentile {p} must be between 0 and 1")
        pset.add(round(p, 10))
        pset.add(round(1 - p, 10))

    return sorted(pset)


def get_stats(
    df: pd.DataFrame,
    stats_cols: Union[list, str],
    group_by_col: str = None,
    percentiles: Optional[list[float]] = None,
):
    """Get descriptive statistics about specific columns of a dataframe.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        stats_cols (Union[list[str], str]): the glucose column name, or a column name,
            or a list of column names for which to get stats.
        group_by_col (str, optional): the name of the column to group values by.
            Defaults to None.
        percentiles (Optional[list[float]], optional): a list of percentiles to plot
            (each value between 0 and 1). Defaults to None.

    Returns:
        pd.Series or pd.DataFrame: descriptive statistics grouped by the given column
    """
    if percentiles is not None:
        percentiles = _normalize_percentiles(percentiles)
    if group_by_col:
        return df.groupby(group_by_col)[stats_cols].describe(percentiles=percentiles)
    return df[stats_cols].describe(percentiles=percentiles)


@autoplot
def plot_compare_by( # pylint: disable=too-many-arguments,too-many-positional-arguments
    df: pd.DataFrame,
    glbl: str = GLUCOSE_COL,
    compare_by: str = _WEEKDAY_COL,
    outliers: bool = False,
    label_map: Union[Callable, Dict] = None,
    method: str = "box",
    sort_vals: bool = False,
    title: Optional[str] = None,
    **kwargs, # pylint: disable=unused-argument
):
    """Compares glucose values by a given field, for example by weekday.
    
    This function uses `@autoplot`. To prevent it from automatically showing
    the plot, call it with `autoplot=False`.

    Args:
        df (pd.DataFrame): dataframe with values to be compared and the comparison field.
        glbl (str, optional): label of the box plot values in the dataframe (Y-axis).
            Defaults to GLUCOSE_COL.
        compare_by (str, optional): field to compare by (X-axis).
            Defaults to _WEEKDAY_COL.
        outliers (bool, optional): wether to show or not show outliers.
            Defaults to False.
        label_map (Union[Callable, Dict], optional): lambda function to map
            the unique values of the compare_by field to some labels. Defaults to None.
        method (str, optional): the method used for comparison. Currently only supports
            box plots as 'box'. Defaults to 'box'.
        sort_vals (bool, optional): wether or not to sort values plotted.
            Defaults to False.
        title (Optional[str], optional): title of the plot, if None a default will be generated.
            Defaults to None.

    Raises:
        NotImplementedError: if the method used for comparison is not implemented.
    """
    all_vals = df[compare_by].unique()
    if sort_vals:
        all_vals.sort()
    if method == "box":
        plt.boxplot(
            [df[df[compare_by] == i][glbl].dropna() for i in all_vals],
            labels=all_vals if label_map is None else [label_map(i) for i in all_vals],
            showfliers=outliers
        )
    else:
        raise NotImplementedError(
            f"Method {method} not implemented for comparison please use one of: 'box'"
        )
    plt.ylabel(f"Trend for {glbl}")
    plt.xlabel(f"{compare_by}")
    if not title:
        title = (
            f"Comparing {glbl} by {compare_by}. "
            f"Outliers are {'shown' if outliers else 'not shown'}."
        )

    plt.title(title)


def get_response_bounds( # pylint: disable=too-many-arguments,too-many-positional-arguments
    df: pd.DataFrame,
    event_time: pd.Timestamp,
    pre_pad_min: int = 20,
    post_pad_min: int = 0,
    resp_time_min: int = 120,
    glbl: str = GLUCOSE_COL,
    t_lbl: str = TIMESTAMP_COL,
):
    """Finds the boundaries of a glucose response (used for plotting) by:
    - Finding the nearest time with a glucose value, 'g_event_time', to the event time 'event_time'.
    - Setting the start of the glucose bounds to: 'g_event_time - pre_pad_min'
    - Setting the end to: 'g_event_time + resp_time_min + post_pad_min'
    Assumes the glucose dataframe is indexed by time

    Args:
        df (pd.DataFrame): The glucose dataframe.
        event_time (pd.Timestamp): Time of the event we want to investigate.
        pre_pad_min (int, optional): Number of minutes used for padding the start time boundary. Defaults to 20.
        post_pad_min (int, optional): Number of minutes used for padding the end time boundary. Defaults to 0.
        resp_time_min (int, optional): The approximate number of minutes it takes for a glucose response to this event.
            Defaults to 120.
        glbl (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
        t_lbl (str, optional): the timestamp column name. Defaults to TIMESTAMP_COL.

    Returns:
        Tuple(datetime, datatime, datetime): A tuple containing:
            - the start time.
            - the end time.
            - the nearest time to the event time with a glucose value in the dataframe.
    # TODO: find nearest takes a lot of time, use something easier
    """
    g_event_time = find_nearest(df, event_time, glbl)
    start = g_event_time - tdel(minutes=pre_pad_min)
    end = g_event_time + tdel(minutes=resp_time_min) + tdel(minutes=post_pad_min)
    return start, end, g_event_time


def plot_response_at_time( # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    glucose_df: pd.DataFrame,
    event_time: pd.Timestamp,
    event_title: Optional[str] = None,
    pre_pad_min: int = 20,
    post_pad_min: int = 0,
    resp_time_min: int = 120,
    glbl: str = GLUCOSE_COL,
    t_lbl: str = TIMESTAMP_COL,
    auc_lim: int = DEFAULT_GLUC_LIMIT,
    show_auc: bool = True,
    use_local_min: bool = False,
):
    """Plots the glucose response around a specific event given by its event time.
    Estimates the start and end of the glucose response to the event.
    TODO: clean inputs AUC/pre-pad, have multi-options large, medium, small

    Args:
        glucose_df (pd.DataFrame): the glucose dataframe.
        event_time (pd.Timestamp): the time of the event to investigate.
        event_title (Optional[str], optional): the name/title of the event. Defaults to None.
        pre_pad_min (int, optional): number of minutes minutes used for padding
            the start of the glucose response. Defaults to 20.
        post_pad_min (int, optional): number of minutes minutes used for padding
            the end of the glucose response. Defaults to 0.
        resp_time_min (int, optional): Approximate number of minutes it takes for a glucose response to this event.
            Defaults to 120.
        glbl (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
        t_lbl (str, optional): the timestamp column name. Defaults to TIMESTAMP_COL.
        auc_lim (int, optional): the limit above which to show the area under the curve
            (if 'use_local_min' this will be overriden). Defaults to DEFAULT_GLUC_LIMIT.
        show_auc (bool, optional): whether or not to show the area under the curve. Defaults to True.
        use_local_min (bool, optional): whether or not to use the local glucose mean to plot the area under the curve.
            Overrides 'auc_lim'. Defaults to False.
    """
    s, e, t = get_response_bounds(
        glucose_df, event_time, pre_pad_min, post_pad_min, resp_time_min, glbl=glbl, t_lbl=t_lbl
    )
    plot_df = glucose_df.loc[s:e][glbl]
    plt.plot(plot_df)
    if show_auc:
        alim = auc_lim if not (use_local_min) else plot_df.mean()
        lim_df = plot_df.map(lambda x: x if x > alim else alim)
        plt.gca()
        plt.axhline(alim, color="red", label="limit", linestyle="--", alpha=0.3)
        plt.fill_between(
            lim_df.index,
            lim_df,
            [alim for a in lim_df.index],
            color="green",
            alpha=0.1,
            label="Estimated glucose quantity consumed",
        )

    plt.axvline(t, color="black", label="Event time", linestyle="--", alpha=0.1)
    if event_title:
        plt.title(event_title)


# Outputs
# - Write to a file
# - Write image plot or responses
def write_glucose(gdf: pd.DataFrame, output_file: str):
    """Writes glucose to a csv file

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        output_file (str): the output file path.
    """
    logger.info("Writing glucose data to %s", output_file)
    gdf.to_csv(output_file)


# Day/Week Metrics
_summary_cols = [GLUCOSE_COL, _AUC_COL, _DG_COL, _DT_COL, _DGDT_COL]


def get_metrics_by_day(
    gdf: pd.DataFrame,
    day_col: str = _DATE_COL,
    percentiles: list | None = None,
    summary_cols: list | None = None,
):
    """
    Get metrics and statistics by day related to specific columns of a dataframe.

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        day_col (str, optional): the day column name. Defaults to _HOUR_COL.
        percentiles (Optional[list[float]], optional): a list of percentiles to plot
            (each value between 0 and 1). Defaults to None.
        summary_cols (Union[list[str], str]): the glucose column name, or a column name,
            or a list of column names for which to get stats. Defaults to _summary_cols.

    Returns:
        pd.Series or pd.DataFrame: descriptive statistics grouped hour
    """
    if summary_cols is None:
        summary_cols = _summary_cols
    return get_stats(gdf, stats_cols=summary_cols, percentiles=percentiles, group_by_col=day_col)


def get_metrics_by_hour(
    gdf: pd.DataFrame,
    hour_col: str = _HOUR_COL,
    percentiles: Optional[list[float]] = None,
    summary_cols: Union[list[str], str] | None = None,
):
    """Get metrics and statistics by hour related to specific columns of a dataframe.

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        hour_col (str, optional): the hour column name. Defaults to _HOUR_COL.
        percentiles (Optional[list[float]], optional): a list of percentiles to plot
            (each value between 0 and 1). Defaults to None.
        summary_cols (Union[list[str], str]): the glucose column name, or a column name,
            or a list of column names for which to get stats. Defaults to _summary_cols.

    Returns:
        pd.Series or pd.DataFrame: descriptive statistics grouped hour
    """
    if summary_cols is None:
        summary_cols = _summary_cols
    return get_stats(gdf, stats_cols=summary_cols, percentiles=percentiles, group_by_col=hour_col)


def get_metrics(
    gdf: pd.DataFrame,
    percentiles: Optional[list[float]] = None,
    summary_cols: Union[list[str], str] | None = None,
    group_by_col: Optional[str] = None,
):
    """Get metrics and statistics related to specific columns of a dataframe.

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        percentiles (Optional[list[float]], optional): a list of percentiles to plot
            (each value between 0 and 1). Defaults to None.
        summary_cols (Union[list[str], str] | None): the glucose column name, or a column name,
            or a list of column names for which to get stats. Defaults to _summary_cols.
        group_by_col (str, optional): the name of the column to group values by.
            Defaults to None.

    Returns:
       pd.Series or pd.DataFrame: descriptive statistics grouped by the given column
    """
    if summary_cols is None:
        summary_cols = _summary_cols
    return get_stats(
        gdf, stats_cols=summary_cols, percentiles=percentiles, group_by_col=group_by_col
    )


def describe_glucose(
    df: pd.DataFrame,
    glucose_col: str = GLUCOSE_COL,
    timestamp_col: str = TIMESTAMP_COL,
    default_unit: str = Units.mmolL.value,
):
    """Describes the glucose DataFrame by providing a summary with the total number of days, start and end dates,
    and overall summary statistics of glucose including the unit of measurement.

    Args:
        df (pd.DataFrame): The glucose dataframe.
        glucose_col (str): Column name for glucose data.
        timestamp_col (str): Column name for timestamp data.
        default_unit (str): Default unit for glucose measurements.
    """
    console = Console()
    # Ensure timestamps are converted to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    start_date = df[timestamp_col].min()
    end_date = df[timestamp_col].max()
    total_days = (end_date - start_date).days + 1

    # Printing the overall summary
    console.print("[bold magenta]Glucose Data Summary[/bold magenta]")
    console.print(
        f"• Total number of days in the data: [bold green]{total_days}[/bold green]\n"
        f"• Starting at: [bold green]{start_date.strftime('%Y-%m-%d %H:%M')}[/bold green]\n"
        f"• Ending at: [bold green]{end_date.strftime('%Y-%m-%d %H:%M')}[/bold green]"
    )

    glucose_stats = get_stats(df, glucose_col)
    table = Table(title="Glucose Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Measure", style="dim")
    table.add_column(f"Value in {default_unit}")

    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        value = glucose_stats.get(stat, "N/A")
        table.add_row(stat.capitalize(), f"{value:.2f}")

    console.print(table)
    console.print(f"Columns in the data: [bold yellow]{', '.join(df.columns)}[/bold yellow]")
    console.print(
        "[bold magenta]First rows in the data:[/bold magenta]\n",
        df[[timestamp_col, glucose_col]].head(),
    )
