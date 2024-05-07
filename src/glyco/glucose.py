import hashlib
import logging
from datetime import date as date_type, datetime as dt, timedelta as tdel
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table

from glyco.privacy import mask_private_information
from glyco.utils import Devices, Units, end_plot, find_nearest, is_weekend, units_to_mmolL_factor, weekday_map

logger = logging.getLogger(__name__)
"""Warning and Error Messages
"""
error_not_implemented_method = (
    "NOT IMPLEMENTED: Method not yet supported in this version."
)

"""Default Values
"""
TIMESTAMP_COL = 'tsp'
GLUCOSE_COL = 'glucose'
# Default values for column names as found in Freestyle Libre data
DEFAULT_INPUT_TSP_COL = "Device Timestamp"
DEFAULT_INPUT_GLUC_COL = "Historic Glucose mmol/L"

# Default formats
DEFAULT_INPUT_TSP_FMT = "%d-%m-%Y %H:%M"
DEFAULT_OUT_DATE_FMT = "%d-%m-%Y (%A)"

DEFAULT_GLUC_LIMIT = 6 # glucose threshold used for calculating Area Under the Curve
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
_freestyle_rec_type_col = "Record Type" # TODO separate freestyle specific to separate class in utils or devices
_freestyle_serial_number_col = "Serial Number" 
_freestyle_notes_rec_type = 6
_freestyle_glucose_rec_type = 0
_optional_cols = [_meal_note_col, _meal_ref_col]
meal_default_cols = [TIMESTAMP_COL, _meal_ref_col, _meal_ref_col]
_default_shift_hours = 7

_default_private_info_kwargs = {
    'set_start_date': '01-01-2023 00:00',
    'remove_columns': [_freestyle_serial_number_col],
    'replace_columns': [_meal_note_col],
    'replace_func': lambda x: hashlib.sha256(str(x).encode()).hexdigest(),
    'noise_std': 0.2
}

general_date_type = Union[str, pd.Timestamp, date_type]

"""File reading
"""
def read_csv(
    file_path: str,
    timestamp_col: str = DEFAULT_INPUT_TSP_COL,
    timestamp_fmt: str = DEFAULT_INPUT_TSP_FMT,
    glucose_col: str = DEFAULT_INPUT_GLUC_COL,
    glucose_unit: str = Units.mmolL.value,
    unit_autodetect: bool = False,
    calculate_glucose_properties: bool=True,
    glucose_lim: int=DEFAULT_GLUC_LIMIT,  # predefined glucose limit used for AUC calculation
    filter_glucose_rows=False,
    delimiter: str = DEFAULT_CSV_DELIMITER,
    skiprows: int = 0,
    generated_glucose_col: str = GLUCOSE_COL,
    generated_date_col: str = _DATE_COL,
    generated_timestamp_col: str =TIMESTAMP_COL,
    glucose_prep_kwargs: Dict = _default_glucose_prep_kwargs,
    mask_private_info: Optional[bool] = False,
    private_info_kwargs: Optional[Dict] = _default_private_info_kwargs
) -> pd.DataFrame:
    # TODO: reorder by importance to improve UX
    # TODO: assert generated G column different from G column? Or warn if similar
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
        glucose_prep_kwargs (Dict, optional): arugments that can be used 
            to smoothening the glucose curve.
            See the Glucose Prep Arguments section of the Glucose documentation.
            Defaults to _default_glucose_prep_kwargs.
        mask_private_info (bool, optional): choose to mask or not to mask private information.
            This uses the 'mask_private_information' function.
            See the Privacy documentation for more on how this works.
            Defaults to false.
        private_info_kwargs (Optional[Dict], optional): arugments that can be used 
            to mask private information. These are give to the 'mask_private_information' function.
            See the Privacy documentation for more on how this works.
            Defaults to _default_private_info_kwargs.

    Returns:
        pd.DataFrame: The resulting Glucose Dataframe that contains the file data, 
            along with the Generated Glucose Properties
    """
    df = pd.read_csv(
        filepath_or_buffer=file_path,
        delimiter=delimiter,
        skiprows=skiprows
    )

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
        raise ValueError(f"The Glucose column '{glucose_col}' is not in the input columns."\
            "Please provide 'glucose_col' as input.")
    if timestamp_col not in df.columns:
        raise ValueError(f"The Timestamp column '{timestamp_col}' is not in the input columns."\
            "Please provide 'timestamp_col' as input.")
    

def read_df(df: pd.DataFrame,
    timestamp_col: str = DEFAULT_INPUT_TSP_COL,
    timestamp_fmt: str = DEFAULT_INPUT_TSP_FMT,
    glucose_col: str = DEFAULT_INPUT_GLUC_COL,
    glucose_unit: str = Units.mmolL.value,
    unit_autodetect : bool = False,
    calculate_glucose_properties: bool = True,
    glucose_lim: int=DEFAULT_GLUC_LIMIT, 
    filter_glucose_rows=False,
    generated_glucose_col: str = GLUCOSE_COL,
    generated_date_col: str = _DATE_COL,
    generated_timestamp_col: str =TIMESTAMP_COL,
    glucose_prep_kwargs: Dict = _default_glucose_prep_kwargs,
    mask_private_info: Optional[bool] = False,
    private_info_kwargs: Optional[Dict] = _default_private_info_kwargs
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
        glucose_prep_kwargs (Dict, optional): arugments that can be used 
            to smoothening the glucose curve.
            See the Glucose Prep Arguments section of the Glucose documentation.
            Defaults to _default_glucose_prep_kwargs.
        mask_private_info (bool, optional): choose to mask or not to mask private information.
            This uses the 'mask_private_information' function.
            See the Privacy documentation for more on how this works.
            Defaults to false.
        private_info_kwargs (Optional[Dict], optional): arugments that can be used 
            to mask private information. These are give to the 'mask_private_information' function.
            See the Privacy documentation for more on how this works.
            Defaults to _default_private_info_kwargs.

    Returns:
        pd.DataFrame: The resulting Glucose Dataframe that contains the file data, 
            along with the Generated Glucose Properties.
    """
    validate_glucose_columns(df=df, glucose_col=glucose_col, timestamp_col=timestamp_col)
    # df = convert_tsp(ndf=df, tlbl=generated_timestamp_col, tsp_lbl=timestamp_col, timestamp_fmt=timestamp_fmt)
    if unit_autodetect:
        glucose_unit = autodetect_unit(df[glucose_col])
    logger.info("Using the glucose unit (%s)", glucose_unit)
    # filter rows based on the values of a column (filter_val)
    if filter_glucose_rows:
        df = filter_glucose_by_column_val(
            df,
            filter_col=_freestyle_rec_type_col,
            filter_val=_freestyle_glucose_rec_type
        )
    # mask private information
    if mask_private_info:
        df, _, _ = mask_private_information(gdf=df,
        glucose_col=glucose_col,
        tsp_col=timestamp_col,
        tsp_fmt=timestamp_fmt,
        **private_info_kwargs)
    # add calculated glucose properties
    if calculate_glucose_properties:
        df =(
            # 
            prepare_glucose(
                glucose_df=df,
                glucose_col=glucose_col,
                tsp_lbl=timestamp_col,
                timestamp_fmt=timestamp_fmt,
                timestamp_is_formatted=False,
                unit=glucose_unit,
                glbl=generated_glucose_col,
                tlbl=generated_timestamp_col,
                dlbl=generated_date_col,
                **glucose_prep_kwargs
            ).pipe(
                get_properties, 
                glbl=generated_glucose_col,
                tlbl=generated_timestamp_col,
                glim=glucose_lim
            )    
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
    elif fail_on_invalid:
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
    raise NotImplementedError(error_not_implemented_method)


def filter_glucose_by_column_val(
    df: pd.DataFrame, filter_col: str=_freestyle_rec_type_col, filter_val=_freestyle_glucose_rec_type
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
    logger.info(f"Selecting only columns with a {filter_col} with the value {filter_val}")
    return df[df[filter_col] == filter_val]


def add_time_values(df, tlbl: str = TIMESTAMP_COL, tsp_lbl: str = DEFAULT_INPUT_TSP_COL, timestamp_fmt: str = DEFAULT_INPUT_TSP_FMT, dlbl: str = _DATE_COL, weekday_map=weekday_map, timestamp_is_formatted: bool = False):
    """Adds generated time-values to the dataframe based on the timestamp. These include:
    Date, Date string, Hour, Weekday number, Weekday name, Is or is not a weekend day.

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
        ndf = convert_tsp(ndf=ndf, tlbl=tlbl, tsp_lbl=tsp_lbl, timestamp_fmt=timestamp_fmt)
    ndf[dlbl] = ndf[tlbl].dt.date
    ndf[f"{dlbl}_str"] = ndf[dlbl].map(lambda x: x.strftime(DEFAULT_OUT_DATE_FMT) if type(x)==pd.DatetimeIndex else x)
    ndf[_HOUR_COL] = ndf[tlbl].dt.hour
    ndf[_DAYOFWEEK_COL] = ndf[tlbl].dt.weekday
    ndf = ndf.assign()
    ndf[_WEEKDAY_COL] = ndf[_DAYOFWEEK_COL].map(weekday_map)
    ndf[_ISWEEKEND_COL] = ndf[_DAYOFWEEK_COL].map(is_weekend)
    return ndf

def convert_tsp(ndf: pd.DataFrame, tlbl: str, tsp_lbl: str, timestamp_fmt: str) -> None:
    """
    Convert a timestamp column in a DataFrame to datetime using a specified format.

    Args:
        ndf (pd.DataFrame): The DataFrame containing the timestamp column to convert.
        tlbl (str): The label for the new column where the converted timestamps will be stored.
        tsp_lbl (str): The label of the timestamp column to convert.
        timestamp_fmt (str): The format to use for parsing the timestamps.

    Raises:
        ValueError: If the timestamp conversion fails, this exception is raised with a detailed error message.

    Example:
    ```python
    import pandas as pd

    # Assuming you have a DataFrame `ndf`, column labels `tlbl`, `tsp_lbl`, and a valid `timestamp_fmt`.
    convert_tsp(ndf, tlbl, tsp_lbl, timestamp_fmt)
    ```
    """
    df = ndf.copy()
    try:
        df[tlbl] = pd.to_datetime(df[tsp_lbl], format=timestamp_fmt)
        return df
    except ValueError as e:
        raise ValueError(f"Failed to convert timestamp '{tsp_lbl}' using the format '{timestamp_fmt}'. "
                         f"Error: '{e}'. "
                         f"Verify that you are using the correct 'timestamp_fmt' as input")


def prepare_glucose(
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
    extra_shift_in_time: int = _default_shift_hours,
):
    """Parses the glucose data.
    - Creates extra columns for hours, days, etc.
    - Sorts the dataframe by time.
    - Creates columns for shifted time if needed (used for certain computations).
    - Converts units if needed.
    - Adds interpolated glucose measures to fill in the gaps.

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
        weekday_map=weekday_map,
        timestamp_is_formatted=timestamp_is_formatted)

    if extra_shift_in_time:
        df = add_shifted_time(df, tlbl, dlbl, extra_shift_in_time)

    # convert to mmol/L
    df[glbl] = (
        pd.to_numeric(df[glucose_col])
        if unit == Units.mmolL.value
        else convert_to_mmolL(pd.to_numeric(df[glucose_col]), from_unit=unit)
    )

    # index by time and keep time column
    df['idx'] = df[tlbl]
    df = df.set_index('idx').sort_index()
    # interpolate and smoothen glucose
    if interpolate:
        df[glbl] = df[glbl].rolling(window=rolling_avg).mean()
        df[glbl] = df[glbl].interpolate(method=interp_met, order=interp_ord)
        df = df[df[glbl].map(lambda g: g > 0 and g < 30)]
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
    df[f"{shift_dlbl}_str"] = df[shift_dlbl].map(lambda x: x.strftime(DEFAULT_OUT_DATE_FMT))
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
        df (pd.DataFrame): the glucose dataframe.
        glucose_col (str): the glucose column name.
        timestamp_col (str): the timestamp column name.

    Returns:
        pd.DataFrame: the pandas dataframe with extra columns for derivatives
            - _DG_COL: the glucose diff dG.
            - _DT_COL: the time diff dt.
            - _DGDT_COL: the glucose time derivative dG/dt
    """
    df[_DG_COL], df[_DT_COL], df[_DGDT_COL] = compute_derivative(
        df, glucose_col, timestamp_col
    )
    return df


def compute_derivative(df: pd.DataFrame, glucose_col: str, timestamp_col: str):
    """Calculates the glucose time derivative (dG/dt)

    Args:
        df (pd.DataFrame): the glucose dataframe.
        glucose_col (str): the glucose column name.
        timestamp_col (str): the timestamp column name.

    Returns:
        (pd.Series, pd.Series, pd.Series): A tuple containing:
            - The glucose diff dG.
            - The time diff dt.
            - The glucose time derivative dG/dt
    """
    dG = df[glucose_col].diff()
    dt = df[timestamp_col].diff().dt.total_seconds()
    return dG, dt, dG / dt


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
    glbl: str = GLUCOSE_COL,
    tlbl: str = TIMESTAMP_COL,
    glim: float = DEFAULT_GLUC_LIMIT,
):
    """Apply prepare_glucose first

    Args:
        df (pd.DataFrame): _description_
        glbl (str, optional): _description_. Defaults to GLUCOSE_COL.
        tlbl (str, optional): _description_. Defaults to TIMESTAMP_COL.
        glim (float, optional): _description_. Defaults to GLUCOSE_LIMIT_DEFAULT.

    Returns:
        _type_: _description_
    """
    # Generated values: derivative, integral, stats
    # set derivative and area under the curve properties
    df = set_derivative(df, glbl, tlbl)
    df = set_auc(df, glbl, tlbl, glim)
    return df


def convert_to_mmolL(g: float, from_unit: str) -> float:
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
    raise NotImplementedError(error_not_implemented_method)


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
    logger.warning("Using unit autodetection." \
        "This may result in unexpected behavior.")
    cast_glucose_sample = pd.to_numeric(
            (
                glucose_values
                .sample(
                    n=min(100, len(glucose_values)),
                    replace=False
                )
            ), errors='coerce')
    m = cast_glucose_sample.mean()
    if m > 33:
        return Units.mgdL.value
    if m > 6:
        return Units.mmolL.value
    if cast_glucose_sample.std() < 0.4:
        return Units.gL.value
    return Units.mmolL.value


"""Plotting
"""

def plot_glucose(
    df: pd.DataFrame,
    glbl: str = GLUCOSE_COL,
    tlbl: str = TIMESTAMP_COL,
    from_time: Optional[general_date_type] = None,
    to_time: Optional[general_date_type] = None,
):
    """Plots the glucose curve for a given dataframe, and optional time frame

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

    Raises:
        KeyError: if the glucose column is not in the glucose dataframe
    """
    plot_df = df[from_time:to_time]

    if glbl not in plot_df.keys():
        raise KeyError(f"Glucose Column {glbl} does not seem to be in the DataFrame.")
    for d in plot_df.date.unique():
        plt.axvline(d, color="brown", linestyle="--", alpha=0.5)
    plt.axhline(DEFAULT_GLUC_LIMIT)
    plt.axhline(DEFAULT_GLUC_LIMIT - 1)
    plt.axhline(DEFAULT_GLUC_LIMIT + 1)
    plt.axhline(plot_df[glbl].median())
    plt.plot(plot_df[tlbl], plot_df[glbl], label='Glucose in mmol/L')
    plt.xlabel("Time")
    plt.ylabel("Glucose")
    plt.title(f"Glucose variation from: '{plot_df.index[0].date()}'' to:'{plot_df.index[-1].date()}'")


def plot_trend_by_hour(df: pd.DataFrame, glbl: str = GLUCOSE_COL):
    """Plots the glucose hourly trend as an averaged curve for each hour
    with percentile distributions.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        glbl (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
    """
    plot_percentiles(df, stat_col=glbl, group_by_col=_HOUR_COL, percentiles=[0.01, 0.05])


def plot_trend_by_weekday(df: pd.DataFrame, glbl=GLUCOSE_COL):
    """Plots the glucose trend for each weekday (Monday to Sunday) as
    a box plot for each weekday.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        glbl (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
    """
    plot_comparison(df=df, glbl=glbl, compare_by=_WEEKDAY_COL, outliers=False, label_map=None, method='box', sort_vals = False)


def plot_trend_by_day(df: pd.DataFrame, glbl=GLUCOSE_COL):
    """Plots the glucose trend for each weekday (Monday to Sunday) as
    a box plot for each weekday.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        glbl (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
    """
    plot_comparison(df=df, glbl=glbl, compare_by=_DATE_COL, outliers=False, label_map=None, method='box', sort_vals = False)


def plot_percentiles(df: pd.DataFrame, stat_col: str, percentiles: List[float], group_by_col: str=_HOUR_COL, color: str='green', label: str=None):
    """Groups glucose by a column column and plots percentiles of glucose.
    Percentiles are plotted using an area color between the main curve and each percentile.
    Does not show plot.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        stat_col (str): the glucose column name or column for which to get stats (Y-axis).
        percentiles (List[float]): a list of percentiles to plot (each value between 0 and 1)
        group_by_col (str, optional): the name of the column to group values by (X-axis).
            Defaults to _HOUR_COL.
        color (str, optional): the name of the color to use for the percentiles area. 
            Defaults to 'green'.
        label (str, optional): the title of the plot. 
            Defaults to None.
    """
    # TODO use only get_stats and remove get_percentiles_and_stats
    _, _, med, perc_l, perc_h = get_percentiles_and_stats(
        df, percentiles, stat_col, group_by_col)

    stats_df = df.pipe(
        get_stats,
        stats_cols=stat_col, 
        group_by_col=group_by_col, 
        percentiles=percentiles
    )
    

    stats_df['50%'].plot(label='50%')
    
    for i in range(len(percentiles)):
        plt.fill_between(
            # TODO enable changing alpha and label
            med.index, perc_l[i], perc_h[i], color=color, alpha=0.2, label=f"{100*(1-percentiles[i])}th")
    if not label:
        label = 'Trend of {} for the percentiles: {} as well as {}'.format(stat_col,
                                                                         ', '.join(
                                                                             [str(int(i*100)) for i in percentiles]),
                                                                         ', '.join([str(int((1-i)*100)) for i in percentiles]))
    plt.title(label)
    plt.xlabel(group_by_col)
    plt.ylabel(stat_col)

def plot_sleep_trends(df: pd.DataFrame, glbl: str = GLUCOSE_COL, sleep_time_filter_col: str = f'shifted_{_HOUR_COL}', sleep_time_hour: int = 24 - (_default_shift_hours + 2)):
    # filter sleep glucose data
    gdf = df[df[sleep_time_filter_col] >= sleep_time_hour]
    # make new plotting time
    gdf.loc[:, 'sleep_hours'] = gdf[sleep_time_filter_col]- gdf[sleep_time_filter_col].min()
    plot_percentiles(df=gdf,
        stat_col=glbl, group_by_col='sleep_hours', percentiles=[0.01, 0.05], label='Hourly trend of Glucose during Sleep')
    plt.ylabel('Glucose during sleep')
    plt.xlabel('Hours of sleep (from 0-8)')
    end_plot()
    plot_comparison(df=gdf, glbl=glbl, compare_by=f'shifted_{_DATE_COL}_str', outliers=False,
        method='box', sort_vals = False, label='Daily trend of Glucose during Sleep')
    plt.ylabel('Glucose during sleep')
    plt.xlabel('Day (sleep from evening of this day)')
    end_plot()

def plot_day_curve(df: pd.DataFrame, d: str, glbl: str = GLUCOSE_COL, tlbl: str = TIMESTAMP_COL, extended=False):
    """Plots day glucose curve

    :param df: Dataframe containing glucose
    :param glbl: Name of the glucose column
    :return:
    """
    plot_df = df.loc[d]
    #plt.axvline(d, color='brown', linestyle='--', alpha=0.5)
    plt.axhline(plot_df[glbl].mean(), color='red', linestyle='--', alpha=0.5, label='Day Average')
    plt.axhline(df[glbl].mean(), color='brown', linestyle='--', alpha=0.2, label='Your general Average')
    plt.axhline(df[glbl].mean() + 2, color='green', linestyle='--', alpha=0.5, label='Recommended range')
    plt.axhline(df[glbl].mean() - 1, color='green', linestyle='--', alpha=0.5)
    plt.plot(plot_df[tlbl], plot_df[glbl])

    
    if extended:
        # TODO find a cleaner way to do this
        xt = df.loc[d: (dt.strptime(d, '%Y-%m-%d')+tdel(days=1, hours=7)).strftime('%Y-%m-%d %H')]
        plt.plot(xt[tlbl], xt[glbl], color='brown', alpha=0.5, label='sleep')

def get_stats(df: pd.DataFrame, stats_cols: Union[List, str], group_by_col: str = None, percentiles: Optional[List[float]] = None):
    """Get descriptive statistics about specific columns of a dataframe.

    Args:
        df (pd.DataFrame): the glucose dataframe.
        stats_cols (Union[List[str], str]): the glucose column name, or a column name,
            or a list of column names for which to get stats.
        group_by_col (str, optional): the name of the column to group values by. 
            Defaults to None.
        percentiles (Optional[List[float]], optional): a list of percentiles to plot 
            (each value between 0 and 1). Defaults to None.

    Returns:
        pd.Series or pd.DataFrame: descriptive statistics grouped by the given column
    """
    if group_by_col:
        return (df
            .groupby(group_by_col)
            [stats_cols]
            .describe(percentiles=percentiles)
        )


    #     grouped_df = df.groupby(group_by_col)[stats_cols]
    #     r= pd.concat([grouped_df.describe(percentiles=percentiles), grouped_df.sum()], axis=1)
    #     return r.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=[0, 1])
    # return df[stats_cols].describe(percentiles=percentiles),  df[stats_cols].sum()
    return df[stats_cols].describe(percentiles=percentiles)
    
def get_percentiles_and_stats(df: pd.DataFrame, percentiles: List[float], stat_col: str, group_by_col: str):
    """Get descriptive statistics about a specific column of a dataframe.

    Args:
        percentiles (Optional[List[float]], optional): a list of percentiles to plot 
            (each value between 0 and 1). Defaults to None.
        df (pd.DataFrame): the glucose dataframe.
        percentiles (List[float]): a list of percentiles to plot (each value between 0 and 1).
        stat_col (str): the glucose column name or a column name to get stats.
        group_by_col (str): the name of the column to group values by. 

    Returns:
        Tuple[float]: mean, standard deviation, percentiles, 1-percentiles
    """
    # TODO replace completely with get_stats
    grouped = df.groupby([group_by_col])[stat_col]
    mean = grouped.mean()
    med = grouped.median()
    dev = grouped.std()
    perc_l = [grouped.quantile(q) for q in percentiles]
    perc_h = [grouped.quantile(1-q) for q in percentiles]
    return mean, dev, med, perc_l, perc_h # FIXME: use dataclass, do we need mean, med?

def plot_comparison(df: pd.DataFrame, glbl: str=GLUCOSE_COL, compare_by: str=_WEEKDAY_COL, outliers: bool=False, label_map: Union[Callable, Dict]=None, method: str='box', sort_vals: bool = False, label: Optional[str] = None):
    """

    :param df: dataframe containing the values to be compared and the comparison field
    :param glbl: 
    :param compare_by: 
    :param outliers: boolean to show or not show outliers, defaults to False
    :param label_map: 
    defaults to None (showing original)
    :return:

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
        label (Optional[str], optional): title of the plot, if None a default will be generated.
            Defaults to None.

    Raises:
        NotImplementedError: if the method used for comparison is not implemented.
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
    plt.ylabel(f"Trend for {glbl}")
    plt.xlabel(f"{compare_by}")
    if not label:
        label='Comparing {} by {}. Outliers are {}.'.format(glbl, compare_by, 'shown' if outliers else 'not shown')
    plt.title(label)


def get_response_bounds(df: pd.DataFrame, event_time: pd.Timestamp, pre_pad_min: int = 20, post_pad_min: int = 0, resp_time_min: int = 120, glbl: str = GLUCOSE_COL, t_lbl: str = TIMESTAMP_COL):
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
    g_event_time = find_nearest(df, event_time, glbl, t_lbl, n_iter=100)
    start = g_event_time - tdel(minutes=pre_pad_min)
    end = g_event_time + tdel(minutes=resp_time_min) + tdel(minutes=post_pad_min)
    return start, end, g_event_time


def plot_response_at_time(glucose_df: pd.DataFrame, event_time: pd.Timestamp, event_title: Optional[str] = None, pre_pad_min: int = 20, post_pad_min: int = 0, resp_time_min: int = 120, glbl: str = GLUCOSE_COL, t_lbl: str = TIMESTAMP_COL, auc_lim: int=DEFAULT_GLUC_LIMIT, show_auc: bool=True, use_local_min: bool=False):
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
    s, e, t = get_response_bounds(glucose_df, event_time, pre_pad_min, post_pad_min, resp_time_min, glbl=glbl, t_lbl=t_lbl)
    plot_df = glucose_df.loc[s:e][glbl]
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

"""
Outputs
- Write to a file
- Write image plot or responses
"""
def write_glucose(gdf : pd.DataFrame, output_file : str):
    """Writes glucose to a csv file

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        output_file (str): the output file path.
    """
    logger.info("Writing glucose data to %s", output_file)
    gdf.to_csv(output_file)

"""
Day/Week Metrics
"""
_summary_cols = [GLUCOSE_COL, _AUC_COL, _DG_COL, _DT_COL, _DGDT_COL]


def get_metrics_by_day(gdf : pd.DataFrame, day_col : str =_DATE_COL, percentiles: list = None, summary_cols : list = _summary_cols):
    """
    Get metrics and statistics by day related to specific columns of a dataframe.

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        day_col (str, optional): the day column name. Defaults to _HOUR_COL.
        percentiles (Optional[List[float]], optional): a list of percentiles to plot 
            (each value between 0 and 1). Defaults to None.
        summary_cols (Union[List[str], str]): the glucose column name, or a column name,
            or a list of column names for which to get stats. Defaults to _summary_cols.

    Returns:
        pd.Series or pd.DataFrame: descriptive statistics grouped hour
    """
    return get_stats(gdf, stats_cols=summary_cols, percentiles=percentiles, group_by_col=day_col)


def get_metrics_by_hour(gdf : pd.DataFrame, hour_col : str =_HOUR_COL, percentiles : Optional[List[float]] = None, summary_cols : Union[List[str], str] = _summary_cols):
    """Get metrics and statistics by hour related to specific columns of a dataframe.

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        hour_col (str, optional): the hour column name. Defaults to _HOUR_COL.
        percentiles (Optional[List[float]], optional): a list of percentiles to plot 
            (each value between 0 and 1). Defaults to None.
        summary_cols (Union[List[str], str]): the glucose column name, or a column name,
            or a list of column names for which to get stats. Defaults to _summary_cols.

    Returns:
        pd.Series or pd.DataFrame: descriptive statistics grouped hour
    """
    return get_stats(gdf, stats_cols=summary_cols, percentiles=percentiles, group_by_col=hour_col)


def get_metrics(gdf : pd.DataFrame, percentiles : Optional[List[float]] = None, summary_cols : Union[List[str], str] = _summary_cols, group_by_col: Optional[str] = None):
    """Get metrics and statistics related to specific columns of a dataframe.

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        percentiles (Optional[List[float]], optional): a list of percentiles to plot 
            (each value between 0 and 1). Defaults to None.
        summary_cols (Union[List[str], str]): the glucose column name, or a column name,
            or a list of column names for which to get stats. Defaults to _summary_cols.
        group_by_col (str, optional): the name of the column to group values by. 
            Defaults to None.

    Returns:
       pd.Series or pd.DataFrame: descriptive statistics grouped by the given column
    """
    return get_stats(gdf, stats_cols=summary_cols, percentiles=percentiles, group_by_col=group_by_col)

def describe_glucose(df: pd.DataFrame, glucose_col: str = GLUCOSE_COL, timestamp_col: str = TIMESTAMP_COL, default_unit: str = Units.mmolL.value):
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

    for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        value = glucose_stats.get(stat, 'N/A')
        table.add_row(stat.capitalize(), f"{value:.2f}")

    console.print(table)
    console.print(
        f"[bold magenta]First rows in the data:[/bold magenta]\n",
        df[[timestamp_col, glucose_col]].head())



