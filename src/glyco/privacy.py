"""
Deidentify and mask private or sensitive information
"""
import logging
import pandas as pd
from datetime import timedelta as tdel
import numpy as np
import hashlib
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)
default_replace_func = lambda x: hashlib.sha256(str(x).encode()).hexdigest()

def mask_private_information(gdf: pd.DataFrame, remove_columns: List[str], replace_columns: List[str], glucose_col: str, tsp_col: str, tsp_fmt: str, set_start_date: Optional[str] = None, replace_func: Callable = default_replace_func, noise_std: float =0.2):
    """
    Masks private information in a DataFrame.

    Args:
        gdf (pd.DataFrame): The input DataFrame containing private information.
        set_start_date (str): The date str at which to reset the start of the timestamp column (with or without time).
        glucose_col (str): Name of the column containing glucose data to add noise to.
        tsp_col (str): Name of the column containing timestamps.
        tsp_fmt (str): Format of the timestamps in the timestamp column.
        remove_columns (List[str]): List of columns to be removed from the DataFrame.
        replace_columns (List[str]): List of columns to be replaced using the function given in `replace_func`.
        replace_func (Callable): A function to replace values. Defaults to a hash function.
        noise_std (float): standard deviation of the noise to be added to glucose

    Returns:
        gdf (pd.DataFrame): A modified DataFrame with private information masked and data transformations applied.
        added_noise (np.ndarray): An array of noise that was added to glucose values in case you want to substract it later
        replaced_values_orig (pd.DataFrame): The original values that were replaced
    """
    # Replace specified columns with hashed values
    df = gdf.copy()
    logger.info("The values of the columns '(%s)' will be replaced using a hash...", ', '.join(replace_columns))
    replaced_values_orig = df[replace_columns].copy()
    df[replace_columns] = df[replace_columns].applymap(replace_func)
    # Convert 'tsp_col' to datetime if it's not already
    convert_from_str = not pd.api.types.is_datetime64_any_dtype(df[tsp_col])
    if convert_from_str:
        if not pd.api.types.is_string_dtype(df[tsp_col]):
            raise ValueError("Timestamp column '%s' provided must be either datetime or str", tsp_col)
        df[tsp_col] = pd.to_datetime(df[tsp_col], format=tsp_fmt, errors='coerce')
    # Reset the timestamps such that they start at 'set_start_datetime_at'
    if set_start_date:
        logger.info("The min time in the timstamp column '%s' will be shifted to: '%s'...", tsp_col, set_start_date)
        day_diff = (pd.to_datetime(set_start_date, format=tsp_fmt) - df[tsp_col].min()).days + 1
        df[tsp_col] = df[tsp_col] + tdel(days=day_diff)
    else:
        logger.warning("The start date is not reset, "
        "this may be identifyable information. To reset provide 'set_start_date'.")
    # Convert 'tsp_col' back to str if it was
    if convert_from_str:
        # If the original type was str keep it str
        df[tsp_col] = df[tsp_col].dt.strftime(tsp_fmt)
    # df[tsp_col] = pd.to_datetime(set_start_datetime_at) + (df[tsp_col] - df[tsp_col].min())
    # Add noise to the glucose data
    logger.info("Adding noise to the glucose data in the column '%s'...", glucose_col)
    added_noise = np.random.normal(0, noise_std, df.shape[0])
    df[glucose_col] += added_noise
    
    if remove_columns:
        logger.info("Removing the following privacy columns: %s...", ', '.join(remove_columns))
        try:
            df = df.drop(columns=remove_columns, errors='raise')
        except KeyError as err:
            logger.warn(f"One of the columns {remove_columns} was not found, column deletion skipped!")
    return df, added_noise, replaced_values_orig
