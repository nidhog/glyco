"""Handles Meals and Events"""

import os
from os.path import isfile, join
from typing import Callable, Iterable, Optional

from datetime import timedelta as tdel
from datetime import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt

from rich.console import Console
from rich.table import Table

from glyco.glucose import (
    GLUCOSE_COL,
    TIMESTAMP_COL,
    _AUC_COL,
    _AUCLIM_COL,
    _AUCMIN_MIN,
    DEFAULT_INPUT_TSP_FMT,
    GeneralDateType,
    add_time_values,
    ERR_NOT_IMPLEMENTED,
)
from glyco.utils import find_nearest

import logging

logger = logging.getLogger(__name__)

_event_note_col = "event_notes"
_event_ref_col = "event_reference"
_freestyle_rec_type_col = "Record Type"
_freestyle_notes_rec_type = 6
_freestyle_glucose_rec_type = 0
_optional_cols = [_event_note_col, _event_ref_col]
event_default_cols = [TIMESTAMP_COL, _event_note_col, _event_ref_col]


def shift_time_fwd(t, h: int = 1, m: int = 0):
    """Shifts time forward

    Args:
        t (datetime.datetime): _description_
        h (int, optional): number of hours. Defaults to 1.
        m (int, optional): number of minutes. Defaults to 0.

    Returns:
        datetime.datetime: shifted time 't' forward by 'h' hours and 'm' minutes
    """
    return t + tdel(hours=h, minutes=m)


def shift_time_bck(t, h=1, m=0):
    """Shifts time backward

    Args:
        t (datetime.datetime): _description_
        h (int, optional): number of hours. Defaults to 1.
        m (int, optional): number of minutes. Defaults to 0.

    Returns:
        datetime.datetime: shifted time 't' backwards by 'h' hours and 'm' minutes
    """
    return t - tdel(hours=h, minutes=m)


def read_meals(
    events_folder_path: str,
    file_ext: Optional[str] = None,
    shift_time_function: Callable = shift_time_fwd,
    hours: int = 1,
    minutes: int = 0,
):
    """Read meals or other events from a folder. Used for using photos taken as events.
    - Each file with the corresponding extension is assumed to be an event/meal.
    - The time of the event is assumed to be the time of creation of the file.
    - Time can be shifted forwards or backwards to account for different timezones.

    Args:
        events_folder_path (str): the path of the folders where the pictures/files are.
        file_ext (Optional[str], optional): the extension of files. If None all files will be considered. Defaults to None.
        shift_time_function (Callable, optional): function to shift time forwards or backwards. Defaults to shift_time_fwd.
        hours (int, optional): number of hours. Defaults to 1.
        minutes (int, optional): number of minutes. Defaults to 0.

    Returns:
        pd.DataFrame: events dataframe:
            - indexed by the event timestamp.
            - contains the column TIMESTAMP_COL with the timestamp.
            - generated time values columns from from the timestamp.
    """
    event_files = [
        (
            shift_time_function(
                dt.fromtimestamp(os.stat(events_folder_path + f).st_mtime), hours, minutes
            ),
            f,
            f,
        )
        for f in os.listdir(events_folder_path)
        if isfile(join(events_folder_path, f)) and (file_ext is None or f.endswith(file_ext))
    ]
    events_df = pd.DataFrame(event_files, columns=event_default_cols)
    events_df["itsp"] = events_df[TIMESTAMP_COL]
    events_df = events_df.set_index("itsp")
    events_df = add_time_values(
        events_df, tlbl=TIMESTAMP_COL, tsp_lbl=TIMESTAMP_COL, timestamp_is_formatted=True
    )
    return events_df


def validate_event_columns(df: pd.DataFrame, ref_col: str, note_col: str, tsp_col: str):
    """
    Validate the required Event DataFrame columns exist in a DataFrame.

    Checks whether: the timestamp, reference, and (optionally)
    note columns are present in the provided DataFrame.
    A ValueError is raised if a column is missing.

    Args:
        df (pd.DataFrame): Event dataframe.
        ref_col (str): the name of the event reference/identifier column.
        note_col (str): the name of the note column.
            Can be None if no notes are available.
        tsp_col (str): the event timestamps column name.

    Raises:
        ValueError: If the specified `tsp_col`, `ref_col`, or `note_col`
        (when not None) is missing from the DataFrame.
    """
    if tsp_col not in df.columns:
        raise ValueError(
            f"The Timestamp column '{tsp_col}' is not in the event input columns."
            "Please provide 'tsp_col' as input."
        )
    if ref_col not in df.columns:
        raise ValueError(
            f"The Event reference column '{ref_col}' is not in the input columns."
            "Please provide 'ref_col' as input."
        )
    if note_col:
        if note_col not in df.columns:
            raise ValueError(
                f"The Event notes column '{note_col}' is not in the input columns."
                "Please provide 'note_col' as input or give 'None' if the column does not exist."
            )


def read_events_csv(
    file_path: str,
    tsp_col: str = TIMESTAMP_COL,
    ref_col: str = _event_ref_col,
    note_col: str = None,
    timestamp_fmt: str = DEFAULT_INPUT_TSP_FMT,
    timestamp_is_formatted: bool = False,
    delimiter: str = ",",
    skiprows: int = 0,
):
    """
    Load and process an Events Dataframe from a CSV file.

    Reads a CSV file containing glucose event data (meals, activities or other),
    validates the required columns, and returns a standardized Event DataFrame
    ready for analysis.

    Args:
        file_path (str): Path to the CSV file containing event data.
        tsp_col (str, optional): Name of the column containing timestamps.
            Defaults to `TIMESTAMP_COL`.
        ref_col (str, optional): Name of the column containing event references/identifiers.
            Defaults to `_event_ref_col`.
        note_col (str, optional): Name of the column containing event notes.
            If None, the notes column will be duplicated from `ref_col`. Defaults to None.
        timestamp_fmt (str, optional): Format string for parsing timestamps (e.g. "%Y-%m-%d %H:%M:%S").
            Defaults to `DEFAULT_INPUT_TSP_FMT`.
        timestamp_is_formatted (bool, optional): If True, assumes timestamps are already in
            a datetime-compatible format and skips parsing. Defaults to False.
        delimiter (str, optional): Column separator used in the CSV file. Defaults to ",".
        skiprows (int, optional): Number of initial rows to skip when reading the CSV. Defaults to 0.

    Returns:
        pd.DataFrame: A processed DataFrame with standardized timestamp, reference,
        and note columns.
    """
    df = pd.read_csv(filepath_or_buffer=file_path, delimiter=delimiter, skiprows=skiprows)
    df = read_events_df(
        edf=df,
        tsp_col=tsp_col,
        ref_col=ref_col,
        note_col=note_col,
        timestamp_fmt=timestamp_fmt,
        timestamp_is_formatted=timestamp_is_formatted,
    )
    return df


def read_events_df(
    edf: pd.DataFrame,
    tsp_col: str = TIMESTAMP_COL,
    ref_col: str = _event_ref_col,
    note_col: str = None,
    timestamp_fmt: str = DEFAULT_INPUT_TSP_FMT,
    timestamp_is_formatted: bool = False,
):
    """
    Process a DataFrame of events and returns a standardized Events DataFrame.

    Checks for the required event columns, parses and standardizes
    timestamp values, and ensures that reference and note columns are properly set.
    If no note column is provided, it duplicates the reference column as the notes column.

    Args:
        edf (pd.DataFrame): Input DataFrame containing raw event data.
        tsp_col (str, optional): Name of the column containing timestamps.
            Defaults to `TIMESTAMP_COL`.
        ref_col (str, optional): Name of the column containing event references/identifiers.
            Defaults to `_event_ref_col`.
        note_col (str, optional): Name of the column containing event notes.
            If None, the notes column will be duplicated from `ref_col`. Defaults to None.
        timestamp_fmt (str, optional): Format string for parsing timestamps (e.g. "%Y-%m-%d %H:%M:%S").
            Defaults to `DEFAULT_INPUT_TSP_FMT`.
        timestamp_is_formatted (bool, optional): If True, assumes timestamps are already in
            a datetime-compatible format and skips parsing. Defaults to False.

    Returns:
        pd.DataFrame: A standardized Events DataFrame with:
            - Standardized timestamp column (`TIMESTAMP_COL`)
            - Event reference column (`_event_ref_col`)
            - Event notes column (`_event_note_col`)
    """
    validate_event_columns(df=edf, ref_col=ref_col, note_col=note_col, tsp_col=tsp_col)
    events_df = add_time_values(
        df=edf,
        tsp_lbl=tsp_col,
        tlbl=TIMESTAMP_COL,
        timestamp_fmt=timestamp_fmt,
        timestamp_is_formatted=timestamp_is_formatted,
    )
    events_df[_event_ref_col] = events_df[ref_col]
    events_df[_event_note_col] = events_df[ref_col] if note_col is None else events_df[note_col]
    return events_df


"""Freestyle Libre Specific
"""


def infer_events_from_notes(df, filter_notes_map: Callable = None):
    """Assume the notes column is referring to events and use those instead

    filter_notes_map example: `lambda x: False if not x else str(x).startswith('food')`
    """
    events_df = df[df[_freestyle_rec_type_col] == _freestyle_notes_rec_type]
    if filter_notes_map:
        events_df = events_df[events_df[_event_note_col].map(filter_notes_map)]
    return events_df


"""Events
"""
# TODO: use the following
# Event session schema
_default_event_session_seconds = 2 * 60 * 60
_original_timestamp = "timestamp_origin"
_next_event = "dt_next_event"
_prev_event = "dt_prev_event"
is_session_first = "is_session_first"
is_session_last = "is_session_last"
session_first = "session_first"
session_last = "session_last"
session_id = "session_id"
estimated_start = "estimated_start"
estimated_end = "estimated_end"
session_len = "session_len"
estimated_len = "estimated_len"


# TODO: timestamp origin must be a new field
def get_event_sessions(
    events_df: pd.DataFrame,
    glucose_df: pd.DataFrame,
    event_tsp: str = TIMESTAMP_COL,
    session_seconds: int = _default_event_session_seconds,
):
    """Generates an Event Sessions DataFrame from an Events DataFrame and a Glucose DataFrame.
    The glucose dataframe is used for finding the estimated_start, estimated_end and estimated_len values.
    Events are sorted by their corresponding timestamp.

    Args:
        events_df (pd.DataFrame): the events dataframe (with times of events)
        glucose_df (pd.DataFrame): the glucose dataframe
        event_tsp (str, optional): the timestamp in the events dataframe. Defaults to TIMESTAMP_COL.
        session_seconds (int, optional): lenght of a session: if time between two events is lower than
            the session lenght, they are grouped in the same session.
            Defaults to the value of _default_event_session_seconds.

    Returns:
        pd.DataFrame: The Event Sessions dataframe,
            sorted by the event timestamp (first event at the start of the dataframe),
            contains rows per event (NOT per event session),
            contains the following columns:

                - All columns present in the events dataframe.
                - 'session_id' the id of the event session, unique per session, may include multiple events.
                - 'dt_next_event' time to next event. NaN if it does not apply.
                - 'dt_prev_event' time from previous event. NaN if it does not apply
                - 'is_session_first' wether or not this is the first event of the session it belongs to.
                - 'is_session_last' wether or not this is the last event of the session it belongs to.
                - 'session_first' datetime of the first event in the session.
                - 'session_last' datetime of the last event in the session.
                - 'estimated_start' estimated datetime start of the event session (different from the first event)
                - 'estimated_end' estimated datetime end of the event session (different from the last event)
                - 'estimated_len' estimated timedelta lenght of the session in seconds
    """
    # define 10 minutes delta
    delta = 10 * 60
    jump = tdel(seconds=session_seconds + delta)
    delta = tdel(seconds=delta)
    edf = (
        events_df.sort_values(event_tsp)
        .assign(
            **{
                _next_event: lambda x: x[event_tsp].diff().dt.total_seconds(),
                _prev_event: lambda x: x[event_tsp].diff(-1).dt.total_seconds().map(abs),
                is_session_first: lambda x: (x[_next_event].isnull())
                | (x[_next_event] > session_seconds),
                is_session_last: lambda x: (x[_prev_event].isnull())
                | (x[_prev_event] > session_seconds),
            }
        )
        .assign(
            **{
                session_first: lambda x: x.loc[x[is_session_first], event_tsp],
                session_last: lambda x: x.loc[x[is_session_last], event_tsp],
                session_id: lambda x: x.loc[x[is_session_first], event_tsp].rank(method="first"),
            }
        )
        .assign(
            **{
                session_id: lambda x: x[session_id].ffill().astype(int),
                session_last: lambda x: x[session_last].bfill(),
                session_first: lambda x: x[session_first].ffill(),
            }
        )
    )

    g_range = lambda t: glucose_df.loc[t - delta : t + jump].index
    edf = edf.assign(
        **{
            estimated_start: lambda x: x[session_first].map(lambda t: g_range(t).min()),
            estimated_end: lambda x: x[session_last].map(lambda t: g_range(t).max()),
        }
    )

    # fill empty fields
    edf = (
        edf.assign(
            **{
                estimated_start: lambda x: x[estimated_start].ffill(),
                estimated_end: lambda x: x[estimated_end].bfill(),
                # TODO session_len: session_seconds
            }
        )
        .assign(estimated_len=lambda x: (x[estimated_end] - x[estimated_start]).dt.total_seconds())
        .set_index(event_tsp, drop=False)
        .rename_axis("itsp")
    )

    return edf


# This function is no longer in use, for now it is left as a comment in case some of the logic is needed
# def sessionize_events(events_df: pd.DataFrame, gdf: pd.DataFrame, event_timestamp: str = _original_timestamp, session_seconds: int = _default_event_session_seconds):
#     """DEPRECATED
#     TODO REMOVE
#     TODO takes very long to run
#     FIXME OPTIMISE takes too long
#     """
#     edf = events_df.sort_values(event_timestamp)
#     edf[_next_event]= edf[event_timestamp].diff().dt.total_seconds()
#     edf[_prev_event]= edf[event_timestamp].diff(-1).dt.total_seconds().map(abs)
#     # TODO change name to is_session_first
#     edf[is_session_first] = (edf[_next_event].isnull()) | (edf[_next_event] > session_seconds)
#     edf[is_session_last] = (edf[_prev_event].isnull()) | (edf[_prev_event] > session_seconds)
#     # TODO ? find end? edf['event_session_end'] = (edf.dt_next_event.notnull()) & (edf.dt_next_event > session_seconds)
#     edf[session_first] = edf[edf[is_session_first]][event_timestamp]
#     edf[session_last] = edf[edf[is_session_last]][event_timestamp]
#     edf[session_id] = edf[edf[is_session_first]][event_timestamp].rank(method='first').astype(int)
#     edf[session_id] = edf[session_id].fillna(method='ffill').astype(int)

#     # define 10 minutes delta
#     delta = 10 * 60
#     jump = session_seconds + delta
#     edf[estimated_start] = edf[session_first].map(lambda x: x if not type(x)==pd.Timestamp else find_nearest(gdf, x - tdel(seconds=delta), TIMESTAMP_COL)) # TODO replace with search start method
#     # FIXME error here NaT needs to be filled
#     edf[estimated_end] = edf[session_last].map(lambda x: x if not type(x)==pd.Timestamp else find_nearest(gdf, x + tdel(seconds=jump), TIMESTAMP_COL)) # TODO replace with search end method


#     # Fill empty fields
#     edf[session_first] = edf[session_first].fillna(method='ffill')
#     edf[estimated_start] = edf[estimated_start].fillna(method='ffill')
#     edf[session_last] = edf[session_last].fillna(method='bfill')
#     edf[estimated_end] = edf[estimated_end].fillna(method='bfill')
#     edf[session_len] = session_seconds

#     # TODO is it better to resort by index
#     edf.sort_index()
#     return edf


# TODO get events automatically
def get_events_infer(gdf: pd.DataFrame, limit: float = None):
    """Not implemented"""
    raise NotImplementedError(ERR_NOT_IMPLEMENTED)


# TODO this is freestyle libre specific
def get_events_from_fs_notes(gdf: pd.DataFrame, edf: pd.DataFrame):
    """Not implemented"""
    raise NotImplementedError(ERR_NOT_IMPLEMENTED)


def get_events_from_df(gdf: pd.DataFrame, edf: pd.DataFrame):
    """Not implemented"""
    raise NotImplementedError(ERR_NOT_IMPLEMENTED)


def get_events_from_times(gdf: pd.DataFrame, event_times: Iterable):
    """Not implemented"""
    raise NotImplementedError(ERR_NOT_IMPLEMENTED)


def describe_event(gdf, edf, eid):
    """Not implemented"""
    raise NotImplementedError(ERR_NOT_IMPLEMENTED)


"""Plotting

TODO:
* add plotting multiple sessions
* add comparing meals
"""
import matplotlib.pyplot as plt
from glucose import plot_glucose

import matplotlib.pyplot as plt


def plot_compare_sessions(
    gdf: pd.DataFrame,
    edf: pd.DataFrame,
    eid1: int,
    eid2: int,
    glbl: str = GLUCOSE_COL,
    tlbl: str = TIMESTAMP_COL,
    sharey: bool = True,
):
    """Compares two event sessions side by side by plotting their glucose curves,
    with aligned grid lines to make comparison easier.

    Args:
        gdf (pd.DataFrame): The glucose dataframe with timestamp index and glucose values.
        edf (pd.DataFrame): The event sessions dataframe.
        eid1 (int): ID of the first session to plot.
        eid2 (int): ID of the second session to plot.
        glbl (str, optional): Glucose column name. Defaults to GLUCOSE_COL.
        tlbl (str, optional): Timestamp column name. Defaults to TIMESTAMP_COL.
        sharey (bool, optional): If True, both plots share the same y-axis scale for comparison.
            Defaults to True.

    Raises:
        ValueError: If one of the session IDs is not found in `edf`.
    """
    if eid1 not in edf[session_id].unique():
        raise ValueError(f"Session {eid1} not found in sessions DataFrame.")
    if eid2 not in edf[session_id].unique():
        raise ValueError(f"Session {eid2} not found in sessions DataFrame.")

    # Extract sessions
    s1 = edf.loc[edf[session_id] == eid1].iloc[0]
    s2 = edf.loc[edf[session_id] == eid2].iloc[0]

    start1, end1 = pd.to_datetime(s1[session_first]), pd.to_datetime(s1[session_last])
    start2, end2 = pd.to_datetime(s2[session_first]), pd.to_datetime(s2[session_last])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=sharey)

    # Plot first session
    plt.sca(axes[0])
    plot_glucose(
        df=gdf,
        glbl=glbl,
        tlbl=tlbl,
        from_time=start1,
        to_time=end1,
        title=f"Session {eid1}\n{start1.strftime('%Y-%m-%d %H:%M')} → {end1.strftime('%H:%M')}",
    )
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)

    # Plot second session
    plt.sca(axes[1])
    plot_glucose(
        df=gdf,
        glbl=glbl,
        tlbl=tlbl,
        from_time=start2,
        to_time=end2,
        title=f"Session {eid2}\n{start2.strftime('%Y-%m-%d %H:%M')} → {end2.strftime('%H:%M')}",
    )
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

    # Align layout
    plt.tight_layout()
    plt.show()


def plot_day_sessions():
    """Not implemented"""
    raise NotImplementedError(ERR_NOT_IMPLEMENTED)


import matplotlib.pyplot as plt
import pandas as pd
import math


def plot_all_sessions(
    gdf: pd.DataFrame,
    edf: pd.DataFrame,
    glbl: str = GLUCOSE_COL,
    tlbl: str = TIMESTAMP_COL,
    sessions_per_page: int = 6,
    page: int = 0,
):
    """Plots all sessions in pages. Only one page is shown at a time.

    Correctly handles repeated session IDs by grouping and plotting each session once.

    Args:
        gdf (pd.DataFrame): The glucose dataframe with timestamp index and glucose values.
        edf (pd.DataFrame): The event sessions dataframe (can have repeated session IDs).
        glbl (str, optional): Glucose column name. Defaults to GLUCOSE_COL.
        tlbl (str, optional): Timestamp column name. Defaults to TIMESTAMP_COL.
        sessions_per_page (int, optional): Number of sessions per page. Defaults to 6.
        page (int, optional): Page number to display (starting from 0). Defaults to 0.
    """
    # Group by session_id to get unique sessions
    grouped_sessions = (
        edf.groupby(session_id)
        .agg({session_first: "min", session_last: "max"})
        .sort_values(session_first)
    )

    n_sessions = len(grouped_sessions)
    n_pages = math.ceil(n_sessions / sessions_per_page)

    if page < 0 or page >= n_pages:
        raise ValueError(f"Page {page} out of range. There are {n_pages} pages.")

    # Select sessions for this page
    start_idx = page * sessions_per_page
    end_idx = start_idx + sessions_per_page
    page_sessions = grouped_sessions.iloc[start_idx:end_idx]

    n_plots = len(page_sessions)
    n_rows = math.ceil(n_plots / 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), sharey=True)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, (sid, sess) in zip(axes, page_sessions.iterrows()):
        plt.sca(ax)
        start, end = pd.to_datetime(sess[session_first]), pd.to_datetime(sess[session_last])
        plot_glucose(
            df=gdf,
            glbl=glbl,
            tlbl=tlbl,
            from_time=start,
            to_time=end,
            title=f"Session {sid} ({start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%H:%M')})",
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Turn off extra axes if fewer plots than axes
    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.suptitle(f"All sessions — Page {page+1}/{n_pages}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    print(f"Showing page {page+1} of {n_pages} ({n_sessions} sessions total)")
    print(f"To see the next page, call with page={page+1}")


def plot_session_response(
    glucose_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    session_id: int,
    use_notes_as_title: str = False,
    session_title: Optional[str] = None,
    notes_col: str = "Notes",
    show_events: bool = False,
    events_tsp: str = TIMESTAMP_COL,
    glbl: str = GLUCOSE_COL,
    show_auc=True,
):
    """Plots the glucose response during one specific event session given by its session id.

    Args:
        glucose_df (_type_):
        sessions_df (_type_):
        session_id (_type_): the id of the session to plot

    Args:
        glucose_df (pd.DataFrame): the glucose dataframe.
        sessions_df (pd.DataFrame): the event sessions dataframe.
        session_id (int): the id of the session to plot.
        use_notes_as_title (str, optional): whether or not to use the Notes column for the event title.
            Overrides 'session_title'. Defaults to False.
        session_title (Optional[str], optional): the title of the session used in the plot.
            Overriden by 'use_notes_as_title'. Defaults to None.
        notes_col (str, optional): the name of the notes column, only used in combination with 'use_notes_as_title'.
            Defaults to 'Notes'.
        show_events (bool, optional): whether or not to show each specific event in the event session.
            Defaults to False.
        events_tsp (str, optional): timestamp column for event sessions, only used for showing events.
        glbl (str, optional): the glucose column name in the glucose dataframe. Defaults to GLUCOSE_COL.
        show_auc (bool, optional): whether or not to show the area under the curve in the plot. Defaults to True.
    """
    session = sessions_df[sessions_df["session_id"] == session_id]
    start = session.iloc[-1]["estimated_start"]
    end = session.iloc[-1]["estimated_end"]  # TODO move to get_session_bounds
    # TODO add Notes capability
    truncated_glucose = glucose_df[start:end][glbl]
    plt.plot(truncated_glucose)
    if show_auc:
        _plot_auc_above_threshold(truncated_glucose, truncated_glucose.mean())

    plt.axvline(
        session.iloc[-1]["session_first"],
        color="red",
        label="First event",
        linestyle="--",
        alpha=0.1,
    )
    if use_notes_as_title:
        sess_str = "; \n".join([x for x in session[notes_col] if x and isinstance(x, str)])
        session_title = f"Session with the events: ({sess_str})"
    if session_title:
        plt.title(session_title)
    if show_events:
        [
            plt.axvline(session.iloc[i][events_tsp], color="black", linestyle="--", alpha=0.1)
            for i in range(len(session))
        ]
    plt.xticks(rotation=45)


def _plot_auc_above_threshold(values: Iterable, threshold: float):
    """Plot area under the curve between some values and a specific threshold.

    Args:
        values (_type_): values under which to plot the area under the curve (in Y-axis).
        threshold (_type_): threshold above which to plot the area under the curve (in Y-axis).
    """
    lim_df = values.map(lambda x: x if x > threshold else threshold)
    plt.gca()
    plt.axhline(threshold, color="red", label="limit", linestyle="--", alpha=0.3)
    try:
        plt.fill_between(
            lim_df.index,
            lim_df,
            [threshold for a in lim_df.index],
            color="green",
            alpha=0.1,
            label=f"Estimated glucose quantity consumed",
        )
    except TypeError:
        logging.error("Could not fill the response graph")


"""Event Pattern recognition
"""


def auto_recognise_meals(
    gdf: pd.DataFrame,
    g_col: str = GLUCOSE_COL,
    tsp_col: str = TIMESTAMP_COL,
    lim: Optional[float] = None,
):
    """Automatically recognises meals based on the change in glucose values.
    Whenever glucose values start rising above a specific value and drop, this part will be considered a meal session.
    See Automatic meal recognition documentation under [Meals and Events](docs/meals_and_events.md).

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        g_col (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
        tsp_col (str, optional): the timestamp column name. Defaults to TIMESTAMP_COL.
        lim (Optional[float], optional): the glucose limit above which to detect a meal.
            If None, the mean of glucose in the dataframe will be used. Defaults to None.
    """
    if lim is None:
        lim = gdf[g_col].mean()
    g_events = gdf[gdf[g_col] > lim][[g_col, tsp_col]]
    e_sessions = get_event_sessions(
        events_df=g_events, glucose_df=gdf, event_tsp=tsp_col, session_seconds=60 * 60
    )
    # TODO select only some of the events instead of keeping all of them?
    meal_events = e_sessions
    return meal_events


"""Event metrics
"""


def get_event_metrics(gdf, start, end):
    """Not implemented"""
    raise NotImplementedError(ERR_NOT_IMPLEMENTED)


def get_event_auc(gdf: pd.DataFrame, start: GeneralDateType, end: GeneralDateType):
    """Get Area under the curve values for a time range.

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        start (general_date_type): string or datetime or date from which the event started.
        end (general_date_type): string or datetime or date at which the event ended.

    Returns:
        Tuple(float, float, float): Tuple containing Area Under the Curve (AUC) values for:
            - auc_mean: Area Under the Curve above the mean glucose.
            - auc_min: Area Under the Curve above the smallest glucose value.
            - auc_lim: Area Under the Curve above a predefined limit
    """
    d = gdf[start:end].copy()  # TODO use .loc
    auc_min = sum(d[_AUCMIN_MIN]) / (15 * 60)
    auc_mean = sum(d[_AUC_COL]) / (15 * 60)
    auc_lim = sum(d[_AUCLIM_COL]) / (15 * 60)
    return auc_mean, auc_min, auc_lim


_METRIC_AUCLIM_COL = "auc_min"
_METRIC_AUC_COL = "auc_mean"
_METRIC_AUCMIN_MIN = "auc_lim"


def get_sessions_auc(esdf: pd.DataFrame, gdf: pd.DataFrame):
    """Adds area under the curve values to Event Sessions DataFrame.

    Args:
        esdf (pd.DataFrame): the event sessions dataframe.
        gdf (pd.DataFrame): the glucose dataframe.

    Returns:
        pd.DataFrame: the event sessions dataframe with area under the curve columns added:
            - auc_mean: Area Under the Curve above the mean glucose.
            - auc_min: Area Under the Curve above the smallest glucose value.
            - auc_lim: Area Under the Curve above a predefined limit
    """
    edf = esdf.copy()
    sdf = edf[[session_id, session_first, session_last]].groupby(session_id).min()
    sdf["aucs"] = sdf.apply(lambda x: get_event_auc(gdf, x[session_first], x[session_last]), axis=1)
    edf[_METRIC_AUC_COL], edf[_METRIC_AUCMIN_MIN], edf[_METRIC_AUCLIM_COL] = zip(
        *edf.session_id.map(lambda x: sdf.loc[x, "aucs"])
    )

    return edf


def describe_all_sessions(esdf: pd.DataFrame, gdf: pd.DataFrame):
    """Describes the event sessions DataFrame by summarizing the number of sessions,
    their average/median lengths, and glucose response statistics aggregated across sessions.

    Args:
        esdf (pd.DataFrame): The event sessions DataFrame.
        gdf (pd.DataFrame): The glucose DataFrame aligned with the sessions.
    """
    console = Console()

    if esdf.empty:
        console.print("[bold red]No event sessions available.[/bold red]")
        return

    if not pd.api.types.is_datetime64_any_dtype(esdf[session_first]):
        esdf[session_first] = pd.to_datetime(esdf[session_first])
    if not pd.api.types.is_datetime64_any_dtype(esdf[session_last]):
        esdf[session_last] = pd.to_datetime(esdf[session_last])

    total_sessions = esdf[session_id].nunique()
    start_date = esdf[session_first].min()
    end_date = esdf[session_last].max()

    console.print("[bold magenta]Event Sessions Summary[/bold magenta]")
    console.print(
        f"• Total number of sessions: [bold green]{total_sessions}[/bold green]\n"
        f"• Session time span: [bold green]{start_date.strftime('%Y-%m-%d %H:%M')}[/bold green]"
        f" to [bold green]{end_date.strftime('%Y-%m-%d %H:%M')}[/bold green]"
    )

    if estimated_len in esdf.columns:
        avg_len = esdf.groupby(session_id)[estimated_len].first().mean() / 60
        med_len = esdf.groupby(session_id)[estimated_len].first().median() / 60
        console.print(
            f"• Average session length: [bold yellow]{avg_len:.1f} min[/bold yellow]\n"
            f"• Median session length: [bold yellow]{med_len:.1f} min[/bold yellow]"
        )

    table = Table(title="Session Glucose Response", show_header=True, header_style="bold magenta")
    table.add_column("Session ID", style="dim")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Mean Glucose")
    table.add_column("Max Glucose")

    for sid, group in esdf.groupby(session_id):
        start = group[session_first].iloc[0]
        end = group[session_last].iloc[0]
        truncated = gdf.loc[start:end, GLUCOSE_COL]
        if truncated.empty:
            continue
        table.add_row(
            str(sid),
            start.strftime("%Y-%m-%d %H:%M"),
            end.strftime("%Y-%m-%d %H:%M"),
            f"{truncated.mean():.2f}",
            f"{truncated.max():.2f}",
        )

    console.print(table)
    console.print(f"Columns in sessions data: [bold yellow]{', '.join(esdf.columns)}[/bold yellow]")
    console.print(
        f"[bold magenta]First rows in the sessions DataFrame:[/bold magenta]\n", esdf.head()
    )


def describe_session(gdf: pd.DataFrame, edf: pd.DataFrame, eid: int):
    """Describes a single session (e.g.: meal or activity) by providing metadata, duration,
    and detailed glucose statistics for that specific session.

    Args:
        gdf (pd.DataFrame): The glucose DataFrame with timestamp index and glucose column.
        edf (pd.DataFrame): The event sessions DataFrame.
        eid (int): The event/session ID to describe.
    """
    console = Console()

    if edf.empty or eid not in edf[session_id].unique():
        console.print(f"[bold red]Session with ID {eid} not found.[/bold red]")
        return

    # Extract the session row
    meal = edf[edf[session_id] == eid].iloc[0]

    start = pd.to_datetime(meal[session_first])
    end = pd.to_datetime(meal[session_last])
    duration_min = (end - start).total_seconds() / 60

    console.print(f"[bold magenta]Session {eid}[/bold magenta]")
    console.print(
        f"• Start: [bold green]{start.strftime('%Y-%m-%d %H:%M')}[/bold green]\n"
        f"• End: [bold green]{end.strftime('%Y-%m-%d %H:%M')}[/bold green]\n"
        f"• Duration: [bold yellow]{duration_min:.1f} min[/bold yellow]"
    )
    truncated = gdf.loc[start:end, GLUCOSE_COL]

    if truncated.empty:
        console.print("[bold red]No glucose readings found for this session.[/bold red]")
        return

    mean_glucose = truncated.mean()
    peak_glucose = truncated.max()
    min_glucose = truncated.min()
    delta_glucose = peak_glucose - min_glucose

    table = Table(
        title=f"Glucose Stats for Session {eid}", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Mean Glucose", f"{mean_glucose:.2f}")
    table.add_row("Peak Glucose", f"{peak_glucose:.2f}")
    table.add_row("Min Glucose", f"{min_glucose:.2f}")
    table.add_row("Delta (Peak - Min)", f"{delta_glucose:.2f}")

    console.print(table)

    console.print(
        f"[bold magenta]First glucose rows in session {eid}:[/bold magenta]\n", truncated.head()
    )
