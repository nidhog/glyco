"""Handles Meals and Events"""

import os
from os.path import isfile, join
from typing import Callable, Iterable, List, Optional

from datetime import timedelta as tdel
from datetime import datetime as dt

import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from rich.console import Console
from rich.table import Table

from .glucose import (
    GLUCOSE_COL,
    TIMESTAMP_COL,
    _AUC_COL,
    _AUCLIM_COL,
    _AUCMIN_MIN,
    _MEAL_NOTE_COL,
    DEFAULT_INPUT_TSP_FMT,
    GeneralDateType,
    add_time_values,
    ERR_NOT_IMPLEMENTED,
)
from .utils import autoplot, find_nearest

import logging

logger = logging.getLogger(__name__)

_event_note_col = "event_notes"
_event_ref_col = "event_reference"
_freestyle_rec_type_col = "Record Type"
_freestyle_notes_rec_type = 6
# _freestyle_glucose_rec_type = 0 TODO remove unused
# _optional_cols = [_event_note_col, _event_ref_col]
event_default_cols = [TIMESTAMP_COL, _event_note_col, _event_ref_col]


def _shift_time_fwd(t, h: int = 1, m: int = 0):
    """Shifts time forward

    Args:
        t (datetime.datetime): _description_
        h (int, optional): number of hours. Defaults to 1.
        m (int, optional): number of minutes. Defaults to 0.

    Returns:
        datetime.datetime: shifted time 't' forward by 'h' hours and 'm' minutes
    """
    return t + tdel(hours=h, minutes=m)


def _shift_time_bck(t, h=1, m=0):
    """Shifts time backward

    Args:
        t (datetime.datetime): _description_
        h (int, optional): number of hours. Defaults to 1.
        m (int, optional): number of minutes. Defaults to 0.

    Returns:
        datetime.datetime: shifted time 't' backwards by 'h' hours and 'm' minutes
    """
    return t - tdel(hours=h, minutes=m)


def read_meals_from_folder(
    events_folder_path: str,
    file_ext: Optional[str] = None,
    shift_time_function: Callable = _shift_time_fwd,
    shift_by_hours: int = 1,
    shift_by_minutes: int = 0,
):
    """Read meals or other events from a folder that contains photos.
    Used when you have meal photos stored in a folder.
    - Each file with the corresponding extension is assumed to be an event/meal.
    - The time of the event is assumed to be the time of creation of the file.
    - Time can be shifted forwards or backwards to account for different timezones.

    Args:
        events_folder_path (str): the path of the folders where the pictures/files are.
        file_ext (Optional[str], optional): the extension of files. If None all files will be considered. Defaults to None.
        shift_time_function (Callable, optional): function to shift time forwards or backwards. Defaults to _shift_time_fwd.
            This is because the timestamp of the files may not be in the correct timezone.
        shift_by_hours (int, optional): number of hours to shift by. Defaults to 1.
            This is because the timestamp of the files may not be in the correct timezone.
        shift_by_minutes (int, optional): number of minutes to shift by. Defaults to 0.
            This is because the timestamp of the files may not be in the correct timezone.

    Returns:
        pd.DataFrame: events dataframe:
            - indexed by the event timestamp.
            - contains the column TIMESTAMP_COL with the timestamp.
            - generated time values columns from from the timestamp.
    """
    event_files = [
        (
            shift_time_function(
                dt.fromtimestamp(os.stat(events_folder_path + f).st_mtime), shift_by_hours, shift_by_minutes
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
def get_events_from_notes(df, filter_notes_map: Callable = None):
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
# _original_timestamp = "timestamp_origin" TODO remove unused
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


def get_event_sessions(
    events_df: pd.DataFrame,
    glucose_df: pd.DataFrame,
    event_tsp: str = TIMESTAMP_COL,
    session_seconds: int = _default_event_session_seconds,
    delta_seconds: int = 10 * 60
):
    """Generates an Event Sessions DataFrame from: an Events DataFrame, and a Glucose DataFrame.
        - The events dataframe is used for finding the sessions based on the time between events.
        (Events are sorted by their corresponding timestamp)
        - The glucose dataframe is used for finding the estimated_start, estimated_end and estimated_len values.

    Args:
        events_df (pd.DataFrame): the events dataframe (with times of events)
        glucose_df (pd.DataFrame): the glucose dataframe
        event_tsp (str, optional): the timestamp in the events dataframe. Defaults to TIMESTAMP_COL.
        session_seconds (int, optional): lenght of a session: if time between two events is lower than
            the session lenght, they are grouped in the same session.
            Defaults to the value of _default_event_session_seconds.
        delta_seconds (int, optional): delta time in seconds to look for glucose values
            before and after the session, it is used to find the estimated_start and estimated_end values.
            The estimated_start is the nearest glucose timestamp before (session_first - delta)
            and the estimated_end is the nearest glucose timestamp after (session_last + delta).
            Defaults to 10 minutes (10 * 60).

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
    jump = tdel(seconds=session_seconds + delta_seconds)
    delta = tdel(seconds=delta_seconds)
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
            }
        )
        .assign(estimated_len=lambda x: (x[estimated_end] - x[estimated_start]).dt.total_seconds())
        .set_index(event_tsp, drop=False)
        .rename_axis("itsp")
    )

    return edf


def read_events_from_times(glucose_df: pd.DataFrame, event_times: Iterable):
    """Create an Event Session DataFrame from a list of event timestamps.
    Usage:
    ```python
    event_times = [dt(2023, 1, 1, 8, 0), dt(2023, 1, 1, 12, 30), dt(2023, 1, 1, 19, 0)]
    sessions_df = get_events_from_times(glucose_df, event_times)
    ```
    Args:
        glucose_df (pd.DataFrame): The glucose dataframe.
        event_times (Iterable): Iterable of event timestamps.
    Returns:
        pd.DataFrame: The Event Session Dataframe."""
    times = list(event_times)
    raw_events = pd.DataFrame({TIMESTAMP_COL: times, _event_ref_col: times})
    tsp_is_formatted = pd.api.types.is_datetime64_any_dtype(raw_events[TIMESTAMP_COL])
    events_df = read_events_df(
        edf=raw_events,
        tsp_col=TIMESTAMP_COL,
        ref_col=_event_ref_col,
        timestamp_is_formatted=tsp_is_formatted,
    )
    return get_event_sessions(events_df=events_df, glucose_df=glucose_df, event_tsp=TIMESTAMP_COL)


"""Plotting"""
def plot_compare_two_sessions(
    glucose_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    session_id_left: int,
    session_id_right: int,
    glucose_col: str = GLUCOSE_COL,
    timestamp_col: str = TIMESTAMP_COL,
    share_y_axis: bool = False,
    use_notes_as_title: bool = False,
    notes_col: str = _MEAL_NOTE_COL,
    show_events: bool = False
):
    """Compares two event sessions side by side by plotting their glucose curves,
    with aligned grid lines to make comparison easier.

    Args:
        glucose_df (pd.DataFrame): The glucose dataframe with timestamp index and glucose values.
        sessions_df (pd.DataFrame): The event sessions dataframe.
        session_id_left (int): ID of the first session to plot.
        session_id_right (int): ID of the second session to plot.
        glucose_col (str, optional): Glucose column name. Defaults to GLUCOSE_COL.
        timestamp_col (str, optional): Timestamp column name. Defaults to TIMESTAMP_COL.
        share_y_axis (bool, optional): If True, both plots share the same y-axis scale for comparison.
            Defaults to False.
        use_notes_as_title (bool, optional): Whether to use event notes as session titles.
            Defaults to False.
        notes_col (str, optional): The name of the notes column, only used in combination with 'use_notes_as_title'.
            Defaults to 'Notes'.
        show_events (bool, optional): Whether to show individual events within sessions (as lines).
            Defaults to False.

    Raises:
        ValueError: If one of the session IDs is not found in `edf`.
    """
    if session_id_left not in sessions_df[session_id].unique():
        raise ValueError(f"Session {session_id_left} not found in sessions DataFrame.")
    if session_id_right not in sessions_df[session_id].unique():
        raise ValueError(f"Session {session_id_right} not found in sessions DataFrame.")

    # Extract sessions
    s1 = sessions_df.loc[sessions_df[session_id] == session_id_left].iloc[0]
    s2 = sessions_df.loc[sessions_df[session_id] == session_id_right].iloc[0]
    start1, end1 = pd.to_datetime(s1[session_first]), pd.to_datetime(s1[session_last])
    start2, end2 = pd.to_datetime(s2[session_first]), pd.to_datetime(s2[session_last])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=share_y_axis)

    # Plot first session
    plt.sca(axes[0])
    plot_session_response(
        glucose_df=glucose_df,
        sessions_df=sessions_df,
        session_id=session_id_left,
        session_title=f"Session {session_id_left}\n{start1.strftime('%Y-%m-%d %H:%M')} → {end1.strftime('%H:%M')}",
        glbl=glucose_col,
        events_tsp=timestamp_col,
        use_notes_as_title=use_notes_as_title,
        notes_col=notes_col,
        ylim_source="session",
        show_events=show_events,
        autoplot=False
    )
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)

    # Plot second session
    plt.sca(axes[1])
    plot_session_response(
        glucose_df=glucose_df,
        sessions_df=sessions_df,
        session_id=session_id_right,
        session_title=f"Session {session_id_right}\n{start2.strftime('%Y-%m-%d %H:%M')} → {end2.strftime('%H:%M')}",
        glucose_col=glucose_col,
        timestamp_col=timestamp_col,
        events_tsp=timestamp_col,
        use_notes_as_title=use_notes_as_title,
        notes_col=notes_col,
        ylim_source="session",
        show_events=show_events,
        autoplot=False
    )
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    # Align layout
    plt.tight_layout()
    plt.show()


def plot_all_sessions(
    gdf: pd.DataFrame,
    edf: pd.DataFrame,
    glbl: str = GLUCOSE_COL,
    tlbl: str = TIMESTAMP_COL,
    sessions_per_page: int = 6,
    page: int = 0,
    use_notes_as_title: bool = False,
    notes_col: str = _MEAL_NOTE_COL,
    show_events: bool = False,
    share_y_axis: bool = False,
):
    """Plots all sessions in pages. Only one page is shown at a time.

    Correctly handles repeated session IDs by grouping and plotting each session once.

    Args:
        glucose_df (pd.DataFrame): The glucose dataframe with timestamp index and glucose values.
        sessions_df (pd.DataFrame): The event sessions dataframe (can have repeated session IDs).
        glucose_col (str, optional): Glucose column name. Defaults to GLUCOSE_COL.
        timestamp_col (str, optional): Timestamp column name. Defaults to TIMESTAMP_COL.
        sessions_per_page (int, optional): Number of sessions per page. Defaults to 6.
        page (int, optional): Page number to display (starting from 0). Defaults to 0.
        use_notes_as_title (bool, optional): Whether to use event notes as session titles.
            Defaults to False.
        notes_col (str, optional): The name of the notes column, only used in combination with 'use_notes_as_title'.
            Defaults to 'Notes'.
        show_events (bool, optional): Whether to show individual events within sessions (as lines).
            Defaults to False.
        share_y_axis (bool, optional): Share y-axis across subplots. Defaults to False.
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
    _, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), sharey=share_y_axis)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, (sid, sess) in zip(axes, page_sessions.iterrows()):
        plt.sca(ax)
        start, end = pd.to_datetime(sess[session_first]), pd.to_datetime(sess[session_last])
        plot_session_response(
            glucose_df=gdf,
            sessions_df=edf,
            session_id=sid,
            session_title=f"Session {sid} ({start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%H:%M')})",
            glbl=glbl,
            events_tsp=tlbl,
            use_notes_as_title=use_notes_as_title,
            notes_col=notes_col,
            ylim_source="session",
            show_events=show_events,
            autoplot=False
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)

    # Turn off extra axes if fewer plots than axes
    for ax in axes[n_plots:]:
        ax.axis("off")
    plt.suptitle(f"All sessions — Page {page+1}/{n_pages}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    print(f"Showing page {page+1} of {n_pages} ({n_sessions} sessions total)")
    print(f"To see the next page, call with page={page+1}")


@autoplot
def plot_session_response(
    glucose_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    session_id: int,
    use_notes_as_title: str = False,
    session_title: Optional[str] = None,
    notes_col: str = _MEAL_NOTE_COL,
    show_events: bool = False,
    events_tsp_col: str = TIMESTAMP_COL,
    glucose_col: str = GLUCOSE_COL,
    show_auc=True,
    y_lim: Optional[tuple[float, float]] = None,
    y_lim_source: str = "session",
    y_lim_pad_ratio: float = 0.10,
    y_lim_pad_abs: float = 0.5,
    **kwargs
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
        events_tsp_col (str, optional): timestamp column for event sessions, only used for showing events.
        glucose_col (str, optional): the glucose column name in the glucose dataframe. Defaults to GLUCOSE_COL.
        show_auc (bool, optional): whether or not to show the area under the curve in the plot. Defaults to True.
        y_lim (Optional[tuple[float, float]], optional): y-axis limits as (min, max).
            If None, limits are calculated automatically. Defaults to None.
        y_lim_source (str, optional): source for automatic y-axis limits calculation.
            Can be 'global' (entire glucose dataframe) or 'session' (only the session being plotted).
            Defaults to 'session'.
        y_lim_pad_ratio (float, optional): ratio of padding to add to y-axis limits when calculated automatically.
            Defaults to 0.10 (10%).
        y_lim_pad_abs (float, optional): absolute padding to add to y-axis limits when calculated automatically.
            Defaults to 0.5.
        **kwargs: additional keyword arguments passed to the plotting function.
    """
    session = sessions_df[sessions_df["session_id"] == session_id]
    start = session.iloc[-1]["estimated_start"]
    end = session.iloc[-1]["estimated_end"]  # TODO move to get_session_bounds
    # TODO add Notes capability
    truncated_glucose = glucose_df[start:end][glucose_col]
    plt.plot(truncated_glucose)
    if y_lim is not None:
        plt.ylim(bottom=y_lim[0], top=y_lim[1])
    else:
        if y_lim_source == "global":
            y_series = glucose_df[glucose_col]
        else:
            y_series = truncated_glucose

        y_min = float(y_series.min())
        y_max = float(y_series.max())
        y_range = y_max - y_min
        pad = max(y_lim_pad_abs, y_lim_pad_ratio * (y_range if y_range > 0 else max(abs(y_max), 1.0)))
        plt.ylim(bottom=y_min - pad, top=y_max + pad)
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
            plt.axvline(session.iloc[i][events_tsp_col], color="black", linestyle="--", alpha=0.1)
            for i in range(len(session))
        ]
    plt.xticks(rotation=45)


def _plot_auc_above_threshold(values: Iterable, threshold: float):
    """Plot area under the curve where values are above threshold.
    Args:
        values (_type_): values under which to plot the area under the curve (in Y-axis).
        threshold (_type_): threshold above which to plot the area under the curve (in Y-axis)."""
    if values is None or len(values) == 0:
        return

    s = values.astype(float)
    x = s.index.to_numpy()
    y = s.to_numpy()

    where = y > threshold

    plt.axhline(threshold, color="red", label="limit", linestyle="--", alpha=0.3)

    try:
        plt.fill_between(
            x,
            y,
            threshold,          # scalar baseline (important)
            where=where,        # prevents fill outside segments
            interpolate=True,   # neat crossing at threshold
            alpha=0.1,
            label="Estimated glucose quantity consumed",
        )
    except TypeError:
        logging.exception("Could not fill the response graph")



"""Event Pattern recognition
"""
def infer_events_from_glucose_excursions(
    glucose_df: pd.DataFrame,
    glucose_col: str = GLUCOSE_COL,
    timestamp_col: str = TIMESTAMP_COL,
    detection_threshold: Optional[float] = None,
    session_seconds: int = _default_event_session_seconds,
):
    """Automatically infers events based on the change in glucose values.
    Whenever glucose values start rising above a specific value and drop, this part will be considered an event session.
    See Automatic event recognition documentation under [Meals and Events](docs/meals_and_events.md).

    Args:
        gdf (pd.DataFrame): the glucose dataframe.
        glucose_col (str, optional): the glucose column name. Defaults to GLUCOSE_COL.
        timestamp_col (str, optional): the timestamp column name. Defaults to TIMESTAMP_COL.
        detection_threshold (Optional[float], optional): the glucose limit above which to detect a meal.
            If None, the mean of glucose in the dataframe will be used. Defaults to None.
    """
    if detection_threshold is None:
        detection_threshold = glucose_df[glucose_col].mean()
    g_events = glucose_df[glucose_df[glucose_col] > detection_threshold][[glucose_col, timestamp_col]]
    e_sessions = get_event_sessions(
        events_df=g_events, glucose_df=glucose_df, event_tsp=timestamp_col, session_seconds=session_seconds
    )
    # TODO select only some of the events instead of keeping all of them?
    # TODO select only important columns?
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


def describe_all_sessions(event_session_df: pd.DataFrame, glucose_df: pd.DataFrame, display_first_rows: bool = False):
    """Describes the event sessions DataFrame by summarizing the number of sessions,
    their average/median lengths, and glucose response statistics aggregated across sessions.

    Args:
        event_session_df (pd.DataFrame): The event sessions DataFrame.
        glucose_df (pd.DataFrame): The glucose DataFrame aligned with the sessions.
    """
    console = Console()

    if event_session_df.empty:
        console.print("[bold red]No event sessions available.[/bold red]")
        return

    if not pd.api.types.is_datetime64_any_dtype(event_session_df[session_first]):
        event_session_df[session_first] = pd.to_datetime(event_session_df[session_first])
    if not pd.api.types.is_datetime64_any_dtype(event_session_df[session_last]):
        event_session_df[session_last] = pd.to_datetime(event_session_df[session_last])

    total_sessions = event_session_df[session_id].nunique()
    total_events = len(event_session_df)
    start_date = event_session_df[session_first].min()
    end_date = event_session_df[session_last].max()

    console.print("[bold magenta]Event Sessions Summary[/bold magenta]")
    console.print(
        f"• Total number of sessions: [bold green]{total_sessions}[/bold green]\n"
        f"• Total number of events: [bold green]{total_events}[/bold green]\n"
        f"• Session time span: [bold green]{start_date.strftime('%Y-%m-%d %H:%M')}[/bold green]"
        f" to [bold green]{end_date.strftime('%Y-%m-%d %H:%M')}[/bold green]"
    )

    if estimated_len in event_session_df.columns:
        avg_len = event_session_df.groupby(session_id)[estimated_len].first().mean() / 60
        med_len = event_session_df.groupby(session_id)[estimated_len].first().median() / 60
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

    for sid, group in event_session_df.groupby(session_id):
        start = group[session_first].iloc[0]
        end = group[session_last].iloc[0]
        truncated = glucose_df.loc[start:end, GLUCOSE_COL]
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
    console.print(f"Columns in sessions data: [bold yellow]{', '.join(event_session_df.columns)}[/bold yellow]")
    if display_first_rows:
        console.print(
            f"[bold magenta]First rows in the sessions DataFrame:[/bold magenta]\n", event_session_df.head()
            )


def describe_session(gdf: pd.DataFrame,
                     edf: pd.DataFrame,
                     eid: int,
                     check_notes_col: bool = False,
                     notes_col: str = _MEAL_NOTE_COL):
    """Describes a single session (e.g.: meal or activity) by providing metadata, duration,
    and detailed glucose statistics for that specific session.

    Args:
        gdf (pd.DataFrame): The glucose DataFrame with timestamp index and glucose column.
        edf (pd.DataFrame): The event sessions DataFrame.
        eid (int): The event/session ID to describe.
        check_notes_col (bool, optional): Whether to check and display notes from the notes column.
            Defaults to False.
        notes_col (str, optional): The name of the notes column to check for event notes.
            Defaults to _MEAL_NOTE_COL.
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
    if check_notes_col and notes_col in edf.columns:
        session_notes = edf.loc[edf[session_id] == eid, notes_col]
        notes = [note for note in session_notes if isinstance(note, str) and note]
        if notes:
            console.print(
                f"• Events: [bold cyan]{'; '.join(notes)}[/bold cyan]"
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


def get_session_metrics(glucose_df, event_session_df, metrics_list=None):
    """
    Computes specified metrics for each event session using glucose data.
    Returns a new dataframe with added columns (does not mutate inputs).
    This can be used to compute a variety of glucose response metrics
    for different meals and comparing these meals to each other.

    Args:
        glucose_df (pd.DataFrame): The glucose DataFrame with timestamp index and glucose column.
        event_session_df (pd.DataFrame): The event sessions DataFrame.
        metrics_list (Optional[List[str]], optional): List of metrics to compute.
            If None, computes all available metrics. Defaults to None.
            All available metrics can be found in the glucose event metrics documentation.
            These include:
                * Baseline glucose: baseline_glucose
                * Summary Statistics:
                mean_glucose, median_glucose, std_glucose, coefficient_of_variation, peak_glucose, min_glucose, delta_glucose
                * Time in ranges (TIR/TAR/TBR): time_in_range_min, time_in_range_pct,
                time_above_10_min, time_above_10_pct, time_above_13_9_min, time_above_13_9_pct,
                time_below_3_9_min, time_below_3_9_pct, time_below_3_0_min, time_below_3_0_pct
                * Time-based: time_to_peak, time_to_baseline
                * Area under the curve: iauc_0_2h, iauc_0_4h, iauc_session
                * Rise/fall rates: max_rise_rate, max_fall_rate
                * Estimates of standard metrics: mage, lability_index
    """
    # TODO take functions out and reuse slicing etc.
    def ensure_datetime_index(gdf):
        if isinstance(gdf.index, pd.DatetimeIndex):
            return gdf.sort_index()
        ts = pd.to_datetime(gdf[TIMESTAMP_COL], errors="coerce")
        return gdf.assign(**{TIMESTAMP_COL: ts}).set_index(TIMESTAMP_COL).sort_index()

    def slice_df(gdf, start, end):
        if pd.isna(start) or pd.isna(end):
            return gdf.iloc[0:0]
        return gdf.loc[pd.to_datetime(start):pd.to_datetime(end)]

    def dt_minutes(ts):
        t = ts.to_numpy(dtype="datetime64[ns]")
        if len(t) < 2:
            return np.zeros(len(t))
        dt = (t[1:] - t[:-1]).astype("timedelta64[s]").astype(float) / 60.0
        dt = np.clip(dt, 0, np.inf)
        return np.append(dt, 0.0)

    def iauc(ts, g, baseline):
        t0 = ts.iloc[0]
        x = (ts - t0).dt.total_seconds().to_numpy() / 60.0
        y = np.clip(g.to_numpy() - baseline, 0, np.inf)
        if len(x) < 2:
            return np.nan
        return float(np.trapezoid(y, x))

    def rise_fall_rates(ts, g):
        t = ts.to_numpy(dtype="datetime64[ns]")
        y = g.to_numpy(dtype=float)
        if len(y) < 2:
            return np.nan, np.nan
        dt = (t[1:] - t[:-1]).astype("timedelta64[s]").astype(float) / 60.0
        dg = y[1:] - y[:-1]
        valid = (dt > 0) & np.isfinite(dg)
        if not valid.any():
            return np.nan, np.nan
        rate = dg[valid] / dt[valid]
        return float(rate.max()), float(rate.min())

    def mage(ts, g):
        y = g.to_numpy(dtype=float)
        if len(y) < 5:
            return np.nan
        sd = np.nanstd(y)
        if sd == 0 or not np.isfinite(sd):
            return np.nan
        dy = np.diff(y)
        s = np.sign(dy)
        idx = [0]
        for i in range(1, len(s)):
            if s[i-1] > 0 and s[i] <= 0:
                idx.append(i)
            elif s[i-1] < 0 and s[i] >= 0:
                idx.append(i)
        idx.append(len(y) - 1)
        idx = sorted(set(idx))
        exc = []
        for a, b in zip(idx[:-1], idx[1:]):
            amp = abs(y[b] - y[a])
            if amp > sd:
                exc.append(amp)
        return float(np.mean(exc)) if exc else np.nan

    def lability_index(ts, g):
        t = ts.to_numpy(dtype="datetime64[ns]")
        y = g.to_numpy(dtype=float)
        if len(y) < 2:
            return np.nan
        dt_h = (t[1:] - t[:-1]).astype("timedelta64[s]").astype(float) / 3600.0
        dg = y[1:] - y[:-1]
        valid = (dt_h > 0) & np.isfinite(dg)
        if not valid.any():
            return np.nan
        return float(np.mean((dg[valid] ** 2) / dt_h[valid]))

    # TODO: only calculate requested metrics
    if metrics_list is None:
        metrics_list = [
            "baseline_glucose",
            "mean_glucose", "median_glucose", "std_glucose", "coefficient_of_variation",
            "min_glucose", "peak_glucose", "delta_glucose", "incremental_peak",
            "time_to_peak", "time_to_baseline",
            "iauc_0_2h", "iauc_0_4h", "iauc_session",
            "time_in_range_min", "time_in_range_pct",
            "time_above_10_min", "time_above_10_pct",
            "time_above_13_9_min", "time_above_13_9_pct",
            "time_below_3_9_min", "time_below_3_9_pct",
            "time_below_3_0_min", "time_below_3_0_pct",
            "max_rise_rate", "max_fall_rate",
            "mage", "lability_index",
        ]

    # TODO: define glucose thresholds as parameters or settings
    LOW_39 = 3.9
    LOW_30 = 3.0
    HIGH_10 = 10.0
    HIGH_139 = 13.9
    RECOVERY_DELTA = 0.2

    gdf = ensure_datetime_index(glucose_df)
    edf = event_session_df.copy()

    session_bounds = (
        edf.groupby(session_id)
        .agg(
            sess_first=(session_first, "min"),
            sess_last=(session_last, "max"),
            est_start=(estimated_start, "min"),
            est_end=(estimated_end, "max"),
        )
        .reset_index()
    )

    per_session = {}

    for row in session_bounds.itertuples(index=False):
        sid = int(row.session_id)
        seg = slice_df(gdf, row.est_start, row.est_end)

        out = {m: np.nan for m in metrics_list}

        if seg.empty:
            per_session[sid] = out
            continue

        g = pd.to_numeric(seg[GLUCOSE_COL], errors="coerce").dropna()
        ts = g.index.to_series()

        if g.empty:
            per_session[sid] = out
            continue

        pre = slice_df(gdf, row.sess_first - pd.Timedelta(minutes=20), row.sess_first)
        baseline = float(pd.to_numeric(pre[GLUCOSE_COL], errors="coerce").mean()) if not pre.empty else float(g.iloc[0])

        mean_g = g.mean()
        med_g = g.median()
        std_g = g.std(ddof=1)
        cv = (std_g / mean_g) * 100 if mean_g != 0 else np.nan

        min_g = g.min()
        peak_g = g.max()
        delta_g = peak_g - min_g
        inc_peak = peak_g - baseline

        peak_time = g.idxmax()
        ttp = (peak_time - row.sess_first).total_seconds() / 60.0

        post = slice_df(gdf, peak_time, row.sess_first + pd.Timedelta(hours=4))
        post_g = pd.to_numeric(post[GLUCOSE_COL], errors="coerce")
        t_rec = np.nan
        if not post_g.empty:
            hit = post_g <= (baseline + RECOVERY_DELTA)
            if hit.any():
                t_rec = (hit.index[0] - row.sess_first).total_seconds() / 60.0

        # these are used for iauc calculation
        seg_2h = slice_df(gdf, row.sess_first, row.sess_first + pd.Timedelta(hours=2))
        seg_4h = slice_df(gdf, row.sess_first, row.sess_first + pd.Timedelta(hours=4))

        iauc_2 = (
            iauc(
                seg_2h.index.to_series(),
                pd.to_numeric(seg_2h[GLUCOSE_COL], errors="coerce"),
                baseline,
            )
            if not seg_2h.empty
            else np.nan
        )

        iauc_4 = (
            iauc(
                seg_4h.index.to_series(),
                pd.to_numeric(seg_4h[GLUCOSE_COL], errors="coerce"),
                baseline,
            )
            if not seg_4h.empty
            else np.nan
        )

        iauc_sess = iauc(ts, g, baseline)

        dt = dt_minutes(ts)
        total_min = dt.sum()

        gv = g.to_numpy()
        tir = ((gv >= LOW_39) & (gv <= HIGH_10))
        tar10 = gv > HIGH_10
        tar139 = gv > HIGH_139
        tbr39 = gv < LOW_39
        tbr30 = gv < LOW_30

        def mins(mask): return float(dt[mask].sum()) if total_min > 0 else np.nan
        def pct(x): return (x / total_min) * 100 if total_min > 0 else np.nan

        max_rise, max_fall = rise_fall_rates(ts, g)

        vals = dict(
            baseline_glucose=baseline,
            mean_glucose=mean_g,
            median_glucose=med_g,
            std_glucose=std_g,
            coefficient_of_variation=cv,
            min_glucose=min_g,
            peak_glucose=peak_g,
            delta_glucose=delta_g,
            incremental_peak=inc_peak,
            time_to_peak=ttp,
            time_to_baseline=t_rec,
            iauc_0_2h=iauc_2,
            iauc_0_4h=iauc_4,
            iauc_session=iauc_sess,
            time_in_range_min=mins(tir),
            time_in_range_pct=pct(mins(tir)),
            time_above_10_min=mins(tar10),
            time_above_10_pct=pct(mins(tar10)),
            time_above_13_9_min=mins(tar139),
            time_above_13_9_pct=pct(mins(tar139)),
            time_below_3_9_min=mins(tbr39),
            time_below_3_9_pct=pct(mins(tbr39)),
            time_below_3_0_min=mins(tbr30),
            time_below_3_0_pct=pct(mins(tbr30)),
            max_rise_rate=max_rise,
            max_fall_rate=max_fall,
            mage=mage(ts, g),
            lability_index=lability_index(ts, g),
        )

        for k in out:
            out[k] = vals.get(k, np.nan)

        per_session[sid] = out

    # get metrics in the list
    out_df = edf.copy()
    for m in metrics_list:
        out_df[m] = out_df[session_id].map(lambda sid: per_session.get(int(sid), {}).get(m, np.nan))

    return out_df
