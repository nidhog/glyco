from asyncio import events
import os
from os.path import isfile, join

from datetime import timedelta as tdel
from datetime import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt

from typing import Callable, Iterable

from glyco.glucose import DEFAULT_INPUT_TSP_COL, DEFAULT_INPUT_TSP_FMT, add_time_values
from .utils import find_nearest, autoplot

from .constants import GLUCOSE_COL, TIMESTAMP_COL
from typedframe import TypedDataFrame, DATE_TIME_DTYPE

# TODO: complete and use schema
class EventSession(TypedDataFrame):
    schema = {
    }

class MealSession(EventSession):
    schema = {
    }

_event_note_col = 'event_notes'
_event_ref_col = 'event_reference'
_freestyle_rec_type_col = "Record Type"
_freestyle_notes_rec_type = 6
_freestyle_glucose_rec_type = 0
_optional_cols = [_event_note_col, _event_ref_col]
event_default_cols = [TIMESTAMP_COL, _event_note_col, _event_ref_col]

def shift_time_fwd(t, h=1, m=0):
    return t + tdel(hours=h, minutes=m)


def shift_time_bck(t, h=1, m=0):
    return t - tdel(hours=h, minutes=m)


def read_meals(events_folder_path, file_ext: str = None, shift_time_function=shift_time_fwd, hours=1, minutes=0):
    event_files = [
        (
            shift_time_function(
                dt.fromtimestamp(os.stat(events_folder_path + f).st_mtime), hours, minutes
            ),
            f,
            f
        )
        for f in os.listdir(events_folder_path)
        if isfile(join(events_folder_path, f)) and (file_ext is None or f.endswith(file_ext))
    ]
    events_df = pd.DataFrame(event_files, columns=event_default_cols)
    events_df.index = events_df[TIMESTAMP_COL]
    events_df = add_time_values(events_df, tlbl=TIMESTAMP_COL, tsp_lbl=TIMESTAMP_COL, timestamp_is_formatted=True)
    return events_df


def validate_event_columns(df: pd.DataFrame, ref_col: str, note_col: str, tsp_col: str):
    if tsp_col not in df.columns:
        raise ValueError(f"The Timestamp column '{tsp_col}' is not in the event input columns."\
            "Please provide 'tsp_col' as input.")
    if ref_col not in df.columns:
        raise ValueError(f"The Event reference column '{ref_col}' is not in the input columns."\
            "Please provide 'ref_col' as input.")
    if note_col:
        if note_col not in df.columns:
            raise ValueError(f"The Event notes column '{note_col}' is not in the input columns."\
                "Please provide 'note_col' as input or give 'None' if the column does not exist.")
    
def read_events_csv(file_path, 
                    tsp_col: str = TIMESTAMP_COL, 
                    ref_col: str = _event_ref_col, 
                    note_col: str = None, 
                    tsp_fmt : str = DEFAULT_INPUT_TSP_FMT, 
                    timestamp_is_formatted: bool = False, 
                    delimiter: str = ",",
                    skiprows: int = 0
                    ):
    # TODO NEXT finish
    df = pd.read_csv(
        filepath_or_buffer=file_path,
        delimiter=delimiter,
        skiprows=skiprows
    )
    df = read_events_df(
        edf=df, 
        tsp_col=tsp_col, 
        ref_col=ref_col, 
        note_col=note_col, 
        tsp_fmt=tsp_fmt, 
        timestamp_is_formatted=timestamp_is_formatted
        )
    return df

def read_events_df(edf: pd.DataFrame, tsp_col: str = TIMESTAMP_COL, ref_col: str = _event_ref_col, note_col: str = None, tsp_fmt : str = DEFAULT_INPUT_TSP_FMT, timestamp_is_formatted: bool = False):
    validate_event_columns(df = edf, ref_col=ref_col, note_col=note_col, tsp_col=tsp_col)
    events_df = add_time_values(df=edf, tsp_lbl=tsp_col, tlbl=TIMESTAMP_COL, timestamp_fmt=tsp_fmt, timestamp_is_formatted=timestamp_is_formatted)
    events_df[_event_ref_col] = events_df[ref_col]
    events_df[_event_note_col] = events_df[ref_col] if note_col is None else events_df[note_col]
    return events_df
    



"""Freestyle Libre Specific
"""
def infer_events_from_notes(df, filter_notes_map: Callable = None):
    """
    filter_notes_map example: `lambda x: False if not x else str(x).startswith('food')`
    """
    events_df = df[df[_freestyle_rec_type_col]==_freestyle_notes_rec_type]
    if filter_notes_map:
        events_df = events_df[events_df[_event_note_col].map(filter_notes_map)]
    return events_df 

"""Events
"""
# TODO: use the following
# Event session schema
_default_event_session_seconds = 2*60*60
_original_timestamp = 'timestamp_origin'
_next_event = 'dt_next_event'
_prev_event = 'dt_prev_event'
is_session_first = 'is_session_first'
is_session_last = 'is_session_last'
session_first = 'session_first'
session_last = 'session_last'
session_id = 'session_id'
estimated_start = 'estimated_start'
estimated_end = 'estimated_end'
session_len = 'session_len'

# TODO: timestamp origin must be a new field
def sessionize_events(events_df: pd.DataFrame, gdf: pd.DataFrame, event_timestamp: str = _original_timestamp, session_seconds: int = _default_event_session_seconds):    
    """
    TODO document
    TODO takes very long to run
    FIXME OPTIMISE takes too long
    """
    edf = events_df.sort_values(event_timestamp)
    # TODO define column names outside
    # TODO check if the index is conserved: assert all(edf.index == events_df.index)
    edf[_next_event]= edf[event_timestamp].diff().dt.total_seconds()
    edf[_prev_event]= edf[event_timestamp].diff(-1).dt.total_seconds().map(abs)
    # TODO change name to is_session_first
    edf[is_session_first] = (edf[_next_event].isnull()) | (edf[_next_event] > session_seconds)
    edf[is_session_last] = (edf[_prev_event].isnull()) | (edf[_prev_event] > session_seconds)
    # TODO ? find end? edf['event_session_end'] = (edf.dt_next_event.notnull()) & (edf.dt_next_event > session_seconds)
    edf[session_first] = edf[edf[is_session_first]][event_timestamp]
    edf[session_last] = edf[edf[is_session_last]][event_timestamp]
    edf[session_id] = edf[edf[is_session_first]][event_timestamp].rank(method='first').astype(int)
    edf[session_id] = edf[session_id].fillna(method='ffill').astype(int)

    delta = 10 * 60
    jump = session_seconds + delta
    edf[estimated_start] = edf[session_first].map(lambda x: x if not type(x)==pd.Timestamp else find_nearest(gdf, x - tdel(seconds=delta), TIMESTAMP_COL)) # TODO replace with search start method
    # FIXME error here NaT needs to be filled
    edf[estimated_end] = edf[session_last].map(lambda x: x if not type(x)==pd.Timestamp else find_nearest(gdf, x + tdel(seconds=jump), TIMESTAMP_COL)) # TODO replace with search end method
    


    # Fill empty fields
    edf[session_first] = edf[session_first].fillna(method='ffill')
    edf[estimated_start] = edf[estimated_start].fillna(method='ffill')
    edf[session_last] = edf[session_last].fillna(method='bfill')
    edf[estimated_end] = edf[estimated_end].fillna(method='bfill')
    edf[session_len] = session_seconds
    
    # TODO is it better to resort by index
    edf.sort_index()
    return edf

# TODO get events automatically
def get_events_infer(gdf: pd.DataFrame, limit: float = None):
    events = None
    return events

# TODO this is freestyle libre specific
def get_events_from_fs_notes(gdf: pd.DataFrame, edf: pd.DataFrame):
    events = None
    return events

def get_events_from_df(gdf: pd.DataFrame, edf: pd.DataFrame):
    events = None
    return events

def get_events_from_times(gdf: pd.DataFrame, event_times: Iterable):
    events = None
    return events

def get_event_metrics(gdf, edf, eid):
    metrics = None
    return metrics

def describe_event(gdf, edf, eid):
    metrics = None
    return metrics



"""Plotting

TODO: 
* add plotting multiple sessions
* add comparing meals
"""
def plot_session_response(glucose_df, sessions_df, session_id, session_title=None, use_notes_as_title=False, notes_col='Notes', show_events=False, glbl=GLUCOSE_COL, pre_pad_min=20, post_pad_min=0, resp_time_min=120, t_lbl=None, show_auc=True):
    """Plots the response to a specific event given by its event time.
    """
    session = sessions_df[sessions_df['session_id']==session_id]
    start = session.iloc[-1]['estimated_start']
    end = session.iloc[-1]['estimated_end'] # TODO move to get_session_bounds
    # TODO add Notes capability
    truncated_glucose = glucose_df[start:end][glbl]
    plt.plot(truncated_glucose)
    if show_auc:
        plot_auc_above_threshold(truncated_glucose, truncated_glucose.mean())

    plt.axvline(session.iloc[-1]['session_first'], color='red', label='First event', linestyle='--', alpha=0.1)
    if use_notes_as_title:
        session_title = f"Session with the events: ({'; '.join([x for x in session[notes_col]])})"
    if session_title:
        plt.title(session_title)
    if show_events:
        [plt.axvline(session.iloc[i]['t'], color='black', linestyle='--', alpha=0.1) for i in range(len(session))]
    plt.xticks(rotation=45)


def plot_auc_above_threshold(values, threshold):
        lim_df = values.map(lambda x: x if x>threshold else threshold)
        plt.gca()
        plt.axhline(threshold, color='red', label='limit', linestyle='--', alpha=0.3)
        plt.fill_between(lim_df.index,  lim_df, [threshold for a in lim_df.index], color='green', alpha=0.1, label=f"Estimated glucose quantity consumed")

