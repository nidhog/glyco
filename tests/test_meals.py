import pandas as pd
import pytest
from datetime import datetime, timedelta

from glyco import meals


def test_shift_time_fwd_and_bck():
    t = datetime(2023, 1, 1, 12, 0, 0)
    assert meals._shift_time_fwd(t, h=2, m=30) == t + timedelta(hours=2, minutes=30)
    assert meals._shift_time_bck(t, h=1, m=15) == t - timedelta(hours=1, minutes=15)


def test_validate_event_columns_success():
    df = pd.DataFrame({
        "timestamp": [datetime(2023, 1, 1, 12, 0)],
        "event_reference": ["meal1"],
        "event_notes": ["note1"]
    })
    # Should not raise
    meals.validate_event_columns(df, "event_reference", "event_notes", "timestamp")


def test_validate_event_columns_failure():
    df = pd.DataFrame({"foo": [1]})
    with pytest.raises(ValueError):
        meals.validate_event_columns(df, "event_reference", "event_notes", "timestamp")


def test_read_events_df_minimal():
    df = pd.DataFrame({
        "timestamp": [datetime(2023, 1, 1, 12, 0)],
        "event_reference": ["meal1"],
    })
    result = meals.read_events_df(df, tsp_col="timestamp", ref_col="event_reference")
    assert "timestamp" in result.columns
    assert "event_notes" in result.columns
    assert "event_reference" in result.columns