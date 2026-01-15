"""Test glucose utils"""

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from glyco.glucose import compute_derivative, set_derivative, add_shifted_time, get_stats

def test_add_shifted_time():
    # note: 2023-01-01 is a Sunday
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 01:00:00', '2023-01-01 02:00:00', '2023-01-02 01:00:00']),
        'date': ['2023-01-01', '2023-01-01', '2023-01-02']
    })

    # shift of 7 hours
    result_df = add_shifted_time(df, 'timestamp', 'date', 7)

    assert 'shifted_timestamp' in result_df.columns
    assert 'shifted_date' in result_df.columns
    assert 'shifted_date_str' in result_df.columns
    assert 'shifted_hour' in result_df.columns
    assert 'shifted_weekday_number' in result_df.columns
    assert 'shifted_weekday_name' in result_df.columns
    assert 'shifted_is_weekend' in result_df.columns

    # check the values in the new columns are shifted
    assert result_df.loc[0, 'shifted_timestamp'] == pd.to_datetime('2022-12-31 18:00:00')
    assert result_df.loc[0, 'shifted_date'] == pd.to_datetime('2022-12-31').date()
    assert result_df.loc[0, 'shifted_date_str'] == '31-12-2022 (Saturday)'
    assert result_df.loc[0, 'shifted_hour'] == 18
    assert result_df.loc[0, 'shifted_weekday_number'] == 5 
    assert result_df.loc[0, 'shifted_weekday_name'] == 'Sat'
    assert result_df.loc[0, 'shifted_is_weekend']

def test_get_stats():
    df = pd.DataFrame({
        'glucose': [4, 4.5, 5, 5.2, 4.9],
        'category': ['A', 'B', 'A', 'B', 'A']
    })

    # test without group_by_col
    result = get_stats(df=df, stats_cols=['glucose'], percentiles=[0.25, 0.75])
    expected_columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    assert all(dim in result.index for dim in expected_columns)

    # test with group_by_col
    result_grouped = get_stats(df=df, stats_cols=['glucose'], group_by_col='category', percentiles=[0.25, 0.75])
    expected_columns_grouped = [('glucose', 'count'), ('glucose', 'mean'), ('glucose', 'std'), ('glucose', 'min'),
                                ('glucose', '25%'), ('glucose', '50%'), ('glucose', '75%'), ('glucose', 'max')]
    assert all(col in result_grouped.columns for col in expected_columns_grouped)


def test_compute_derivative_basic_seconds():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-01-01 00:00:00", "2023-01-01 00:01:00", "2023-01-01 00:03:00"]
            ),
            "glucose": [100, 110, 140],
        }
    )

    dg, dt, dgdt = compute_derivative(
        df, glucose_col="glucose", timestamp_col="timestamp", time_unit="s", sort_by_time=True
    )

    expected_dg = pd.Series([np.nan, 10.0, 30.0], index=df.index)
    expected_dt = pd.Series([np.nan, 60.0, 120.0], index=df.index)
    expected_dgdt = pd.Series([np.nan, 10.0 / 60.0, 30.0 / 120.0], index=df.index)

    assert_series_equal(dg, expected_dg, check_names=False)
    assert_series_equal(dt, expected_dt, check_names=False)
    assert_series_equal(dgdt, expected_dgdt, check_names=False)


def test_compute_derivative_duplicate_timestamps_yield_nan_derivative():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-01-01 00:00:00", "2023-01-01 00:00:00", "2023-01-01 00:01:00"]
            ),
            "glucose": [100, 105, 115],
        }
    )

    dg, dt, dgdt = compute_derivative(
        df, glucose_col="glucose", timestamp_col="timestamp", time_unit="s", sort_by_time=True
    )

    # After sorting, the middle row has dt==0 => derivative should be NaN (not inf)
    assert np.isnan(dgdt.iloc[1])
    assert dt.iloc[1] == 0.0


def test_compute_derivative_out_of_order_is_handled_by_sorting():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-01-01 00:02:00", "2023-01-01 00:00:00", "2023-01-01 00:01:00"]
            ),
            "glucose": [120, 100, 110],
        }
    )

    dg, dt, dgdt = compute_derivative(
        df, glucose_col="glucose", timestamp_col="timestamp", time_unit="s", sort_by_time=True
    )

    # In sorted order: (00:00->00:01->00:02) glucose (100->110->120)
    # So derivatives should be finite and non-negative (except first NaN)
    assert np.isnan(dgdt.iloc[0])
    assert (dgdt.iloc[1:] >= 0).all()
    assert (dt.iloc[1:] > 0).all()


def test_set_derivative_adds_columns_and_aligns_sorted_assignment():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-01-01 00:02:00", "2023-01-01 00:00:00", "2023-01-01 00:01:00"]
            ),
            "glucose": [120, 100, 110],
        }
    )

    out = set_derivative(
        df,
        glucose_col="glucose",
        timestamp_col="timestamp",
        time_unit="s",
        sort_by_time=True,
        copy=True,
    )

    # Uses module constants if present; otherwise defaults ("dG", "dt_s", "dGdt_per_s")
    dg_col = getattr(__import__("glyco.glucose", fromlist=["_DG_COL"]), "_DG_COL", "dG")
    dt_col = getattr(__import__("glyco.glucose", fromlist=["_DT_COL"]), "_DT_COL", "dt_s")
    dgdt_col = getattr(__import__("glyco.glucose", fromlist=["_DGDT_COL"]), "_DGDT_COL", "dGdt_per_s")

    assert dg_col in out.columns
    assert dt_col in out.columns
    assert dgdt_col in out.columns

    # Verify correctness by comparing to compute_derivative aligned to sorted timestamp order
    dg, dt, dgdt = compute_derivative(
        df, glucose_col="glucose", timestamp_col="timestamp", time_unit="s", sort_by_time=True
    )

    sorted_idx = df.sort_values("timestamp", kind="mergesort").index

    assert_series_equal(
        out.loc[sorted_idx, dg_col].reset_index(drop=True),
        dg.reset_index(drop=True),
        check_names=False,
    )
    assert_series_equal(
        out.loc[sorted_idx, dt_col].reset_index(drop=True),
        dt.reset_index(drop=True),
        check_names=False,
    )
    assert_series_equal(
        out.loc[sorted_idx, dgdt_col].reset_index(drop=True),
        dgdt.reset_index(drop=True),
        check_names=False,
    )
