"""Test glucose utils"""
import pandas as pd
from datetime import timedelta
from glyco.glucose import add_shifted_time, get_stats

def test_add_shifted_time():
    # Create a sample DataFrame 2023-01-01 is a Sunday
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 01:00:00', '2023-01-01 02:00:00', '2023-01-02 01:00:00']),
        'date': ['2023-01-01', '2023-01-01', '2023-01-02']
    })

    # Call the function with a shift of 7 hours
    result_df = add_shifted_time(df, 'timestamp', 'date', 7)

    # Assert that the new columns are shifted back 7 hours
    assert 'shifted_timestamp' in result_df.columns
    assert 'shifted_date' in result_df.columns
    assert 'shifted_date_str' in result_df.columns
    assert 'shifted_hour' in result_df.columns
    assert 'shifted_weekday_number' in result_df.columns
    assert 'shifted_weekday_name' in result_df.columns
    assert 'shifted_is_weekend' in result_df.columns

    # Check the values in the new columns for the first row
    assert result_df.loc[0, 'shifted_timestamp'] == pd.to_datetime('2022-12-31 18:00:00')
    assert result_df.loc[0, 'shifted_date'] == pd.to_datetime('2022-12-31').date()
    assert result_df.loc[0, 'shifted_date_str'] == '31-12-2022 (Saturday)'
    assert result_df.loc[0, 'shifted_hour'] == 18
    assert result_df.loc[0, 'shifted_weekday_number'] == 5 
    assert result_df.loc[0, 'shifted_weekday_name'] == 'Sat'
    assert result_df.loc[0, 'shifted_is_weekend']

def test_get_stats():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'glucose': [4, 4.5, 5, 5.2, 4.9],
        'category': ['A', 'B', 'A', 'B', 'A']
    })

    # Test case 1: without group_by_col
    result = get_stats(df=df, stats_cols=['glucose'], percentiles=[0.25, 0.75])
    # Assert that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert that the expected columns are present in the result DataFrame's index
    expected_columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    assert all(dim in result.index for dim in expected_columns)

    # Test case 2: with group_by_col
    result_grouped = get_stats(df=df, stats_cols=['glucose'], group_by_col='category', percentiles=[0.25, 0.75])
    
    # Assert that the result is a DataFrame
    assert isinstance(result_grouped, pd.DataFrame)

    # Assert that the expected columns are present in the result DataFrame
    expected_columns_grouped = [('glucose', 'count'), ('glucose', 'mean'), ('glucose', 'std'), ('glucose', 'min'),
                                ('glucose', '25%'), ('glucose', '50%'), ('glucose', '75%'), ('glucose', 'max')]
    assert all(col in result_grouped.columns for col in expected_columns_grouped)

