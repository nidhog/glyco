import pandas as pd
import numpy as np
import hashlib
from datetime import datetime

from glyco.privacy import mask_private_information

def test_mask_private_information_no_changes():
    """Test with no masking options, should return the original DataFrame except for glucose"""
    data = {'glucose': [100, 110, 120],
            'tsp': ['2023-01-01 00:00', '2023-01-01 00:15', '2023-01-01 00:30'],
            'notes': ['test1', 'test2', 'test3']}
    gdf = pd.DataFrame(data)
    original_df = gdf.copy()
    masked_df, _, replaced = mask_private_information(gdf, [], [], 'glucose', 'tsp', '%Y-%m-%d %H:%M', set_start_date=None)
    assert masked_df[['tsp', 'notes']].equals(original_df[['tsp', 'notes']])

    assert replaced is None

def test_mask_private_information_remove_columns():
    """Test removing columns"""
    data = {'glucose': [100, 110, 120], 'tsp': ['2023-01-01 00:00', '2023-01-01 00:15', '2023-01-01 00:30'], 'notes': ['test1', 'test2', 'test3'], 'serial': [1, 2, 3]}
    gdf = pd.DataFrame(data)
    masked_df, _, _ = mask_private_information(
        gdf,
        ['notes', 'serial'],
        [], 'glucose',
        'tsp',
        '%Y-%m-%d %H:%M',
        set_start_date=None)
    assert 'notes' not in masked_df.columns
    assert 'serial' not in masked_df.columns
    assert 'glucose' in masked_df.columns
    assert 'tsp' in masked_df.columns

def test_mask_private_information_replace_columns():
    """Test replacing column values"""
    data = {'glucose': [100, 110, 120], 'tsp': ['2023-01-01 00:00', '2023-01-01 00:15', '2023-01-01 00:30'], 'notes': ['test1', 'test2', 'test3']}
    gdf = pd.DataFrame(data)
    masked_df, _, replaced = mask_private_information(gdf, [], ['notes'], 'glucose', 'tsp', '%Y-%m-%d %H:%M', set_start_date=None)
    assert 'notes' in masked_df.columns
    assert not masked_df['notes'].equals(gdf['notes'])
    assert replaced is not None
    assert replaced.equals(gdf[['notes']])
    for note in masked_df['notes']:
        try:
            hashlib.sha256(str(replaced['notes'].iloc[masked_df['notes'].to_list().index(note)]).encode()).hexdigest()
        except ValueError:
            assert False

def test_mask_private_information_set_start_date():
    """Test setting a new start date"""
    data = {'glucose': [100, 110, 120], 'tsp': ['2023-01-15 10:00', '2023-01-15 10:15', '2023-01-15 10:30']}
    gdf = pd.DataFrame(data)
    new_start_date = '2023-02-01 00:00'
    masked_df, _, _ = mask_private_information(gdf, [], [], 'glucose', 'tsp', '%Y-%m-%d %H:%M', set_start_date=new_start_date)
    expected_start=datetime.strptime('2023-02-01 10:00', '%Y-%m-%d %H:%M')

    assert pd.to_datetime(masked_df['tsp'].min()).date() == expected_start.date()

def test_mask_private_information_add_noise():
    """Test adding noise to glucose data"""
    data = {'glucose': [100, 110, 120], 'tsp': ['2023-01-01 00:00', '2023-01-01 00:15', '2023-01-01 00:30']}
    gdf = pd.DataFrame(data)
    masked_df, noise, _ = mask_private_information(gdf, [], [], 'glucose', 'tsp', '%Y-%m-%d %H:%M', set_start_date=None, noise_std=0.5)
    assert not masked_df['glucose'].equals(gdf['glucose'])
    assert np.allclose(masked_df['glucose'], gdf['glucose'] + noise)
    assert np.std(noise) > 0

def test_mask_private_information_all_options():
    """Test all options combined"""
    data = {'glucose': [100, 110, 120], 'tsp': ['2023-01-15 10:00', '2023-01-15 10:15', '2023-01-15 10:30'], 'notes': ['test1', 'test2', 'test3'], 'serial': [1, 2, 3]}
    gdf = pd.DataFrame(data)
    new_start_date = '2023-02-01 00:00'
    masked_df, noise, replaced = mask_private_information(gdf, ['serial'], ['notes'], 'glucose', 'tsp', '%Y-%m-%d %H:%M', set_start_date=new_start_date, noise_std=0.5)
    assert 'serial' not in masked_df.columns
    assert not masked_df['notes'].equals(gdf['notes'])
    assert not masked_df['glucose'].equals(gdf['glucose'])
    assert pd.to_datetime(masked_df['tsp'].min()).date() == pd.to_datetime(new_start_date).date()
    assert replaced is not None
    assert replaced.equals(gdf[['notes']])
    assert np.std(noise) > 0
