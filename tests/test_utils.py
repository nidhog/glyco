import pandas as pd
import matplotlib.pyplot as plt

from glyco import utils


def test_is_weekend():
    # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    assert utils.is_weekend(5) is True
    assert utils.is_weekend(6) is True
    assert utils.is_weekend(0) is False
    assert utils.is_weekend(4) is False


def test_find_nearest_returns_correct_index():
    idx = pd.date_range("2023-01-01", periods=5, freq="h")
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]}, index=idx)
    pivot = idx[2] + pd.Timedelta(minutes=10)  # closer to index[2]
    nearest = utils.find_nearest(df, pivot, "val")
    assert nearest == idx[2]


def test_find_nearest_skips_nan():
    idx = pd.date_range("2023-01-01", periods=3, freq="h")
    df = pd.DataFrame({"val": [1, float("nan"), 3]}, index=idx)
    pivot = idx[1]
    nearest = utils.find_nearest(df, pivot, "val")
    # should skip NaN and return closest non-nan
    assert nearest in [idx[0], idx[2]]


def test_init_plot_sets_ylim():
    utils.init_plot(l=4, w=3, gmin=0, gmax=10)
    ax = plt.gca()
    ylims = ax.get_ylim()
    assert ylims == (0, 10)
