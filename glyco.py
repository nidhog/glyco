from datetime import timedelta as tdel
import enum
import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np
import scipy as sp
from scipy import io, signal

# define default values
# TODO
G_LBL = 'G'  # Generated glucose
T_LBL = 't'  # Generated time TODO: handle different tmz, UTC etc.
DATE_LBL = 'date'  # TODO

DT = 'dt'
DG = 'dg'
DGDT = 'dg_dt'
AUC = 'auc_mean'
AUCLIM = 'auc_lim'
AUCMIN = 'auc_min'
G_LIM = 6

# time derivatives
DLBL = 'date'
HLBL = 'hour'
DAYOFWEEK = 'dayofweek'
WEEKDAYLBL = 'weekday'
WEEKENDLBL = 'is_weekend'

# TODO move into utils.py + unit conversion etc.
weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed',
               3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}


def is_weekend(x): return 1 if x % 7 > 4 else 0


class Devices(enum.Enum):
    ABBOTT = "ABBOTT"  # FreeStyle Libre


class Units(enum.Enum):
    MMOL = "mmol/L"
    MGDL = "mg/dL"
    GL = "g/L"


def get_time_labels(device):
    # TODO
    if device == Devices.ABBOTT:
        timestamp = 'Device Timestamp'
        format = '%d-%m-%Y %H:%M'
        return timestamp, format


def get_glucose_label(device):
    # TODO
    if device == Devices.ABBOTT:
        return 'Historic Glucose mmol/L'


def read_from_csv(file_path,
                  device,  # TODO define type
                  skiprows=0,
                  delimiter=',',
                  tsp_lbl='Device Timestamp', tsp_fmt='%d-%m-%Y %H:%M',
                  glucose_lbl='Historic Glucose mmol/L',
                  set_labels_by_device=False,
                  unit=Units.MMOL,
                  glbl=G_LBL,
                  tlbl=T_LBL,
                  glim=G_LIM,  # static glucose used for auc
                  dlbl=DATE_LBL,
                  set_deriv=True, set_itg=True,
                  interpolate=True, interp_met='polynomial', interp_ord=2, rolling_avg=3):
    if not is_valid_entry(device, unit):
        # TODO handle error
        return
    if set_labels_by_device:
        tsp_lbl, tsp_fmt = get_time_labels(device)
        glucose_lbl = get_glucose_label(device)

    df = pd.read_csv(file_path, skiprows=skiprows, delimiter=delimiter)
    # TODO log reading, size etc.
    return get_glucose_df(df, tsp_lbl, tsp_fmt, glucose_lbl, 
                        unit, glbl, tlbl, glim, dlbl, set_deriv, set_itg,
                        interpolate, interp_met, interp_ord, rolling_avg)


def get_glucose_df(df,
                   tsp_lbl='Device Timestamp',
                   tsp_fmt='%d-%m-%Y %H:%M', glucose_lbl='Historic Glucose mmol/L',
                   unit=Units.MMOL,
                   glbl=G_LBL,
                   tlbl=T_LBL,
                   glim=G_LIM,  # static glucose used for auc
                   dlbl=DATE_LBL,
                   set_deriv=True, set_itg=True,
                   interpolate=True, interp_met='polynomial', interp_ord=2, rolling_avg=3):

    # derive time, date, hour etc. from timestamp
    df[tlbl] = pd.to_datetime(df[tsp_lbl], format=tsp_fmt)
    df[dlbl] = df[tlbl].dt.date
    df[HLBL] = df[tlbl].dt.hour
    df[DAYOFWEEK] = df[tlbl].dt.weekday
    df[WEEKDAYLBL] = df[DAYOFWEEK].map(weekday_map)
    df[WEEKENDLBL] = df[DAYOFWEEK].map(is_weekend)

    # convert to mmol/L
    df[glbl] = df[glucose_lbl] if unit == Units.MMOL \
        else convert_unit(df[glucose_lbl], unit, Units.MMOL)

    # index by time and keep time column
    df = df.set_index(tlbl)
    df[tlbl] = df.index
    df.sort_index(inplace=True)

    # derive
    if set_deriv:
        df = set_derivative(df, glbl, tlbl)

    # set area under the curve
    if set_itg:
        df = set_auc(df, glbl, tlbl, glim)

    # interpolate and smoothen glucose
    
    if interpolate:
        # TODO handle Nan
        # TODO FIX missing values should still be nan otherwise averaged
        df[glbl] = df[glbl].rolling(window=rolling_avg).mean()
        # TODO smoothen methods
        df[glbl] = df[glbl].interpolate(method=interp_met, order=interp_ord)
        df = df[df[glbl].map(lambda g: g>0 and g<30)]
    return df

def plot_compare_multi(df, start, end, patient_ids = None, normalize=False):
    # TODO check right columns
    if type(df.index) != pd.MultiIndex:
        # TODO handle exception
        return
    if patient_ids is None:
        # TODO log
        patient_ids = df.index.get_level_values(0).unique()
    
    begin_plot()
    for i in patient_ids:
        # TODO skip patient if not exists?
        
        d=df.loc[i].loc[start:end][G_LBL]
        nd = (d - d.mean())/d.std()
        plt.plot(nd if normalize else d, label=i)
        # print(i, d[G_LBL].iloc[0:10].mean(), df.loc[i].date.min(), df.loc[i].date.max())
    end_plot()

DEFAULT_FS = 125 # TODO move utils
def fourier_transform(s, fs=DEFAULT_FS):
    # TODO move to utils.py
    freqs = np.fft.rfftfreq(len(s), 1/fs)
    fft = np.fft.rfft(s)
    return freqs, fft

def bandpass_filter(order, signal, pass_band, fs=DEFAULT_FS):
    b, a = sp.signal.butter(order, pass_band, btype='bandpass', fs=fs)
    return sp.signal.filtfilt(b, a, signal)

def plot_filtered(s, f_band, fs=DEFAULT_FS):
    begin_plot()
    filtered_s = bandpass_filter(s, f_band, fs)
    plt.plot(s)
    plt.plot((filtered_s+s.mean()), color='black')
    end_plot()

def plot_fft(s, t, freqs = None, fft = None):
    if freqs is None or fft is None:
        freqs, fft = fourier_transform(s)
    begin_plot()
    plt.subplot(2,1,1)
    plt.plot(t, s)
    plt.title('Time-Domain')
    plt.xlabel('Time (sec)')
    plt.subplot(2,1,2)
    plt.plot(freqs, np.abs(fft))
    plt.title('Frequency-Domain')
    plt.xlabel('Frequency (Hz)')
    end_plot()

def set_derivative(df, glbl, tlbl):
    df[DG] = df[glbl].diff()
    df[DT] = df[tlbl].diff().dt.total_seconds()
    df[DGDT] = df.dg / df.dt
    return df


def set_auc(df, glbl, tlbl, g_lim):
    """
    Sets derivative automatically if not done
    :param df:
    :param glbl:
    :param tlbl:
    :param g_lim:
    :return:
    """
    if DG not in df.columns:
        df = set_derivative(df, glbl, tlbl)
    mean_g = df[glbl].mean()
    min_g = df[glbl].min()
    g_above_mean = df[glbl].map(lambda x: mean_g if x < mean_g else x)
    g_above_lim = df[glbl].map(lambda x: g_lim if x < g_lim else x)
    g_above_min = df[glbl].map(lambda x: min_g if x < min_g else x)
    df[AUC] = (g_above_mean-mean_g) * df[DT]
    df[AUCLIM] = (g_above_lim-g_lim) * df[DT]
    df[AUCMIN] = (g_above_min-min_g) * df[DT]
    return df


def is_valid_entry(device, unit):
    # TODO device in Devices, unit in Units
    return True


def convert_unit(g, from_unit, to_unit):
    # TODO all conversions
    if from_unit == to_unit:
        return g
    if from_unit == Units.MMOL:
        pass
    pass


def begin_plot(l=8, w=6):
    """Initialize plot

    :param l: length, defaults to 8
    :param w: width, defaults to 6
    :return:
    """
    plt.figure(num=None, figsize=(l, w), dpi=120, facecolor='w', edgecolor='k')


def end_plot(r=45, legend=True):
    """End plot by rotating xticks, adding legend and showing the plot

    :param r:
    :return:
    """
    plt.xticks(rotation=r)
    if legend:
        plt.legend()
    plt.show()


def plot_day_trend(df, glbl):
    """

    :param df: Dataframe containing glucose
    :param glbl: Name of the glucose column
    :return:
    """
    plot_percentiles(df, lbl=glbl, percentiles=[0.01, 0.05])


def plot_percentiles(df, lbl, percentiles, group_by=HLBL, color='green'):
    # TODO manual plot shape
    begin_plot()
    mean, dev, med, perc_l, perc_h = get_percentile_dist(
        df, percentiles, lbl, group_by)
    plt.plot(med)
    for i in range(len(percentiles)):
        plt.fill_between(
            med.index, perc_l[i], perc_h[i], color=color, alpha=0.2, label='Ho')
    plt.title('Trend of {} for the percentiles: {} as well as {}'.format(lbl,
                                                                         ', '.join(
                                                                             [str(int(i*100)) for i in percentiles]),
                                                                         ', '.join([str(int((1-i)*100)) for i in percentiles])))
    plt.xlabel(group_by)
    plt.ylabel(lbl)
    end_plot(legend=False)


def get_percentile_dist(df, percentiles, lbl, group_by):
    grouped = df.groupby([group_by])[lbl]
    mean = grouped.mean()
    med = grouped.median()
    dev = grouped.std()
    perc_l = [grouped.quantile(q) for q in percentiles]
    perc_h = [grouped.quantile(1-q) for q in percentiles]
    return mean, dev, med, perc_l, perc_h


def plot_box_comparison(df, lbl=G_LBL, compare_by=WEEKENDLBL, outliers=False, label_map=None):
    """

    :param df: dataframe containing the values to be compared and the comparison field
    :param lbl: label of the box plot values in the dataframe (Y-axis, defaults to the glucose label G_LBL)
    :param compare_by: field to compare by (X-axis, defaults to weekend label WEEKENDLBL)
    :param outliers: boolean to show or not show outliers, defaults to False
    :param label_map: lambda function to map the unique values of the compare_by field to some labels,
    defaults to None (showing original)
    :return:
    """
    begin_plot()
    all_vals = df[compare_by].unique()
    all_vals.sort()
    plt.boxplot([df[df[compare_by] == i][lbl].dropna() for i in all_vals],
                labels=all_vals if label_map is None else [
                    label_map(i) for i in all_vals],
                showfliers=outliers)

    plt.title('Comparing {} by {}. Outliers are {}.'.
              format(lbl, compare_by, 'shown' if outliers else 'not shown'))
    end_plot()


def describe(df, g_lbl):
    """

    :param df:
    :param g_lbl:
    :return:
    """
    T = """ Dataset contains:
    {} days
    
    Glucose summary:
    - mean: {}
    - std : {}
    - 68th percentile: {}
    - 95th percentile: {}
    
    """
    pass
def get_response_bounds(df, event_time, pre_pad_min=20, post_pad_min=0, resp_time_min=120, glbl=G_LBL, t_lbl=None):
    df_time = find_nearest_time(df, event_time, glbl, t_lbl)
    # TODO assert indexed by time
    start = df_time - tdel(minutes=pre_pad_min)
    end = df_time + tdel(minutes=resp_time_min) + tdel(minutes=post_pad_min)
    return start, end, df_time

def plot_response_to_event(df, event_time, pre_pad_min=20, post_pad_min=0, resp_time_min=120, glbl=G_LBL, t_lbl=None):
    s, e, t = get_response_bounds(df, event_time, pre_pad_min, post_pad_min, resp_time_min, glbl=G_LBL, t_lbl=None)
    plt.plot(df.loc[s:e][glbl])
    plt.axvline(t)


def find_nearest_time(df, time_pivot, glbl=G_LBL, t_lbl=None, niter=100):
    # TODO smaller time_l by number of hours
    # time_l = list(df[df.t> time_pivot - tdel(hours=3)])
    if t_lbl is None:
        time_l = list(df.index)
    else:
        time_l = list(df[t_lbl])
    time_list = time_l.copy()
    for i in range(niter):
        m = min(time_list, key=lambda x: abs(x - time_pivot))
        gval=df.loc[m][glbl]
        if type(gval)==pd.Series:
            gval=gval[0]
        if pd.isna(gval):
            time_list.remove(m)
        else:
            return m

# TODO detect periods of usage and empty periods in time, days with missing data etc.
# Data quality assessment
# TODO extend pandas dataframe instead

# TODO get meals and exercise from csv

# TODO calibration

# TODO use notes as tags
