"""
Glyco

Glyco is a glucose data analysis library.
"""

from __future__ import annotations

# core glucose functions
from .glucose import (
    read_csv,
    read_df,
    prepare_glucose,
    get_properties,
    get_stats,
    get_metrics,
    get_metrics_by_day,
    get_metrics_by_hour,
    describe_glucose,
    plot_glucose,
    plot_trend_by_hour,
    plot_trend_by_day,
    plot_trend_by_weekday,
    plot_percentiles,
    plot_compare_by,
    plot_response_at_time,
    get_area_under_curve,
    set_derivative,
    set_auc,
    compute_derivative,
    GlucosePrepKwargs,
    PrivateInfoKwargs,
    DerivativeCols,
    GLUCOSE_COL,
    TIMESTAMP_COL,
    DEFAULT_INPUT_TSP_FMT,
    GeneralDateType,
)

# Meals / events
from .meals import (
    read_meals_from_folder,
    read_events_csv,
    read_events_df,
    get_event_sessions,
    read_events_from_times,
    infer_events_from_glucose_excursions,
    plot_all_sessions,
    plot_session_response,
    plot_compare_two_sessions,
    describe_all_sessions,
    describe_session,
    get_session_metrics,
)

# Privacy
from .privacy import mask_private_information, default_replace_func

# Utilities (only the ones you want public)
from .utils import Units, Devices, init_plot, end_plot, autoplot

__all__ = [
    # glucose
    "read_csv", "read_df", "prepare_glucose", "get_properties", "get_stats",
    "get_metrics", "get_metrics_by_day", "get_metrics_by_hour", "describe_glucose",
    "plot_glucose", "plot_trend_by_hour", "plot_trend_by_day", "plot_trend_by_weekday",
    "plot_percentiles", "plot_compare_by", "plot_response_at_time",
    "get_area_under_curve", "set_derivative", "set_auc", "compute_derivative",
    "GlucosePrepKwargs", "PrivateInfoKwargs", "DerivativeCols",
    "GLUCOSE_COL", "TIMESTAMP_COL", "DEFAULT_INPUT_TSP_FMT", "GeneralDateType",

    # meals/events
    "read_meals_from_folder", "read_events_csv", "read_events_df",
    "get_event_sessions", "read_events_from_times", "infer_events_from_glucose_excursions",
    "plot_all_sessions", "plot_session_response", "plot_compare_two_sessions",
    "describe_all_sessions", "describe_session", "get_session_metrics",

    # privacy
    "mask_private_information",

    # utils
    "Units", "Devices", "init_plot", "end_plot", "autoplot",
]