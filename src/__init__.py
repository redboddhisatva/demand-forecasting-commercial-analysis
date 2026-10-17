"""
Forecast Engine Package
=======================
Modular pipeline for recurring demand forecasting, 3-layer classical decomposition,
ensemble modeling, Monte Carlo risk evaluation, and executive revenue bridges.
"""

from .decomposition import run_3layer_decomposition
from .models import (
    predict_damped_naive,
    predict_trend_seasonal,
    predict_exponential_smoothing,
    run_walk_forward_backtest,
    project_recurring_forward_direct,
)
from .simulation import run_empirical_bootstrap_simulation
from .finance import (
    calc_gross_to_net_revenue,
    evaluate_commercial_action_scenarios,
    build_bottom_up_grid_forecast,
    build_channel_revenue_exposure_table,
    build_brand_revenue_exposure_table,
)
from .metrics import (
    calc_wmape,
    calc_bias,
    calc_mae,
    calc_rmse,
    evaluate_forecasts,
)

__all__ = [
    "calc_wmape",
    "calc_bias",
    "calc_mae",
    "calc_rmse",
    "evaluate_forecasts",
    "run_3layer_decomposition",
    "predict_damped_naive",
    "predict_trend_seasonal",
    "predict_exponential_smoothing",
    "run_walk_forward_backtest",
    "project_recurring_forward_direct",
    "run_empirical_bootstrap_simulation",
    "calc_gross_to_net_revenue",
    "evaluate_commercial_action_scenarios",
    "build_bottom_up_grid_forecast",
    "build_channel_revenue_exposure_table",
    "build_brand_revenue_exposure_table",
]
