"""
Forecast Evaluation Metrics
============================
Evaluation metrics for demand forecasting:
- WMAPE (Weighted Mean Absolute Percentage Error)
- Forecast Bias (%)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
"""

import numpy as np


def calc_wmape(actuals, forecasts):
    """Calculate Weighted Mean Absolute Percentage Error (volume-weighted)."""
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    sum_actual = np.sum(actuals)
    if sum_actual == 0:
        return 0.0
    return (np.sum(np.abs(actuals - forecasts)) / sum_actual) * 100.0


def calc_bias(actuals, forecasts):
    """
    Calculate Forecast Bias in percent.
    Positive value indicates over-forecasting.
    Negative value indicates under-forecasting.
    """
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    sum_actual = np.sum(actuals)
    if sum_actual == 0:
        return 0.0
    return (np.sum(forecasts - actuals) / sum_actual) * 100.0


def calc_mae(actuals, forecasts):
    """Calculate Mean Absolute Error."""
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    return np.mean(np.abs(actuals - forecasts))


def calc_rmse(actuals, forecasts):
    """Calculate Root Mean Squared Error."""
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    return np.sqrt(np.mean((actuals - forecasts) ** 2))


def evaluate_forecasts(actuals, forecasts):
    """Compute all standard forecast accuracy metrics."""
    return {
        "WMAPE": calc_wmape(actuals, forecasts),
        "Bias_pct": calc_bias(actuals, forecasts),
        "MAE": calc_mae(actuals, forecasts),
        "RMSE": calc_rmse(actuals, forecasts),
    }
