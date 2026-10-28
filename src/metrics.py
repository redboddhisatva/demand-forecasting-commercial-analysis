import numpy as np


def calc_wmape(actuals, forecasts):
    """Volume-weighted Mean Absolute Percentage Error (%)."""
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    sum_actual = np.sum(actuals)
    if sum_actual == 0:
        return 0.0
    return float((np.sum(np.abs(actuals - forecasts)) / sum_actual) * 100.0)


def calc_bias(actuals, forecasts):
    """Percentage forecast bias; positive values indicate over-forecasting."""
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    sum_actual = np.sum(actuals)
    if sum_actual == 0:
        return 0.0
    return float((np.sum(forecasts - actuals) / sum_actual) * 100.0)


def calc_mae(actuals, forecasts):
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    return float(np.mean(np.abs(actuals - forecasts)))


def calc_rmse(actuals, forecasts):
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    return float(np.sqrt(np.mean((actuals - forecasts) ** 2)))


def evaluate_forecasts(actuals, forecasts):
    return {
        "WMAPE": calc_wmape(actuals, forecasts),
        "Bias_pct": calc_bias(actuals, forecasts),
        "MAE": calc_mae(actuals, forecasts),
        "RMSE": calc_rmse(actuals, forecasts),
    }
