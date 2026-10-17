"""
Ensemble Forecasting Engine
============================
Three intuitive, transparent forecasting models:
1. Damped Naive (Seasonal lag with damped local momentum, phi = 0.70)
2. Simple Trend + Seasonality (Linear trend plus detrended monthly seasonal lifts)
3. Exponential Smoothing (Holt-Winters additive with b = 0.0 initial trend)

Combines predictions via Simple Average or Inverse-RMSE weighting.
Includes walk-forward out-of-sample backtesting and direct forward projection.
"""

import numpy as np
import pandas as pd
from .metrics import calc_rmse, evaluate_forecasts


def predict_damped_naive(train_y, train_months, target_month, phi=0.70, h=1):
    """
    Seasonal Naive with damped local momentum.
    Uses observation from 12 months ago plus a damped YoY drift term.
    """
    n = len(train_y)
    if n < 12:
        return float(train_y[-1])
    s_lag = float(train_y[-12 + ((h - 1) % 12)])
    if n >= 15:
        # Year-over-year monthly drift from the most recent quarter
        yoy_diff = np.mean(train_y[-3:]) - np.mean(train_y[-15:-12])
        local_drift = float(yoy_diff) / 12.0
    else:
        local_drift = 0.0
    return max(0.0, s_lag + (phi ** h) * local_drift)


def predict_trend_seasonal(train_y, train_months, target_month, h=1):
    """
    Simple Trend plus Seasonal Lift.
    1. Fits linear trend line on training data: trend = intercept + slope * t
    2. Computes detrended residuals and averages them by calendar month.
    3. Projects step h ahead: trend_at_step_h + seasonal_lift[target_month]
    """
    n = len(train_y)
    train_y = np.asarray(train_y, dtype=float)
    train_months = np.asarray(train_months, dtype=int)

    t_idx = np.arange(n, dtype=float)
    if n >= 2:
        slope_t, intercept = np.polyfit(t_idx, train_y, 1)
    else:
        slope_t = 0.0
        intercept = float(train_y[0])

    detrended = train_y - (intercept + slope_t * t_idx)

    # Average lift per calendar month (1 to 12)
    month_lifts = {}
    for m in range(1, 13):
        mask = (train_months == m)
        if np.sum(mask) > 0:
            month_lifts[m] = float(np.mean(detrended[mask]))
        else:
            month_lifts[m] = 0.0

    target_t = n + (h - 1)
    target_lift = month_lifts.get(target_month, 0.0)
    pred = intercept + (slope_t * target_t) + target_lift
    return max(0.0, float(pred))


def predict_exponential_smoothing(train_y, train_months, target_month, alpha=0.15, beta=0.05, gamma=0.15, h=1):
    """
    Holt-Winters Additive model.
    Initializes baseline level as first-year mean and initial trend b = 0.0
    to prevent quarter-end spikes from being mistaken for annual growth.
    """
    y = np.asarray(train_y, dtype=float)
    n = len(y)
    if n < 12:
        return float(y[-1])

    l = float(np.mean(y[:12]))
    b = 0.0  # Set initial trend to zero to avoid mistaking month 12 spike for annual trend
    s = list(y[:12] - l)

    for t in range(12, n):
        s_prev = s[t - 12]
        new_l = alpha * (y[t] - s_prev) + (1.0 - alpha) * (l + b)
        new_b = beta * (new_l - l) + (1.0 - beta) * b
        new_s = gamma * (y[t] - new_l) + (1.0 - gamma) * s_prev
        s.append(new_s)
        l, b = new_l, new_b

    s_target = s[n - 12 + ((h - 1) % 12)]
    pred = l + (h * b) + s_target
    return max(0.0, float(pred))


def run_walk_forward_backtest(y, months, dates, min_train=12):
    """
    Strict 1-step-ahead walk-forward out-of-sample backtest.
    Expands the training window sequentially without look-ahead bias.
    """
    y = np.asarray(y, dtype=float)
    months = np.asarray(months, dtype=int)
    n = len(y)

    preds_naive = []
    preds_ols = []
    preds_hw = []
    actuals = []
    test_dates = []
    test_months = []

    for t in range(min_train, n):
        train_y = y[:t]
        train_m = months[:t]
        target_m = months[t]

        p_naive = predict_damped_naive(train_y, train_m, target_m, phi=0.70, h=1)
        p_ols = predict_trend_seasonal(train_y, train_m, target_m, h=1)
        p_hw = predict_exponential_smoothing(train_y, train_m, target_m, h=1)

        preds_naive.append(p_naive)
        preds_ols.append(p_ols)
        preds_hw.append(p_hw)
        actuals.append(y[t])
        test_dates.append(dates[t])
        test_months.append(target_m)

    preds_naive = np.array(preds_naive)
    preds_ols = np.array(preds_ols)
    preds_hw = np.array(preds_hw)
    actuals = np.array(actuals)

    # Simple Average Ensemble
    preds_simple_avg = (preds_naive + preds_ols + preds_hw) / 3.0

    df_backtest = pd.DataFrame({
        "period_date": test_dates,
        "month": test_months,
        "actual": actuals,
        "pred_damped_naive": preds_naive,
        "pred_trend_seasonal": preds_ols,
        "pred_exponential_smoothing": preds_hw,
        "pred_combined": preds_simple_avg,
        "resid_combined": actuals - preds_simple_avg,
    })

    metrics = {
        "Damped_Naive": evaluate_forecasts(actuals, preds_naive),
        "Trend_Seasonal": evaluate_forecasts(actuals, preds_ols),
        "Exponential_Smoothing": evaluate_forecasts(actuals, preds_hw),
        "Combined_Forecast": evaluate_forecasts(actuals, preds_simple_avg),
    }

    return df_backtest, metrics


def project_recurring_forward_direct(y, months, n_forward=6):
    """
    Direct forward projection for n_forward steps from historical series.
    Does not feed forecasts back into training to avoid error compounding.
    """
    y = np.asarray(y, dtype=float)
    months = np.asarray(months, dtype=int)
    last_month = months[-1]

    projections = []
    for h in range(1, n_forward + 1):
        target_m = (last_month + h - 1) % 12 + 1

        p_naive = predict_damped_naive(y, months, target_m, phi=0.70, h=h)
        p_ols = predict_trend_seasonal(y, months, target_m, h=h)
        p_hw = predict_exponential_smoothing(y, months, target_m, h=h)
        p_combo = (p_naive + p_ols + p_hw) / 3.0

        projections.append({
            "step": h,
            "target_month": target_m,
            "pred_damped_naive": p_naive,
            "pred_trend_seasonal": p_ols,
            "pred_exponential_smoothing": p_hw,
            "point_forecast": p_combo,
        })

    return pd.DataFrame(projections)
