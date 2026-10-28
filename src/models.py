import numpy as np
import pandas as pd
from .metrics import evaluate_forecasts


def predict_seasonal_naive(train_y, h=1):
    """Plain seasonal naive benchmark: repeats the same calendar month from 12 months prior."""
    train_y = np.asarray(train_y, dtype=float)
    n = len(train_y)
    if n < 12:
        return float(train_y[-1])
    s_lag = float(train_y[-12 + ((h - 1) % 12)])
    return max(0.0, s_lag)


def predict_damped_naive(train_y, phi=0.70, h=1):
    """Seasonal naive baseline plus cumulative damped YoY drift over horizon h."""
    train_y = np.asarray(train_y, dtype=float)
    n = len(train_y)
    if n < 12:
        return float(train_y[-1])
    s_lag = float(train_y[-12 + ((h - 1) % 12)])
    if n >= 15:
        yoy_diff = np.mean(train_y[-3:]) - np.mean(train_y[-15:-12])
        monthly_drift = float(yoy_diff) / 12.0
    else:
        monthly_drift = 0.0
    damped_sum = sum(phi ** i for i in range(1, h + 1))
    return max(0.0, s_lag + damped_sum * monthly_drift)


def predict_trend_seasonal(train_y, train_months, target_month, h=1):
    """OLS linear trend fitted on deseasonalized series plus zero-centered monthly lifts."""
    train_y = np.asarray(train_y, dtype=float)
    train_months = np.asarray(train_months, dtype=int)
    n = len(train_y)
    if n < 12:
        return float(train_y[-1])

    overall_mean = float(np.mean(train_y))
    month_lifts = {}
    for m in range(1, 13):
        mask = train_months == m
        month_lifts[m] = float(np.mean(train_y[mask]) - overall_mean) if np.sum(mask) > 0 else 0.0

    lift_offset = float(np.mean(list(month_lifts.values())))
    for m in month_lifts:
        month_lifts[m] -= lift_offset

    # Deseasonalize before fitting OLS slope so quarter-end placement in the window does not bias trend.
    deseasonalized = np.array([train_y[i] - month_lifts[train_months[i]] for i in range(n)])
    t_idx = np.arange(n, dtype=float)
    slope_t, intercept = np.polyfit(t_idx, deseasonalized, 1)

    target_t = n + (h - 1)
    pred = intercept + (slope_t * target_t) + month_lifts.get(int(target_month), 0.0)
    return max(0.0, float(pred))


def predict_exponential_smoothing(train_y, alpha=0.15, beta=0.05, gamma=0.15, h=1):
    """Additive Holt-Winters with flat initial slope so month-12 spikes stay in seasonal factors."""
    y = np.asarray(train_y, dtype=float)
    n = len(y)
    if n < 12:
        return float(y[-1])

    l = float(np.mean(y[:12]))
    b = 0.0
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
    """Expanding-window 1-step-ahead out-of-sample backtest over closed historical periods."""
    y = np.asarray(y, dtype=float)
    months = np.asarray(months, dtype=int)
    n = len(y)

    preds_snaive = []
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

        preds_snaive.append(predict_seasonal_naive(train_y, h=1))
        preds_naive.append(predict_damped_naive(train_y, phi=0.70, h=1))
        preds_ols.append(predict_trend_seasonal(train_y, train_m, target_m, h=1))
        preds_hw.append(predict_exponential_smoothing(train_y, h=1))
        actuals.append(y[t])
        test_dates.append(dates[t])
        test_months.append(target_m)

    preds_snaive = np.array(preds_snaive)
    preds_naive = np.array(preds_naive)
    preds_ols = np.array(preds_ols)
    preds_hw = np.array(preds_hw)
    actuals = np.array(actuals)
    preds_simple_avg = (preds_naive + preds_ols + preds_hw) / 3.0

    df_backtest = pd.DataFrame(
        {
            "period_date": test_dates,
            "month": test_months,
            "actual": actuals,
            "pred_seasonal_naive": preds_snaive,
            "pred_damped_naive": preds_naive,
            "pred_trend_seasonal": preds_ols,
            "pred_exponential_smoothing": preds_hw,
            "pred_combined": preds_simple_avg,
            "resid_combined": actuals - preds_simple_avg,
        }
    )

    metrics = {
        "Seasonal_Naive": evaluate_forecasts(actuals, preds_snaive),
        "Damped_Naive": evaluate_forecasts(actuals, preds_naive),
        "Trend_Seasonal": evaluate_forecasts(actuals, preds_ols),
        "Exponential_Smoothing": evaluate_forecasts(actuals, preds_hw),
        "Combined_Forecast": evaluate_forecasts(actuals, preds_simple_avg),
    }

    return df_backtest, metrics


def collect_multistep_residuals(y, months, min_train=12, max_h=6):
    """Collect expanding-window out-of-sample residuals for each forecast horizon h = 1..max_h."""
    y = np.asarray(y, dtype=float)
    months = np.asarray(months, dtype=int)
    n = len(y)

    residuals_by_h = []
    for h in range(1, max_h + 1):
        errs = []
        for t in range(min_train, n - h + 1):
            train_y = y[:t]
            train_m = months[:t]
            target_m = months[t + h - 1]
            p_naive = predict_damped_naive(train_y, phi=0.70, h=h)
            p_ols = predict_trend_seasonal(train_y, train_m, target_m, h=h)
            p_hw = predict_exponential_smoothing(train_y, h=h)
            p_combo = (p_naive + p_ols + p_hw) / 3.0
            errs.append(y[t + h - 1] - p_combo)
        residuals_by_h.append(np.asarray(errs, dtype=float))
    return residuals_by_h


def evaluate_granular_backtest(fact_recurring_df, min_train=12):
    """Evaluate 12-month walk-forward backtest WMAPE across all 16 Brand x Channel series."""
    rec_only = fact_recurring_df[fact_recurring_df["demand_stream"] == "Recurring"].copy()
    rec_only["period_date"] = pd.to_datetime(rec_only["period_date"])

    rows = []
    for (b, c), grp in rec_only.groupby(["brand_code", "channel_code"]):
        grp = grp.sort_values("period_date")
        _, m = run_walk_forward_backtest(
            grp["stream_qty_pcs"].values,
            grp["period_date"].dt.month.values,
            grp["period_date"].dt.strftime("%Y-%m-%d").values,
            min_train=min_train,
        )
        row = {"brand_code": b, "channel_code": c}
        for model_name, vals in m.items():
            row[model_name] = vals["WMAPE"]
        rows.append(row)
    return pd.DataFrame(rows)


def project_recurring_forward_direct(y, months, n_forward=6):
    """Multi-step direct projection without feeding predicted values back into training."""
    y = np.asarray(y, dtype=float)
    months = np.asarray(months, dtype=int)
    last_month = months[-1]

    projections = []
    for h in range(1, n_forward + 1):
        target_m = (last_month + h - 1) % 12 + 1
        p_snaive = predict_seasonal_naive(y, h=h)
        p_naive = predict_damped_naive(y, phi=0.70, h=h)
        p_ols = predict_trend_seasonal(y, months, target_m, h=h)
        p_hw = predict_exponential_smoothing(y, h=h)
        p_combo = (p_naive + p_ols + p_hw) / 3.0

        projections.append(
            {
                "step": h,
                "target_month": target_m,
                "pred_seasonal_naive": p_snaive,
                "pred_damped_naive": p_naive,
                "pred_trend_seasonal": p_ols,
                "pred_exponential_smoothing": p_hw,
                "point_forecast": p_combo,
            }
        )

    return pd.DataFrame(projections)
