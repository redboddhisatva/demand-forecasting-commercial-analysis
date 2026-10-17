"""
Three-Layer Classical Decomposition Module
===========================================
Separates demand into Level, Seasonality, and Noise:
  Actual Demand = Level + Seasonality + Noise

Uses simple moving average arithmetic instead of complex black-box algorithms:
1. Level: Centered 12-month rolling average to capture the baseline trend.
2. Seasonality: Calendar month and quarter-end average deviation from the level.
3. Noise: Residual deviation, evaluated using the Coefficient of Variation (CV)
   and empirical +/- 1 sigma and +/- 2 sigma confidence envelopes.
"""

import numpy as np
import pandas as pd


def run_3layer_decomposition(series, period=12):
    """
    Computes transparent 3-layer classical decomposition.

    Parameters:
    -----------
    series : pd.Series
        Monthly demand series with datetime or period index.
    period : int
        Seasonal cycle length (default 12 for annual seasonality).

    Returns:
    --------
    df_decomp : pd.DataFrame
        DataFrame with columns ['observed', 'level', 'seasonality', 'noise']
    summary_metrics : dict
        Dictionary of level mean, seasonality lifts, and noise standard deviation.
    """
    s = pd.Series(series.values, index=series.index, dtype=float)
    n = len(s)

    # 1. Level: Centered 12-month moving average
    level = s.rolling(period, min_periods=max(4, period // 2), center=True).mean()
    # Fill ends cleanly with backfill and forward fill
    level = level.bfill().ffill()

    # 2. Seasonality: Average deviation per calendar month
    detrended = s - level
    month_indices = s.index.month if hasattr(s.index, 'month') else (np.arange(n) % period) + 1
    monthly_lifts = detrended.groupby(month_indices).mean()

    # Map monthly lifts back to series
    seasonality = pd.Series(month_indices, index=s.index).map(monthly_lifts)

    # 3. Noise: Residual envelope
    noise = s - (level + seasonality)

    df_decomp = pd.DataFrame({
        "observed": s,
        "level": level,
        "seasonality": seasonality,
        "noise": noise,
    }, index=s.index)

    # Calculate summary metrics
    sigma_noise = float(noise.std(ddof=1)) if n > 1 else 0.0
    mean_level = float(level.mean())
    cv_noise = (sigma_noise / mean_level * 100.0) if mean_level > 0 else 0.0

    # Categorize forecastability by CV of noise
    if cv_noise <= 5.0:
        category = "High"
        recommendation = "Low noise; highly reliable for automated baseline forecasting"
    elif cv_noise <= 10.0:
        category = "Moderate"
        recommendation = "Moderate noise; standard baseline with planner review"
    else:
        category = "Low"
        recommendation = "High noise; maintain safety stock buffers"

    summary_metrics = {
        "mean_level": mean_level,
        "sigma_noise": sigma_noise,
        "cv_noise_pct": cv_noise,
        "category": category,
        "recommendation": recommendation,
        "monthly_lifts": monthly_lifts.to_dict(),
    }

    return df_decomp, summary_metrics
