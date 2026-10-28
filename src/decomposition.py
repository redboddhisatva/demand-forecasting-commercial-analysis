import numpy as np
import pandas as pd


def run_3layer_decomposition(series, period=12):
    """Decompose a monthly series into additive level, seasonality, and residual noise."""
    s = pd.Series(series.values, index=series.index, dtype=float)
    n = len(s)

    # Use a 2x12 moving average when period is even so calendar months are weighted symmetrically.
    level = s.rolling(period, min_periods=max(4, period // 2), center=True).mean()
    if period % 2 == 0:
        level = level.rolling(2, min_periods=1, center=True).mean()
    level = level.bfill().ffill()

    detrended = s - level
    month_indices = s.index.month if hasattr(s.index, "month") else (np.arange(n) % period) + 1
    monthly_lifts = detrended.groupby(month_indices).mean()
    # Zero-center seasonal factors so they do not shift the underlying level.
    monthly_lifts = monthly_lifts - monthly_lifts.mean()

    seasonality = pd.Series(month_indices, index=s.index).map(monthly_lifts)
    noise = s - (level + seasonality)

    df_decomp = pd.DataFrame(
        {
            "observed": s,
            "level": level,
            "seasonality": seasonality,
            "noise": noise,
        },
        index=s.index,
    )

    sigma_noise = float(noise.std(ddof=1)) if n > 1 else 0.0
    mean_level = float(level.mean())
    cv_noise = (sigma_noise / mean_level * 100.0) if mean_level > 0 else 0.0

    if cv_noise <= 5.0:
        category = "High"
        recommendation = "Low residual variance; suitable for statistical baseline projection"
    elif cv_noise <= 10.0:
        category = "Moderate"
        recommendation = "Moderate variance; pair statistical baseline with commercial review"
    else:
        category = "Low"
        recommendation = "High variance; maintain wider safety stock buffers"

    summary_metrics = {
        "mean_level": mean_level,
        "sigma_noise": sigma_noise,
        "cv_noise_pct": cv_noise,
        "category": category,
        "recommendation": recommendation,
        "monthly_lifts": monthly_lifts.to_dict(),
    }

    return df_decomp, summary_metrics
