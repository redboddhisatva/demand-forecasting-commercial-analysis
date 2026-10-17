"""
Monte Carlo Simulation via Empirical Bootstrapping
===================================================
Simulates forward demand uncertainty by bootstrapping out-of-sample forecast residuals:
1. Retains true empirical operational error properties (skewness and fat tails).
2. Incorporates regime persistence and horizon scaling (sqrt(h)) so uncertainty
   widens naturally over time rather than canceling out when summed over 6 months.
3. Computes executive planning percentiles:
   - P10 (Bear): Financial Downside Floor (Minimum Revenue Commitment)
   - P50 (Base): Operational Baseline Point Forecast
   - P90 (Bull): Supply Chain Safety Stock Buffer / Factory Capacity Ceiling
"""

import numpy as np
import pandas as pd


def run_empirical_bootstrap_simulation(
    point_forecasts,
    residuals,
    n_simulations=5000,
    random_seed=42,
    target_values=None,
):
    """
    Simulates forward demand trajectories by bootstrapping residuals.

    Parameters:
    -----------
    point_forecasts : array-like, shape (H,)
        Baseline point forecasts for H future steps.
    residuals : array-like
        Historical out-of-sample forecast residuals (actual - forecast).
    n_simulations : int
        Number of Monte Carlo paths (default 5,000).
    random_seed : int
        Seed for reproducibility.
    target_values : array-like, optional
        Target or budget values for H steps to compute P(Simulated >= Target).

    Returns:
    --------
    sim_paths : np.ndarray, shape (n_simulations, H)
        All simulated trajectories.
    summary_df : pd.DataFrame
        Table containing P10, P50, P90, Mean, Std, and Target Achievement Probability.
    tot_summary : dict
        Aggregate 6-month portfolio metrics.
    """
    rng = np.random.default_rng(random_seed)
    point_forecasts = np.asarray(point_forecasts, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    h = len(point_forecasts)

    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) == 0:
        residuals = np.array([0.0])
    else:
        # Center residuals so mean is zero to preserve point forecast baseline
        residuals = residuals - np.mean(residuals)

    # 1. Persistent regime shock: captures whether the half-year is overall strong, neutral, or weak
    regime_shock = rng.choice(residuals, size=(n_simulations, 1)) * 0.70

    # 2. Monthly variation scaled by sqrt(step / 2) to model widening horizon uncertainty
    horizon_scaling = np.sqrt(np.arange(1, h + 1, dtype=float) / 2.0)
    monthly_noise = rng.choice(residuals, size=(n_simulations, h)) * 0.70 * horizon_scaling

    # Simulated trajectories
    sim_paths = point_forecasts + regime_shock + monthly_noise
    sim_paths = np.maximum(0.0, sim_paths)

    # Percentiles per period
    p10 = np.percentile(sim_paths, 10, axis=0)
    p25 = np.percentile(sim_paths, 25, axis=0)
    p50 = np.percentile(sim_paths, 50, axis=0)
    p75 = np.percentile(sim_paths, 75, axis=0)
    p90 = np.percentile(sim_paths, 90, axis=0)
    mean_sim = np.mean(sim_paths, axis=0)
    std_sim = np.std(sim_paths, axis=0)

    summary_data = {
        "step": np.arange(1, h + 1),
        "point_forecast": point_forecasts,
        "p10_bear": p10,
        "p25": p25,
        "p50_base": p50,
        "p75": p75,
        "p90_bull": p90,
        "sim_mean": mean_sim,
        "sim_std": std_sim,
    }

    if target_values is not None:
        target_values = np.asarray(target_values, dtype=float)
        prob_target = np.mean(sim_paths >= target_values, axis=0) * 100.0
        summary_data["target_value"] = target_values
        summary_data["prob_achieve_target_pct"] = prob_target

    summary_df = pd.DataFrame(summary_data)

    # Aggregate total across all H steps per simulation
    tot_sim = np.sum(sim_paths, axis=1)
    tot_summary = {
        "total_point_forecast": float(np.sum(point_forecasts)),
        "total_p10_bear": float(np.percentile(tot_sim, 10)),
        "total_p50_base": float(np.percentile(tot_sim, 50)),
        "total_p90_bull": float(np.percentile(tot_sim, 90)),
        "total_mean": float(np.mean(tot_sim)),
        "total_std": float(np.std(tot_sim)),
    }

    if target_values is not None:
        tot_target = float(np.sum(target_values))
        tot_prob = float(np.mean(tot_sim >= tot_target) * 100.0)
        tot_summary["total_target"] = tot_target
        tot_summary["total_prob_achieve_target_pct"] = tot_prob

    return sim_paths, summary_df, tot_summary
