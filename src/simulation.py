import numpy as np
import pandas as pd


def run_empirical_bootstrap_simulation(
    point_forecasts,
    residuals,
    n_simulations=5000,
    random_seed=42,
    target_values=None,
):
    """Bootstrap multi-step out-of-sample forecast residuals directly at each horizon h = 1..H."""
    rng = np.random.default_rng(random_seed)
    point_forecasts = np.asarray(point_forecasts, dtype=float)
    h = len(point_forecasts)

    if isinstance(residuals, (list, tuple)) and len(residuals) == h and isinstance(residuals[0], (list, tuple, np.ndarray)):
        horizon_residuals = []
        for r_h in residuals:
            arr = np.asarray(r_h, dtype=float)
            arr = arr[~np.isnan(arr)]
            horizon_residuals.append(arr - np.mean(arr) if len(arr) > 0 else np.array([0.0]))
    else:
        arr = np.asarray(residuals, dtype=float)
        arr = arr[~np.isnan(arr)]
        centered = arr - np.mean(arr) if len(arr) > 0 else np.array([0.0])
        horizon_residuals = [centered for _ in range(h)]

    sim_paths = np.zeros((n_simulations, h), dtype=float)
    for step_idx in range(h):
        r_step = horizon_residuals[step_idx]
        draw_idx = rng.integers(0, len(r_step), size=n_simulations)
        sim_paths[:, step_idx] = np.maximum(0.0, point_forecasts[step_idx] + r_step[draw_idx])

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
        summary_data["target_value"] = target_values
        summary_data["prob_achieve_target_pct"] = np.mean(sim_paths >= target_values, axis=0) * 100.0

    summary_df = pd.DataFrame(summary_data)

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
        tot_summary["total_target"] = tot_target
        tot_summary["total_prob_achieve_target_pct"] = float(np.mean(tot_sim >= tot_target) * 100.0)

    return sim_paths, summary_df, tot_summary
