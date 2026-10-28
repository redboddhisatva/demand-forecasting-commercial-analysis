import numpy as np
import pandas as pd
from .models import collect_multistep_residuals


def calc_gross_to_net_revenue(volume_pcs, base_list_price, contra_rate):
    volume_pcs = np.asarray(volume_pcs, dtype=float)
    base_list_price = np.asarray(base_list_price, dtype=float)
    contra_rate = np.asarray(contra_rate, dtype=float)

    gross_sales = volume_pcs * base_list_price
    contra_spend = gross_sales * contra_rate
    net_sales = gross_sales - contra_spend

    return {
        "gross_sales": gross_sales,
        "contra_spend": contra_spend,
        "net_sales": net_sales,
    }


def estimate_channel_ped(fact_sales_df, dim_calendar_df=None):
    """Estimate channel Price Elasticity of Demand via log-log OLS on non-settlement months."""
    df = fact_sales_df.copy()
    if dim_calendar_df is not None and "is_quarter_end" in dim_calendar_df.columns:
        cal_map = dim_calendar_df.set_index("period_date")["is_quarter_end"].to_dict()
        df["is_quarter_end"] = df["period_date"].map(cal_map)
    else:
        df["is_quarter_end"] = df["month"].isin([3, 6, 9, 12]).astype(int)

    rows = []
    for ch, grp in df.groupby("channel_code"):
        m_grp = (
            grp.groupby(["period_date", "month", "is_quarter_end"])
            .agg({"qty_pcs": "sum", "gross_sales_amt": "sum", "net_sales_amt": "sum"})
            .reset_index()
            .sort_values("period_date")
        )
        m_mean = m_grp.groupby("month")["qty_pcs"].transform("mean")
        m_grp["q_deseason"] = m_grp["qty_pcs"] / (m_mean / m_grp["qty_pcs"].mean())
        m_grp["net_price"] = m_grp["net_sales_amt"] / m_grp["qty_pcs"]

        # Exclude quarter-end rebate settlement months so lump-sum accruals do not distort unit price.
        reg = m_grp[m_grp["is_quarter_end"] == 0]
        log_p = np.log(reg["net_price"].values)
        log_q = np.log(reg["q_deseason"].values)
        n_obs = len(reg)

        slope, intercept = np.polyfit(log_p, log_q, 1)
        y_hat = intercept + slope * log_p
        sxx = float(np.sum((log_p - np.mean(log_p)) ** 2))
        resid_ss = float(np.sum((log_q - y_hat) ** 2))
        std_err = float(np.sqrt((resid_ss / max(1, n_obs - 2)) / sxx)) if sxx > 0 else 0.0

        r_val = float(np.corrcoef(log_p, log_q)[0, 1])
        r2 = float(r_val ** 2)

        rows.append(
            {
                "channel_code": ch,
                "estimated_ped": round(float(slope), 2),
                "std_error": round(std_err, 2),
                "r_squared": round(r2, 2),
                "n_obs": n_obs,
            }
        )
    return pd.DataFrame(rows)


def evaluate_commercial_action_scenarios(
    channel_code,
    channel_name,
    base_monthly_vol,
    base_gross_price,
    contra_rate,
    ped_coefficient,
):
    """Evaluate list price changes and contra rebate cuts under constant-elasticity demand."""
    scenarios = []
    base_gross = base_monthly_vol * base_gross_price
    base_net_price = base_gross_price * (1.0 - contra_rate)
    base_net = base_monthly_vol * base_net_price

    scenarios.append(
        {
            "channel_code": channel_code,
            "channel_name": channel_name,
            "scenario": "Baseline",
            "net_price_chg_pct": 0.0,
            "vol_chg_pct": 0.0,
            "contra_rate_pct": contra_rate * 100.0,
            "monthly_vol_pcs": base_monthly_vol,
            "net_revenue_jpy": base_net,
            "net_rev_delta_jpy": 0.0,
        }
    )

    for pct in (0.02, 0.05):
        new_gross_price = base_gross_price * (1.0 + pct)
        new_net_price = new_gross_price * (1.0 - contra_rate)
        vol_ratio = (new_net_price / base_net_price) ** ped_coefficient
        new_vol = max(0.0, base_monthly_vol * vol_ratio)
        new_net = new_vol * new_net_price
        scenarios.append(
            {
                "channel_code": channel_code,
                "channel_name": channel_name,
                "scenario": f"+{int(pct * 100)}% List Price Hike",
                "net_price_chg_pct": round(((new_net_price / base_net_price) - 1.0) * 100.0, 2),
                "vol_chg_pct": round((vol_ratio - 1.0) * 100.0, 1),
                "contra_rate_pct": contra_rate * 100.0,
                "monthly_vol_pcs": new_vol,
                "net_revenue_jpy": new_net,
                "net_rev_delta_jpy": new_net - base_net,
            }
        )

    # Unconditional -2pp contra cut raises effective net price by +2pp / (1 - contra_rate)
    tightened_contra = max(0.0, contra_rate - 0.02)
    rebate_net_price = base_gross_price * (1.0 - tightened_contra)
    rebate_vol_ratio = (rebate_net_price / base_net_price) ** ped_coefficient
    rebate_vol = max(0.0, base_monthly_vol * rebate_vol_ratio)
    rebate_net_rev = rebate_vol * rebate_net_price
    scenarios.append(
        {
            "channel_code": channel_code,
            "channel_name": channel_name,
            "scenario": "Unconditional -2pp Contra Cut",
            "net_price_chg_pct": round(((rebate_net_price / base_net_price) - 1.0) * 100.0, 2),
            "vol_chg_pct": round((rebate_vol_ratio - 1.0) * 100.0, 1),
            "contra_rate_pct": tightened_contra * 100.0,
            "monthly_vol_pcs": rebate_vol,
            "net_revenue_jpy": rebate_net_rev,
            "net_rev_delta_jpy": rebate_net_rev - base_net,
        }
    )

    return pd.DataFrame(scenarios)


def build_bottom_up_grid_forecast(
    fact_recurring_df,
    fact_target_df,
    dim_product_df,
    dim_channel_df,
    forecast_func,
    bootstrap_func,
    future_start_date="2026-10-01",
    n_forward=6,
):
    """Forecast all 16 Brand x Channel series using series-specific multi-step backtest residuals."""
    rec_only = fact_recurring_df[fact_recurring_df["demand_stream"] == "Recurring"].copy()
    non_rec_only = fact_recurring_df[fact_recurring_df["demand_stream"] == "Non-Recurring"].copy()
    fut_target = fact_target_df[fact_target_df["period_date"] >= future_start_date].copy()

    prod_map = dim_product_df.set_index("brand_code").to_dict("index")
    chan_map = dim_channel_df.set_index("channel_code").to_dict("index")

    brands = sorted(rec_only["brand_code"].unique())
    channels = sorted(rec_only["channel_code"].unique())

    records = []
    for b in brands:
        if b not in prod_map:
            raise KeyError(f"Brand '{b}' missing from dim_product_df")
        p_info = prod_map[b]
        base_price = float(p_info["base_list_price"])
        brand_name = str(p_info["brand_name"])
        modality = str(p_info["modality"])

        for c in channels:
            if c not in chan_map:
                raise KeyError(f"Channel '{c}' missing from dim_channel_df")
            c_info = chan_map[c]
            contra_rate = float(c_info["target_contra_rate"])
            channel_name = str(c_info["channel_name"])
            net_price = base_price * (1.0 - contra_rate)

            sub = (
                rec_only[(rec_only["brand_code"] == b) & (rec_only["channel_code"] == c)]
                .sort_values("period_date")
                .reset_index(drop=True)
            )
            sub["period_date"] = pd.to_datetime(sub["period_date"])
            sub["Month"] = sub["period_date"].dt.month

            nr_sub = (
                non_rec_only[(non_rec_only["brand_code"] == b) & (non_rec_only["channel_code"] == c)]
                .sort_values("period_date")
                .reset_index(drop=True)
            )
            hist_nr_pcs = float(nr_sub["stream_qty_pcs"].iloc[-n_forward:].sum())
            hist_nr_jpy = hist_nr_pcs * net_price

            y = sub["stream_qty_pcs"].values.astype(float)
            m = sub["Month"].values

            tgt_sub = (
                fut_target[(fut_target["brand_code"] == b) & (fut_target["channel_code"] == c)]
                .sort_values("period_date")
                .reset_index(drop=True)
            )
            target_pcs = float(tgt_sub["target_qty_pcs"].sum())
            target_jpy = float(tgt_sub["target_net_amt"].sum())

            fwd_df = forecast_func(y, m, n_forward=n_forward)
            point_fwd = fwd_df["point_forecast"].values

            series_multistep_resids = collect_multistep_residuals(y, m, min_train=12, max_h=n_forward)

            sim_paths, _, tot_mc = bootstrap_func(
                point_fwd,
                series_multistep_resids,
                n_simulations=5000,
                random_seed=42,
                target_values=tgt_sub["target_qty_pcs"].values,
            )

            sim_tot_pcs = np.sum(sim_paths, axis=1)
            sim_tot_jpy = sim_tot_pcs * net_price

            rec_p50_pcs = tot_mc["total_point_forecast"]
            rec_p10_pcs = tot_mc["total_p10_bear"]
            rec_p90_pcs = tot_mc["total_p90_bull"]

            gap_pcs = target_pcs - rec_p50_pcs
            stretch_pcs = gap_pcs - hist_nr_pcs
            rec_p50_jpy = rec_p50_pcs * net_price
            rec_p10_jpy = rec_p10_pcs * net_price
            rec_p90_jpy = rec_p90_pcs * net_price
            gap_jpy = target_jpy - rec_p50_jpy
            stretch_jpy = gap_jpy - hist_nr_jpy

            records.append(
                {
                    "brand_code": b,
                    "brand_name": brand_name,
                    "modality": modality,
                    "channel_code": c,
                    "channel_name": channel_name,
                    "base_list_price": base_price,
                    "contra_rate": contra_rate,
                    "net_price": net_price,
                    "target_pcs": target_pcs,
                    "rec_p50_pcs": rec_p50_pcs,
                    "rec_p10_pcs": rec_p10_pcs,
                    "rec_p90_pcs": rec_p90_pcs,
                    "gap_pcs": gap_pcs,
                    "hist_nr_pcs": hist_nr_pcs,
                    "stretch_pcs": stretch_pcs,
                    "target_jpy": target_jpy,
                    "rec_p50_jpy": rec_p50_jpy,
                    "rec_p10_jpy": rec_p10_jpy,
                    "rec_p90_jpy": rec_p90_jpy,
                    "gap_jpy": gap_jpy,
                    "hist_nr_jpy": hist_nr_jpy,
                    "stretch_jpy": stretch_jpy,
                    "sim_tot_pcs": sim_tot_pcs,
                    "sim_tot_jpy": sim_tot_jpy,
                }
            )

    return pd.DataFrame(records)


def _aggregate_exposure_table(grid_df, group_cols, first_cols, total_meta):
    """Aggregate bottom-up forecast grid by group_cols and compute true group-level percentiles from simulated paths."""
    sum_cols = [
        "target_pcs",
        "rec_p50_pcs",
        "gap_pcs",
        "hist_nr_pcs",
        "stretch_pcs",
        "target_jpy",
        "rec_p50_jpy",
        "gap_jpy",
        "hist_nr_jpy",
        "stretch_jpy",
    ]

    rows = []
    for keys, grp in grid_df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for fc in first_cols:
            row[fc] = grp[fc].iloc[0]
        for sc in sum_cols:
            row[sc] = float(grp[sc].sum())

        grp_sim_pcs = np.sum(np.vstack(grp["sim_tot_pcs"].values), axis=0)
        grp_sim_jpy = np.sum(np.vstack(grp["sim_tot_jpy"].values), axis=0)
        row["rec_p10_pcs"] = float(np.percentile(grp_sim_pcs, 10))
        row["rec_p90_pcs"] = float(np.percentile(grp_sim_pcs, 90))
        row["rec_p10_jpy"] = float(np.percentile(grp_sim_jpy, 10))
        row["rec_p90_jpy"] = float(np.percentile(grp_sim_jpy, 90))
        row["gap_pct"] = (row["gap_pcs"] / row["target_pcs"]) * 100.0
        rows.append(row)

    agg_df = pd.DataFrame(rows)

    tot_sim_pcs = np.sum(np.vstack(grid_df["sim_tot_pcs"].values), axis=0)
    tot_sim_jpy = np.sum(np.vstack(grid_df["sim_tot_jpy"].values), axis=0)

    tot_row = dict(total_meta)
    for sc in sum_cols:
        tot_row[sc] = float(agg_df[sc].sum())
    tot_row["rec_p10_pcs"] = float(np.percentile(tot_sim_pcs, 10))
    tot_row["rec_p90_pcs"] = float(np.percentile(tot_sim_pcs, 90))
    tot_row["rec_p10_jpy"] = float(np.percentile(tot_sim_jpy, 10))
    tot_row["rec_p90_jpy"] = float(np.percentile(tot_sim_jpy, 90))
    tot_row["gap_pct"] = (tot_row["gap_pcs"] / tot_row["target_pcs"]) * 100.0

    return pd.concat([agg_df, pd.DataFrame([tot_row])], ignore_index=True)


def build_channel_revenue_exposure_table(grid_df):
    gross_total = float((grid_df["rec_p50_pcs"] * grid_df["base_list_price"]).sum())
    weighted_contra = 1.0 - (float(grid_df["rec_p50_jpy"].sum()) / gross_total) if gross_total > 0 else 0.0
    return _aggregate_exposure_table(
        grid_df,
        group_cols=["channel_code", "channel_name"],
        first_cols=["contra_rate"],
        total_meta={
            "channel_code": "TOTAL",
            "channel_name": "Portfolio Total",
            "contra_rate": weighted_contra,
        },
    )


def build_brand_revenue_exposure_table(grid_df):
    return _aggregate_exposure_table(
        grid_df,
        group_cols=["brand_code", "brand_name", "modality"],
        first_cols=["base_list_price"],
        total_meta={
            "brand_code": "TOTAL",
            "brand_name": "Portfolio Total",
            "modality": "All Modalities",
            "base_list_price": float(grid_df["base_list_price"].mean()),
        },
    )
