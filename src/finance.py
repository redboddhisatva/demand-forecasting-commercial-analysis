"""
Financial Integration and Commercial Revenue Bridge
===================================================
Connects physical volume forecasts to financial revenue in Japanese Yen (JPY):
1. Gross-to-Net Bridge:
   Gross Revenue = Volume (Pcs) * Base List Price
   Trade Contra Spend = Gross Revenue * Contra Rate
   Net Revenue = Gross Revenue - Trade Contra Spend
2. Channel Commercial Scenario Testing:
   Evaluates impact of price adjustments and trade rebate reductions across channels
   using Price Elasticity of Demand (PED) coefficients.
3. Commercial Gap to Budget Target:
   Builds bottom-up forecast tables aggregated by Brand and Channel in both
   Physical Pieces and Million JPY, receiving real forecast model outputs.
"""

import numpy as np
import pandas as pd


def calc_gross_to_net_revenue(volume_pcs, base_list_price, contra_rate):
    """
    Computes Gross Revenue, Trade Contra Spend, and Net Revenue in JPY.
    """
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


def evaluate_commercial_action_scenarios(
    channel_code,
    channel_name,
    base_monthly_vol,
    base_gross_price,
    contra_rate,
    ped_coefficient,
):
    """
    Evaluates practical commercial scenarios for pricing and rebate adjustments.
    Scenario 1: Baseline current state
    Scenario 2: +2% Price Adjustment
    Scenario 3: +5% Price Adjustment
    Scenario 4: 2% Rebate Cut (tightening contra spending)
    """
    scenarios = []

    # Scenario 1: Baseline
    base_gross = base_monthly_vol * base_gross_price
    base_contra = base_gross * contra_rate
    base_net = base_gross - base_contra
    scenarios.append({
        "channel_code": channel_code,
        "channel_name": channel_name,
        "scenario": "Baseline",
        "price_adj_pct": 0.0,
        "contra_rate_pct": contra_rate * 100.0,
        "monthly_vol_pcs": base_monthly_vol,
        "net_revenue_jpy": base_net,
        "net_rev_delta_jpy": 0.0,
    })

    # Scenario 2: +2% Price Adjustment
    vol_chg_2 = ped_coefficient * 0.02
    new_vol_2 = max(0.0, base_monthly_vol * (1.0 + vol_chg_2))
    new_price_2 = base_gross_price * 1.02
    new_net_2 = new_vol_2 * new_price_2 * (1.0 - contra_rate)
    scenarios.append({
        "channel_code": channel_code,
        "channel_name": channel_name,
        "scenario": "+2% Price Adjustment",
        "price_adj_pct": +2.0,
        "contra_rate_pct": contra_rate * 100.0,
        "monthly_vol_pcs": new_vol_2,
        "net_revenue_jpy": new_net_2,
        "net_rev_delta_jpy": new_net_2 - base_net,
    })

    # Scenario 3: +5% Price Adjustment
    vol_chg_5 = ped_coefficient * 0.05
    new_vol_5 = max(0.0, base_monthly_vol * (1.0 + vol_chg_5))
    new_price_5 = base_gross_price * 1.05
    new_net_5 = new_vol_5 * new_price_5 * (1.0 - contra_rate)
    scenarios.append({
        "channel_code": channel_code,
        "channel_name": channel_name,
        "scenario": "+5% Price Adjustment",
        "price_adj_pct": +5.0,
        "contra_rate_pct": contra_rate * 100.0,
        "monthly_vol_pcs": new_vol_5,
        "net_revenue_jpy": new_net_5,
        "net_rev_delta_jpy": new_net_5 - base_net,
    })

    # Scenario 4: Tightening Contra Rebate (-2% points in contra rate)
    tightened_contra = max(0.0, contra_rate - 0.02)
    new_net_rebate = base_gross * (1.0 - tightened_contra)
    scenarios.append({
        "channel_code": channel_code,
        "channel_name": channel_name,
        "scenario": "Rebate Optimization (-2% Contra)",
        "price_adj_pct": 0.0,
        "contra_rate_pct": tightened_contra * 100.0,
        "monthly_vol_pcs": base_monthly_vol,
        "net_revenue_jpy": new_net_rebate,
        "net_rev_delta_jpy": new_net_rebate - base_net,
    })

    return pd.DataFrame(scenarios)


def build_bottom_up_grid_forecast(
    fact_recurring_df,
    fact_target_df,
    dim_product_df,
    dim_channel_df,
    forecast_func,
    bootstrap_func,
    future_start_date='2026-10-01',
    n_forward=6,
):
    """
    Computes bottom-up 6-month forward forecasts and Monte Carlo risk percentiles
    for each combination of (brand_code, channel_code) using genuine forecast models.

    Returns:
    --------
    grid_df : pd.DataFrame
        Detailed table containing Target, P50 Base, P10 Floor, P90 Buffer, and Gap
        in both Pieces and JPY for each Brand x Channel pair.
    """
    rec_only = fact_recurring_df[fact_recurring_df['demand_stream'] == 'Recurring'].copy()
    fut_target = fact_target_df[fact_target_df['period_date'] >= future_start_date].copy()

    prod_map = dim_product_df.set_index('brand_code').to_dict('index')
    chan_map = dim_channel_df.set_index('channel_code').to_dict('index')

    brands = sorted(rec_only['brand_code'].unique())
    channels = sorted(rec_only['channel_code'].unique())

    records = []
    for b in brands:
        p_info = prod_map.get(b, {})
        base_price = p_info.get('base_list_price', 30.0)
        brand_name = p_info.get('brand_name', b)
        modality = p_info.get('modality', '')

        for c in channels:
            c_info = chan_map.get(c, {})
            contra_rate = c_info.get('target_contra_rate', 0.05)
            channel_name = c_info.get('channel_name', c)
            net_price = base_price * (1.0 - contra_rate)

            # Filter historical recurring data
            sub = rec_only[(rec_only['brand_code'] == b) & (rec_only['channel_code'] == c)].sort_values('period_date').reset_index(drop=True)
            sub['period_date'] = pd.to_datetime(sub['period_date'])
            sub['Month'] = sub['period_date'].dt.month

            y = sub['stream_qty_pcs'].values.astype(float)
            m = sub['Month'].values

            # Filter future target
            tgt_sub = fut_target[(fut_target['brand_code'] == b) & (fut_target['channel_code'] == c)].sort_values('period_date').reset_index(drop=True)
            target_pcs = float(tgt_sub['target_qty_pcs'].sum())
            target_jpy = float(tgt_sub['target_net_amt'].sum())

            # Real model projection
            fwd_df = forecast_func(y, m, n_forward=n_forward)
            point_fwd = fwd_df['point_forecast'].values

            # Historical residuals for bootstrap
            t_idx = np.arange(len(y))
            slope_t, intercept = np.polyfit(t_idx, y, 1) if len(y) >= 2 else (0.0, float(y[0]))
            detrended = y - (intercept + slope_t * t_idx)
            m_lifts = {m_val: float(np.mean(detrended[m == m_val])) for m_val in range(1, 13) if np.sum(m == m_val) > 0}
            preds_hist = np.array([intercept + slope_t * t + m_lifts.get(m[t], 0.0) for t in range(len(y))])
            residuals = y - preds_hist

            # Monte Carlo empirical bootstrap
            _, _, tot_mc = bootstrap_func(
                point_fwd,
                residuals,
                n_simulations=5000,
                random_seed=42,
                target_values=tgt_sub['target_qty_pcs'].values
            )

            rec_p50_pcs = tot_mc['total_point_forecast']
            rec_p10_pcs = tot_mc['total_p10_bear']
            rec_p90_pcs = tot_mc['total_p90_bull']

            gap_pcs = target_pcs - rec_p50_pcs
            rec_p50_jpy = rec_p50_pcs * net_price
            rec_p10_jpy = rec_p10_pcs * net_price
            rec_p90_jpy = rec_p90_pcs * net_price
            gap_jpy = target_jpy - rec_p50_jpy

            records.append({
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
                "target_jpy": target_jpy,
                "rec_p50_jpy": rec_p50_jpy,
                "rec_p10_jpy": rec_p10_jpy,
                "rec_p90_jpy": rec_p90_jpy,
                "gap_jpy": gap_jpy,
            })

    return pd.DataFrame(records)


def build_channel_revenue_exposure_table(grid_df):
    """
    Aggregates bottom-up forecast results by Channel for Sales Leadership.
    """
    ch_agg = grid_df.groupby(['channel_code', 'channel_name']).agg({
        'contra_rate': 'first',
        'target_pcs': 'sum',
        'rec_p50_pcs': 'sum',
        'rec_p10_pcs': 'sum',
        'rec_p90_pcs': 'sum',
        'gap_pcs': 'sum',
        'target_jpy': 'sum',
        'rec_p50_jpy': 'sum',
        'rec_p10_jpy': 'sum',
        'rec_p90_jpy': 'sum',
        'gap_jpy': 'sum',
    }).reset_index()

    ch_agg['gap_pct'] = (ch_agg['gap_pcs'] / ch_agg['target_pcs']) * 100.0

    tot_row = pd.DataFrame([{
        "channel_code": "TOTAL",
        "channel_name": "Portfolio Total",
        "contra_rate": 1.0 - (ch_agg['rec_p50_jpy'].sum() / (grid_df['rec_p50_pcs'] * grid_df['base_list_price']).sum()) if len(grid_df) > 0 and (grid_df['rec_p50_pcs'] * grid_df['base_list_price']).sum() > 0 else 0.0,
        "target_pcs": ch_agg['target_pcs'].sum(),
        "rec_p50_pcs": ch_agg['rec_p50_pcs'].sum(),
        "rec_p10_pcs": ch_agg['rec_p10_pcs'].sum(),
        "rec_p90_pcs": ch_agg['rec_p90_pcs'].sum(),
        "gap_pcs": ch_agg['gap_pcs'].sum(),
        "target_jpy": ch_agg['target_jpy'].sum(),
        "rec_p50_jpy": ch_agg['rec_p50_jpy'].sum(),
        "rec_p10_jpy": ch_agg['rec_p10_jpy'].sum(),
        "rec_p90_jpy": ch_agg['rec_p90_jpy'].sum(),
        "gap_jpy": ch_agg['gap_jpy'].sum(),
        "gap_pct": (ch_agg['gap_pcs'].sum() / ch_agg['target_pcs'].sum()) * 100.0,
    }])

    return pd.concat([ch_agg, tot_row], ignore_index=True)


def build_brand_revenue_exposure_table(grid_df):
    """
    Aggregates bottom-up forecast results by Brand for Commercial Leadership.
    """
    b_agg = grid_df.groupby(['brand_code', 'brand_name', 'modality']).agg({
        'base_list_price': 'first',
        'target_pcs': 'sum',
        'rec_p50_pcs': 'sum',
        'rec_p10_pcs': 'sum',
        'rec_p90_pcs': 'sum',
        'gap_pcs': 'sum',
        'target_jpy': 'sum',
        'rec_p50_jpy': 'sum',
        'rec_p10_jpy': 'sum',
        'rec_p90_jpy': 'sum',
        'gap_jpy': 'sum',
    }).reset_index()

    b_agg['gap_pct'] = (b_agg['gap_pcs'] / b_agg['target_pcs']) * 100.0

    tot_row = pd.DataFrame([{
        "brand_code": "TOTAL",
        "brand_name": "Portfolio Total",
        "modality": "All Modalities",
        "base_list_price": b_agg['base_list_price'].mean(),
        "target_pcs": b_agg['target_pcs'].sum(),
        "rec_p50_pcs": b_agg['rec_p50_pcs'].sum(),
        "rec_p10_pcs": b_agg['rec_p10_pcs'].sum(),
        "rec_p90_pcs": b_agg['rec_p90_pcs'].sum(),
        "gap_pcs": b_agg['gap_pcs'].sum(),
        "target_jpy": b_agg['target_jpy'].sum(),
        "rec_p50_jpy": b_agg['rec_p50_jpy'].sum(),
        "rec_p10_jpy": b_agg['rec_p10_jpy'].sum(),
        "rec_p90_jpy": b_agg['rec_p90_jpy'].sum(),
        "gap_jpy": b_agg['gap_jpy'].sum(),
        "gap_pct": (b_agg['gap_pcs'].sum() / b_agg['target_pcs'].sum()) * 100.0,
    }])

    return pd.concat([b_agg, tot_row], ignore_index=True)
