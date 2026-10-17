# Recurring Demand Forecasting & Commercial Gap Analysis
**Scope:** Japan Contact Lens Portfolio | 24M Historical Actuals (Oct 2024 - Sep 2026) & 6M Forward Plan (Oct 2026 - Mar 2027)

---

## Project Overview & Introduction

In the contact lens industry, monthly shipment volume is driven by two very different customer behaviors:
1. **Recurring Demand:** Steady, repeat refill orders from existing patients who wear daily or bi-weekly disposable lenses.
2. **Non-Recurring Demand:** One-off volume spikes driven by hospital tender wins, new optical retail store openings, and seasonal distributor bulk deals.

When a company forecasts total shipments using a single time-series model, the algorithm mistakes one-off promotional spikes for permanent demand growth. At the same time, commercial leadership cannot see how much of the annual sales target will come in organically versus how much the sales team needs to actively hunt.

In this analysis, I solve this problem by separating the two demand streams and running a two-stage planning workflow across 4 brands and 4 commercial channels:
* **Stage 1 - Volume Forecasting & Target Gap (in Pieces):** I isolate historical recurring demand, validate a combined time-series model over a 12-month walk-forward backtest, project the 6-month recurring baseline (Oct 2026 - Mar 2027) with a 5,000-run Monte Carlo uncertainty band, and calculate the non-recurring volume gap that the sales team must close by channel and brand.
* **Stage 2 - Gross-to-Net Financials & Pricing Policy (in JPY):** Once the physical volume numbers are locked, I convert pieces into Gross Sales and apply channel-specific contra rates (trade rebates and allowances) to show the exact Net Revenue gap in Japanese Yen, alongside channel price elasticity (PED) rules.

### Key Findings (Oct 2026 - Mar 2027 Planning Horizon)

* **Volume Baseline vs. Budget Target:** Organic recurring refills are projected to deliver **741.44M pcs (P50)** against the 6-month corporate target of **1,014.18M pcs** (covering **73.1%** of target). Under a P10 downside scenario, organic volume drops to **717.70M pcs**, while the P90 upside boundary reaches **761.40M pcs**.
* **Sales Team Volume Quota:** The remaining **272.74M pcs (26.9% of target)** cannot be met organically and must be closed through active commercial initiatives: **111.92M pcs** in Retail Chains, **90.78M pcs** in Online, **42.74M pcs** in Eye Clinics, and **27.30M pcs** in Direct Accounts.
* **Net Revenue Gap (JPY):** After deducting channel contra spend, the P50 recurring baseline translates to **¥21,870.7M JPY** against the **¥30,232.6M JPY** net sales budget, leaving an **¥8,361.8M JPY** revenue gap, with **Brand B** accounting for **60.6% (¥5,069.7M JPY)** of the total shortfall.

---

## 1. Historical Demand Split & Signal Decomposition (Pieces)

Across the 24-month historical window (Oct 2024 - Sep 2026), recurring patient refills generated **74.3% (2,933.08M pcs)** of total shipments, while non-recurring commercial orders generated **25.7% (1,016.47M pcs)**. Brand B has the highest reliance on non-recurring promotional volume (42.1% non-recurring), whereas specialty astigmatism lenses (Brand D) rely primarily on steady prescription refills (87.7% recurring).

| Brand Code | Brand Name | Modality | 24M Recurring (M pcs) | 24M Non-Recurring (M pcs) | 24M Total (M pcs) | Recurring Share (%) |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **BRAND_A** | Brand A | 1-Day Spherical | 1,601.20M | 350.13M | 1,951.33M | 82.1% |
| **BRAND_B** | Brand B | 1-Day Premium | 650.70M | 473.45M | 1,124.15M | 57.9% |
| **BRAND_C** | Brand C | 2-Week Mainstream | 455.77M | 161.38M | 617.15M | 73.9% |
| **BRAND_D** | Brand D | 1-Day Toric | 225.41M | 31.51M | 256.92M | 87.7% |
| **Total** | **Portfolio Total** | **All Modalities** | **2,933.08M** | **1,016.47M** | **3,949.55M** | **74.3%** |

Before fitting forecasting models, I decompose the monthly recurring volume series into three additive components (`src/decomposition.py`):

$$\text{Actual Volume}_t = \text{Level}_t + \text{Seasonality}_t + \text{Residual}_t$$

* **Level (12M Centered Moving Average):** Tracks the underlying organic baseline (**122.23M pcs/month**) without seasonal distortion.
* **Seasonality:** Captures monthly deviations from the trend line. Quarter-end months (March, June, September, December) consistently show positive lifts of **+13.0M to +16.6M pcs** driven by quarterly retail restocking.
* **Residual Noise:** Unexplained monthly variance has a standard deviation of **1.40M pcs** and a Coefficient of Variation ($\text{CV} = \sigma / \mu$) of **1.1%**, confirming a clean signal suitable for statistical projection.

![Signal Decomposition](figures/chapter1_3layer_decomposition_diagnostics.png)

---

## 2. Model Backtesting (12-Month Walk-Forward Validation)

I evaluate three time-series models and their equal-weighted combination over the last 12 closed months (Oct 2025 - Sep 2026) in `src/models.py`. Models are re-estimated sequentially each month using only prior data to mirror real production planning:

1. **Damped Naive ($\phi = 0.70$):** Uses the same calendar month from 12 months prior plus a year-over-year local drift term damped at $\phi = 0.70$ per step:
   $$\hat{Y}_{t+h} = Y_{t+h-12} + \phi^h \cdot \text{Drift}_{\text{YoY}}$$
   * **Why I chose $\phi = 0.70$:** Contact lens shipments experience strong quarterly restocking spikes every 3 months (Mar, Jun, Sep, Dec). If $\phi$ is set too high (e.g., $0.90$), a strong quarter-end surge lingers for the entire 6-month horizon ($0.90^6 = 53.1\%$), causing systematic over-forecasting. Setting $\phi = 0.70$ preserves **70.0%** of recent momentum in Month 1 and **49.0%** in Month 2, decays to **34.3%** by the end of the first quarter (Month 3), and fades to **11.8%** by Month 6 - acting as a conservative stabilizer against the un-damped linear trend model.
2. **Linear Trend + Seasonality (OLS):** Fits a linear trend on historical volume and adds the average detrended seasonal lift for each calendar month.
3. **Exponential Smoothing (Holt-Winters Additive):** Updates local level and 12-month seasonal indices dynamically, initialized with a flat slope ($b_0 = 0$) so quarter-end spikes are absorbed into seasonal factors rather than inflating the trend.
4. **Combined Forecast (Simple Average):** Averages the three models (1/3 weight each).

| Model | WMAPE (%) | Forecast Bias (%) | MAE (Million Pcs) | RMSE (Million Pcs) | Operational Characteristic |
|:---|:---:|:---:|:---:|:---:|:---|
| **Damped Naive ($\phi = 0.70$)** | 1.84% | +0.73% | 2.24M | 2.75M | Dampens local momentum over multi-month horizons |
| **Trend + Seasonality (OLS)** | 2.92% | +1.96% | 3.55M | 4.00M | Captures fixed calendar-month seasonal swings |
| **Exponential Smoothing (HW)** | 1.70% | +0.20% | 2.06M | 2.83M | Strongest aggregate fit; adapts to recent level shifts |
| **Combined Forecast (Simple Avg)** | **2.00%** | **+0.96%** | **2.44M** | **2.91M** | **Selected Production Model: Diversifies risk across all 16 Brand × Channel series** |

*Note on Model Selection:* While Exponential Smoothing achieved the lowest error on the aggregated portfolio series (1.70% WMAPE), no single model consistently wins across all 16 individual Brand × Channel series. I use the **Combined Forecast (2.00% WMAPE)** in production to balance the un-damped OLS trend with the $\phi = 0.70$ Damped Naive anchor over a 6-month rolling horizon.

![Walk-Forward Backtest](figures/chapter2_walk_forward_backtest.png)

---

## 3. 6-Month Recurring Volume Forecast & Uncertainty Range (Pieces)

I project recurring demand for Oct 2026 - Mar 2027 and run a **5,000-iteration Empirical Bootstrap Monte Carlo simulation** (`src/simulation.py`) using centered out-of-sample backtest residuals (scaled by $\sqrt{h}$ across the forecast horizon) to establish operational planning bounds:

* **Why 5,000 Iterations (Instead of 1,000 or 10,000)?**
  1. **Tail Percentile Stability:** Estimating tail risk bounds like **P10** and **P90** requires enough simulated paths in the outer 10% tails. The binomial standard error of a quantile indicator scales at $\sqrt{q(1-q)/N}$. At $N = 1,000$, the tail standard error is nearly $1.0\%$, which causes noticeable run-to-run jitter. At **$N = 5,000$**, I get 500 observations in each 10% tail and the standard error drops to **0.42%**, stabilizing P10 and P90 to two decimal places.
  2. **Empirical Bootstrap Efficiency:** Because I am resampling from a finite pool of historical monthly residuals rather than a theoretical continuous distribution, increasing $N$ from 5,000 to 10,000 only reduces the standard error by $0.12\%$ (from $0.42\%$ to $0.30\%$) while doubling runtime across all 16 Brand × Channel series. Thus, 5,000 iterations is the optimal threshold for convergence and speed.

* **P50 Baseline (741.44M pcs):** Expected recurring volume used as the baseline demand plan.
* **P10 Downside Floor (717.70M pcs):** Conservative volume floor (90% exceedance probability) for downside financial risk assessment.
* **P90 Upside Ceiling (761.40M pcs):** Upper demand boundary used by Supply Chain to size safety stock and factory capacity buffers.

| Forward Period | Recurring P10 Floor (M pcs) | Recurring P50 Base (M pcs) | Recurring P90 Buffer (M pcs) | Corporate Target (M pcs) | Non-Recurring Gap to Close (M pcs) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **2026-10** | 112.78M | 116.52M | 119.66M | 158.92M | **42.40M** |
| **2026-11** | 111.24M | 114.92M | 118.02M | 159.87M | **44.95M** |
| **2026-12** | 134.26M | 138.70M | 142.43M | 186.71M | **48.01M** |
| **2027-01** | 114.50M | 118.29M | 121.47M | 157.60M | **39.31M** |
| **2027-02** | 110.72M | 114.39M | 117.47M | 159.84M | **45.45M** |
| **2027-03** | 134.20M | 138.62M | 142.35M | 191.24M | **52.62M** |
| **6M Total** | **717.70M** | **741.44M** | **761.40M** | **1,014.18M** | **272.74M (26.9%)** |

![Forward Fan Chart](figures/chapter3_monte_carlo_fan_chart.png)

---

## 4. Non-Recurring Volume Gap by Channel & Brand (Pieces)

Subtracting the bottom-up P50 recurring forecast from the corporate budget target defines the **272.74M pcs** non-recurring volume quota that the commercial sales teams must generate over the next 6 months:

### A. 6-Month Volume Gap by Channel (Million Pcs)
| Channel Code | Channel Name | Target Budget (M pcs) | Recurring Base P50 (M pcs) | Non-Recurring Sales Quota (M pcs) | Gap Share (%) | Commercial Execution Focus |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **CH_RETAIL_CHAIN** | Optical Retail Chains | 409.06M | 297.15M | **111.92M** | 27.4% | Quarter-end tier volume rebates & new store rollouts |
| **CH_ONLINE** | E-Commerce / Online | 349.32M | 258.54M | **90.78M** | 26.0% | Multi-box subscription bundles (maintain list price) |
| **CH_CLINIC_INST** | Eye Clinics & Hospitals | 154.60M | 111.85M | **42.74M** | 27.6% | Hospital tender contracts & trial fitting kits |
| **CH_DIRECT_ACC** | Direct Key Accounts | 101.20M | 73.90M | **27.30M** | 27.0% | Annual B2B volume agreements |
| **Total** | **All Channels** | **1,014.18M** | **741.44M** | **272.74M** | **26.9%** | **Total Non-Recurring Volume Quota** |

### B. 6-Month Volume Gap by Brand (Million Pcs)
| Brand Code | Brand Name | Modality | Target Budget (M pcs) | Recurring Base P50 (M pcs) | Non-Recurring Sales Quota (M pcs) | Gap Share (%) |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **BRAND_A** | Brand A | 1-Day Spherical | 495.94M | 405.80M | **90.14M** | 18.2% |
| **BRAND_B** | Brand B | 1-Day Premium | 297.61M | 164.57M | **133.04M** | 44.7% |
| **BRAND_C** | Brand C | 2-Week Mainstream | 156.13M | 113.69M | **42.44M** | 27.2% |
| **BRAND_D** | Brand D | 1-Day Toric | 64.50M | 57.37M | **7.13M** | 11.1% |
| **Total** | **Portfolio Total** | **All Modalities** | **1,014.18M** | **741.44M** | **272.74M** | **26.9%** |

---

## 5. Gross-to-Net Financials: Converting Volume to Net Revenue (JPY)

### What is "Contra" and Why Do I Use a Channel Contra Rate?
In commercial accounting, **Gross Sales** is the invoice value of shipped lenses at Base List Price ($\text{Volume} \times \text{List Price}$). However, the business does not keep 100% of Gross Sales. **Contra Revenue** (trade spend deductions) represents commercial rebates and allowances subtracted from Gross Sales to get **Net Revenue**:
* **Volume Rebates:** Retrospective payouts to optical retail chains and distributors when they hit quarterly or annual volume tiers.
* **Promotional & Co-Op Allowances:** Bill-backs from online platforms and retail chains for seasonal campaigns and multi-pack discounts.
* **Return & Exchange Allowances:** Credits issued for expired stock or patient prescription exchanges.

**Why I forecast in Pieces first and apply a Contra Rate (%) at the end:**  
In the accounting ledger, Finance books lump-sum rebate settlements at the close of each quarter (March, June, September, December). As the chart below shows, this accounting practice causes ledger Net Sales to drop sharply in quarter-ending months even though physical shipments are peaking. If I trained a time-series model directly on ledger Net Sales, the model would forecast artificial revenue drops every quarter-end. By forecasting physical pieces first and then applying each channel's **Target Contra Rate (%)** (`src/finance.py`) - such as **22.0%** in promotion-heavy Online vs. **10.5%** in Eye Clinics - the revenue plan stays aligned with real physical demand:

$$\text{Net Revenue (JPY)} = \text{Volume (Pcs)} \times \text{Base List Price (JPY)} \times (1 - \text{Channel Contra Rate})$$

![Physical Volume vs Net Sales Ledger](figures/chapter1_accounting_trap_physical_vs_net.png)

---

## 6. Revenue Gap by Brand & Channel (Sales Team Targets in Million JPY)

Applying the Gross-to-Net calculation to my bottom-up volume projections gives the 6-month Net Revenue baseline and the exact revenue gap that each sales team needs to close:

### A. 6-Month Revenue Gap by Brand (Million JPY)
| Brand Code | Brand Name | Target Vol (M pcs) | Rec Base P50 (M pcs) | Vol Gap (M pcs) | Target Net Rev (M JPY) | Rec Base P50 (M JPY) | Downside P10 (M JPY) | Revenue Gap to Close (M JPY) | Rev Gap (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BRAND_A** | Brand A | 495.94M | 405.80M | **90.14M** | ¥12,823.1M | ¥11,079.1M | ¥10,726.6M | **¥1,744.0M** | 13.6% |
| **BRAND_B** | Brand B | 297.61M | 164.57M | **133.04M** | ¥10,762.7M | ¥5,693.0M | ¥5,496.0M | **¥5,069.7M** | 47.1% |
| **BRAND_C** | Brand C | 156.13M | 113.69M | **42.44M** | ¥4,085.3M | ¥2,902.3M | ¥2,802.9M | **¥1,183.0M** | 29.0% |
| **BRAND_D** | Brand D | 64.50M | 57.37M | **7.13M** | ¥2,561.5M | ¥2,196.3M | ¥2,114.8M | **¥365.2M** | 14.3% |
| **TOTAL** | **Portfolio Total** | **1,014.18M** | **741.44M** | **272.74M** | **¥30,232.6M** | **¥21,870.7M** | **¥21,140.3M** | **¥8,361.8M** | **27.7%** |

### B. 6-Month Revenue Gap by Channel (Million JPY)
| Channel Code | Channel Name | Channel Contra Rate | Target Net Rev (M JPY) | Recurring Base P50 (M JPY) | Downside P10 Floor (M JPY) | Sales Team Revenue Gap (M JPY) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **CH_RETAIL_CHAIN** | Optical Retail Chains | 18.5% | ¥12,378.9M | ¥9,124.9M | ¥8,853.0M | **¥3,253.9M** |
| **CH_ONLINE** | E-Commerce / Online | 22.0% | ¥10,005.3M | ¥6,859.0M | ¥6,599.6M | **¥3,146.3M** |
| **CH_CLINIC_INST** | Eye Clinics & Hospitals | 10.5% | ¥4,735.6M | ¥3,534.7M | ¥3,412.7M | **¥1,200.9M** |
| **CH_DIRECT_ACC** | Direct Key Accounts | 15.0% | ¥3,112.8M | ¥2,352.1M | ¥2,275.0M | **¥760.7M** |
| **TOTAL** | **Portfolio Total** | **17.5% (Weighted)** | **¥30,232.6M** | **¥21,870.7M** | **¥21,140.3M** | **¥8,361.8M** |

![Monte Carlo Revenue Distribution](figures/chapter3_monte_carlo_revenue_density.png)

![Executive Revenue Summary](figures/chapter4_executive_revenue_bridge.png)

---

## 7. Channel Price Elasticity (PED) & Trade Spend Policy

### How I Calculate Price Elasticity of Demand (PED)
Price Elasticity of Demand measures how sensitive a channel's shipment volume ($Q$) is to changes in the effective net selling price ($P$):

$$\text{PED}_c = \frac{\%\Delta Q_c}{\%\Delta P_c} = \frac{(Q_1 - Q_0) / Q_0}{(P_1 - P_0) / P_0}$$

To estimate $\text{PED}_c$ from the 24-month historical dataset without seasonal distortion, I first remove the monthly seasonal index from shipped volume ($Q_{c,t}^{\text{adj}}$) and compute the realized net price per piece after contra deductions ($P_{c,t}^{\text{net}} = \text{Net Sales}_{c,t} / Q_{c,t}$). I then fit a log-log Ordinary Least Squares (OLS) regression for each channel:

$$\ln\left(Q_{c,t}^{\text{adj}}\right) = \alpha_c + \text{PED}_c \cdot \ln\left(P_{c,t}^{\text{net}}\right) + \varepsilon_{c,t}$$

When simulating a price change of $\Delta P\%$ in `src/finance.py`, the projected volume response is calculated as:

$$Q_{\text{new}} = Q_{\text{base}} \times \left(1 + \text{PED}_c \cdot \Delta P\%\right)$$

### Channel Pricing & Rebate Scenarios
Using these PED coefficients, I evaluate how a **+5% list price increase** vs. a **-2% contra rebate reduction** impacts monthly net revenue across channels:

| Channel Code | Channel Name | PED | Avg Monthly Net Rev | +5% Price Hike Impact (Monthly) | -2% Contra Rate Reduction (Monthly) | Commercial Pricing & Rebate Policy |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **CH_CLINIC_INST** | Eye Clinics & Hospitals | **0.00** | ¥854.0M | **+¥42.8M** (0% vol loss) | **+¥19.0M** | **Margin Expansion:** Demand is prescription-driven and price-inelastic; reduce contra rebates or adjust list prices without volume attrition. |
| **CH_RETAIL_CHAIN** | Optical Retail Chains | **-1.11** | ¥1,496.8M | -¥12.4M (-5.6% vol loss) | **+¥36.8M** | **Conditional Volume Tiers:** Hold base list price; pay contra rebates strictly on incremental volume above the P50 baseline. |
| **CH_DIRECT_ACC** | Direct Key Accounts | **-1.43** | ¥377.0M | -¥9.6M (-7.2% vol loss) | **+¥8.8M** | **Contract Rebate Caps:** Cap off-invoice discounts; structure annual rebate tiers to lock in the 27.30M pcs gap. |
| **CH_ONLINE** | E-Commerce / Online | **-4.51** | ¥941.6M | **-¥186.8M** (-22.6% vol loss) | **+¥24.2M** | **Price Protection:** Highly price-elastic; avoid list price increases. Close the 90.78M pcs gap using multi-box bundle promotions. |

---

## 8. Commercial & Supply Chain Recommendations

1. **Baseline Financial & Inventory Planning:**
   - Anchor rolling 6-month FP&A revenue projections on my P50 recurring baseline (**741.44M pcs / ¥21,870.7M JPY**), using the P10 floor (**¥21,140.3M JPY**) as the downside risk reserve.
   - Set Supply Chain raw material and factory capacity buffers at the **P90 threshold (761.40M pcs)** to protect fill rates during quarter-end peaks.
2. **Sales Quota Allocation (Closing the 272.74M Pcs / ¥8,361.8M JPY Gap):**
   - **Retail Chains (111.92M pcs / ¥3,253.9M JPY):** Focus on Q3 and Q4 quarter-end volume tier incentives, conditional on accounts exceeding their historical recurring run-rate.
   - **Online (90.78M pcs / ¥3,146.3M JPY):** Drive volume through 90-pack and multi-box subscription bundles rather than unit price cuts.
   - **Eye Clinics (42.74M pcs / ¥1,200.9M JPY) & Brand B Focus:** **Brand B** accounts for **¥5,069.7M JPY (60.6%)** of the total company revenue gap. Prioritize hospital tenders and practitioner fitting programs for Brand B.
   - **Direct Accounts (27.30M pcs / ¥760.7M JPY):** Secure upfront volume commitments during annual contract renewals.

---

## 9. Repository Structure

```text
Forecast/
├── data/                                             # Commercial & demand datasets
│   ├── dim_calendar.csv                              # Calendar dimension (30 monthly periods)
│   ├── dim_channel.csv                               # Channels, PED coefficients, target contra rates
│   ├── dim_product.csv                               # Brand metadata, pack sizes, list prices
│   ├── fact_recurring_split.csv                      # Recurring vs. non-recurring volume split
│   ├── fact_sales_transaction.csv                    # Sales transactions, gross and net revenue
│   ├── fact_target_budget.csv                        # Corporate budget volume and revenue targets
│   └── fact_trade_contra.csv                         # Trade spend and contra deductions breakdown
├── notebooks/
│   └── recurring_demand_forecast_evaluation.ipynb    # End-to-end analysis & forecasting notebook
├── src/                                              # Core forecasting & financial modules
│   ├── __init__.py
│   ├── decomposition.py                              # Classical 3-layer decomposition (Level, Seasonal, Noise)
│   ├── models.py                                     # Damped Naive, Trend+Seasonality, Holt-Winters
│   ├── simulation.py                                 # Empirical bootstrap Monte Carlo simulation
│   ├── finance.py                                    # Bottom-up Brand x Channel forecast & Gross-to-Net bridge
│   └── metrics.py                                    # WMAPE, Bias, MAE, RMSE evaluation functions
├── figures/                                          # Output charts and diagnostics
│   ├── chapter1_3layer_decomposition_diagnostics.png
│   ├── chapter1_accounting_trap_physical_vs_net.png
│   ├── chapter2_walk_forward_backtest.png
│   ├── chapter3_monte_carlo_fan_chart.png
│   ├── chapter3_monte_carlo_revenue_density.png
│   └── chapter4_executive_revenue_bridge.png
├── requirements.txt
└── README.md
```

---

## 10. Running the Analysis

Install dependencies:
```bash
pip install -r requirements.txt
```

Launch the Jupyter Notebook:
```bash
jupyter notebook notebooks/recurring_demand_forecast_evaluation.ipynb
```
