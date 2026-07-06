# Crypto Macro Crisis Radar

**Crypto Macro Crisis Radar** is a research pipeline for detecting stress regimes in crypto markets by combining crypto market structure, macroeconomic indicators, cross-asset behavior, stablecoin liquidity, news and narrative risk, anomaly detection, and machine learning.

The main idea behind the project is simple: crypto drawdowns rarely happen because of one isolated variable. A real risk regime usually comes from several layers moving together — market stress, liquidity pressure, macro conditions, risk-off behavior in traditional assets, and negative narrative pressure. This project turns those layers into a structured daily regime report and a set of validation studies that explain whether the signal has economic meaning.

---

## Executive Summary

The latest included run, dated **2026-05-30**, classified the market as **Neutral** with a **model-adjusted final risk score of 36.93**.

That result makes sense because BTC was under pressure, but the broader confirmation layers were not all aligned. BTC had a **30-day drawdown of -10.26%**, while crypto stress and liquidity stress were moderate. However, cross-asset risk was low, equity risk was low, and the macro engine did not show major economic-cycle stress. In other words, the system detected weakness, but not enough broad confirmation to classify the environment as a crisis regime.

| Metric | Latest Value |
|---|---:|
| Market Regime | Neutral |
| Model-Adjusted Final Risk Score | 36.93 |
| Rule-Based Risk Score | 45.19 |
| Model Crisis Probability | 12.15% |
| Anomaly Score | 94.27 |
| Macro Risk | 45.37 |
| Macro Engine Stress | 29.81 |
| Cross-Asset Risk | 13.43 |
| Crypto Market Stress | 38.36 |
| Liquidity Stress | 46.93 |
| News / Narrative Risk | 60.50 |

The most important interpretation is that the system did not simply react to BTC being down. It looked for confirmation across macro, liquidity, traditional markets, narrative risk, and model behavior before assigning a stronger stress regime.

---

## Project Goal

The goal of this project is to build an interpretable early-warning framework for crypto market stress.

Instead of forecasting the exact price of BTC or ETH, the system asks a more useful research question:

> Are current market, macro, liquidity, cross-asset, and narrative conditions starting to resemble periods that have historically preceded short-term crypto stress?

This makes the project closer to a **market-regime research system** than a pure price-prediction model. The final output is written like an analyst note, with scores, drivers, scenario watches, and validation results.

Main report output:

```text
outputs/reports/latest_market_regime_report.md
```

Sample report included in this repository:

```text
examples/reports/latest_market_regime_report.md
```

---

## What the System Measures

The system combines several independent signal groups:

| Signal Group | What It Captures |
|---|---|
| Crypto Market Stress | BTC/ETH returns, volatility, drawdowns, volume shocks, BTC-ETH correlation |
| Macro Financial Risk | VIX, Treasury yields, yield curve pressure, Fed Funds, dollar/liquidity proxies |
| Macro Engine | GDP, unemployment, CPI, industrial production, retail sales, private investment, fiscal pressure |
| Cross-Asset Confirmation | S&P 500, Nasdaq, gold, oil, safe-haven behavior, commodity stress |
| Stablecoin Liquidity | Stablecoin market-cap changes and liquidity stress |
| News / Narrative Risk | GDELT-based news risk and category-level narrative pressure |
| Anomaly Detection | Whether the current market configuration looks unusual versus history |
| Machine Learning Risk | Probability of a forward-looking mini-crisis label |

The final score is not one black-box prediction. It is a blended regime score that separates rule-based risk, anomaly behavior, and model-estimated crisis probability.

---

## Target Definition

The supervised learning target is a forward-looking mini-crisis label:

```text
mini_crisis_next_7d = 1 if BTC's minimum forward 7-day return <= -7%
```

This turns the problem into a **risk detection task**. The system is trying to identify environments where BTC is more exposed to sharp near-term weakness, rather than predicting the next daily return.

---

## Methodology

The pipeline follows a full research workflow:

1. Load or update raw market, macro, cross-asset, liquidity, and news data.
2. Build a master feature table with market and macro stress features.
3. Create forward-looking stress labels based on BTC future drawdowns.
4. Train an anomaly model to detect unusual market configurations.
5. Train a baseline crisis classifier using time-aware validation.
6. Generate scored regime history.
7. Produce a daily Markdown market-regime report.
8. Run research modules: event studies, lag/shock analysis, VAR checks, robustness tests, and ablations.

The pipeline is intentionally modular. Data collection, feature engineering, scoring, modeling, reporting, and research validation are separated into different scripts and source modules so the project can be extended without rewriting the full system.

---

## Latest Market Regime Result

The included sample run is dated **2026-05-30**.

### Market Snapshot

| Metric | Value |
|---|---:|
| BTC Price | $73,765.18 |
| BTC 1D Return | 0.54% |
| BTC 7D Return | -3.76% |
| BTC 30D Drawdown | -10.26% |
| BTC 30D Volatility | 26.84% |
| S&P 500 7D Return | 1.43% |
| Nasdaq 7D Return | 2.39% |
| Gold 7D Return | 1.59% |
| Oil WTI 7D Return | -9.57% |

### Interpretation

The system classified the market as **Neutral**, even though BTC was in a meaningful drawdown. That classification is important because it shows the model is not just a drawdown detector.

BTC weakness was visible, but cross-asset confirmation was low at **13.43**, equity risk was low at **8.06**, and macro engine stress was low at **29.81**. Liquidity stress was moderate at **46.93**, narrative risk was elevated at **60.50**, and the anomaly score was high at **94.27**. The final result was a market that deserved monitoring, but not a full risk-off or crisis classification.

The project therefore behaves in a reasonable way: it recognizes stress, but it waits for broader confirmation before escalating the regime.

---

## Model Results

The current classifier is the **Ablation-Selected Risk Classifier v1**. The selected feature set is:

```text
refined_no_macro_engine
```

The macro engine was excluded from the classifier because ablation testing showed that the classifier performed better without that feature group. However, the macro engine remains active in the rule-based score, daily report, VAR analysis, dashboard, and interpretation layer.

### Final Classifier Test Metrics

| Metric | Value |
|---|---:|
| Tuned Threshold | 0.36 |
| ROC-AUC | 0.6209 |
| Average Precision | 0.1820 |
| Brier Score | 0.1813 |
| Accuracy | 0.3643 |
| Balanced Accuracy | 0.6126 |
| Precision | 0.1772 |
| Recall | 0.9589 |
| F1 | 0.2991 |
| Positive Prediction Rate | 0.7655 |

The model is tuned for **high recall**, which means it is better at catching most future mini-crisis events than staying highly selective. That creates false positives, but it is a reasonable tradeoff for an early-warning research system. In market risk work, missing a stress event can be more costly analytically than flagging too many environments for further review.

The confusion matrix also reflects this behavior:

```text
True negatives: 118
False positives: 325
False negatives: 3
True positives: 70
```

The classifier caught **70 of 73** positive mini-crisis cases in the test set, which explains the high recall of **95.89%**. The cost is low precision because many flagged observations did not become mini-crisis events. This is one of the most important findings of the project: the system is useful as a risk monitor, but the classification threshold needs to be interpreted as an alerting threshold, not as a high-conviction prediction.

---

## Ablation Results

Ablation testing was one of the most useful parts of the project because it tested whether every feature group was actually helping.

The second ablation study used **walk-forward validation** instead of one static split. Each fold used a train → validation → test structure, and the classification threshold was tuned on validation before being evaluated on the next test window.

### Full Model Walk-Forward Average

| Metric | Mean | Std |
|---|---:|---:|
| Tuned Threshold | 0.2020 | 0.1499 |
| Test ROC-AUC | 0.5275 | 0.1474 |
| Test Average Precision | 0.1548 | 0.0675 |
| Test Balanced Accuracy | 0.5176 | 0.0202 |
| Test Precision | 0.1233 | 0.0371 |
| Test Recall | 0.9962 | 0.0086 |
| Test F1 | 0.2178 | 0.0588 |
| Default 0.50 F1 | 0.0471 | 0.1052 |

The full model had very high recall, but weak ranking power. The default 0.50 threshold performed poorly, which is why threshold tuning was necessary.

### Refined Model Experiments

| Experiment | ROC-AUC | Avg Precision | F1 | Recall | AUC Δ vs Full | F1 Δ vs Full |
|---|---:|---:|---:|---:|---:|---:|
| refined_no_macro_engine | 0.6248 | 0.1848 | 0.2035 | 0.8750 | 0.0973 | -0.0144 |
| refined_core_market_macro_liquidity | 0.6050 | 0.1776 | 0.1912 | 0.8264 | 0.0775 | -0.0266 |
| refined_market_cross_liquidity_narrative | 0.5939 | 0.1727 | 0.2215 | 0.8658 | 0.0664 | 0.0037 |

The best ranking improvement came from **refined_no_macro_engine**, which improved ROC-AUC from **0.5275** to **0.6248**. That is why the final classifier uses this refined feature set.

### Standalone Signal Value by Feature Group

| Only Group | ROC-AUC | Avg Precision | Balanced Accuracy | F1 | Recall |
|---|---:|---:|---:|---:|---:|
| macro_financial | 0.5898 | 0.1826 | 0.5316 | 0.2067 | 0.5480 |
| cross_asset | 0.5739 | 0.1815 | 0.5040 | 0.2109 | 0.8992 |
| crypto_market | 0.5623 | 0.1429 | 0.5188 | 0.1863 | 0.7193 |
| liquidity | 0.5599 | 0.1709 | 0.5156 | 0.2179 | 0.8397 |
| macro_engine | 0.5064 | 0.1603 | 0.4541 | 0.1252 | 0.5852 |
| news_narrative | 0.5000 | 0.1195 | 0.5000 | 0.2120 | 1.0000 |

This result is useful. It suggests that **macro financial variables, cross-asset behavior, crypto market stress, and liquidity** carry more standalone predictive value than the slower-moving macro engine or the current narrative feature layer.

That does not mean the macro engine is useless. It means the macro engine is more valuable as an interpretation layer than as a direct short-horizon classifier input. For a 7-day BTC stress label, fast-moving market and liquidity variables naturally carry more direct signal.

### Top Feature Importances

The most important full-model features across walk-forward folds included:

| Rank | Feature | Mean Importance |
|---:|---|---:|
| 1 | epu_z_90d | 0.0397 |
| 2 | bitcoin_vol_30d | 0.0257 |
| 3 | industrial_production_6m_chg | 0.0240 |
| 4 | treasury_10y | 0.0220 |
| 5 | bitcoin_vol_7d | 0.0214 |
| 6 | industrial_production_yoy | 0.0209 |
| 7 | real_gdp_yoy | 0.0208 |
| 8 | btc_eth_corr_30d | 0.0206 |
| 9 | monetary_tightening_score | 0.0191 |
| 10 | ethereum_drawdown_90d | 0.0189 |

The feature importance profile is also reasonable: economic policy uncertainty, BTC volatility, industrial production, Treasury yields, and cross-crypto correlation all appear as meaningful stress indicators.

---

## Event Study Results

The event study evaluates how BTC, ETH, and the risk score behaved after detected stress events.

Total complete events studied: **372**

| Event Type | Count |
|---|---:|
| btc_drawdown_warning | 70 |
| cross_asset_risk | 62 |
| market_anomaly | 59 |
| risk_off_score | 48 |
| crypto_stress | 45 |
| monetary_tightening | 32 |
| inflation_pressure | 30 |
| liquidity_stress | 21 |
| narrative_shock | 4 |
| macro_engine_stress | 1 |

### Event Type Performance

| Event Type | Count | Avg BTC 7D | Adverse Rate | Signal Quality |
|---|---:|---:|---:|---|
| narrative_shock | 4 | -1.42% | 50.00% | Mixed |
| liquidity_stress | 21 | -1.01% | 52.38% | Mixed |
| risk_off_score | 48 | 0.23% | 58.33% | Mixed |
| inflation_pressure | 30 | 0.75% | 46.67% | Mixed / neutral |
| cross_asset_risk | 62 | 1.06% | 46.77% | Mixed |
| btc_drawdown_warning | 70 | 1.14% | 54.29% | Mixed |
| market_anomaly | 59 | 3.37% | 42.37% | Relief / mean-reversion |

The event study showed that single event categories are not enough by themselves. Many event types produced mixed outcomes. This supports the overall design of the project: risk should be evaluated through multiple confirming layers instead of one trigger.

The top adverse events were still economically intuitive. For example, cross-asset risk on **2018-01-29** was followed by a **-37.92%** BTC 7-day return, and the June 2022 stress period also appeared strongly in the adverse-event list.

---

## Lag and Shock Analysis

The lag and shock module studied whether current risk variables were associated with future BTC returns.

The strongest negative relationship with BTC 7-day forward return was the model crisis probability:

| Predictor | Lag Days | Pearson | Spearman | Observations |
|---|---:|---:|---:|---:|
| model_crisis_probability | 0 | -0.26 | -0.32 | 3430 |
| model_crisis_probability | 1 | -0.24 | -0.28 | 3429 |
| model_crisis_probability | 3 | -0.18 | -0.21 | 3427 |
| final_risk_score_with_model | 0 | -0.16 | -0.17 | 3430 |
| final_risk_score_with_model | 1 | -0.15 | -0.16 | 3429 |

The shock analysis was one of the strongest validations in the project.

| Shock Type | Events | Avg BTC 7D | BTC Adverse Rate | Avg Risk Δ |
|---|---:|---:|---:|---:|
| Model crisis probability shock | 50 | -10.64% | 98.00% | 1.15 |
| Final model-adjusted risk shock | 50 | -4.57% | 70.00% | -4.66 |
| Rule-based risk shock | 51 | -2.50% | 64.71% | -6.37 |
| High narrative breadth shock | 8 | -1.98% | 50.00% | -8.55 |
| Oil volatility shock | 29 | -1.37% | 65.52% | -3.03 |

The model crisis probability shock is especially important: across **50 events**, BTC averaged **-10.64%** over the next 7 days, with a **98.00% adverse rate**. That is the clearest evidence that the model risk layer captured meaningful stress information in the historical sample.

---

## VAR and Robustness Results

The VAR modules tested whether changes in the risk scores added information to BTC and ETH return equations across different lag specifications.

### BTC Return Equation

| Causing Variable | Robustness | Significant Count | Borderline Count |
|---|---|---:|---:|
| d_final_risk | Strong | 3 | 3 |
| eth_ret_1d | Strong | 3 | 3 |
| d_crypto_stress | Strong | 3 | 3 |
| d_cross_asset | Strong | 3 | 3 |
| d_macro_engine | Strong | 3 | 3 |
| d_blended_narrative_risk | Moderate / borderline | 1 | 2 |
| d_liquidity_stress | Weak | 0 | 0 |

### ETH Return Equation

| Causing Variable | Robustness | Significant Count | Borderline Count |
|---|---|---:|---:|
| d_final_risk | Strong | 3 | 3 |
| d_crypto_stress | Strong | 3 | 3 |
| d_macro_engine | Strong | 3 | 3 |
| d_liquidity_stress | Moderate | 2 | 3 |
| d_cross_asset | Moderate | 2 | 2 |
| btc_ret_1d | Moderate / borderline | 1 | 2 |
| d_blended_narrative_risk | Weak | 0 | 0 |

The VAR results do not prove economic causality, but they do show that the risk-score changes were not random noise inside the time-series framework. Final risk, crypto stress, cross-asset stress, and macro engine changes were robustly informative across the BTC return equation.

---

## Does the Project Make Sense?

Yes, the project makes sense as a professional research.

The strongest evidence is that the system produced consistent findings across multiple validation layers:

- The latest regime classification was reasonable: BTC was weak, but broad stress confirmation was not strong enough to escalate the regime.
- Ablation testing improved the model by removing noisy/redundant feature groups from the classifier.
- The refined classifier improved ROC-AUC from **0.5275** to **0.6248** in walk-forward testing.
- The final classifier caught **70 of 73** positive mini-crisis cases in the test set.
- Model crisis probability shocks were followed by an average BTC 7-day return of **-10.64%** with a **98.00% adverse rate**.
- VAR robustness checks found that final risk, crypto stress, cross-asset stress, and macro engine changes were consistently informative in the BTC equation.

The project is not perfect, and that is also part of the value. The results show a realistic research tradeoff: high recall, low precision, and noisy regimes. That is normal in financial-market data. The important part is that the project does not hide that weakness. It uses ablations, event studies, shock analysis, and VAR diagnostics to understand what is working and what still needs improvement.

The current version is strongest as a **risk-monitoring and research framework**. The next natural step would be to improve calibration, add richer narrative data, and test whether the regime scores improve simulated risk-management decisions over time.

---

## Repository Structure

```text
crypto-macro-crisis-radar/
├── config.yaml
├── requirements.txt
├── Makefile
├── .env.example
├── data/
│   ├── raw/                  # small public-source snapshot for reproducible demo
│   ├── features/             # generated locally
│   ├── processed/            # generated locally
│   └── events/               # manual event template
├── examples/
│   ├── reports/              # sample daily market-regime report
│   └── research/             # sample ablation, VAR, event-study, dashboard outputs
├── models/                   # generated model artifacts
├── outputs/                  # generated reports and scores
├── scripts/                  # runnable pipeline and research scripts
└── src/                      # reusable project modules
```

Generated model files, processed data, and output artifacts are ignored by Git so the repository stays clean.

---

## Setup

Recommended Python version: **Python 3.11 or 3.12**

```bash
git clone https://github.com/JAAA05/crypto-macro-crisis-radar.git
cd crypto-macro-crisis-radar
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You can also use the Makefile:

```bash
make setup
```

---

## Quick Demo With Included Data

The repository includes a small raw-data snapshot so the project can run without downloading fresh data first.

```bash
python scripts/run_pipeline.py --skip-data
```

Or:

```bash
make demo
```

This generates the core local artifacts:

```text
data/features/master_features.csv
data/processed/scored_regime_history.csv
models/anomaly_model.joblib
models/crisis_classifier.joblib
outputs/scores/baseline_model_report.json
outputs/reports/latest_market_regime_report.md
```

---

## Run With Fresh or Updated Data

Create an environment file:

```bash
cp .env.example .env
```

Add your FRED API key:

```text
FRED_API_KEY=your_key_here
```

Run an incremental data update:

```bash
python scripts/run_pipeline.py --incremental-data
```

Force a full data refresh:

```bash
python scripts/run_pipeline.py --force-data
```

If public crypto data endpoints are rate-limited, add a CoinGecko key to `.env` as well.

---

## Run the Research Suite

After the main pipeline has generated the scored regime history, run:

```bash
python scripts/run_research_suite.py
```

Or:

```bash
make research
```

This runs the deeper validation modules:

```text
scripts/05_event_study.py
scripts/06_lag_shock_analysis.py
scripts/07_var_analysis.py
scripts/07b_var_robustness.py
scripts/09_ablation_study.py
scripts/09b_ablation_study_v2.py
scripts/08_build_research_dashboard.py
```

Main research outputs:

```text
outputs/event_studies/event_study_report.md
outputs/econometrics/lag_shock_report.md
outputs/econometrics/var_lite_report.md
outputs/econometrics/var_robustness/var_robustness_report.md
outputs/ablation/ablation_report.md
outputs/ablation_v2/ablation_v2_report.md
outputs/dashboard/crypto_macro_research_dashboard.md
outputs/dashboard/key_findings.json
```

Sample versions are included in:

```text
examples/research/
```

---

## Main Commands

```bash
# Install dependencies
make setup

# Reproduce the demo from included raw data
python scripts/run_pipeline.py --skip-data

# Update only new raw data rows, then rebuild features/models/report
python scripts/run_pipeline.py --incremental-data

# Force full data refresh
python scripts/run_pipeline.py --force-data

# Run extra research modules after the main pipeline
python scripts/run_research_suite.py

# Remove generated artifacts
make clean
```

---

## Key Outputs

| Output | Description |
|---|---|
| `latest_market_regime_report.md` | Daily analyst-style market regime report |
| `baseline_model_report.json` | Classifier metrics, threshold, feature set, confusion matrix |
| `scored_regime_history.csv` | Historical daily risk scores and regime labels |
| `ablation_v2_report.md` | Walk-forward ablation study with feature-group tests |
| `event_study_report.md` | Post-event BTC/ETH behavior after detected stress events |
| `lag_shock_report.md` | Lag correlations and shock-response summaries |
| `var_lite_report.md` | VAR-style time-series diagnostics |
| `var_robustness_report.md` | VAR robustness across different lag choices |
| `crypto_macro_research_dashboard.md` | Combined research dashboard and project summary |

---

## Current Strengths

- Combines crypto, macro, liquidity, cross-asset, narrative, anomaly, and machine-learning signals in one pipeline.
- Keeps the model interpretable by separating rule-based scoring from classifier probability.
- Uses walk-forward validation instead of relying only on a random split.
- Includes ablation testing to identify noisy or redundant feature groups.
- Produces analyst-style Markdown reports that are easy to review in GitHub.
- Includes event studies, shock analysis, and VAR robustness to evaluate whether the signals have historical meaning.

---

## Current Improvements to Make

- Improve category-level narrative retrieval so Fed, inflation, regulation, exchange, stablecoin, ETF, and geopolitical narratives use real article counts more consistently.
- Add a visual dashboard for scores, events, ablations, and VAR impulse responses.
- Add calibration plots and threshold analysis for the classifier.
- Expand walk-forward testing across more market regimes.
- Add portfolio-style backtesting to evaluate whether regime scores improve risk management decisions.
- Add model cards for the classifier and anomaly model.

---

## Conclusion

Crypto Macro Crisis Radar is a complete end-to-end market-regime research system. It collects multiple layers of market and macro data, builds stress features, trains an interpretable risk classifier, scores the latest regime, and validates the signal through ablations, event studies, shock analysis, and VAR robustness checks.

The results are not perfect, but they are realistic and useful. The project shows that crypto stress detection works better when BTC price action is analyzed together with liquidity, macro pressure, cross-asset confirmation, narrative risk, and anomaly behavior. The most important finding is that the model is strong as a monitoring layer: it captured most mini-crisis cases in the test set and produced meaningful adverse shock results, while also exposing where precision and narrative data need improvement.

