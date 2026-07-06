# Crypto Macro Crisis Radar — Research Dashboard



This dashboard summarizes the latest research outputs from the crypto macro crisis radar pipeline.



## Latest Market Regime Snapshot

| Metric | Value |
|---|---|
| Date | 2026-05-30 |
| Market Regime | Neutral |
| Model-Adjusted Final Risk Score | 36.93 |
| Rule-Based Risk Score | 45.19 |
| Model Crisis Probability | 12.15% |
| BTC Price | $73,765.18 |
| BTC 7D Return | -3.76% |
| BTC 30D Drawdown | -10.26% |
| Macro Engine Stress | 29.81 |
| Cross-Asset Risk | 13.43 |
| Crypto Market Stress | 38.36 |
| Liquidity Stress | 46.93 |
| Blended News/Narrative Risk | 60.50 |
| Anomaly Score | 94.27 |

**Interpretation:** The latest regime is **Neutral** with a model-adjusted final risk score of **36.93**. BTC is in a **-10.26%** 30-day drawdown, but the broader regime depends on whether macro, cross-asset, liquidity, narrative, and crypto stress confirm that weakness.


## Model Architecture Summary

| Metric | Value |
|---|---|
| Active Classifier | Ablation-Selected Risk Classifier v1 |
| Classifier Feature Set | refined_no_macro_engine |
| Excluded From Supervised Classifier | macro_engine feature group |
| Tuned Classification Threshold | 0.36 |
| Rule-Based Risk Score | 45.19 |
| Model Crisis Probability | 12.15% |
| Model-Adjusted Final Risk Score | 36.93 |
| Current Market Regime | Neutral |
| Test ROC-AUC | 0.62 |
| Test Average Precision | 0.18 |
| Test Balanced Accuracy | 0.61 |
| Test Precision | 0.18 |
| Test Recall | 0.96 |
| Test F1 | 0.30 |
| Default 0.50 Threshold F1 | 0.00 |

**Architecture Note:** Ablation v2 showed that refined_no_macro_engine improved walk-forward ROC-AUC relative to the full feature set. Macro Engine remains active in rule-based scoring, daily report, VAR, dashboard, and interpretation layers.


## Narrative AI Summary

| Metric | Value |
|---|---|
| Narrative AI Risk | 60.50 |
| Crypto Narrative Risk | 60.50 |
| Macro Narrative Risk | 60.50 |
| Blended News/Narrative Risk | 60.50 |
| General News Risk | 60.50 |
| Fallback Used | Yes |
| Category-Level Articles | 0 |
| Elevated Narrative Categories | 0 |
| High Narrative Categories | 0 |

**Narrative Data Note:** The category-level Narrative AI feed is currently using the general GDELT news-risk fallback. This keeps the system from treating missing category data as falsely neutral, but the category breakdown should be improved in a future data-source upgrade.


## Event Study Highlights

Total complete events studied: **372**

| Event Type | Count | Avg BTC 7D | Adverse Rate | Signal Quality |
|---|---|---|---|---|
| narrative_shock | 4 | -1.42% | 50.00% | Mixed |
| liquidity_stress | 21 | -1.01% | 52.38% | Mixed |
| risk_off_score | 48 | 0.23% | 58.33% | Mixed |
| inflation_pressure | 30 | 0.75% | 46.67% | Mixed / neutral |
| cross_asset_risk | 62 | 1.06% | 46.77% | Mixed |
| monetary_tightening | 32 | 1.09% | 34.38% | Mixed |
| btc_drawdown_warning | 70 | 1.14% | 54.29% | Mixed |
| crypto_stress | 45 | 1.69% | 46.67% | Mixed |

**Interpretation:** The event-study module separates stress signals that historically preceded further weakness from signals that appeared closer to relief/recovery periods.


## Lag & Shock Analysis Highlights

### Strongest Negative Relationships with BTC 7-Day Forward Return

| Predictor | Lag | Pearson | Spearman | Relationship |
|---|---|---|---|---|
| model_crisis_probability | 0 | -0.26 | -0.32 | Strong negative relationship |
| model_crisis_probability | 1 | -0.24 | -0.28 | Strong negative relationship |
| model_crisis_probability | 3 | -0.18 | -0.21 | Moderate negative relationship |
| final_risk_score_with_model | 0 | -0.16 | -0.17 | Moderate negative relationship |
| industrial_production_yoy | 14 | -0.16 | -0.18 | Moderate negative relationship |
| industrial_production_yoy | 7 | -0.16 | -0.18 | Moderate negative relationship |
| final_risk_score_with_model | 1 | -0.15 | -0.16 | Moderate negative relationship |
| industrial_production_yoy | 3 | -0.15 | -0.16 | Moderate negative relationship |

### Most Adverse 7-Day Shock Types

| Shock | Events | Avg BTC 7D | BTC Adverse Rate | Avg Risk Δ |
|---|---|---|---|---|
| Model crisis probability shock | 50 | -10.64% | 98.00% | 1.15 |
| Final model-adjusted risk shock | 50 | -4.57% | 70.00% | -4.66 |
| Rule-based risk shock | 51 | -2.50% | 64.71% | -6.37 |
| High narrative breadth shock | 8 | -1.98% | 50.00% | -8.55 |
| Oil volatility shock | 29 | -1.37% | 65.52% | -3.03 |
| News / narrative risk shock | 8 | -1.23% | 50.00% | -7.93 |
| Narrative AI risk shock | 8 | -1.23% | 50.00% | -7.93 |
| Crypto narrative risk shock | 8 | -1.23% | 50.00% | -7.93 |
| Macro narrative risk shock | 8 | -1.23% | 50.00% | -7.93 |
| Blended news/narrative risk shock | 8 | -1.23% | 50.00% | -7.93 |

**Interpretation:** Lag/shock analysis provides diagnostic evidence about which stress indicators are most associated with future weakness.

## VAR Robustness Highlights

### BTC Return Equation

| Causing Variable | Robustness | Significant Count | Borderline Count |
|---|---|---|---|
| d_final_risk | Strong | 3 | 3 |
| eth_ret_1d | Strong | 3 | 3 |
| d_crypto_stress | Strong | 3 | 3 |
| d_cross_asset | Strong | 3 | 3 |
| d_macro_engine | Strong | 3 | 3 |
| d_blended_narrative_risk | Moderate / borderline | 1 | 2 |
| d_liquidity_stress | Weak | 0 | 0 |

### ETH Return Equation

| Causing Variable | Robustness | Significant Count | Borderline Count |
|---|---|---|---|
| d_final_risk | Strong | 3 | 3 |
| d_crypto_stress | Strong | 3 | 3 |
| d_macro_engine | Strong | 3 | 3 |
| d_liquidity_stress | Moderate | 2 | 3 |
| d_cross_asset | Moderate | 2 | 2 |
| btc_ret_1d | Moderate / borderline | 1 | 2 |
| d_blended_narrative_risk | Weak | 0 | 0 |

### Top Robust Impulse Responses

| Response | Impulse | Step | Avg IRF | Direction Consistent |
|---|---|---|---|---|
| btc_ret_1d | eth_ret_1d | 1 | -0.14 | True |
| eth_ret_1d | d_final_risk | 7 | -0.08 | False |
| btc_ret_1d | d_final_risk | 1 | 0.08 | True |
| btc_ret_1d | d_final_risk | 7 | -0.06 | False |
| btc_ret_1d | d_final_risk | 3 | -0.07 | True |
| btc_ret_1d | d_crypto_stress | 7 | 0.05 | False |
| eth_ret_1d | d_crypto_stress | 7 | 0.05 | False |
| btc_ret_1d | d_blended_narrative_risk | 1 | -0.05 | True |
| eth_ret_1d | d_final_risk | 1 | 0.05 | True |
| btc_ret_1d | d_crypto_stress | 1 | -0.05 | True |

**Interpretation:** The VAR robustness check tests whether dynamic relationships survive across different lag specifications.

## Project Strengths

- Combines **macro variables**, **cross-asset confirmation**, **crypto market stress**, **liquidity**, **Narrative AI**, **anomaly detection**, and **machine learning** into one research pipeline.
- Uses multiple validation layers: daily regime report, event study, lag/shock analysis, VAR Lite, VAR robustness, and ablation testing.
- Keeps the system interpretable by separating rule-based scoring from model-adjusted probabilities.
- Uses an Ablation-Selected Risk Classifier instead of blindly relying on the largest feature set.
- Avoids treating missing category-level narrative data as neutral by using a transparent fallback flag.
- Produces research outputs that can be cited in a README, portfolio, or technical interview.


## Current Limitations

- This is a research system, not a trading bot and not a trading recommendation engine.
- Narrative AI currently uses a general GDELT fallback when category-level article data is unavailable.
- VAR and correlation analyses show statistical associations, not true economic causality.
- Some macro variables are slow-moving and may be revised after initial release.
- Crypto regimes are highly unstable; model performance should be evaluated through continued walk-forward testing before any live use.


## Recommended Next Steps

1. **Fix category-level Narrative AI retrieval** so that Fed, inflation, regulation, exchange, stablecoin, ETF, and geopolitical narratives are separated with real article counts.
2. **Add dashboard visualizations** for risk scores, event-study results, ablation results, and VAR impulse responses.
3. **Add walk-forward backtesting** to test whether regime changes would have improved simulated risk management.
4. **Create a README and GitHub packaging layer** explaining methodology, folder structure, and research findings.
5. **Add model cards** describing assumptions, limitations, and safe use boundaries.
