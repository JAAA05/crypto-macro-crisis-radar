# Ablation Study v2 Report

This report improves the first ablation by using walk-forward validation and threshold tuning.

Instead of relying on one final 20% split and a fixed 0.50 threshold, each fold uses train → validation → test windows. The classification threshold is selected on the validation window and evaluated on the next test window.

## Walk-Forward Fold Design

| Fold | Train Rows | Validation Rows | Test Rows | Train End | Validation End | Test End |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1718 | 343 | 343 | 1718 | 2061 | 2404 |
| 2 | 1976 | 343 | 343 | 1976 | 2319 | 2662 |
| 3 | 2234 | 343 | 343 | 2234 | 2577 | 2920 |
| 4 | 2492 | 343 | 343 | 2492 | 2835 | 3178 |
| 5 | 2751 | 343 | 343 | 2751 | 3094 | 3437 |

## Full Model — Walk-Forward Average

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

## Refined Model Experiments

| Experiment | ROC-AUC | Avg Precision | F1 | Recall | AUC Δ vs Full | F1 Δ vs Full |
|---|---:|---:|---:|---:|---:|---:|
| refined_no_macro_engine | 0.6248 | 0.1848 | 0.2035 | 0.8750 | 0.0973 | -0.0144 |
| refined_core_market_macro_liquidity | 0.6050 | 0.1776 | 0.1912 | 0.8264 | 0.0775 | -0.0266 |
| refined_market_cross_liquidity_narrative | 0.5939 | 0.1727 | 0.2215 | 0.8658 | 0.0664 | 0.0037 |

## Drop-One-Group Ablation — Walk-Forward Average

A larger AUC loss means the removed group was more useful to the classifier.

| Removed Group | ROC-AUC | AUC Loss vs Full | Avg Precision | AP Loss vs Full | F1 | F1 Loss vs Full |
|---|---:|---:|---:|---:|---:|---:|
| liquidity | 0.5118 | 0.0157 | 0.1632 | -0.0084 | 0.2136 | 0.0043 |
| macro_financial | 0.5203 | 0.0072 | 0.1721 | -0.0174 | 0.2141 | 0.0037 |
| cross_asset | 0.5223 | 0.0053 | 0.1661 | -0.0113 | 0.1721 | 0.0457 |
| news_narrative | 0.5268 | 0.0007 | 0.1477 | 0.0071 | 0.1961 | 0.0217 |
| crypto_market | 0.5492 | -0.0217 | 0.1853 | -0.0305 | 0.1990 | 0.0188 |
| macro_engine | 0.6248 | -0.0973 | 0.1848 | -0.0300 | 0.2035 | 0.0144 |

## Only-One-Group Models — Walk-Forward Average

This shows standalone predictive value by layer.

| Only Group | ROC-AUC | Avg Precision | Balanced Accuracy | F1 | Recall |
|---|---:|---:|---:|---:|---:|
| macro_financial | 0.5898 | 0.1826 | 0.5316 | 0.2067 | 0.5480 |
| cross_asset | 0.5739 | 0.1815 | 0.5040 | 0.2109 | 0.8992 |
| crypto_market | 0.5623 | 0.1429 | 0.5188 | 0.1863 | 0.7193 |
| liquidity | 0.5599 | 0.1709 | 0.5156 | 0.2179 | 0.8397 |
| macro_engine | 0.5064 | 0.1603 | 0.4541 | 0.1252 | 0.5852 |
| news_narrative | 0.5000 | 0.1195 | 0.5000 | 0.2120 | 1.0000 |

## Top Full-Model Feature Importances Across Walk-Forward Folds

| Rank | Feature | Mean Importance | Std | Folds |
|---:|---|---:|---:|---:|
| 1 | epu_z_90d | 0.0397 | 0.0035 | 5 |
| 2 | bitcoin_vol_30d | 0.0257 | 0.0028 | 5 |
| 3 | industrial_production_6m_chg | 0.0240 | 0.0036 | 5 |
| 4 | treasury_10y | 0.0220 | 0.0036 | 5 |
| 5 | bitcoin_vol_7d | 0.0214 | 0.0025 | 5 |
| 6 | industrial_production_yoy | 0.0209 | 0.0016 | 5 |
| 7 | real_gdp_yoy | 0.0208 | 0.0032 | 5 |
| 8 | btc_eth_corr_30d | 0.0206 | 0.0045 | 5 |
| 9 | monetary_tightening_score | 0.0191 | 0.0025 | 5 |
| 10 | ethereum_drawdown_90d | 0.0189 | 0.0042 | 5 |
| 11 | unemployment_rate | 0.0184 | 0.0054 | 5 |
| 12 | oil_vol_z_90d | 0.0184 | 0.0039 | 5 |
| 13 | private_investment_yoy | 0.0181 | 0.0075 | 5 |
| 14 | ethereum_vol_30d | 0.0175 | 0.0076 | 5 |
| 15 | unemployment_z_3y | 0.0172 | 0.0022 | 5 |
| 16 | fed_balance_sheet_90d_chg | 0.0166 | 0.0062 | 5 |
| 17 | cpi_yoy | 0.0165 | 0.0028 | 5 |
| 18 | fed_funds_rate | 0.0164 | 0.0056 | 5 |
| 19 | stablecoin_mcap_z_90d | 0.0164 | 0.0019 | 5 |
| 20 | fiscal_stress_score | 0.0161 | 0.0016 | 5 |
| 21 | gold | 0.0161 | 0.0029 | 5 |
| 22 | oil_wti | 0.0161 | 0.0022 | 5 |
| 23 | macro_engine_stress_index | 0.0158 | 0.0013 | 5 |
| 24 | treasury_2y | 0.0150 | 0.0052 | 5 |
| 25 | treasury_2y_change_6m | 0.0140 | 0.0013 | 5 |

## Interpretation Guide

- Walk-forward validation is more realistic than a single train/test split for market data.
- Threshold tuning prevents the model from being judged only at the default 0.50 probability cutoff.
- ROC-AUC and Average Precision evaluate ranking quality; F1/recall/precision evaluate classification behavior after thresholding.
- If refined models beat the full model, the full model may have noisy or redundant feature groups.
- This is research validation, not a trading signal.