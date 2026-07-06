# Lag & Shock Analysis Report

This report studies lag relationships and shock responses using the scored market-regime history.

It is designed for research and interpretation, not trading recommendations.

## Files Generated

- Lag correlation results: `outputs/econometrics/lag_correlation_results.csv`
- Shock analysis results: `outputs/econometrics/shock_analysis_results.csv`

## BTC 7-Day Forward Return — Strongest Negative Lag Relationships

Higher predictor values were associated with weaker BTC 7-day forward returns.

| Predictor | Lag Days | Pearson | Spearman | N | Relationship |
|---|---:|---:|---:|---:|---|
| model_crisis_probability | 0 | -0.26 | -0.32 | 3430 | Strong negative relationship |
| model_crisis_probability | 1 | -0.24 | -0.28 | 3429 | Strong negative relationship |
| model_crisis_probability | 3 | -0.18 | -0.21 | 3427 | Moderate negative relationship |
| final_risk_score_with_model | 0 | -0.16 | -0.17 | 3430 | Moderate negative relationship |
| industrial_production_yoy | 14 | -0.16 | -0.18 | 3051 | Moderate negative relationship |
| industrial_production_yoy | 7 | -0.16 | -0.18 | 3058 | Moderate negative relationship |
| final_risk_score_with_model | 1 | -0.15 | -0.16 | 3429 | Moderate negative relationship |
| industrial_production_yoy | 3 | -0.15 | -0.16 | 3062 | Moderate negative relationship |
| btc_eth_corr_30d | 30 | -0.15 | -0.06 | 3385 | Moderate negative relationship |
| industrial_production_yoy | 1 | -0.14 | -0.15 | 3064 | Moderate negative relationship |
| industrial_production_yoy | 0 | -0.14 | -0.15 | 3065 | Moderate negative relationship |
| industrial_production_yoy | 30 | -0.14 | -0.17 | 3035 | Moderate negative relationship |
| stablecoin_mcap_7d_chg | 14 | -0.14 | -0.00 | 3077 | Moderate negative relationship |
| private_investment_yoy | 7 | -0.13 | -0.12 | 3058 | Moderate negative relationship |
| real_gdp_yoy | 7 | -0.13 | -0.12 | 3058 | Moderate negative relationship |

## BTC 7-Day Forward Return — Strongest Positive Lag Relationships

Higher predictor values were associated with stronger BTC 7-day forward returns.

| Predictor | Lag Days | Pearson | Spearman | N | Relationship |
|---|---:|---:|---:|---:|---|
| growth_slowdown_score | 7 | 0.17 | 0.22 | 3423 | Moderate positive relationship |
| growth_slowdown_score | 14 | 0.16 | 0.21 | 3416 | Moderate positive relationship |
| growth_slowdown_score | 3 | 0.16 | 0.21 | 3427 | Moderate positive relationship |
| growth_slowdown_score | 1 | 0.15 | 0.20 | 3429 | Moderate positive relationship |
| growth_slowdown_score | 0 | 0.15 | 0.20 | 3430 | Moderate positive relationship |
| growth_slowdown_score | 30 | 0.14 | 0.18 | 3400 | Moderate positive relationship |
| stablecoin_mcap_7d_chg | 30 | 0.09 | 0.04 | 3061 | Weak / neutral relationship |
| bitcoin_ret_7d | 7 | 0.07 | 0.07 | 3416 | Weak / neutral relationship |
| ethereum_vol_30d | 14 | 0.07 | 0.03 | 3401 | Weak / neutral relationship |
| bitcoin_ret_7d | 3 | 0.07 | 0.04 | 3420 | Weak / neutral relationship |
| commodity_pressure_score | 14 | 0.07 | 0.07 | 3416 | Weak / neutral relationship |
| bitcoin_vol_7d | 14 | 0.07 | 0.00 | 3411 | Weak / neutral relationship |
| ethereum_vol_30d | 7 | 0.07 | 0.03 | 3408 | Weak / neutral relationship |
| bitcoin_drawdown_90d | 7 | 0.07 | 0.06 | 3394 | Weak / neutral relationship |
| bitcoin_ret_7d | 1 | 0.07 | 0.03 | 3422 | Weak / neutral relationship |

## 7-Day Shock Response Summary

| Shock | Direction | Events | Avg BTC 7D | Median BTC 7D | BTC Adverse Rate | Avg ETH 7D | Avg Risk Δ | Stress Continuation Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Model crisis probability shock | high | 50 | -10.64% | -8.99% | 98.00% | -6.71% | 1.15 | 62.00% |
| Final model-adjusted risk shock | high | 50 | -4.57% | -1.60% | 70.00% | -5.21% | -4.66 | 24.00% |
| Rule-based risk shock | high | 51 | -2.50% | -1.33% | 64.71% | -4.88% | -6.37 | 9.80% |
| High narrative breadth shock | high | 8 | -1.98% | -0.25% | 50.00% | -4.23% | -8.55 | 12.50% |
| Oil volatility shock | high | 29 | -1.37% | -1.74% | 65.52% | -0.28% | -3.03 | 24.14% |
| News / narrative risk shock | high | 8 | -1.23% | -0.25% | 50.00% | -3.66% | -7.93 | 25.00% |
| Narrative AI risk shock | high | 8 | -1.23% | -0.25% | 50.00% | -3.66% | -7.93 | 25.00% |
| Crypto narrative risk shock | high | 8 | -1.23% | -0.25% | 50.00% | -3.66% | -7.93 | 25.00% |
| Macro narrative risk shock | high | 8 | -1.23% | -0.25% | 50.00% | -3.66% | -7.93 | 25.00% |
| Blended news/narrative risk shock | high | 8 | -1.23% | -0.25% | 50.00% | -3.66% | -7.93 | 25.00% |
| Elevated narrative breadth shock | high | 8 | -1.23% | -0.25% | 50.00% | -3.66% | -7.93 | 25.00% |
| Anomaly shock | high | 48 | -0.93% | -0.58% | 58.33% | -2.90% | -4.49 | 18.75% |
| Commodity pressure shock | high | 47 | -0.83% | -1.62% | 55.32% | -1.44% | -0.87 | 44.68% |
| Liquidity stress shock | high | 31 | -0.80% | -0.56% | 58.06% | -2.08% | -0.96 | 32.26% |
| Inflation pressure shock | high | 35 | -0.06% | -1.16% | 60.00% | 0.91% | 0.31 | 60.00% |
| Monetary tightening shock | high | 28 | 0.56% | -0.38% | 57.14% | 0.54% | -1.12 | 42.86% |
| Dollar strength shock | high | 46 | 0.58% | -0.82% | 52.17% | -0.68% | -1.83 | 41.30% |
| S&P 500 selloff shock | low | 75 | 0.71% | 0.91% | 48.00% | 0.58% | -5.27 | 17.33% |
| Crypto stress shock | high | 60 | 0.79% | 1.74% | 41.67% | 0.09% | -4.78 | 20.00% |
| Oil downside shock | low | 70 | 0.80% | 0.12% | 50.00% | 2.39% | -2.23 | 31.43% |
| Nasdaq selloff shock | low | 68 | 1.03% | 0.80% | 47.06% | 0.43% | -4.40 | 22.06% |
| Equity risk shock | high | 65 | 1.16% | 1.28% | 44.62% | -0.37% | -5.18 | 16.92% |
| Macro Engine stress shock | high | 28 | 1.29% | 0.54% | 46.43% | 2.26% | -1.66 | 32.14% |
| Cross-asset risk shock | high | 65 | 1.30% | 0.55% | 47.69% | 0.59% | -5.15 | 18.46% |
| Safe-haven demand shock | high | 65 | 1.85% | 1.28% | 44.62% | 2.18% | -4.67 | 20.00% |
| Labor stress shock | high | 28 | 1.85% | 0.52% | 50.00% | 1.99% | 0.59 | 50.00% |
| Traditional macro risk shock | high | 42 | 1.94% | -0.40% | 52.38% | -1.08% | -3.16 | 30.95% |
| BTC 30D drawdown shock | low | 44 | 2.59% | 2.65% | 36.36% | 4.49% | -3.58 | 29.55% |
| VIX volatility shock | high | 55 | 2.62% | 1.74% | 40.00% | 3.53% | -3.99 | 21.82% |
| BTC volatility shock | high | 32 | 3.74% | 1.07% | 43.75% | 3.33% | -0.93 | 37.50% |
| Growth slowdown shock | high | 49 | 5.55% | 5.60% | 34.69% | 10.03% | 0.64 | 57.14% |

## Most Adverse 7-Day Shock Types

| Shock | Events | Avg BTC 7D | Avg ETH 7D | Avg Risk Δ |
|---|---:|---:|---:|---:|
| Model crisis probability shock | 50 | -10.64% | -6.71% | 1.15 |
| Final model-adjusted risk shock | 50 | -4.57% | -5.21% | -4.66 |
| Rule-based risk shock | 51 | -2.50% | -4.88% | -6.37 |
| High narrative breadth shock | 8 | -1.98% | -4.23% | -8.55 |
| Oil volatility shock | 29 | -1.37% | -0.28% | -3.03 |
| News / narrative risk shock | 8 | -1.23% | -3.66% | -7.93 |
| Narrative AI risk shock | 8 | -1.23% | -3.66% | -7.93 |
| Crypto narrative risk shock | 8 | -1.23% | -3.66% | -7.93 |
| Macro narrative risk shock | 8 | -1.23% | -3.66% | -7.93 |
| Blended news/narrative risk shock | 8 | -1.23% | -3.66% | -7.93 |

## Strongest Risk-Build Shock Types

| Shock | Events | Avg BTC 7D | Avg ETH 7D | Avg Risk Δ |
|---|---:|---:|---:|---:|
| Model crisis probability shock | 50 | -10.64% | -6.71% | 1.15 |
| Growth slowdown shock | 49 | 5.55% | 10.03% | 0.64 |
| Labor stress shock | 28 | 1.85% | 1.99% | 0.59 |
| Inflation pressure shock | 35 | -0.06% | 0.91% | 0.31 |
| Commodity pressure shock | 47 | -0.83% | -1.44% | -0.87 |
| BTC volatility shock | 32 | 3.74% | 3.33% | -0.93 |
| Liquidity stress shock | 31 | -0.80% | -2.08% | -0.96 |
| Monetary tightening shock | 28 | 0.56% | 0.54% | -1.12 |
| Macro Engine stress shock | 28 | 1.29% | 2.26% | -1.66 |
| Dollar strength shock | 46 | 0.58% | -0.68% | -1.83 |

## Interpretation Notes

- Correlations describe association, not causality.
- Lag days mean the predictor is shifted backward before comparing it with future BTC/ETH returns.
- A negative correlation means higher predictor values were associated with weaker future returns.
- A positive correlation means higher predictor values were associated with stronger future returns.
- Shock analysis uses extreme quantiles to study what happened after unusually high or low values.
- Results should be interpreted as research diagnostics, not trading signals.