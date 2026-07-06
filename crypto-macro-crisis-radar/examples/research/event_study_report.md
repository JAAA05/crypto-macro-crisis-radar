# Event Study Report

This report evaluates how BTC, ETH, and risk scores behaved around detected or manually provided events.

## Files Generated

- Detailed results: `outputs/event_studies/event_study_results.csv`
- Event type summary: `outputs/event_studies/event_study_summary_by_type.csv`
- Event insights: `outputs/event_studies/event_study_insights_by_type.csv`

## Event Count

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

## Event Type Performance Summary

| Event Type | Count | Avg BTC 7D | Median BTC 7D | Avg ETH 7D | Avg Risk Δ 7D | Adverse Rate | Stress Continuation Rate | Signal Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| narrative_shock | 4 | -1.42% | 0.96% | -3.42% | -8.73 | 50.00% | 0.00% | Mixed |
| liquidity_stress | 21 | -1.01% | -0.09% | -2.24% | -2.09 | 52.38% | 33.33% | Mixed |
| risk_off_score | 48 | 0.23% | -2.33% | 0.22% | -4.03 | 58.33% | 27.08% | Mixed |
| inflation_pressure | 30 | 0.75% | 0.48% | 4.17% | -0.88 | 46.67% | 40.00% | Mixed / neutral |
| cross_asset_risk | 62 | 1.06% | 0.84% | 2.79% | -3.27 | 46.77% | 30.65% | Mixed |
| monetary_tightening | 32 | 1.09% | 1.68% | 0.92% | -0.73 | 34.38% | 34.38% | Mixed |
| btc_drawdown_warning | 70 | 1.14% | -0.22% | 3.82% | -1.15 | 54.29% | 35.71% | Mixed |
| crypto_stress | 45 | 1.69% | 0.43% | 1.24% | -3.54 | 46.67% | 22.22% | Mixed |
| market_anomaly | 59 | 3.37% | 1.47% | 5.14% | -3.58 | 42.37% | 28.81% | Relief / mean-reversion |
| macro_engine_stress | 1 | 10.56% | 10.56% | 27.34% | -3.06 | 0.00% | 0.00% | Relief / mean-reversion |

## Latest Events

| Event Date | Type | Name | Regime | BTC t0→t+7 | Risk Score t0→t+7 |
|---|---|---|---|---:|---:|
| 2026-02-24 | btc_drawdown_warning | BTC 30D drawdown below -10% | Neutral | 6.66% | 10.02 |
| 2026-03-02 | market_anomaly | High anomaly score | Risk-Off | -0.58% | -11.78 |
| 2026-03-02 | narrative_shock | News / narrative risk shock | Risk-Off | -0.58% | -11.78 |
| 2026-03-21 | cross_asset_risk | Cross-asset risk confirmation | Neutral | -3.76% | 3.99 |
| 2026-03-24 | risk_off_score | Model-adjusted risk score above 50 | Risk-Off | -3.28% | -5.98 |
| 2026-03-27 | btc_drawdown_warning | BTC 30D drawdown below -10% | Risk-Off | 0.91% | -10.04 |
| 2026-04-01 | market_anomaly | High anomaly score | Neutral | 4.37% | -11.04 |
| 2026-04-01 | narrative_shock | News / narrative risk shock | Neutral | 4.37% | -11.04 |
| 2026-05-01 | market_anomaly | High anomaly score | Neutral | 2.50% | -3.21 |
| 2026-05-01 | narrative_shock | News / narrative risk shock | Neutral | 2.50% | -3.21 |

## Top Adverse Events

Events followed by the weakest BTC 7-day returns.

| Event Date | Type | Name | Regime | BTC t0→t+7 | ETH t0→t+7 | Risk Score t0→t+7 |
|---|---|---|---|---:|---:|---:|
| 2018-01-29 | cross_asset_risk | Cross-asset risk confirmation | Risk-Off | -37.92% | -40.33% | 4.49 |
| 2022-06-11 | btc_drawdown_warning | BTC 30D drawdown below -10% | Risk-Off | -33.25% | -35.10% | 1.29 |
| 2022-06-10 | cross_asset_risk | Cross-asset risk confirmation | Risk-Off | -29.64% | -34.62% | 0.29 |
| 2021-05-12 | crypto_stress | Crypto market stress shock | Risk-Off | -25.79% | -36.13% | -2.53 |
| 2022-06-08 | liquidity_stress | Liquidity stress shock | Risk-Off | -25.23% | -30.95% | 9.74 |
| 2018-03-23 | market_anomaly | High anomaly score | Risk-Off | -23.29% | -27.36% | -9.09 |
| 2022-06-12 | crypto_stress | Crypto market stress shock | Risk-Off | -22.60% | -21.37% | 0.10 |
| 2021-05-10 | cross_asset_risk | Cross-asset risk confirmation | Risk-Off | -21.99% | -16.87% | 3.01 |
| 2022-11-02 | risk_off_score | Model-adjusted risk score above 50 | Risk-Off | -21.14% | -27.54% | 2.93 |
| 2018-01-09 | btc_drawdown_warning | BTC 30D drawdown below -10% | Risk-Off | -20.10% | -16.28% | 2.75 |

## Top Relief / Recovery Events

Events followed by the strongest BTC 7-day returns.

| Event Date | Type | Name | Regime | BTC t0→t+7 | ETH t0→t+7 | Risk Score t0→t+7 |
|---|---|---|---|---:|---:|---:|
| 2017-07-15 | market_anomaly | High anomaly score | Risk-Off | 43.83% | 37.54% | 0.92 |
| 2017-07-15 | risk_off_score | Model-adjusted risk score above 50 | Risk-Off | 43.83% | 37.54% | 0.92 |
| 2017-07-15 | crypto_stress | Crypto market stress shock | Risk-Off | 43.83% | 37.54% | 0.92 |
| 2019-05-07 | cross_asset_risk | Cross-asset risk confirmation | Neutral | 39.01% | 30.59% | 5.72 |
| 2017-11-12 | risk_off_score | Model-adjusted risk score above 50 | Risk-Off | 36.45% | 15.46% | -7.55 |
| 2017-05-17 | cross_asset_risk | Cross-asset risk confirmation | Neutral | 34.18% | 125.67% | -1.13 |
| 2018-02-10 | risk_off_score | Model-adjusted risk score above 50 | Risk-Off | 30.11% | 14.78% | -7.03 |
| 2018-02-10 | market_anomaly | High anomaly score | Risk-Off | 30.11% | 14.78% | -7.03 |
| 2017-12-10 | btc_drawdown_warning | BTC 30D drawdown below -10% | Neutral | 26.74% | 63.61% | 0.88 |
| 2021-09-28 | inflation_pressure | Inflation pressure shock | Risk-Off | 25.52% | 25.29% | -3.31 |

## Top Risk-Build Events

Events followed by the largest increase in model-adjusted risk score.

| Event Date | Type | Name | Regime | BTC t0→t+7 | ETH t0→t+7 | Risk Score t0→t+7 |
|---|---|---|---|---:|---:|---:|
| 2025-11-11 | btc_drawdown_warning | BTC 30D drawdown below -10% | Neutral | -9.80% | -8.63% | 15.47 |
| 2025-03-28 | crypto_stress | Crypto market stress shock | Neutral | -0.62% | -4.21% | 14.87 |
| 2025-03-28 | cross_asset_risk | Cross-asset risk confirmation | Neutral | -0.62% | -4.21% | 14.87 |
| 2025-03-28 | btc_drawdown_warning | BTC 30D drawdown below -10% | Neutral | -0.62% | -4.21% | 14.87 |
| 2018-10-04 | cross_asset_risk | Cross-asset risk confirmation | Neutral | -6.00% | -14.79% | 14.49 |
| 2026-01-25 | btc_drawdown_warning | BTC 30D drawdown below -10% | Risk-Off | -11.17% | -19.40% | 12.98 |
| 2022-02-25 | inflation_pressure | Inflation pressure shock | Neutral | -0.18% | -5.29% | 12.65 |
| 2019-07-27 | btc_drawdown_warning | BTC 30D drawdown below -10% | Neutral | 14.14% | 7.26% | 12.04 |
| 2022-10-29 | monetary_tightening | Monetary tightening shock | Neutral | 2.29% | 0.43% | 10.51 |
| 2026-02-24 | btc_drawdown_warning | BTC 30D drawdown below -10% | Neutral | 6.66% | 7.07% | 10.02 |

## Interpretation Notes

- **Adverse Rate** is the share of events followed by a negative BTC 7-day return.
- **Stress Continuation Rate** is the share of events followed by a higher model-adjusted risk score after 7 days.
- Positive BTC/ETH post-event returns suggest recovery or relief after an event.
- Negative BTC/ETH post-event returns suggest the event was followed by market weakness.
- Positive risk-score changes after an event indicate that stress continued rising.
- Negative risk-score changes after an event indicate that stress faded after the event.

This is a research tool, not a trading recommendation.