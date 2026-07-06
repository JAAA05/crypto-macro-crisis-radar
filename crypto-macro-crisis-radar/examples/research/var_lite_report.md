# VAR Lite Report

This report fits a lightweight Vector Autoregression model using standardized stationary-style variables.

The model is designed for research diagnostics: lag interactions, Granger-style causality tests, impulse responses, and short-horizon forecasts.

## Model Setup

- Selected lag order: **14**
- Number of observations used: **3422**
- Number of equations: **8**

## Variables

| Variable | Meaning |
|---|---|
| btc_ret_1d | BTC daily return |
| eth_ret_1d | ETH daily return |
| d_crypto_stress | Daily change in crypto stress score |
| d_final_risk | Daily change in model-adjusted final risk score |
| d_macro_engine | Daily change in macro engine stress index |
| d_cross_asset | Daily change in cross-asset risk score |
| d_liquidity_stress | Daily change in liquidity stress score |
| d_blended_narrative_risk | Daily change in blended news/narrative risk score |

## Granger-Style Causality Tests — Target: BTC Return

Lower p-values suggest the causing variable adds lagged information for the BTC return equation. This is not proof of true causality.

| Causing Variable | Test Statistic | P-Value | 5% Conclusion |
|---|---:|---:|---|
| d_final_risk | 5.2685 | 0.0000 | reject_no_causality |
| eth_ret_1d | 4.7828 | 0.0000 | reject_no_causality |
| d_crypto_stress | 3.2176 | 0.0000 | reject_no_causality |
| d_cross_asset | 2.9111 | 0.0002 | reject_no_causality |
| d_macro_engine | 2.4247 | 0.0021 | reject_no_causality |
| d_blended_narrative_risk | 1.0810 | 0.3691 | fail_to_reject |
| d_liquidity_stress | 0.9862 | 0.4642 | fail_to_reject |

## Granger-Style Causality Tests — Target: ETH Return

| Causing Variable | Test Statistic | P-Value | 5% Conclusion |
|---|---:|---:|---|
| d_final_risk | 3.8475 | 0.0000 | reject_no_causality |
| d_crypto_stress | 2.6607 | 0.0007 | reject_no_causality |
| d_macro_engine | 2.0303 | 0.0125 | reject_no_causality |
| d_cross_asset | 1.9240 | 0.0197 | reject_no_causality |
| d_liquidity_stress | 1.6500 | 0.0587 | fail_to_reject |
| btc_ret_1d | 1.3737 | 0.1564 | fail_to_reject |
| d_blended_narrative_risk | 0.8439 | 0.6212 | fail_to_reject |

## Largest Impulse Responses

Values are in standardized units because the VAR input variables were standardized.

| Step | Response | Impulse | IRF Value |
|---:|---|---|---:|
| 7 | eth_ret_1d | d_final_risk | -0.1375 |
| 1 | btc_ret_1d | eth_ret_1d | -0.1362 |
| 7 | btc_ret_1d | d_final_risk | -0.1172 |
| 3 | btc_ret_1d | d_final_risk | -0.0883 |
| 7 | eth_ret_1d | d_crypto_stress | 0.0802 |
| 7 | btc_ret_1d | d_crypto_stress | 0.0799 |
| 1 | btc_ret_1d | d_final_risk | 0.0718 |
| 3 | btc_ret_1d | d_cross_asset | 0.0584 |
| 7 | eth_ret_1d | d_blended_narrative_risk | 0.0508 |
| 1 | btc_ret_1d | d_blended_narrative_risk | -0.0491 |
| 1 | eth_ret_1d | d_final_risk | 0.0481 |
| 7 | eth_ret_1d | d_cross_asset | 0.0464 |
| 3 | btc_ret_1d | d_crypto_stress | 0.0460 |
| 1 | eth_ret_1d | d_blended_narrative_risk | -0.0421 |
| 3 | eth_ret_1d | d_final_risk | -0.0415 |
| 3 | eth_ret_1d | d_cross_asset | 0.0392 |
| 7 | btc_ret_1d | d_blended_narrative_risk | 0.0384 |
| 1 | btc_ret_1d | d_crypto_stress | -0.0382 |
| 3 | eth_ret_1d | d_macro_engine | -0.0344 |
| 3 | eth_ret_1d | d_crypto_stress | 0.0339 |

## VAR Forecast Path

Forecast values are standardized expected changes/returns from the VAR model.

| Step | BTC Ret | ETH Ret | Δ Final Risk | Δ Crypto Stress | Δ Liquidity Stress | Δ Blended Narrative |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2594 | 0.1538 | -0.7196 | -0.4227 | 0.4320 | -0.7986 |
| 2 | 0.1799 | 0.3766 | 0.7852 | 0.5791 | 0.0882 | 1.4033 |
| 3 | 0.0987 | -0.1088 | 0.1808 | 0.5502 | -0.1852 | -0.9632 |
| 4 | -0.3079 | -0.2901 | 0.0921 | -0.3827 | 0.2116 | 0.2804 |
| 5 | -0.1470 | -0.1296 | -0.2670 | -0.0002 | -0.0875 | -0.3829 |
| 6 | -0.1032 | -0.2409 | -0.1257 | 0.5519 | -0.1955 | -0.4733 |
| 7 | -0.3147 | -0.0767 | 0.7094 | -0.0315 | 0.1725 | 1.0763 |
| 8 | -0.2344 | -0.1979 | -0.6943 | -0.4553 | 0.0696 | -1.0230 |
| 9 | 0.0347 | -0.1301 | 0.3290 | 0.3866 | -0.0948 | 0.7442 |
| 10 | -0.0410 | 0.1025 | 0.3147 | 0.5858 | -0.0491 | 0.0984 |
| 11 | -0.0844 | -0.1130 | -0.5801 | -0.4084 | -0.0123 | -0.8391 |
| 12 | -0.0541 | -0.1215 | 0.2584 | 0.0298 | -0.0412 | 0.7080 |
| 13 | 0.0073 | 0.0174 | -0.1106 | 0.3065 | 0.0799 | -0.4258 |
| 14 | 0.0478 | 0.0683 | -0.2260 | -0.4670 | 0.0456 | 0.0766 |

## Interpretation Notes

- VAR models require careful interpretation because crypto and macro variables can be non-stationary, noisy, and regime-dependent.
- This VAR Lite version uses returns and score changes to reduce non-stationarity.
- Granger-style tests show whether lagged values add information inside this model, not true economic causality.
- Impulse responses are based on standardized variables, so they show relative dynamic effects rather than direct dollar predictions.
- This is a research diagnostic, not a trading recommendation.