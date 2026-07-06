# VAR Robustness Report

This report compares VAR Lite results across multiple maximum lag settings.

The goal is to check whether the main conclusions survive changes in lag specification.

## Files Generated

- Robustness outputs folder: `outputs/econometrics/var_robustness`
- Run summary: `outputs/econometrics/var_robustness/var_robustness_run_summary.csv`
- BTC Granger robustness: `outputs/econometrics/var_robustness/var_robustness_btc_granger.csv`
- ETH Granger robustness: `outputs/econometrics/var_robustness/var_robustness_eth_granger.csv`
- IRF robustness: `outputs/econometrics/var_robustness/var_robustness_irf.csv`

## Run Summary

| Maxlags Requested | Selected Lag Order | Observations | Equations |
|---:|---:|---:|---:|
| 14 | 14 | 3422 | 8 |
| 7 | 7 | 3429 | 8 |
| 5 | 5 | 3431 | 8 |

## BTC Return Equation — Granger Robustness

| Causing Variable | Robustness | Significant Count | Borderline Count | P-Values by Maxlags |
|---|---|---:|---:|---|
| d_final_risk | Strong | 3 | 3 | maxlags 14: <0.0001, maxlags 7: <0.0001, maxlags 5: <0.0001 |
| eth_ret_1d | Strong | 3 | 3 | maxlags 14: <0.0001, maxlags 7: <0.0001, maxlags 5: <0.0001 |
| d_crypto_stress | Strong | 3 | 3 | maxlags 14: <0.0001, maxlags 7: <0.0001, maxlags 5: 0.0006 |
| d_cross_asset | Strong | 3 | 3 | maxlags 14: 0.0002, maxlags 7: <0.0001, maxlags 5: 0.0465 |
| d_macro_engine | Strong | 3 | 3 | maxlags 14: 0.0021, maxlags 7: 0.0060, maxlags 5: 0.0077 |
| d_blended_narrative_risk | Moderate / borderline | 1 | 2 | maxlags 14: 0.3691, maxlags 7: 0.0707, maxlags 5: 0.0207 |
| d_liquidity_stress | Weak | 0 | 0 | maxlags 14: 0.4642, maxlags 7: 0.2096, maxlags 5: 0.2410 |

## ETH Return Equation — Granger Robustness

| Causing Variable | Robustness | Significant Count | Borderline Count | P-Values by Maxlags |
|---|---|---:|---:|---|
| d_final_risk | Strong | 3 | 3 | maxlags 14: <0.0001, maxlags 7: <0.0001, maxlags 5: 0.0218 |
| d_crypto_stress | Strong | 3 | 3 | maxlags 14: 0.0007, maxlags 7: <0.0001, maxlags 5: 0.0175 |
| d_macro_engine | Strong | 3 | 3 | maxlags 14: 0.0125, maxlags 7: 0.0066, maxlags 5: 0.0110 |
| d_liquidity_stress | Moderate | 2 | 3 | maxlags 14: 0.0587, maxlags 7: 0.0061, maxlags 5: 0.0280 |
| d_cross_asset | Moderate | 2 | 2 | maxlags 14: 0.0197, maxlags 7: 0.0034, maxlags 5: 0.3637 |
| btc_ret_1d | Moderate / borderline | 1 | 2 | maxlags 14: 0.1564, maxlags 7: 0.0228, maxlags 5: 0.0602 |
| d_blended_narrative_risk | Weak | 0 | 0 | maxlags 14: 0.6212, maxlags 7: 0.2293, maxlags 5: 0.1609 |

## Top Robust Impulse Responses

Impulse responses are averaged across the robustness runs. Values are in standardized units.

| Response | Impulse | Step | Avg IRF | Median IRF | Avg Abs IRF | Direction Consistent? |
|---|---|---:|---:|---:|---:|---|
| btc_ret_1d | eth_ret_1d | 1 | -0.1365 | -0.1362 | 0.1365 | True |
| eth_ret_1d | d_final_risk | 7 | -0.0822 | -0.1117 | 0.0840 | False |
| btc_ret_1d | d_final_risk | 1 | 0.0815 | 0.0827 | 0.0815 | True |
| btc_ret_1d | d_final_risk | 7 | -0.0645 | -0.0885 | 0.0726 | False |
| btc_ret_1d | d_final_risk | 3 | -0.0719 | -0.0655 | 0.0719 | True |
| btc_ret_1d | d_crypto_stress | 7 | 0.0458 | 0.0697 | 0.0540 | False |
| eth_ret_1d | d_crypto_stress | 7 | 0.0458 | 0.0681 | 0.0530 | False |
| btc_ret_1d | d_blended_narrative_risk | 1 | -0.0521 | -0.0522 | 0.0521 | True |
| eth_ret_1d | d_final_risk | 1 | 0.0506 | 0.0481 | 0.0506 | True |
| btc_ret_1d | d_crypto_stress | 1 | -0.0502 | -0.0511 | 0.0502 | True |
| btc_ret_1d | d_cross_asset | 3 | 0.0416 | 0.0351 | 0.0416 | True |
| eth_ret_1d | d_blended_narrative_risk | 1 | -0.0412 | -0.0414 | 0.0412 | True |
| eth_ret_1d | d_macro_engine | 3 | -0.0359 | -0.0356 | 0.0359 | True |
| eth_ret_1d | d_blended_narrative_risk | 7 | 0.0317 | 0.0468 | 0.0334 | False |
| btc_ret_1d | d_cross_asset | 1 | -0.0322 | -0.0330 | 0.0322 | True |
| btc_ret_1d | d_crypto_stress | 3 | 0.0320 | 0.0271 | 0.0320 | True |
| btc_ret_1d | d_macro_engine | 3 | -0.0278 | -0.0283 | 0.0278 | True |
| btc_ret_1d | d_blended_narrative_risk | 7 | 0.0224 | 0.0364 | 0.0275 | False |
| eth_ret_1d | d_cross_asset | 7 | 0.0267 | 0.0340 | 0.0269 | False |
| eth_ret_1d | d_final_risk | 3 | -0.0266 | -0.0214 | 0.0266 | True |
| btc_ret_1d | eth_ret_1d | 3 | -0.0260 | -0.0266 | 0.0260 | True |
| eth_ret_1d | d_cross_asset | 3 | 0.0256 | 0.0210 | 0.0256 | True |
| btc_ret_1d | d_macro_engine | 7 | 0.0220 | 0.0335 | 0.0236 | False |
| btc_ret_1d | eth_ret_1d | 7 | -0.0206 | -0.0327 | 0.0234 | False |
| btc_ret_1d | d_blended_narrative_risk | 3 | 0.0233 | 0.0200 | 0.0233 | True |
| eth_ret_1d | btc_ret_1d | 7 | 0.0212 | 0.0311 | 0.0212 | True |
| eth_ret_1d | d_crypto_stress | 3 | 0.0199 | 0.0142 | 0.0199 | True |
| eth_ret_1d | btc_ret_1d | 1 | -0.0193 | -0.0172 | 0.0193 | True |
| eth_ret_1d | d_cross_asset | 1 | -0.0176 | -0.0181 | 0.0176 | True |
| eth_ret_1d | d_liquidity_stress | 3 | -0.0170 | -0.0165 | 0.0170 | True |

## Main Robustness Interpretation

- **Strong** means the variable was significant at the 5% level across all VAR specifications.
- **Moderate** means the variable was significant in most specifications.
- **Borderline** means the variable had at least one result below the 10% level but was not consistently significant.
- Stable results across lag settings are more credible than results that only appear in one VAR specification.

This is a research diagnostic, not a trading recommendation.