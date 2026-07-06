from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class LagShockConfig:
    horizons: tuple[int, ...] = (1, 3, 7, 14)
    lags: tuple[int, ...] = (0, 1, 3, 7, 14, 30)
    min_obs_corr: int = 250
    min_events_shock: int = 3
    shock_quantile_high: float = 0.90
    shock_quantile_low: float = 0.10
    min_gap_days_between_shocks: int = 14


PREDICTOR_COLUMNS = [
    # Final / model scores
    "rule_based_risk_score",
    "final_risk_score",
    "final_risk_score_with_model",
    "model_crisis_probability",
    "anomaly_score",

    # Macro / financial conditions
    "macro_risk_score",
    "vix",
    "vix_z_90d",
    "yield_curve_10y_2y",
    "yield_curve_inversion_flag",
    "treasury_10y",
    "treasury_2y",
    "fed_funds_rate",
    "dollar_ret_20d",
    "dollar_z_90d",
    "fed_balance_sheet_90d_chg",
    "m2_90d_chg",
    "epu_z_90d",

    # Macro Engine
    "macro_engine_stress_index",
    "growth_slowdown_score",
    "labor_stress_score",
    "inflation_pressure_score",
    "monetary_tightening_score",
    "fiscal_stress_score",
    "real_gdp_yoy",
    "industrial_production_yoy",
    "retail_sales_yoy",
    "private_investment_yoy",
    "unemployment_change_3m",
    "unemployment_change_6m",
    "cpi_yoy",
    "core_cpi_yoy",

    # Cross-asset
    "cross_asset_risk_score",
    "equity_risk_score",
    "safe_haven_demand_score",
    "commodity_pressure_score",
    "sp500_ret_7d",
    "nasdaq_ret_7d",
    "gold_ret_7d",
    "oil_ret_7d",
    "oil_vol_30d",

    # Crypto market state
    "crypto_stress_score",
    "bitcoin_ret_1d",
    "bitcoin_ret_3d",
    "bitcoin_ret_7d",
    "bitcoin_vol_7d",
    "bitcoin_vol_30d",
    "bitcoin_drawdown_30d",
    "bitcoin_drawdown_90d",
    "ethereum_ret_1d",
    "ethereum_ret_3d",
    "ethereum_ret_7d",
    "ethereum_vol_7d",
    "ethereum_vol_30d",
    "ethereum_drawdown_30d",
    "ethereum_drawdown_90d",
    "btc_eth_corr_30d",

    # Liquidity / original narrative
    "liquidity_stress_score",
    "stablecoin_mcap_7d_chg",
    "stablecoin_mcap_30d_chg",
    "stablecoin_mcap_z_90d",
    "news_article_count",
    "news_negative_hits",
    "news_positive_hits",
    "news_risk_score",

    # Narrative AI v2
    "narrative_ai_risk_score",
    "crypto_narrative_risk_score",
    "macro_narrative_risk_score",
    "blended_news_narrative_risk_score",
    "narrative_elevated_category_count",
    "narrative_high_category_count",
    "narrative_ai_fallback_used",
    "narrative_ai_category_article_total",
    "fed_policy_enhanced_risk_score",
    "inflation_enhanced_risk_score",
    "recession_enhanced_risk_score",
    "fiscal_policy_enhanced_risk_score",
    "crypto_regulation_enhanced_risk_score",
    "exchange_risk_enhanced_risk_score",
    "stablecoin_risk_enhanced_risk_score",
    "etf_institutional_enhanced_risk_score",
    "geopolitical_enhanced_risk_score",
    "market_sentiment_enhanced_risk_score",
]


SHOCK_SPECS = [
    # High-risk score shocks
    {
        "column": "final_risk_score_with_model",
        "direction": "high",
        "name": "Final model-adjusted risk shock",
    },
    {
        "column": "rule_based_risk_score",
        "direction": "high",
        "name": "Rule-based risk shock",
    },
    {
        "column": "model_crisis_probability",
        "direction": "high",
        "name": "Model crisis probability shock",
    },
    {
        "column": "anomaly_score",
        "direction": "high",
        "name": "Anomaly shock",
    },

    # Macro / macro engine shocks
    {
        "column": "macro_risk_score",
        "direction": "high",
        "name": "Traditional macro risk shock",
    },
    {
        "column": "macro_engine_stress_index",
        "direction": "high",
        "name": "Macro Engine stress shock",
    },
    {
        "column": "inflation_pressure_score",
        "direction": "high",
        "name": "Inflation pressure shock",
    },
    {
        "column": "monetary_tightening_score",
        "direction": "high",
        "name": "Monetary tightening shock",
    },
    {
        "column": "labor_stress_score",
        "direction": "high",
        "name": "Labor stress shock",
    },
    {
        "column": "growth_slowdown_score",
        "direction": "high",
        "name": "Growth slowdown shock",
    },

    # Cross-asset shocks
    {
        "column": "cross_asset_risk_score",
        "direction": "high",
        "name": "Cross-asset risk shock",
    },
    {
        "column": "equity_risk_score",
        "direction": "high",
        "name": "Equity risk shock",
    },
    {
        "column": "safe_haven_demand_score",
        "direction": "high",
        "name": "Safe-haven demand shock",
    },
    {
        "column": "commodity_pressure_score",
        "direction": "high",
        "name": "Commodity pressure shock",
    },
    {
        "column": "vix_z_90d",
        "direction": "high",
        "name": "VIX volatility shock",
    },
    {
        "column": "dollar_z_90d",
        "direction": "high",
        "name": "Dollar strength shock",
    },
    {
        "column": "sp500_ret_7d",
        "direction": "low",
        "name": "S&P 500 selloff shock",
    },
    {
        "column": "nasdaq_ret_7d",
        "direction": "low",
        "name": "Nasdaq selloff shock",
    },
    {
        "column": "oil_ret_7d",
        "direction": "low",
        "name": "Oil downside shock",
    },
    {
        "column": "oil_vol_30d",
        "direction": "high",
        "name": "Oil volatility shock",
    },

    # Crypto / liquidity shocks
    {
        "column": "crypto_stress_score",
        "direction": "high",
        "name": "Crypto stress shock",
    },
    {
        "column": "liquidity_stress_score",
        "direction": "high",
        "name": "Liquidity stress shock",
    },

    # Original narrative shock
    {
        "column": "news_risk_score",
        "direction": "high",
        "name": "News / narrative risk shock",
        "threshold_override": 65.0,
    },

    # Narrative AI v2 shocks
    {
        "column": "narrative_ai_risk_score",
        "direction": "high",
        "name": "Narrative AI risk shock",
        "threshold_override": 65.0,
    },
    {
        "column": "crypto_narrative_risk_score",
        "direction": "high",
        "name": "Crypto narrative risk shock",
        "threshold_override": 65.0,
    },
    {
        "column": "macro_narrative_risk_score",
        "direction": "high",
        "name": "Macro narrative risk shock",
        "threshold_override": 65.0,
    },
    {
        "column": "blended_news_narrative_risk_score",
        "direction": "high",
        "name": "Blended news/narrative risk shock",
        "threshold_override": 65.0,
    },
    {
        "column": "narrative_elevated_category_count",
        "direction": "high",
        "name": "Elevated narrative breadth shock",
        "threshold_override": 1.0,
    },
    {
        "column": "narrative_high_category_count",
        "direction": "high",
        "name": "High narrative breadth shock",
        "threshold_override": 1.0,
    },

    # BTC-specific shocks
    {
        "column": "bitcoin_drawdown_30d",
        "direction": "low",
        "name": "BTC 30D drawdown shock",
    },
    {
        "column": "bitcoin_vol_30d",
        "direction": "high",
        "name": "BTC volatility shock",
    },
]


def safe_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def prepare_forward_returns(df: pd.DataFrame, config: LagShockConfig) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    for price_col, prefix in [
        ("bitcoin_price", "btc"),
        ("ethereum_price", "eth"),
    ]:
        if price_col not in out.columns:
            continue

        price = pd.to_numeric(out[price_col], errors="coerce")

        for horizon in config.horizons:
            out[f"{prefix}_fwd_return_{horizon}d"] = price.shift(-horizon) / price - 1

    return out


def relationship_label(corr: float) -> str:
    if pd.isna(corr):
        return "Insufficient data"
    if corr <= -0.20:
        return "Strong negative relationship"
    if corr <= -0.10:
        return "Moderate negative relationship"
    if corr < 0.10:
        return "Weak / neutral relationship"
    if corr < 0.20:
        return "Moderate positive relationship"
    return "Strong positive relationship"


def analyze_lag_correlations(
    df: pd.DataFrame,
    config: LagShockConfig,
) -> pd.DataFrame:
    rows = []

    predictors = [col for col in PREDICTOR_COLUMNS if col in df.columns]

    for predictor in predictors:
        predictor_series = pd.to_numeric(df[predictor], errors="coerce")

        for lag in config.lags:
            lagged_predictor = predictor_series.shift(lag)

            for asset_prefix in ["btc", "eth"]:
                for horizon in config.horizons:
                    target_col = f"{asset_prefix}_fwd_return_{horizon}d"

                    if target_col not in df.columns:
                        continue

                    target = pd.to_numeric(df[target_col], errors="coerce")

                    pair = pd.DataFrame(
                        {
                            "x": lagged_predictor,
                            "y": target,
                        }
                    ).dropna()

                    n_obs = len(pair)

                    if n_obs < config.min_obs_corr:
                        continue

                    if pair["x"].nunique(dropna=True) < 2:
                        continue

                    if pair["y"].nunique(dropna=True) < 2:
                        continue

                    pearson = pair["x"].corr(pair["y"], method="pearson")
                    spearman = pair["x"].corr(pair["y"], method="spearman")

                    rows.append(
                        {
                            "predictor": predictor,
                            "lag_days": lag,
                            "target_asset": asset_prefix,
                            "forward_horizon_days": horizon,
                            "n_obs": n_obs,
                            "pearson_corr": pearson,
                            "spearman_corr": spearman,
                            "abs_pearson_corr": abs(pearson) if not pd.isna(pearson) else np.nan,
                            "relationship": relationship_label(pearson),
                        }
                    )

    results = pd.DataFrame(rows)

    if results.empty:
        return results

    results = results.sort_values(
        ["target_asset", "forward_horizon_days", "abs_pearson_corr"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return results


def detect_shock_indices(
    df: pd.DataFrame,
    column: str,
    direction: str,
    config: LagShockConfig,
    threshold_override: float | None = None,
) -> tuple[list[int], float]:
    if column not in df.columns:
        return [], np.nan

    values = pd.to_numeric(df[column], errors="coerce")

    if threshold_override is not None:
        threshold = float(threshold_override)
    else:
        if direction == "high":
            threshold = values.quantile(config.shock_quantile_high)
        elif direction == "low":
            threshold = values.quantile(config.shock_quantile_low)
        else:
            raise ValueError("direction must be 'high' or 'low'")

    if direction == "high":
        condition = values >= threshold
    elif direction == "low":
        condition = values <= threshold
    else:
        raise ValueError("direction must be 'high' or 'low'")

    condition = condition.fillna(False)

    raw_indices = list(np.where(condition.values)[0])

    if not raw_indices:
        return [], threshold

    filtered = []
    last_idx = None

    for idx in raw_indices:
        if last_idx is None or idx - last_idx >= config.min_gap_days_between_shocks:
            filtered.append(idx)
            last_idx = idx

    return filtered, threshold


def analyze_shocks(
    df: pd.DataFrame,
    config: LagShockConfig,
) -> pd.DataFrame:
    rows = []

    for spec in SHOCK_SPECS:
        column = spec["column"]

        if column not in df.columns:
            continue

        direction = spec["direction"]
        shock_name = spec["name"]

        shock_indices, threshold = detect_shock_indices(
            df=df,
            column=column,
            direction=direction,
            config=config,
            threshold_override=spec.get("threshold_override"),
        )

        if len(shock_indices) < config.min_events_shock:
            continue

        for horizon in config.horizons:
            btc_col = f"btc_fwd_return_{horizon}d"
            eth_col = f"eth_fwd_return_{horizon}d"

            if btc_col not in df.columns:
                continue

            event_rows = df.iloc[shock_indices].copy()

            btc_returns = pd.to_numeric(event_rows[btc_col], errors="coerce")
            eth_returns = (
                pd.to_numeric(event_rows[eth_col], errors="coerce")
                if eth_col in event_rows.columns
                else pd.Series(dtype=float)
            )

            risk_now = pd.to_numeric(event_rows.get("final_risk_score_with_model"), errors="coerce")

            future_risk = (
                pd.to_numeric(df["final_risk_score_with_model"].shift(-horizon), errors="coerce")
                .iloc[shock_indices]
                if "final_risk_score_with_model" in df.columns
                else pd.Series(dtype=float)
            )

            risk_delta = future_risk.reset_index(drop=True) - risk_now.reset_index(drop=True)

            valid_btc = btc_returns.dropna()
            valid_eth = eth_returns.dropna()
            valid_risk_delta = risk_delta.dropna()

            if len(valid_btc) < config.min_events_shock:
                continue

            rows.append(
                {
                    "shock_column": column,
                    "shock_name": shock_name,
                    "direction": direction,
                    "threshold": threshold,
                    "horizon_days": horizon,
                    "event_count": int(len(valid_btc)),
                    "avg_btc_return": float(valid_btc.mean()),
                    "median_btc_return": float(valid_btc.median()),
                    "btc_adverse_rate": float((valid_btc < 0).mean()),
                    "avg_eth_return": float(valid_eth.mean()) if len(valid_eth) else np.nan,
                    "median_eth_return": float(valid_eth.median()) if len(valid_eth) else np.nan,
                    "avg_risk_score_change": float(valid_risk_delta.mean()) if len(valid_risk_delta) else np.nan,
                    "median_risk_score_change": float(valid_risk_delta.median()) if len(valid_risk_delta) else np.nan,
                    "stress_continuation_rate": float((valid_risk_delta > 0).mean()) if len(valid_risk_delta) else np.nan,
                }
            )

    results = pd.DataFrame(rows)

    if results.empty:
        return results

    results = results.sort_values(
        ["horizon_days", "avg_btc_return", "avg_risk_score_change"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return results


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def fmt_num(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def build_lag_shock_markdown_report(
    lag_results: pd.DataFrame,
    shock_results: pd.DataFrame,
    lag_csv_name: str,
    shock_csv_name: str,
) -> str:
    lines = [
        "# Lag & Shock Analysis Report",
        "",
        "This report studies lag relationships and shock responses using the scored market-regime history.",
        "",
        "It is designed for research and interpretation, not trading recommendations.",
        "",
        "## Files Generated",
        "",
        f"- Lag correlation results: `{lag_csv_name}`",
        f"- Shock analysis results: `{shock_csv_name}`",
        "",
    ]

    if lag_results.empty:
        lines.extend(["## Lag Correlation Summary", "", "No lag correlation results were generated."])
    else:
        btc_7d = lag_results[
            (lag_results["target_asset"] == "btc")
            & (lag_results["forward_horizon_days"] == 7)
        ].copy()

        top_negative = btc_7d.sort_values("pearson_corr", ascending=True).head(15)
        top_positive = btc_7d.sort_values("pearson_corr", ascending=False).head(15)

        lines.extend(
            [
                "## BTC 7-Day Forward Return — Strongest Negative Lag Relationships",
                "",
                "Higher predictor values were associated with weaker BTC 7-day forward returns.",
                "",
                "| Predictor | Lag Days | Pearson | Spearman | N | Relationship |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )

        for _, row in top_negative.iterrows():
            lines.append(
                f"| {row['predictor']} | "
                f"{int(row['lag_days'])} | "
                f"{fmt_num(row['pearson_corr'])} | "
                f"{fmt_num(row['spearman_corr'])} | "
                f"{int(row['n_obs'])} | "
                f"{row['relationship']} |"
            )

        lines.extend(
            [
                "",
                "## BTC 7-Day Forward Return — Strongest Positive Lag Relationships",
                "",
                "Higher predictor values were associated with stronger BTC 7-day forward returns.",
                "",
                "| Predictor | Lag Days | Pearson | Spearman | N | Relationship |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )

        for _, row in top_positive.iterrows():
            lines.append(
                f"| {row['predictor']} | "
                f"{int(row['lag_days'])} | "
                f"{fmt_num(row['pearson_corr'])} | "
                f"{fmt_num(row['spearman_corr'])} | "
                f"{int(row['n_obs'])} | "
                f"{row['relationship']} |"
            )

    if shock_results.empty:
        lines.extend(["", "## Shock Response Summary", "", "No shock analysis results were generated."])
    else:
        shock_7d = shock_results[shock_results["horizon_days"] == 7].copy()

        lines.extend(
            [
                "",
                "## 7-Day Shock Response Summary",
                "",
                "| Shock | Direction | Events | Avg BTC 7D | Median BTC 7D | BTC Adverse Rate | Avg ETH 7D | Avg Risk Δ | Stress Continuation Rate |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for _, row in shock_7d.iterrows():
            lines.append(
                f"| {row['shock_name']} | "
                f"{row['direction']} | "
                f"{int(row['event_count'])} | "
                f"{fmt_pct(row['avg_btc_return'])} | "
                f"{fmt_pct(row['median_btc_return'])} | "
                f"{fmt_pct(row['btc_adverse_rate'])} | "
                f"{fmt_pct(row['avg_eth_return'])} | "
                f"{fmt_num(row['avg_risk_score_change'])} | "
                f"{fmt_pct(row['stress_continuation_rate'])} |"
            )

        worst_shocks = shock_7d.sort_values("avg_btc_return", ascending=True).head(10)
        risk_build_shocks = shock_7d.sort_values("avg_risk_score_change", ascending=False).head(10)

        lines.extend(
            [
                "",
                "## Most Adverse 7-Day Shock Types",
                "",
                "| Shock | Events | Avg BTC 7D | Avg ETH 7D | Avg Risk Δ |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for _, row in worst_shocks.iterrows():
            lines.append(
                f"| {row['shock_name']} | "
                f"{int(row['event_count'])} | "
                f"{fmt_pct(row['avg_btc_return'])} | "
                f"{fmt_pct(row['avg_eth_return'])} | "
                f"{fmt_num(row['avg_risk_score_change'])} |"
            )

        lines.extend(
            [
                "",
                "## Strongest Risk-Build Shock Types",
                "",
                "| Shock | Events | Avg BTC 7D | Avg ETH 7D | Avg Risk Δ |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for _, row in risk_build_shocks.iterrows():
            lines.append(
                f"| {row['shock_name']} | "
                f"{int(row['event_count'])} | "
                f"{fmt_pct(row['avg_btc_return'])} | "
                f"{fmt_pct(row['avg_eth_return'])} | "
                f"{fmt_num(row['avg_risk_score_change'])} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Correlations describe association, not causality.",
            "- Lag days mean the predictor is shifted backward before comparing it with future BTC/ETH returns.",
            "- A negative correlation means higher predictor values were associated with weaker future returns.",
            "- A positive correlation means higher predictor values were associated with stronger future returns.",
            "- Shock analysis uses extreme quantiles to study what happened after unusually high or low values.",
            "- Results should be interpreted as research diagnostics, not trading signals.",
        ]
    )

    return "\n".join(lines)