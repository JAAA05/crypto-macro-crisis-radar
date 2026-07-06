from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.insights.scenario_engine import build_scenario_analysis


def safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)

    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def fmt_score(value: float) -> str:
    return f"{value:.2f}"


def fmt_pct_from_decimal(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def yes_no(value: float) -> str:
    return "Yes" if value >= 1 else "No"


def build_driver_bullets(latest: pd.Series, previous: pd.Series | None = None) -> list[str]:
    bullets: list[str] = []

    btc_drawdown_30d = safe_float(latest, "bitcoin_drawdown_30d")
    btc_ret_7d = safe_float(latest, "bitcoin_ret_7d")

    macro = safe_float(latest, "macro_risk_score", 50)
    macro_engine = safe_float(latest, "macro_engine_stress_index", 50)
    growth = safe_float(latest, "growth_slowdown_score", 50)
    labor = safe_float(latest, "labor_stress_score", 50)
    inflation = safe_float(latest, "inflation_pressure_score", 50)
    monetary = safe_float(latest, "monetary_tightening_score", 50)
    fiscal = safe_float(latest, "fiscal_stress_score", 50)

    cross_asset = safe_float(latest, "cross_asset_risk_score", 50)
    equity = safe_float(latest, "equity_risk_score", 50)
    safe_haven = safe_float(latest, "safe_haven_demand_score", 50)
    commodity = safe_float(latest, "commodity_pressure_score", 50)

    crypto = safe_float(latest, "crypto_stress_score", 50)
    liquidity = safe_float(latest, "liquidity_stress_score", 50)

    news = safe_float(latest, "news_risk_score", 50)
    narrative_ai = safe_float(latest, "narrative_ai_risk_score", news)
    blended_narrative = safe_float(latest, "blended_news_narrative_risk_score", news)
    narrative_fallback = safe_float(latest, "narrative_ai_fallback_used", 0)
    elevated_narratives = safe_float(latest, "narrative_elevated_category_count", 0)
    high_narratives = safe_float(latest, "narrative_high_category_count", 0)

    anomaly = safe_float(latest, "anomaly_score", 0)
    model_prob = safe_float(latest, "model_crisis_probability", 0)

    if btc_drawdown_30d <= -0.10:
        bullets.append(
            f"BTC remains in a meaningful 30-day drawdown ({btc_drawdown_30d * 100:.2f}%), "
            "which keeps market weakness on the radar."
        )
    elif btc_ret_7d > 0.03:
        bullets.append(
            f"BTC has positive 7-day momentum ({btc_ret_7d * 100:.2f}%), "
            "which supports a more constructive short-term backdrop."
        )
    else:
        bullets.append(
            f"BTC 7-day return is {btc_ret_7d * 100:.2f}%, suggesting no strong directional impulse."
        )

    if macro >= 65:
        bullets.append(f"Traditional macro risk is elevated ({macro:.2f}).")
    elif macro >= 45:
        bullets.append(f"Traditional macro risk is moderate ({macro:.2f}), not extreme but still relevant.")
    else:
        bullets.append(f"Traditional macro risk is relatively contained ({macro:.2f}).")

    if macro_engine >= 65:
        bullets.append(
            f"The Macro Engine stress index is elevated ({macro_engine:.2f}), "
            "suggesting broad economic pressure from growth, labor, inflation, monetary, or fiscal variables."
        )
    elif macro_engine <= 35:
        bullets.append(
            f"The Macro Engine stress index is low ({macro_engine:.2f}), "
            "meaning the broader economic cycle is not confirming major stress."
        )
    else:
        bullets.append(f"The Macro Engine stress index is moderate ({macro_engine:.2f}).")

    if growth >= 65:
        bullets.append(f"Growth slowdown risk is elevated ({growth:.2f}).")
    elif growth <= 35:
        bullets.append(
            f"Growth slowdown risk is low ({growth:.2f}), with GDP, production, retail sales, or investment not showing major weakness."
        )

    if labor >= 65:
        bullets.append(f"Labor market stress is elevated ({labor:.2f}).")
    elif labor <= 35:
        bullets.append(f"Labor market stress is low ({labor:.2f}).")

    if inflation >= 65:
        bullets.append(f"Inflation pressure is elevated ({inflation:.2f}), which can keep monetary policy restrictive.")
    elif inflation <= 35:
        bullets.append(f"Inflation pressure is low-to-moderate ({inflation:.2f}).")
    else:
        bullets.append(f"Inflation pressure is moderate ({inflation:.2f}).")

    if monetary >= 65:
        bullets.append(f"Monetary tightening pressure is elevated ({monetary:.2f}).")
    elif monetary <= 40:
        bullets.append(f"Monetary tightening pressure is contained ({monetary:.2f}).")

    if fiscal >= 65:
        bullets.append(f"Fiscal stress is elevated ({fiscal:.2f}).")
    elif fiscal <= 35:
        bullets.append(f"Fiscal stress is low ({fiscal:.2f}).")

    if cross_asset >= 65:
        bullets.append(
            f"Cross-asset risk is elevated ({cross_asset:.2f}), showing confirmation from equities, safe-haven assets, or commodities."
        )
    elif cross_asset <= 30:
        bullets.append(
            f"Cross-asset risk is low ({cross_asset:.2f}), meaning traditional markets are not confirming a broad risk-off regime."
        )
    else:
        bullets.append(f"Cross-asset risk is neutral/moderate ({cross_asset:.2f}).")

    if equity >= 65:
        bullets.append(f"Equity risk is elevated ({equity:.2f}), suggesting weakness in S&P 500/Nasdaq conditions.")
    elif equity <= 30:
        bullets.append(f"Equity risk is low ({equity:.2f}), with S&P 500/Nasdaq not showing strong stress.")

    if safe_haven >= 65:
        bullets.append(
            f"Safe-haven demand is elevated ({safe_haven:.2f}), suggesting investors may be seeking protection."
        )
    elif safe_haven <= 30:
        bullets.append(
            f"Safe-haven demand is low ({safe_haven:.2f}), so defensive rotation is not strongly visible."
        )

    if commodity >= 65:
        bullets.append(
            f"Commodity pressure is elevated ({commodity:.2f}), which may point to inflation/geopolitical stress."
        )
    elif commodity <= 30:
        bullets.append(
            f"Commodity pressure is low ({commodity:.2f}), so oil is not currently adding strong macro stress."
        )

    if crypto >= 65:
        bullets.append(f"Crypto market stress is elevated ({crypto:.2f}), confirming pressure inside digital assets.")
    elif crypto <= 40:
        bullets.append(f"Crypto market stress is low-to-moderate ({crypto:.2f}).")
    else:
        bullets.append(f"Crypto market stress is moderate ({crypto:.2f}).")

    if liquidity >= 65:
        bullets.append(f"Liquidity stress is elevated ({liquidity:.2f}), which could pressure risk assets.")
    elif liquidity <= 40:
        bullets.append(f"Liquidity stress is contained ({liquidity:.2f}).")
    else:
        bullets.append(f"Liquidity stress is moderate ({liquidity:.2f}).")

    if blended_narrative >= 70:
        bullets.append(
            f"Narrative AI risk is elevated ({blended_narrative:.2f}), meaning qualitative/news pressure is important today."
        )
    elif blended_narrative >= 55:
        bullets.append(
            f"Narrative AI risk is neutral-to-moderate ({blended_narrative:.2f})."
        )
    else:
        bullets.append(
            f"Narrative AI risk is low/neutral ({blended_narrative:.2f})."
        )

    if narrative_fallback >= 1:
        bullets.append(
            "Narrative AI is currently using the general GDELT news-risk fallback because category-level narrative data was unavailable."
        )

    if elevated_narratives >= 1:
        bullets.append(
            f"Elevated narrative breadth is present across {int(elevated_narratives)} category/categorized fallback signals."
        )

    if high_narratives >= 1:
        bullets.append(
            f"High narrative breadth is present across {int(high_narratives)} category/categorized fallback signals."
        )

    if anomaly >= 70:
        bullets.append(f"Anomaly score is high ({anomaly:.2f}), meaning today's market configuration looks unusual.")
    elif anomaly >= 50:
        bullets.append(f"Anomaly score is moderate ({anomaly:.2f}), so some signals are unusual but not extreme.")
    elif anomaly <= 30:
        bullets.append(f"Anomaly score is low ({anomaly:.2f}), so the current setup does not look statistically extreme.")

    bullets.append(f"The model-estimated mini-crisis probability is {model_prob:.2f}%.")

    if previous is not None:
        prev_regime = str(previous.get("market_regime", "Unknown"))
        latest_regime = str(latest.get("market_regime", "Unknown"))

        if prev_regime != latest_regime:
            bullets.append(f"Regime changed from {prev_regime} to {latest_regime}.")
        else:
            bullets.append(f"Regime remained {latest_regime} compared with the previous observation.")

    return bullets


def main() -> None:
    scored_path = ROOT / "data/processed/scored_regime_history.csv"

    if not scored_path.exists():
        raise FileNotFoundError(
            "data/processed/scored_regime_history.csv not found. "
            "Run scripts/03_train_baseline.py first."
        )

    df = pd.read_csv(scored_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else None

    scenario = build_scenario_analysis(latest, previous)

    date = latest["date"].date()
    regime = latest.get("market_regime", "Unknown")

    macro = safe_float(latest, "macro_risk_score", 50)
    macro_engine = safe_float(latest, "macro_engine_stress_index", 50)
    growth = safe_float(latest, "growth_slowdown_score", 50)
    labor = safe_float(latest, "labor_stress_score", 50)
    inflation = safe_float(latest, "inflation_pressure_score", 50)
    monetary = safe_float(latest, "monetary_tightening_score", 50)
    fiscal = safe_float(latest, "fiscal_stress_score", 50)

    cross_asset = safe_float(latest, "cross_asset_risk_score", 50)
    equity = safe_float(latest, "equity_risk_score", 50)
    safe_haven = safe_float(latest, "safe_haven_demand_score", 50)
    commodity = safe_float(latest, "commodity_pressure_score", 50)

    news = safe_float(latest, "news_risk_score", 50)
    narrative_ai = safe_float(latest, "narrative_ai_risk_score", news)
    crypto_narrative = safe_float(latest, "crypto_narrative_risk_score", narrative_ai)
    macro_narrative = safe_float(latest, "macro_narrative_risk_score", narrative_ai)
    blended_narrative = safe_float(latest, "blended_news_narrative_risk_score", news)
    narrative_fallback = safe_float(latest, "narrative_ai_fallback_used", 0)
    narrative_article_total = safe_float(latest, "narrative_ai_category_article_total", 0)
    elevated_narratives = safe_float(latest, "narrative_elevated_category_count", 0)
    high_narratives = safe_float(latest, "narrative_high_category_count", 0)

    fed_policy = safe_float(latest, "fed_policy_enhanced_risk_score", narrative_ai)
    inflation_narrative = safe_float(latest, "inflation_enhanced_risk_score", narrative_ai)
    recession_narrative = safe_float(latest, "recession_enhanced_risk_score", narrative_ai)
    fiscal_narrative = safe_float(latest, "fiscal_policy_enhanced_risk_score", narrative_ai)
    crypto_regulation = safe_float(latest, "crypto_regulation_enhanced_risk_score", narrative_ai)
    exchange_risk = safe_float(latest, "exchange_risk_enhanced_risk_score", narrative_ai)
    stablecoin_risk = safe_float(latest, "stablecoin_risk_enhanced_risk_score", narrative_ai)
    etf_institutional = safe_float(latest, "etf_institutional_enhanced_risk_score", narrative_ai)
    geopolitical = safe_float(latest, "geopolitical_enhanced_risk_score", narrative_ai)
    market_sentiment = safe_float(latest, "market_sentiment_enhanced_risk_score", narrative_ai)

    crypto = safe_float(latest, "crypto_stress_score", 50)
    liquidity = safe_float(latest, "liquidity_stress_score", 50)
    anomaly = safe_float(latest, "anomaly_score", 0)
    model_prob = safe_float(latest, "model_crisis_probability", 0)
    rule_based = safe_float(latest, "rule_based_risk_score", safe_float(latest, "final_risk_score", 50))
    final_model = safe_float(latest, "final_risk_score_with_model", rule_based)

    btc_price = safe_float(latest, "bitcoin_price")
    btc_ret_1d = safe_float(latest, "bitcoin_ret_1d")
    btc_ret_7d = safe_float(latest, "bitcoin_ret_7d")
    btc_drawdown_30d = safe_float(latest, "bitcoin_drawdown_30d")
    btc_vol_30d = safe_float(latest, "bitcoin_vol_30d")

    sp500_ret_7d = safe_float(latest, "sp500_ret_7d")
    nasdaq_ret_7d = safe_float(latest, "nasdaq_ret_7d")
    gold_ret_7d = safe_float(latest, "gold_ret_7d")
    oil_ret_7d = safe_float(latest, "oil_ret_7d")

    real_gdp_yoy = safe_float(latest, "real_gdp_yoy")
    unemployment_rate = safe_float(latest, "unemployment_rate")
    cpi_yoy = safe_float(latest, "cpi_yoy")
    core_cpi_yoy = safe_float(latest, "core_cpi_yoy")
    retail_sales_yoy = safe_float(latest, "retail_sales_yoy")
    private_investment_yoy = safe_float(latest, "private_investment_yoy")

    bullets = build_driver_bullets(latest, previous)

    report = f"""# Crypto Macro Crisis Radar — Latest Report

**Date:** {date}

## Market Regime

**{regime}**

## AI Market Analyst Summary

{scenario["summary"]}

## Scores

| Component | Score |
|---|---:|
| Macro Risk | {fmt_score(macro)} |
| Macro Engine Stress | {fmt_score(macro_engine)} |
| Cross-Asset Risk | {fmt_score(cross_asset)} |
| Equity Risk | {fmt_score(equity)} |
| Safe-Haven Demand | {fmt_score(safe_haven)} |
| Commodity Pressure | {fmt_score(commodity)} |
| News / Narrative Risk | {fmt_score(news)} |
| Narrative AI Risk | {fmt_score(narrative_ai)} |
| Blended News/Narrative Risk | {fmt_score(blended_narrative)} |
| Crypto Market Stress | {fmt_score(crypto)} |
| Liquidity Stress | {fmt_score(liquidity)} |
| Anomaly Score | {fmt_score(anomaly)} |
| Model Crisis Probability | {model_prob:.2f}% |
| Rule-Based Risk Score | {fmt_score(rule_based)} |
| Model-Adjusted Final Risk Score | {fmt_score(final_model)} |

## Narrative AI Snapshot

| Component | Value |
|---|---:|
| Narrative AI Risk | {fmt_score(narrative_ai)} |
| Crypto Narrative Risk | {fmt_score(crypto_narrative)} |
| Macro Narrative Risk | {fmt_score(macro_narrative)} |
| Blended News/Narrative Risk | {fmt_score(blended_narrative)} |
| General News Risk | {fmt_score(news)} |
| Category-Level Articles | {int(narrative_article_total)} |
| Fallback Used | {yes_no(narrative_fallback)} |
| Elevated Narrative Categories | {int(elevated_narratives)} |
| High Narrative Categories | {int(high_narratives)} |

## Narrative Category Breakdown

| Category | Score |
|---|---:|
| Fed Policy | {fmt_score(fed_policy)} |
| Inflation Narrative | {fmt_score(inflation_narrative)} |
| Recession Narrative | {fmt_score(recession_narrative)} |
| Fiscal Policy | {fmt_score(fiscal_narrative)} |
| Crypto Regulation | {fmt_score(crypto_regulation)} |
| Exchange Risk | {fmt_score(exchange_risk)} |
| Stablecoin Risk | {fmt_score(stablecoin_risk)} |
| ETF / Institutional | {fmt_score(etf_institutional)} |
| Geopolitical | {fmt_score(geopolitical)} |
| Market Sentiment | {fmt_score(market_sentiment)} |

## Macro Engine Snapshot

| Component | Value |
|---|---:|
| Growth Slowdown Score | {fmt_score(growth)} |
| Labor Stress Score | {fmt_score(labor)} |
| Inflation Pressure Score | {fmt_score(inflation)} |
| Monetary Tightening Score | {fmt_score(monetary)} |
| Fiscal Stress Score | {fmt_score(fiscal)} |
| Real GDP YoY | {fmt_pct_from_decimal(real_gdp_yoy)} |
| Unemployment Rate | {unemployment_rate:.2f}% |
| CPI YoY | {fmt_pct_from_decimal(cpi_yoy)} |
| Core CPI YoY | {fmt_pct_from_decimal(core_cpi_yoy)} |
| Retail Sales YoY | {fmt_pct_from_decimal(retail_sales_yoy)} |
| Private Investment YoY | {fmt_pct_from_decimal(private_investment_yoy)} |

## Market Snapshot

| Metric | Value |
|---|---:|
| BTC Price | {fmt_money(btc_price)} |
| BTC 1D Return | {fmt_pct_from_decimal(btc_ret_1d)} |
| BTC 7D Return | {fmt_pct_from_decimal(btc_ret_7d)} |
| BTC 30D Drawdown | {fmt_pct_from_decimal(btc_drawdown_30d)} |
| BTC 30D Volatility | {fmt_pct_from_decimal(btc_vol_30d)} |

## Cross-Asset Snapshot

| Metric | Value |
|---|---:|
| S&P 500 7D Return | {fmt_pct_from_decimal(sp500_ret_7d)} |
| Nasdaq 7D Return | {fmt_pct_from_decimal(nasdaq_ret_7d)} |
| Gold 7D Return | {fmt_pct_from_decimal(gold_ret_7d)} |
| Oil WTI 7D Return | {fmt_pct_from_decimal(oil_ret_7d)} |

## Main Drivers

"""

    for bullet in bullets:
        report += f"- {bullet}\n"

    report += "\n## Scenario Watch\n\n"

    scenario_watch = scenario["scenario_watch"]
    if isinstance(scenario_watch, dict):
        for title, text in scenario_watch.items():
            report += f"### {title}\n\n{text}\n\n"

    report += "## What Would Change the Regime?\n\n"

    triggers = scenario["regime_change_triggers"]
    if isinstance(triggers, list):
        for trigger in triggers:
            report += f"- {trigger}\n"

    report += "\n## Risk Flags to Monitor\n\n"

    risk_flags = scenario["risk_flags"]
    if isinstance(risk_flags, list):
        for flag in risk_flags:
            report += f"- {flag}\n"

    report += """
## Interpretation

This report is a research signal, not a trading recommendation. The system combines macroeconomic cycle variables, monetary/fiscal pressure, cross-asset confirmation, crypto market stress, liquidity, Narrative AI risk, anomaly detection, and a baseline classifier to estimate the current crypto market regime.

## Suggested research action

- **Risk-On / Neutral:** monitor opportunities, but validate with macro, event, liquidity, narrative, and cross-asset context.
- **Risk-Off:** reduce simulated exposure and track whether stress is spreading across macro data, equities, commodities, safe-haven assets, narrative channels, and crypto.
- **Mini-Crisis Watch / Crisis:** observe only, document drivers, and evaluate whether the model gave early warning.
"""

    output_path = ROOT / "outputs/reports/latest_market_regime_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    print(report)
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()