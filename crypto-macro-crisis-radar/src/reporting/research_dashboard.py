from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def fmt_score(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def fmt_pct_decimal(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def fmt_pct_score(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2f}%"


def fmt_money(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"${value:,.2f}"


def yes_no(value: float) -> str:
    return "Yes" if safe_float(value, 0) >= 1 else "No"


def read_csv_if_exists(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path, parse_dates=parse_dates)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_latest_row(scored: pd.DataFrame) -> pd.Series:
    if scored.empty:
        raise ValueError("scored_regime_history.csv is empty or missing.")

    scored = scored.copy()
    scored["date"] = pd.to_datetime(scored["date"])
    scored = scored.sort_values("date").reset_index(drop=True)

    return scored.iloc[-1]


def build_latest_snapshot(latest: pd.Series) -> pd.DataFrame:
    rows = [
        {
            "section": "Market Regime",
            "metric": "Date",
            "value": str(pd.to_datetime(latest.get("date")).date()),
        },
        {
            "section": "Market Regime",
            "metric": "Regime",
            "value": str(latest.get("market_regime", "Unknown")),
        },
        {
            "section": "Market Regime",
            "metric": "Model-Adjusted Final Risk Score",
            "value": fmt_score(safe_float(latest.get("final_risk_score_with_model"))),
        },
        {
            "section": "Market Regime",
            "metric": "Rule-Based Risk Score",
            "value": fmt_score(safe_float(latest.get("rule_based_risk_score"))),
        },
        {
            "section": "Market Regime",
            "metric": "Model Crisis Probability",
            "value": fmt_pct_score(safe_float(latest.get("model_crisis_probability"))),
        },
        {
            "section": "Crypto",
            "metric": "BTC Price",
            "value": fmt_money(safe_float(latest.get("bitcoin_price"))),
        },
        {
            "section": "Crypto",
            "metric": "BTC 7D Return",
            "value": fmt_pct_decimal(safe_float(latest.get("bitcoin_ret_7d"))),
        },
        {
            "section": "Crypto",
            "metric": "BTC 30D Drawdown",
            "value": fmt_pct_decimal(safe_float(latest.get("bitcoin_drawdown_30d"))),
        },
        {
            "section": "Risk Scores",
            "metric": "Macro Engine Stress",
            "value": fmt_score(safe_float(latest.get("macro_engine_stress_index"))),
        },
        {
            "section": "Risk Scores",
            "metric": "Cross-Asset Risk",
            "value": fmt_score(safe_float(latest.get("cross_asset_risk_score"))),
        },
        {
            "section": "Risk Scores",
            "metric": "Crypto Market Stress",
            "value": fmt_score(safe_float(latest.get("crypto_stress_score"))),
        },
        {
            "section": "Risk Scores",
            "metric": "Liquidity Stress",
            "value": fmt_score(safe_float(latest.get("liquidity_stress_score"))),
        },
        {
            "section": "Risk Scores",
            "metric": "Narrative AI Risk",
            "value": fmt_score(safe_float(latest.get("narrative_ai_risk_score"))),
        },
        {
            "section": "Risk Scores",
            "metric": "Blended News/Narrative Risk",
            "value": fmt_score(safe_float(latest.get("blended_news_narrative_risk_score"))),
        },
        {
            "section": "Risk Scores",
            "metric": "Anomaly Score",
            "value": fmt_score(safe_float(latest.get("anomaly_score"))),
        },
        {
            "section": "Narrative AI",
            "metric": "Fallback Used",
            "value": yes_no(safe_float(latest.get("narrative_ai_fallback_used"))),
        },
        {
            "section": "Narrative AI",
            "metric": "Category-Level Articles",
            "value": str(safe_int(latest.get("narrative_ai_category_article_total"))),
        },
    ]

    return pd.DataFrame(rows)


def markdown_table_from_records(records: list[dict], columns: list[str]) -> str:
    if not records:
        return "_No data available._"

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")

    for record in records:
        values = [str(record.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_latest_regime_section(latest: pd.Series) -> str:
    date = pd.to_datetime(latest.get("date")).date()
    regime = str(latest.get("market_regime", "Unknown"))

    final_score = safe_float(latest.get("final_risk_score_with_model"))
    rule_score = safe_float(latest.get("rule_based_risk_score"))
    model_prob = safe_float(latest.get("model_crisis_probability"))

    btc_price = safe_float(latest.get("bitcoin_price"))
    btc_7d = safe_float(latest.get("bitcoin_ret_7d"))
    btc_dd_30d = safe_float(latest.get("bitcoin_drawdown_30d"))

    macro_engine = safe_float(latest.get("macro_engine_stress_index"))
    cross_asset = safe_float(latest.get("cross_asset_risk_score"))
    crypto_stress = safe_float(latest.get("crypto_stress_score"))
    liquidity = safe_float(latest.get("liquidity_stress_score"))
    narrative = safe_float(latest.get("blended_news_narrative_risk_score"))
    anomaly = safe_float(latest.get("anomaly_score"))

    rows = [
        {"Metric": "Date", "Value": str(date)},
        {"Metric": "Market Regime", "Value": regime},
        {"Metric": "Model-Adjusted Final Risk Score", "Value": fmt_score(final_score)},
        {"Metric": "Rule-Based Risk Score", "Value": fmt_score(rule_score)},
        {"Metric": "Model Crisis Probability", "Value": fmt_pct_score(model_prob)},
        {"Metric": "BTC Price", "Value": fmt_money(btc_price)},
        {"Metric": "BTC 7D Return", "Value": fmt_pct_decimal(btc_7d)},
        {"Metric": "BTC 30D Drawdown", "Value": fmt_pct_decimal(btc_dd_30d)},
        {"Metric": "Macro Engine Stress", "Value": fmt_score(macro_engine)},
        {"Metric": "Cross-Asset Risk", "Value": fmt_score(cross_asset)},
        {"Metric": "Crypto Market Stress", "Value": fmt_score(crypto_stress)},
        {"Metric": "Liquidity Stress", "Value": fmt_score(liquidity)},
        {"Metric": "Blended News/Narrative Risk", "Value": fmt_score(narrative)},
        {"Metric": "Anomaly Score", "Value": fmt_score(anomaly)},
    ]

    table = markdown_table_from_records(rows, ["Metric", "Value"])

    return f"""## Latest Market Regime Snapshot

{table}

**Interpretation:** The latest regime is **{regime}** with a model-adjusted final risk score of **{fmt_score(final_score)}**. BTC is in a **{fmt_pct_decimal(btc_dd_30d)}** 30-day drawdown, but the broader regime depends on whether macro, cross-asset, liquidity, narrative, and crypto stress confirm that weakness.
"""


def build_model_architecture_section(model_report: dict, latest: pd.Series) -> str:
    classifier_name = model_report.get("classifier_name", "Unknown")
    classifier_feature_set = model_report.get("classifier_feature_set", "Unknown")
    excluded = model_report.get("excluded_from_classifier", "None")
    reason = model_report.get("reason_for_exclusion", "")
    macro_note = model_report.get("macro_engine_usage_note", "")

    tuned_threshold = model_report.get("tuned_threshold", None)

    test_metrics = model_report.get("test_metrics_tuned_threshold", {})
    default_metrics = model_report.get("test_metrics_default_threshold_0_50", {})

    rule_based = safe_float(latest.get("rule_based_risk_score"))
    model_prob = safe_float(latest.get("model_crisis_probability"))
    final_score = safe_float(latest.get("final_risk_score_with_model"))
    regime = str(latest.get("market_regime", "Unknown"))

    rows = [
        {"Metric": "Active Classifier", "Value": classifier_name},
        {"Metric": "Classifier Feature Set", "Value": classifier_feature_set},
        {"Metric": "Excluded From Supervised Classifier", "Value": excluded},
        {
            "Metric": "Tuned Classification Threshold",
            "Value": "" if tuned_threshold is None else f"{float(tuned_threshold):.2f}",
        },
        {"Metric": "Rule-Based Risk Score", "Value": fmt_score(rule_based)},
        {"Metric": "Model Crisis Probability", "Value": fmt_pct_score(model_prob)},
        {"Metric": "Model-Adjusted Final Risk Score", "Value": fmt_score(final_score)},
        {"Metric": "Current Market Regime", "Value": regime},
        {
            "Metric": "Test ROC-AUC",
            "Value": fmt_score(safe_float(test_metrics.get("roc_auc"))),
        },
        {
            "Metric": "Test Average Precision",
            "Value": fmt_score(safe_float(test_metrics.get("average_precision"))),
        },
        {
            "Metric": "Test Balanced Accuracy",
            "Value": fmt_score(safe_float(test_metrics.get("balanced_accuracy"))),
        },
        {
            "Metric": "Test Precision",
            "Value": fmt_score(safe_float(test_metrics.get("precision"))),
        },
        {
            "Metric": "Test Recall",
            "Value": fmt_score(safe_float(test_metrics.get("recall"))),
        },
        {
            "Metric": "Test F1",
            "Value": fmt_score(safe_float(test_metrics.get("f1"))),
        },
        {
            "Metric": "Default 0.50 Threshold F1",
            "Value": fmt_score(safe_float(default_metrics.get("f1"))),
        },
    ]

    table = markdown_table_from_records(rows, ["Metric", "Value"])

    explanation_parts = []

    if reason:
        explanation_parts.append(reason)

    if macro_note:
        explanation_parts.append(macro_note)

    explanation = " ".join(explanation_parts)

    if not explanation:
        explanation = (
            "The supervised classifier is one layer of the system. "
            "The final market regime also depends on rule-based risk scoring, anomaly detection, "
            "macro context, cross-asset confirmation, liquidity, and narrative risk."
        )

    return f"""## Model Architecture Summary

{table}

**Architecture Note:** {explanation}
"""


def build_narrative_section(latest: pd.Series) -> str:
    narrative_ai = safe_float(latest.get("narrative_ai_risk_score"))
    crypto_narrative = safe_float(latest.get("crypto_narrative_risk_score"))
    macro_narrative = safe_float(latest.get("macro_narrative_risk_score"))
    blended = safe_float(latest.get("blended_news_narrative_risk_score"))
    news = safe_float(latest.get("news_risk_score"))
    fallback = safe_float(latest.get("narrative_ai_fallback_used"))
    category_articles = safe_int(latest.get("narrative_ai_category_article_total"))
    elevated = safe_int(latest.get("narrative_elevated_category_count"))
    high = safe_int(latest.get("narrative_high_category_count"))

    rows = [
        {"Metric": "Narrative AI Risk", "Value": fmt_score(narrative_ai)},
        {"Metric": "Crypto Narrative Risk", "Value": fmt_score(crypto_narrative)},
        {"Metric": "Macro Narrative Risk", "Value": fmt_score(macro_narrative)},
        {"Metric": "Blended News/Narrative Risk", "Value": fmt_score(blended)},
        {"Metric": "General News Risk", "Value": fmt_score(news)},
        {"Metric": "Fallback Used", "Value": yes_no(fallback)},
        {"Metric": "Category-Level Articles", "Value": str(category_articles)},
        {"Metric": "Elevated Narrative Categories", "Value": str(elevated)},
        {"Metric": "High Narrative Categories", "Value": str(high)},
    ]

    table = markdown_table_from_records(rows, ["Metric", "Value"])

    if fallback >= 1:
        note = (
            "The category-level Narrative AI feed is currently using the general GDELT news-risk fallback. "
            "This keeps the system from treating missing category data as falsely neutral, but the category "
            "breakdown should be improved in a future data-source upgrade."
        )
    else:
        note = (
            "Category-level Narrative AI data is available, so the narrative breakdown is based on thematic article signals."
        )

    return f"""## Narrative AI Summary

{table}

**Narrative Data Note:** {note}
"""


def build_event_study_section(event_insights: pd.DataFrame, event_results: pd.DataFrame) -> str:
    if event_insights.empty:
        return """## Event Study Highlights

_No event study insight file found. Run `python scripts/05_event_study.py` first._
"""

    insights = event_insights.copy()

    if "avg_btc_return_7d" in insights.columns:
        insights["avg_btc_return_7d_num"] = pd.to_numeric(
            insights["avg_btc_return_7d"],
            errors="coerce",
        )
    else:
        insights["avg_btc_return_7d_num"] = np.nan

    top_adverse = insights.sort_values("avg_btc_return_7d_num", ascending=True).head(8)

    rows = []

    for _, row in top_adverse.iterrows():
        rows.append(
            {
                "Event Type": row.get("event_type", ""),
                "Count": safe_int(row.get("count")),
                "Avg BTC 7D": fmt_pct_decimal(safe_float(row.get("avg_btc_return_7d"))),
                "Adverse Rate": fmt_pct_decimal(safe_float(row.get("adverse_rate_7d"))),
                "Signal Quality": row.get("signal_quality", ""),
            }
        )

    table = markdown_table_from_records(
        rows,
        ["Event Type", "Count", "Avg BTC 7D", "Adverse Rate", "Signal Quality"],
    )

    total_events = len(event_results) if not event_results.empty else safe_int(insights["count"].sum())

    return f"""## Event Study Highlights

Total complete events studied: **{total_events}**

{table}

**Interpretation:** The event-study module separates stress signals that historically preceded further weakness from signals that appeared closer to relief/recovery periods.
"""


def build_lag_shock_section(lag_results: pd.DataFrame, shock_results: pd.DataFrame) -> str:
    if lag_results.empty and shock_results.empty:
        return """## Lag & Shock Analysis Highlights

_No lag/shock files found. Run `python scripts/06_lag_shock_analysis.py` first._
"""

    lines = ["## Lag & Shock Analysis Highlights", ""]

    if not lag_results.empty:
        lag = lag_results.copy()

        required = {"target_asset", "forward_horizon_days", "pearson_corr"}
        if required.issubset(set(lag.columns)):
            btc_7d = lag[
                (lag["target_asset"] == "btc")
                & (lag["forward_horizon_days"] == 7)
            ].copy()

            top_negative = btc_7d.sort_values("pearson_corr", ascending=True).head(8)

            rows = []
            for _, row in top_negative.iterrows():
                rows.append(
                    {
                        "Predictor": row.get("predictor", ""),
                        "Lag": safe_int(row.get("lag_days")),
                        "Pearson": fmt_score(safe_float(row.get("pearson_corr"))),
                        "Spearman": fmt_score(safe_float(row.get("spearman_corr"))),
                        "Relationship": row.get("relationship", ""),
                    }
                )

            lines.extend(
                [
                    "### Strongest Negative Relationships with BTC 7-Day Forward Return",
                    "",
                    markdown_table_from_records(
                        rows,
                        ["Predictor", "Lag", "Pearson", "Spearman", "Relationship"],
                    ),
                    "",
                ]
            )

    if not shock_results.empty:
        shock = shock_results.copy()

        if "horizon_days" in shock.columns:
            shock_7d = shock[shock["horizon_days"] == 7].copy()
        else:
            shock_7d = shock.copy()

        shock_7d["avg_btc_return_num"] = pd.to_numeric(
            shock_7d.get("avg_btc_return"),
            errors="coerce",
        )

        most_adverse = shock_7d.sort_values("avg_btc_return_num", ascending=True).head(10)

        rows = []
        for _, row in most_adverse.iterrows():
            rows.append(
                {
                    "Shock": row.get("shock_name", ""),
                    "Events": safe_int(row.get("event_count")),
                    "Avg BTC 7D": fmt_pct_decimal(safe_float(row.get("avg_btc_return"))),
                    "BTC Adverse Rate": fmt_pct_decimal(safe_float(row.get("btc_adverse_rate"))),
                    "Avg Risk Δ": fmt_score(safe_float(row.get("avg_risk_score_change"))),
                }
            )

        lines.extend(
            [
                "### Most Adverse 7-Day Shock Types",
                "",
                markdown_table_from_records(
                    rows,
                    ["Shock", "Events", "Avg BTC 7D", "BTC Adverse Rate", "Avg Risk Δ"],
                ),
                "",
            ]
        )

    lines.append(
        "**Interpretation:** Lag/shock analysis provides diagnostic evidence about which stress indicators are most associated with future weakness."
    )

    return "\n".join(lines)


def build_var_section(
    btc_robustness: pd.DataFrame,
    eth_robustness: pd.DataFrame,
    irf_robustness: pd.DataFrame,
) -> str:
    if btc_robustness.empty and eth_robustness.empty:
        return """## VAR Robustness Highlights

_No VAR robustness files found. Run `python scripts/07b_var_robustness.py` first._
"""

    lines = ["## VAR Robustness Highlights", ""]

    if not btc_robustness.empty:
        btc_rows = []

        for _, row in btc_robustness.iterrows():
            btc_rows.append(
                {
                    "Causing Variable": row.get("causing_variable", ""),
                    "Robustness": row.get("robustness", ""),
                    "Significant Count": safe_int(row.get("significant_count")),
                    "Borderline Count": safe_int(row.get("borderline_count")),
                }
            )

        lines.extend(
            [
                "### BTC Return Equation",
                "",
                markdown_table_from_records(
                    btc_rows,
                    ["Causing Variable", "Robustness", "Significant Count", "Borderline Count"],
                ),
                "",
            ]
        )

    if not eth_robustness.empty:
        eth_rows = []

        for _, row in eth_robustness.iterrows():
            eth_rows.append(
                {
                    "Causing Variable": row.get("causing_variable", ""),
                    "Robustness": row.get("robustness", ""),
                    "Significant Count": safe_int(row.get("significant_count")),
                    "Borderline Count": safe_int(row.get("borderline_count")),
                }
            )

        lines.extend(
            [
                "### ETH Return Equation",
                "",
                markdown_table_from_records(
                    eth_rows,
                    ["Causing Variable", "Robustness", "Significant Count", "Borderline Count"],
                ),
                "",
            ]
        )

    if not irf_robustness.empty:
        irf = irf_robustness.copy()

        if "avg_abs_irf" in irf.columns:
            irf["avg_abs_irf_num"] = pd.to_numeric(irf["avg_abs_irf"], errors="coerce")
            top_irf = irf.sort_values("avg_abs_irf_num", ascending=False).head(10)
        else:
            top_irf = irf.head(10)

        rows = []
        for _, row in top_irf.iterrows():
            rows.append(
                {
                    "Response": row.get("response", ""),
                    "Impulse": row.get("impulse", ""),
                    "Step": safe_int(row.get("step")),
                    "Avg IRF": fmt_score(safe_float(row.get("avg_irf"))),
                    "Direction Consistent": str(row.get("direction_consistent", "")),
                }
            )

        lines.extend(
            [
                "### Top Robust Impulse Responses",
                "",
                markdown_table_from_records(
                    rows,
                    ["Response", "Impulse", "Step", "Avg IRF", "Direction Consistent"],
                ),
                "",
            ]
        )

    lines.append(
        "**Interpretation:** The VAR robustness check tests whether dynamic relationships survive across different lag specifications."
    )

    return "\n".join(lines)


def build_project_strengths_section() -> str:
    return """## Project Strengths

- Combines **macro variables**, **cross-asset confirmation**, **crypto market stress**, **liquidity**, **Narrative AI**, **anomaly detection**, and **machine learning** into one research pipeline.
- Uses multiple validation layers: daily regime report, event study, lag/shock analysis, VAR Lite, VAR robustness, and ablation testing.
- Keeps the system interpretable by separating rule-based scoring from model-adjusted probabilities.
- Uses an Ablation-Selected Risk Classifier instead of blindly relying on the largest feature set.
- Avoids treating missing category-level narrative data as neutral by using a transparent fallback flag.
- Produces research outputs that can be cited in a README, portfolio, or technical interview.
"""


def build_limitations_section() -> str:
    return """## Current Limitations

- This is a research system, not a trading bot and not a trading recommendation engine.
- Narrative AI currently uses a general GDELT fallback when category-level article data is unavailable.
- VAR and correlation analyses show statistical associations, not true economic causality.
- Some macro variables are slow-moving and may be revised after initial release.
- Crypto regimes are highly unstable; model performance should be evaluated through continued walk-forward testing before any live use.
"""


def build_next_steps_section() -> str:
    return """## Recommended Next Steps

1. **Fix category-level Narrative AI retrieval** so that Fed, inflation, regulation, exchange, stablecoin, ETF, and geopolitical narratives are separated with real article counts.
2. **Add dashboard visualizations** for risk scores, event-study results, ablation results, and VAR impulse responses.
3. **Add walk-forward backtesting** to test whether regime changes would have improved simulated risk management.
4. **Create a README and GitHub packaging layer** explaining methodology, folder structure, and research findings.
5. **Add model cards** describing assumptions, limitations, and safe use boundaries.
"""


def build_key_findings_json(
    latest: pd.Series,
    model_report: dict,
    event_insights: pd.DataFrame,
    shock_results: pd.DataFrame,
    btc_robustness: pd.DataFrame,
    eth_robustness: pd.DataFrame,
) -> dict:
    findings: dict = {}

    findings["latest_regime"] = {
        "date": str(pd.to_datetime(latest.get("date")).date()),
        "market_regime": str(latest.get("market_regime", "Unknown")),
        "final_risk_score_with_model": safe_float(latest.get("final_risk_score_with_model")),
        "rule_based_risk_score": safe_float(latest.get("rule_based_risk_score")),
        "model_crisis_probability": safe_float(latest.get("model_crisis_probability")),
    }

    findings["model_architecture"] = {
        "classifier_name": model_report.get("classifier_name", "Unknown"),
        "classifier_feature_set": model_report.get("classifier_feature_set", "Unknown"),
        "excluded_from_classifier": model_report.get("excluded_from_classifier", ""),
        "tuned_threshold": model_report.get("tuned_threshold"),
        "reason_for_exclusion": model_report.get("reason_for_exclusion", ""),
        "macro_engine_usage_note": model_report.get("macro_engine_usage_note", ""),
        "test_metrics_tuned_threshold": model_report.get("test_metrics_tuned_threshold", {}),
    }

    if not event_insights.empty and "avg_btc_return_7d" in event_insights.columns:
        tmp = event_insights.copy()
        tmp["avg_btc_return_7d"] = pd.to_numeric(tmp["avg_btc_return_7d"], errors="coerce")
        top = tmp.sort_values("avg_btc_return_7d", ascending=True).head(5)

        findings["most_adverse_event_types"] = top[
            [
                c
                for c in [
                    "event_type",
                    "count",
                    "avg_btc_return_7d",
                    "adverse_rate_7d",
                    "signal_quality",
                ]
                if c in top.columns
            ]
        ].to_dict(orient="records")

    if not shock_results.empty and "avg_btc_return" in shock_results.columns:
        tmp = shock_results.copy()

        if "horizon_days" in tmp.columns:
            tmp = tmp[tmp["horizon_days"] == 7].copy()

        tmp["avg_btc_return"] = pd.to_numeric(tmp["avg_btc_return"], errors="coerce")
        top = tmp.sort_values("avg_btc_return", ascending=True).head(5)

        findings["most_adverse_shocks_7d"] = top[
            [
                c
                for c in [
                    "shock_name",
                    "event_count",
                    "avg_btc_return",
                    "btc_adverse_rate",
                    "avg_risk_score_change",
                ]
                if c in top.columns
            ]
        ].to_dict(orient="records")

    if not btc_robustness.empty:
        findings["btc_var_robustness"] = btc_robustness[
            [
                c
                for c in [
                    "causing_variable",
                    "robustness",
                    "significant_count",
                    "borderline_count",
                ]
                if c in btc_robustness.columns
            ]
        ].to_dict(orient="records")

    if not eth_robustness.empty:
        findings["eth_var_robustness"] = eth_robustness[
            [
                c
                for c in [
                    "causing_variable",
                    "robustness",
                    "significant_count",
                    "borderline_count",
                ]
                if c in eth_robustness.columns
            ]
        ].to_dict(orient="records")

    return findings


def build_dashboard(root: Path) -> tuple[str, pd.DataFrame, dict]:
    scored = read_csv_if_exists(
        root / "data/processed/scored_regime_history.csv",
        parse_dates=["date"],
    )

    latest = get_latest_row(scored)
    snapshot = build_latest_snapshot(latest)

    model_report = read_json_if_exists(
        root / "outputs/scores/baseline_model_report.json"
    )

    event_insights = read_csv_if_exists(
        root / "outputs/event_studies/event_study_insights_by_type.csv"
    )

    event_results = read_csv_if_exists(
        root / "outputs/event_studies/event_study_results.csv"
    )

    lag_results = read_csv_if_exists(
        root / "outputs/econometrics/lag_correlation_results.csv"
    )

    shock_results = read_csv_if_exists(
        root / "outputs/econometrics/shock_analysis_results.csv"
    )

    btc_robustness = read_csv_if_exists(
        root / "outputs/econometrics/var_robustness/var_robustness_btc_granger.csv"
    )

    eth_robustness = read_csv_if_exists(
        root / "outputs/econometrics/var_robustness/var_robustness_eth_granger.csv"
    )

    irf_robustness = read_csv_if_exists(
        root / "outputs/econometrics/var_robustness/var_robustness_irf.csv"
    )

    report_sections = [
        "# Crypto Macro Crisis Radar — Research Dashboard",
        "",
        "This dashboard summarizes the latest research outputs from the crypto macro crisis radar pipeline.",
        "",
        build_latest_regime_section(latest),
        build_model_architecture_section(model_report, latest),
        build_narrative_section(latest),
        build_event_study_section(event_insights, event_results),
        build_lag_shock_section(lag_results, shock_results),
        build_var_section(btc_robustness, eth_robustness, irf_robustness),
        build_project_strengths_section(),
        build_limitations_section(),
        build_next_steps_section(),
    ]

    dashboard = "\n\n".join(report_sections)

    findings = build_key_findings_json(
        latest=latest,
        model_report=model_report,
        event_insights=event_insights,
        shock_results=shock_results,
        btc_robustness=btc_robustness,
        eth_robustness=eth_robustness,
    )

    return dashboard, snapshot, findings


def save_dashboard_outputs(root: Path) -> dict[str, str]:
    dashboard, snapshot, findings = build_dashboard(root)

    output_dir = root / "outputs/dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard_path = output_dir / "crypto_macro_research_dashboard.md"
    snapshot_path = output_dir / "latest_snapshot.csv"
    findings_path = output_dir / "key_findings.json"

    dashboard_path.write_text(dashboard)
    snapshot.to_csv(snapshot_path, index=False)

    with open(findings_path, "w") as f:
        json.dump(findings, f, indent=2)

    return {
        "dashboard": str(dashboard_path),
        "snapshot": str(snapshot_path),
        "findings": str(findings_path),
    }