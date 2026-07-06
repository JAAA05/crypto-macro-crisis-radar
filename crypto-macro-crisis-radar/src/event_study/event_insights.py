from __future__ import annotations

import numpy as np
import pandas as pd


def safe_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def fmt_num(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def classify_signal_quality(
    avg_btc_7d: float,
    avg_risk_change_7d: float,
    adverse_rate: float,
    stress_continuation_rate: float,
) -> str:
    """Simple descriptive label for event-type behavior."""
    if pd.isna(avg_btc_7d) or pd.isna(avg_risk_change_7d):
        return "Insufficient data"

    if avg_btc_7d < -0.02 and avg_risk_change_7d > 1.0:
        return "Adverse continuation"

    if avg_btc_7d < -0.01 and adverse_rate >= 0.55:
        return "Weakness-prone"

    if avg_btc_7d > 0.02 and avg_risk_change_7d < -1.0:
        return "Relief / mean-reversion"

    if avg_risk_change_7d > 2.0 and stress_continuation_rate >= 0.55:
        return "Stress-building"

    if abs(avg_btc_7d) < 0.01 and abs(avg_risk_change_7d) < 1.0:
        return "Mixed / neutral"

    return "Mixed"


def build_event_type_insights(results: pd.DataFrame) -> pd.DataFrame:
    """Create an interpretable event-type summary.

    Main metrics:
    - avg BTC/ETH return after 7 days
    - avg risk-score change after 7 days
    - adverse rate: share of events followed by negative BTC 7D return
    - stress continuation rate: share of events followed by higher risk score
    """
    if results.empty:
        return pd.DataFrame()

    required_cols = [
        "event_type",
        "btc_return_t0_to_t+7",
        "eth_return_t0_to_t+7",
        "risk_score_change_t0_to_t+7",
    ]

    for col in required_cols:
        if col not in results.columns:
            raise ValueError(f"Missing required column for event insights: {col}")

    df = results.copy()

    df["btc_7d_negative"] = df["btc_return_t0_to_t+7"] < 0
    df["risk_score_7d_higher"] = df["risk_score_change_t0_to_t+7"] > 0

    grouped = (
        df.groupby("event_type")
        .agg(
            count=("event_type", "size"),
            avg_btc_return_7d=("btc_return_t0_to_t+7", "mean"),
            median_btc_return_7d=("btc_return_t0_to_t+7", "median"),
            avg_eth_return_7d=("eth_return_t0_to_t+7", "mean"),
            median_eth_return_7d=("eth_return_t0_to_t+7", "median"),
            avg_risk_score_change_7d=("risk_score_change_t0_to_t+7", "mean"),
            median_risk_score_change_7d=("risk_score_change_t0_to_t+7", "median"),
            adverse_rate_7d=("btc_7d_negative", "mean"),
            stress_continuation_rate_7d=("risk_score_7d_higher", "mean"),
        )
        .reset_index()
    )

    grouped["signal_quality"] = grouped.apply(
        lambda row: classify_signal_quality(
            avg_btc_7d=safe_float(row["avg_btc_return_7d"]),
            avg_risk_change_7d=safe_float(row["avg_risk_score_change_7d"]),
            adverse_rate=safe_float(row["adverse_rate_7d"]),
            stress_continuation_rate=safe_float(row["stress_continuation_rate_7d"]),
        ),
        axis=1,
    )

    grouped = grouped.sort_values(
        ["avg_btc_return_7d", "avg_risk_score_change_7d"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return grouped


def get_top_adverse_events(results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if results.empty or "btc_return_t0_to_t+7" not in results.columns:
        return pd.DataFrame()

    cols = [
        "event_date",
        "event_type",
        "event_name",
        "market_regime_at_event",
        "btc_return_t0_to_t+7",
        "eth_return_t0_to_t+7",
        "risk_score_change_t0_to_t+7",
        "final_risk_score_at_event",
    ]

    existing = [c for c in cols if c in results.columns]

    return (
        results.dropna(subset=["btc_return_t0_to_t+7"])
        .sort_values("btc_return_t0_to_t+7", ascending=True)
        .head(n)[existing]
        .reset_index(drop=True)
    )


def get_top_relief_events(results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if results.empty or "btc_return_t0_to_t+7" not in results.columns:
        return pd.DataFrame()

    cols = [
        "event_date",
        "event_type",
        "event_name",
        "market_regime_at_event",
        "btc_return_t0_to_t+7",
        "eth_return_t0_to_t+7",
        "risk_score_change_t0_to_t+7",
        "final_risk_score_at_event",
    ]

    existing = [c for c in cols if c in results.columns]

    return (
        results.dropna(subset=["btc_return_t0_to_t+7"])
        .sort_values("btc_return_t0_to_t+7", ascending=False)
        .head(n)[existing]
        .reset_index(drop=True)
    )


def get_top_risk_build_events(results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if results.empty or "risk_score_change_t0_to_t+7" not in results.columns:
        return pd.DataFrame()

    cols = [
        "event_date",
        "event_type",
        "event_name",
        "market_regime_at_event",
        "btc_return_t0_to_t+7",
        "eth_return_t0_to_t+7",
        "risk_score_change_t0_to_t+7",
        "final_risk_score_at_event",
    ]

    existing = [c for c in cols if c in results.columns]

    return (
        results.dropna(subset=["risk_score_change_t0_to_t+7"])
        .sort_values("risk_score_change_t0_to_t+7", ascending=False)
        .head(n)[existing]
        .reset_index(drop=True)
    )


def add_markdown_event_rows(
    lines: list[str],
    table: pd.DataFrame,
    title: str,
    sort_note: str,
) -> None:
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            sort_note,
            "",
            "| Event Date | Type | Name | Regime | BTC t0→t+7 | ETH t0→t+7 | Risk Score t0→t+7 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )

    if table.empty:
        lines.append("|  |  | No events available |  |  |  |  |")
        return

    for _, row in table.iterrows():
        btc_7d = safe_float(row.get("btc_return_t0_to_t+7"))
        eth_7d = safe_float(row.get("eth_return_t0_to_t+7"))
        risk_7d = safe_float(row.get("risk_score_change_t0_to_t+7"))

        lines.append(
            f"| {row.get('event_date')} | "
            f"{row.get('event_type')} | "
            f"{row.get('event_name')} | "
            f"{row.get('market_regime_at_event')} | "
            f"{fmt_pct(btc_7d)} | "
            f"{fmt_pct(eth_7d)} | "
            f"{fmt_num(risk_7d)} |"
        )


def build_enhanced_markdown_report(
    results: pd.DataFrame,
    output_csv_name: str,
    summary_csv_name: str,
    insights_csv_name: str,
) -> tuple[str, pd.DataFrame]:
    if results.empty:
        report = """# Event Study Report

No complete events were found. Try lowering thresholds, using `--include-incomplete`, or adding manual events to `data/events/manual_events.csv`.
"""
        return report, pd.DataFrame()

    insights = build_event_type_insights(results)

    total_events = len(results)

    event_counts = results["event_type"].value_counts().reset_index()
    event_counts.columns = ["event_type", "count"]

    latest_events = results.sort_values("event_date").tail(10)

    top_adverse = get_top_adverse_events(results, n=10)
    top_relief = get_top_relief_events(results, n=10)
    top_risk_build = get_top_risk_build_events(results, n=10)

    lines = [
        "# Event Study Report",
        "",
        "This report evaluates how BTC, ETH, and risk scores behaved around detected or manually provided events.",
        "",
        "## Files Generated",
        "",
        f"- Detailed results: `{output_csv_name}`",
        f"- Event type summary: `{summary_csv_name}`",
        f"- Event insights: `{insights_csv_name}`",
        "",
        "## Event Count",
        "",
        f"Total complete events studied: **{total_events}**",
        "",
        "| Event Type | Count |",
        "|---|---:|",
    ]

    for _, row in event_counts.iterrows():
        lines.append(f"| {row['event_type']} | {int(row['count'])} |")

    lines.extend(
        [
            "",
            "## Event Type Performance Summary",
            "",
            "| Event Type | Count | Avg BTC 7D | Median BTC 7D | Avg ETH 7D | Avg Risk Δ 7D | Adverse Rate | Stress Continuation Rate | Signal Quality |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for _, row in insights.iterrows():
        lines.append(
            f"| {row['event_type']} | "
            f"{int(row['count'])} | "
            f"{fmt_pct(safe_float(row['avg_btc_return_7d']))} | "
            f"{fmt_pct(safe_float(row['median_btc_return_7d']))} | "
            f"{fmt_pct(safe_float(row['avg_eth_return_7d']))} | "
            f"{fmt_num(safe_float(row['avg_risk_score_change_7d']))} | "
            f"{fmt_pct(safe_float(row['adverse_rate_7d']))} | "
            f"{fmt_pct(safe_float(row['stress_continuation_rate_7d']))} | "
            f"{row['signal_quality']} |"
        )

    lines.extend(
        [
            "",
            "## Latest Events",
            "",
            "| Event Date | Type | Name | Regime | BTC t0→t+7 | Risk Score t0→t+7 |",
            "|---|---|---|---|---:|---:|",
        ]
    )

    for _, row in latest_events.iterrows():
        btc_7d = safe_float(row.get("btc_return_t0_to_t+7"))
        risk_7d = safe_float(row.get("risk_score_change_t0_to_t+7"))

        lines.append(
            f"| {row.get('event_date')} | {row.get('event_type')} | {row.get('event_name')} | "
            f"{row.get('market_regime_at_event')} | {fmt_pct(btc_7d)} | {fmt_num(risk_7d)} |"
        )

    add_markdown_event_rows(
        lines=lines,
        table=top_adverse,
        title="Top Adverse Events",
        sort_note="Events followed by the weakest BTC 7-day returns.",
    )

    add_markdown_event_rows(
        lines=lines,
        table=top_relief,
        title="Top Relief / Recovery Events",
        sort_note="Events followed by the strongest BTC 7-day returns.",
    )

    add_markdown_event_rows(
        lines=lines,
        table=top_risk_build,
        title="Top Risk-Build Events",
        sort_note="Events followed by the largest increase in model-adjusted risk score.",
    )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- **Adverse Rate** is the share of events followed by a negative BTC 7-day return.",
            "- **Stress Continuation Rate** is the share of events followed by a higher model-adjusted risk score after 7 days.",
            "- Positive BTC/ETH post-event returns suggest recovery or relief after an event.",
            "- Negative BTC/ETH post-event returns suggest the event was followed by market weakness.",
            "- Positive risk-score changes after an event indicate that stress continued rising.",
            "- Negative risk-score changes after an event indicate that stress faded after the event.",
            "",
            "This is a research tool, not a trading recommendation.",
        ]
    )

    report = "\n".join(lines)

    return report, insights