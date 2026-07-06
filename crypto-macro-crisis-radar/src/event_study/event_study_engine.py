from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EventStudyConfig:
    pre_windows: tuple[int, ...] = (-14, -7, -3, -1)
    post_windows: tuple[int, ...] = (1, 3, 7, 14)
    min_gap_days_between_auto_events: int = 7


def safe_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _nearest_available_date(df: pd.DataFrame, event_date: pd.Timestamp) -> pd.Timestamp | None:
    """Return the exact date if available, otherwise the next available date."""
    dates = df["date"]

    exact = df.loc[dates == event_date, "date"]

    if not exact.empty:
        return exact.iloc[0]

    future = df.loc[dates > event_date, "date"]

    if future.empty:
        return None

    return future.iloc[0]


def _get_row_by_date(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    rows = df.loc[df["date"] == date]

    if rows.empty:
        return None

    return rows.iloc[0]


def _get_shifted_row(
    df: pd.DataFrame,
    event_idx: int,
    shift_days: int,
) -> pd.Series | None:
    target_idx = event_idx + shift_days

    if target_idx < 0 or target_idx >= len(df):
        return None

    return df.iloc[target_idx]


def _return_between(
    event_row: pd.Series,
    shifted_row: pd.Series | None,
    price_col: str,
) -> float:
    if shifted_row is None:
        return np.nan

    event_price = safe_float(event_row.get(price_col))
    shifted_price = safe_float(shifted_row.get(price_col))

    if pd.isna(event_price) or pd.isna(shifted_price) or event_price == 0:
        return np.nan

    return shifted_price / event_price - 1


def _change_between(
    event_row: pd.Series,
    shifted_row: pd.Series | None,
    col: str,
) -> float:
    if shifted_row is None:
        return np.nan

    event_value = safe_float(event_row.get(col))
    shifted_value = safe_float(shifted_row.get(col))

    if pd.isna(event_value) or pd.isna(shifted_value):
        return np.nan

    return shifted_value - event_value


def _detect_crossing_events(
    df: pd.DataFrame,
    col: str,
    threshold: float,
    direction: str,
    event_type: str,
    event_name: str,
    min_gap_days: int,
) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()

    values = pd.to_numeric(df[col], errors="coerce")

    if direction == "above":
        condition = values >= threshold
    elif direction == "below":
        condition = values <= threshold
    else:
        raise ValueError("direction must be 'above' or 'below'")

    # Only count the first day of a new stress episode.
    previous_condition = condition.shift(1).fillna(False)
    crosses = condition & (~previous_condition)

    events = df.loc[crosses, ["date"]].copy()

    if events.empty:
        return pd.DataFrame()

    events["event_type"] = event_type
    events["event_name"] = event_name
    events["event_source"] = "auto_threshold"
    events["trigger_column"] = col
    events["trigger_threshold"] = threshold
    events["trigger_direction"] = direction

    filtered_rows = []
    last_date: pd.Timestamp | None = None

    for _, row in events.iterrows():
        current_date = pd.to_datetime(row["date"])

        if last_date is None or (current_date - last_date).days >= min_gap_days:
            filtered_rows.append(row)
            last_date = current_date

    if not filtered_rows:
        return pd.DataFrame()

    return pd.DataFrame(filtered_rows).reset_index(drop=True)


def detect_auto_events(
    scored_df: pd.DataFrame,
    config: EventStudyConfig | None = None,
) -> pd.DataFrame:
    """Detect automatic market events from existing model signals.

    This is not a macro release calendar. It detects stress episodes from the
    model's own indicators, which is useful for event-study backtesting.
    """
    config = config or EventStudyConfig()

    df = scored_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    event_specs = [
        {
            "col": "news_risk_score",
            "threshold": 70,
            "direction": "above",
            "event_type": "narrative_shock",
            "event_name": "News / narrative risk shock",
        },
        {
            "col": "anomaly_score",
            "threshold": 75,
            "direction": "above",
            "event_type": "market_anomaly",
            "event_name": "High anomaly score",
        },
        {
            "col": "crypto_stress_score",
            "threshold": 60,
            "direction": "above",
            "event_type": "crypto_stress",
            "event_name": "Crypto market stress shock",
        },
        {
            "col": "liquidity_stress_score",
            "threshold": 60,
            "direction": "above",
            "event_type": "liquidity_stress",
            "event_name": "Liquidity stress shock",
        },
        {
            "col": "cross_asset_risk_score",
            "threshold": 50,
            "direction": "above",
            "event_type": "cross_asset_risk",
            "event_name": "Cross-asset risk confirmation",
        },
        {
            "col": "macro_engine_stress_index",
            "threshold": 60,
            "direction": "above",
            "event_type": "macro_engine_stress",
            "event_name": "Macro Engine stress shock",
        },
        {
            "col": "inflation_pressure_score",
            "threshold": 65,
            "direction": "above",
            "event_type": "inflation_pressure",
            "event_name": "Inflation pressure shock",
        },
        {
            "col": "monetary_tightening_score",
            "threshold": 65,
            "direction": "above",
            "event_type": "monetary_tightening",
            "event_name": "Monetary tightening shock",
        },
        {
            "col": "bitcoin_drawdown_30d",
            "threshold": -0.10,
            "direction": "below",
            "event_type": "btc_drawdown_warning",
            "event_name": "BTC 30D drawdown below -10%",
        },
        {
            "col": "final_risk_score_with_model",
            "threshold": 50,
            "direction": "above",
            "event_type": "risk_off_score",
            "event_name": "Model-adjusted risk score above 50",
        },
    ]

    frames = []

    for spec in event_specs:
        event_frame = _detect_crossing_events(
            df=df,
            col=spec["col"],
            threshold=float(spec["threshold"]),
            direction=str(spec["direction"]),
            event_type=str(spec["event_type"]),
            event_name=str(spec["event_name"]),
            min_gap_days=config.min_gap_days_between_auto_events,
        )

        if not event_frame.empty:
            frames.append(event_frame)

    if "market_regime" in df.columns:
        regime = df["market_regime"].astype(str)
        stress_regime = regime.isin(["Risk-Off", "Mini-Crisis Watch", "Crisis"])
        new_stress_regime = stress_regime & (~stress_regime.shift(1).fillna(False))

        regime_events = df.loc[new_stress_regime, ["date", "market_regime"]].copy()

        if not regime_events.empty:
            regime_events["event_type"] = "regime_transition"
            regime_events["event_name"] = "Transition into Risk-Off or worse"
            regime_events["event_source"] = "auto_regime"
            regime_events["trigger_column"] = "market_regime"
            regime_events["trigger_threshold"] = np.nan
            regime_events["trigger_direction"] = "regime_transition"
            frames.append(regime_events.drop(columns=["market_regime"]))

    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "event_type",
                "event_name",
                "event_source",
                "trigger_column",
                "trigger_threshold",
                "trigger_direction",
            ]
        )

    events = pd.concat(frames, ignore_index=True)
    events["date"] = pd.to_datetime(events["date"])

    events = (
        events.drop_duplicates(["date", "event_type", "event_name"])
        .sort_values(["date", "event_type"])
        .reset_index(drop=True)
    )

    return events


def load_manual_events(events_path: Path) -> pd.DataFrame:
    if not events_path.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "event_type",
                "event_name",
                "event_source",
                "notes",
            ]
        )

    events = pd.read_csv(events_path, parse_dates=["date"])

    required = ["date", "event_type", "event_name"]

    for col in required:
        if col not in events.columns:
            raise ValueError(f"Manual event file is missing required column: {col}")

    if "event_source" not in events.columns:
        events["event_source"] = "manual"

    if "notes" not in events.columns:
        events["notes"] = ""

    events = events[["date", "event_type", "event_name", "event_source", "notes"]].copy()
    events["date"] = pd.to_datetime(events["date"])

    return events.sort_values("date").reset_index(drop=True)


def combine_events(auto_events: pd.DataFrame, manual_events: pd.DataFrame) -> pd.DataFrame:
    frames = []

    if not auto_events.empty:
        frames.append(auto_events)

    if not manual_events.empty:
        manual = manual_events.copy()
        manual["trigger_column"] = ""
        manual["trigger_threshold"] = np.nan
        manual["trigger_direction"] = ""
        frames.append(manual)

    if not frames:
        return pd.DataFrame()

    events = pd.concat(frames, ignore_index=True, sort=False)

    events["date"] = pd.to_datetime(events["date"])
    events["event_source"] = events["event_source"].fillna("unknown")
    events["event_name"] = events["event_name"].fillna(events["event_type"])

    events = (
        events.drop_duplicates(["date", "event_type", "event_name"])
        .sort_values(["date", "event_type"])
        .reset_index(drop=True)
    )

    return events


def compute_event_study(
    scored_df: pd.DataFrame,
    events: pd.DataFrame,
    config: EventStudyConfig | None = None,
) -> pd.DataFrame:
    config = config or EventStudyConfig()

    df = scored_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])

    results = []

    for _, event in events.iterrows():
        requested_event_date = pd.to_datetime(event["date"])
        actual_event_date = _nearest_available_date(df, requested_event_date)

        if actual_event_date is None:
            continue

        event_row = _get_row_by_date(df, actual_event_date)

        if event_row is None:
            continue

        event_idx = int(event_row.name)

        base = {
            "requested_event_date": requested_event_date.date(),
            "event_date": actual_event_date.date(),
            "event_type": event.get("event_type", ""),
            "event_name": event.get("event_name", ""),
            "event_source": event.get("event_source", ""),
            "trigger_column": event.get("trigger_column", ""),
            "trigger_threshold": event.get("trigger_threshold", np.nan),
            "trigger_direction": event.get("trigger_direction", ""),
            "market_regime_at_event": event_row.get("market_regime", ""),
            "btc_price_at_event": safe_float(event_row.get("bitcoin_price")),
            "eth_price_at_event": safe_float(event_row.get("ethereum_price")),
            "final_risk_score_at_event": safe_float(event_row.get("final_risk_score_with_model")),
            "crypto_stress_at_event": safe_float(event_row.get("crypto_stress_score")),
            "liquidity_stress_at_event": safe_float(event_row.get("liquidity_stress_score")),
            "news_risk_at_event": safe_float(event_row.get("news_risk_score")),
            "cross_asset_risk_at_event": safe_float(event_row.get("cross_asset_risk_score")),
            "macro_engine_stress_at_event": safe_float(event_row.get("macro_engine_stress_index")),
            "anomaly_score_at_event": safe_float(event_row.get("anomaly_score")),
        }

        for window in config.pre_windows:
            pre_row = _get_shifted_row(df, event_idx, window)

            base[f"btc_return_t{window}_to_t0"] = (
                safe_float(event_row.get("bitcoin_price")) / safe_float(pre_row.get("bitcoin_price")) - 1
                if pre_row is not None and safe_float(pre_row.get("bitcoin_price")) not in [0, np.nan]
                else np.nan
            )

            base[f"eth_return_t{window}_to_t0"] = (
                safe_float(event_row.get("ethereum_price")) / safe_float(pre_row.get("ethereum_price")) - 1
                if pre_row is not None and safe_float(pre_row.get("ethereum_price")) not in [0, np.nan]
                else np.nan
            )

            base[f"risk_score_change_t{window}_to_t0"] = (
                safe_float(event_row.get("final_risk_score_with_model"))
                - safe_float(pre_row.get("final_risk_score_with_model"))
                if pre_row is not None
                else np.nan
            )

            base[f"news_risk_change_t{window}_to_t0"] = (
                safe_float(event_row.get("news_risk_score"))
                - safe_float(pre_row.get("news_risk_score"))
                if pre_row is not None
                else np.nan
            )

        for window in config.post_windows:
            post_row = _get_shifted_row(df, event_idx, window)

            base[f"btc_return_t0_to_t+{window}"] = _return_between(
                event_row,
                post_row,
                "bitcoin_price",
            )

            base[f"eth_return_t0_to_t+{window}"] = _return_between(
                event_row,
                post_row,
                "ethereum_price",
            )

            base[f"risk_score_change_t0_to_t+{window}"] = _change_between(
                event_row,
                post_row,
                "final_risk_score_with_model",
            )

            base[f"crypto_stress_change_t0_to_t+{window}"] = _change_between(
                event_row,
                post_row,
                "crypto_stress_score",
            )

            base[f"liquidity_stress_change_t0_to_t+{window}"] = _change_between(
                event_row,
                post_row,
                "liquidity_stress_score",
            )

            base[f"news_risk_change_t0_to_t+{window}"] = _change_between(
                event_row,
                post_row,
                "news_risk_score",
            )

        results.append(base)

    return pd.DataFrame(results)


def summarize_event_study(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    numeric_cols = [
        col
        for col in results.columns
        if col.startswith("btc_return_")
        or col.startswith("eth_return_")
        or col.startswith("risk_score_change_")
        or col.startswith("crypto_stress_change_")
        or col.startswith("liquidity_stress_change_")
        or col.startswith("news_risk_change_")
    ]

    summary = (
        results.groupby("event_type")[numeric_cols]
        .agg(["count", "mean", "median"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    return summary


def build_markdown_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_csv_name: str,
    summary_csv_name: str,
) -> str:
    if results.empty:
        return f"""# Event Study Report

No events were found. Try lowering thresholds or adding manual events to `data/events/manual_events.csv`.
"""

    total_events = len(results)
    event_counts = results["event_type"].value_counts().reset_index()
    event_counts.columns = ["event_type", "count"]

    latest_events = results.sort_values("event_date").tail(10)

    lines = [
        "# Event Study Report",
        "",
        "This report evaluates how BTC, ETH, and risk scores behaved around detected or manually provided events.",
        "",
        "## Files Generated",
        "",
        f"- Detailed results: `{output_csv_name}`",
        f"- Event type summary: `{summary_csv_name}`",
        "",
        "## Event Count",
        "",
        f"Total events studied: **{total_events}**",
        "",
        "| Event Type | Count |",
        "|---|---:|",
    ]

    for _, row in event_counts.iterrows():
        lines.append(f"| {row['event_type']} | {int(row['count'])} |")

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

        btc_7d_str = "" if pd.isna(btc_7d) else f"{btc_7d * 100:.2f}%"
        risk_7d_str = "" if pd.isna(risk_7d) else f"{risk_7d:.2f}"

        lines.append(
            f"| {row.get('event_date')} | {row.get('event_type')} | {row.get('event_name')} | "
            f"{row.get('market_regime_at_event')} | {btc_7d_str} | {risk_7d_str} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Positive BTC/ETH post-event returns suggest recovery or relief after an event.",
            "- Negative BTC/ETH post-event returns suggest the event was followed by market weakness.",
            "- Positive risk-score changes after an event indicate that stress continued rising.",
            "- Negative risk-score changes after an event indicate that stress faded after the event.",
            "",
            "This is a research tool, not a trading recommendation.",
        ]
    )

    return "\n".join(lines)