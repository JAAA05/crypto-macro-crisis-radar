from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.event_study.event_insights import build_enhanced_markdown_report
from src.event_study.event_study_engine import (
    EventStudyConfig,
    combine_events,
    compute_event_study,
    detect_auto_events,
    load_manual_events,
    summarize_event_study,
)


def ensure_manual_event_template(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    template = pd.DataFrame(
        [
            {
                "date": "2022-11-08",
                "event_type": "manual_crypto_event",
                "event_name": "Example: FTX-related market stress",
                "event_source": "manual_template",
                "notes": "Replace or delete this row. Add CPI/FOMC/ETF/regulatory events here.",
            }
        ]
    )

    template.to_csv(path, index=False)


def filter_auto_events(
    events: pd.DataFrame,
    include_regime_transitions: bool = False,
) -> pd.DataFrame:
    """Clean automatic events before running event study.

    By default, regime_transition is excluded because it can produce too many
    repeated events and dominate the event study.
    """
    if events.empty:
        return events

    clean = events.copy()

    if not include_regime_transitions:
        clean = clean[clean["event_type"] != "regime_transition"].copy()

    clean = (
        clean.drop_duplicates(["date", "event_type", "event_name"])
        .sort_values(["date", "event_type"])
        .reset_index(drop=True)
    )

    return clean


def drop_incomplete_event_windows(
    results: pd.DataFrame,
    required_post_window: int = 7,
) -> pd.DataFrame:
    """Remove events that do not have enough future data.

    Example:
    If required_post_window=7, the event must have BTC return t0→t+7.
    """
    if results.empty:
        return results

    required_col = f"btc_return_t0_to_t+{required_post_window}"

    if required_col not in results.columns:
        print(f"[WARN] Required post-window column not found: {required_col}")
        return results

    before = len(results)
    clean = results.dropna(subset=[required_col]).copy()
    after = len(clean)

    print(
        f"Dropped incomplete events without t+{required_post_window} BTC return: "
        f"{before - after:,}"
    )

    return clean.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Event Study Engine v1.")
    parser.add_argument(
        "--manual-events",
        default="data/events/manual_events.csv",
        help="Path to optional manual event CSV.",
    )
    parser.add_argument(
        "--no-template",
        action="store_true",
        help="Do not create a manual event template if missing.",
    )
    parser.add_argument(
        "--min-gap-days",
        type=int,
        default=30,
        help="Minimum number of days between repeated auto events of the same type.",
    )
    parser.add_argument(
        "--include-regime-transitions",
        action="store_true",
        help="Include automatic regime_transition events. Default is to exclude them.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include events without enough future data for t+7 windows.",
    )
    parser.add_argument(
        "--required-post-window",
        type=int,
        default=7,
        help="Required future window for complete events. Default requires t+7.",
    )

    args = parser.parse_args()

    scored_path = ROOT / "data/processed/scored_regime_history.csv"

    if not scored_path.exists():
        raise FileNotFoundError(
            "data/processed/scored_regime_history.csv not found. "
            "Run scripts/run_pipeline.py --skip-data first."
        )

    manual_events_path = ROOT / args.manual_events

    if not args.no_template:
        ensure_manual_event_template(manual_events_path)

    print("Loading scored regime history...")
    scored = pd.read_csv(scored_path, parse_dates=["date"])
    scored = scored.sort_values("date").reset_index(drop=True)

    config = EventStudyConfig(
        min_gap_days_between_auto_events=int(args.min_gap_days),
    )

    print("Detecting automatic events...")
    auto_events = detect_auto_events(scored, config=config)
    print(f"Detected raw auto events: {len(auto_events):,}")

    auto_events = filter_auto_events(
        auto_events,
        include_regime_transitions=bool(args.include_regime_transitions),
    )

    print(f"Auto events after cleaning: {len(auto_events):,}")

    print("Loading manual events...")
    manual_events = load_manual_events(manual_events_path)

    if not manual_events.empty:
        manual_events = manual_events[
            manual_events["event_name"] != "Example: FTX-related market stress"
        ].copy()

    print(f"Manual events loaded: {len(manual_events):,}")

    events = combine_events(auto_events, manual_events)

    if events.empty:
        print("No events found after cleaning.")
        return

    print(f"Total events before event-window filtering: {len(events):,}")

    outputs_dir = ROOT / "outputs/event_studies"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    event_list_path = outputs_dir / "event_list.csv"
    events.to_csv(event_list_path, index=False)

    print("Computing event study metrics...")
    results = compute_event_study(scored, events, config=config)

    if not args.include_incomplete:
        results = drop_incomplete_event_windows(
            results,
            required_post_window=int(args.required_post_window),
        )

    if results.empty:
        print("No complete event-study results after filtering.")
        return

    detailed_path = outputs_dir / "event_study_results.csv"
    results.to_csv(detailed_path, index=False)

    summary = summarize_event_study(results)
    summary_path = outputs_dir / "event_study_summary_by_type.csv"
    summary.to_csv(summary_path, index=False)

    report_path = outputs_dir / "event_study_report.md"
    insights_path = outputs_dir / "event_study_insights_by_type.csv"

    report, insights = build_enhanced_markdown_report(
        results=results,
        output_csv_name="outputs/event_studies/event_study_results.csv",
        summary_csv_name="outputs/event_studies/event_study_summary_by_type.csv",
        insights_csv_name="outputs/event_studies/event_study_insights_by_type.csv",
    )

    insights.to_csv(insights_path, index=False)
    report_path.write_text(report)

    print(report)
    print(f"\nSaved event list to: {event_list_path}")
    print(f"Saved detailed results to: {detailed_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved insights to: {insights_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()