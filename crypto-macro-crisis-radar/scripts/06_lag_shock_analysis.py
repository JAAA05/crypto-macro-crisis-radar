from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.econometrics.lag_shock_analysis import (
    LagShockConfig,
    analyze_lag_correlations,
    analyze_shocks,
    build_lag_shock_markdown_report,
    prepare_forward_returns,
)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lag & Shock Analysis v1.")
    parser.add_argument(
        "--input",
        default="data/processed/scored_regime_history.csv",
        help="Path to scored regime history CSV.",
    )
    parser.add_argument(
        "--horizons",
        default="1,3,7,14",
        help="Forward return horizons in days, comma-separated.",
    )
    parser.add_argument(
        "--lags",
        default="0,1,3,7,14,30",
        help="Predictor lag days, comma-separated.",
    )
    parser.add_argument(
        "--min-obs-corr",
        type=int,
        default=250,
        help="Minimum observations required for correlation results.",
    )
    parser.add_argument(
        "--min-events-shock",
        type=int,
        default=3,
        help="Minimum shock events required for shock summary.",
    )
    parser.add_argument(
        "--min-gap-days",
        type=int,
        default=14,
        help="Minimum index-day gap between shock episodes.",
    )

    args = parser.parse_args()

    input_path = ROOT / args.input

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run scripts/run_pipeline.py --skip-data first."
        )

    print("Loading scored regime history...")
    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    config = LagShockConfig(
        horizons=parse_int_tuple(args.horizons),
        lags=parse_int_tuple(args.lags),
        min_obs_corr=int(args.min_obs_corr),
        min_events_shock=int(args.min_events_shock),
        min_gap_days_between_shocks=int(args.min_gap_days),
    )

    print(f"Horizons: {config.horizons}")
    print(f"Lags: {config.lags}")

    print("Preparing forward returns...")
    df = prepare_forward_returns(df, config=config)

    outputs_dir = ROOT / "outputs/econometrics"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Running lag correlation analysis...")
    lag_results = analyze_lag_correlations(df, config=config)

    lag_path = outputs_dir / "lag_correlation_results.csv"
    lag_results.to_csv(lag_path, index=False)
    print(f"Saved lag correlations to {lag_path}")

    print("Running shock response analysis...")
    shock_results = analyze_shocks(df, config=config)

    shock_path = outputs_dir / "shock_analysis_results.csv"
    shock_results.to_csv(shock_path, index=False)
    print(f"Saved shock analysis to {shock_path}")

    report = build_lag_shock_markdown_report(
        lag_results=lag_results,
        shock_results=shock_results,
        lag_csv_name="outputs/econometrics/lag_correlation_results.csv",
        shock_csv_name="outputs/econometrics/shock_analysis_results.csv",
    )

    report_path = outputs_dir / "lag_shock_report.md"
    report_path.write_text(report)

    print(report)
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()