from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.econometrics.var_model import (
    VarLiteConfig,
    extract_irf,
    fit_var_model,
    make_forecast,
    prepare_var_dataset,
    run_granger_tests,
    select_lag_order,
    standardize_var_data,
    summarize_var_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VAR Lite v1 analysis.")
    parser.add_argument(
        "--input",
        default="data/processed/scored_regime_history.csv",
        help="Path to scored regime history CSV.",
    )
    parser.add_argument(
        "--maxlags",
        type=int,
        default=14,
        help="Maximum lag order to test.",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=14,
        help="Forecast steps.",
    )
    parser.add_argument(
        "--irf-steps",
        type=int,
        default=14,
        help="Impulse response steps.",
    )

    args = parser.parse_args()

    input_path = ROOT / args.input

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run scripts/run_pipeline.py --skip-data first."
        )

    outputs_dir = ROOT / "outputs/econometrics"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scored regime history...")
    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    config = VarLiteConfig(
        maxlags=int(args.maxlags),
        forecast_steps=int(args.forecast_steps),
        irf_steps=int(args.irf_steps),
    )

    print("Preparing VAR dataset...")
    var_raw = prepare_var_dataset(df)

    if len(var_raw) < config.min_rows:
        raise ValueError(
            f"Not enough rows for VAR Lite. Rows={len(var_raw)}, required={config.min_rows}"
        )

    var_raw_path = outputs_dir / "var_lite_dataset_raw.csv"
    var_raw.to_csv(var_raw_path, index=False)

    print("Standardizing VAR dataset...")
    var_data, scaling = standardize_var_data(var_raw)

    var_data_path = outputs_dir / "var_lite_dataset_standardized.csv"
    scaling_path = outputs_dir / "var_lite_scaling.csv"

    var_data.to_csv(var_data_path, index=False)
    scaling.to_csv(scaling_path, index=False)

    print("Selecting lag order...")
    lag_order, lag_selection = select_lag_order(
        data=var_data,
        maxlags=config.maxlags,
    )

    lag_selection_path = outputs_dir / "var_lag_order_selection.csv"
    lag_selection.to_csv(lag_selection_path, index=False)

    print(f"Selected lag order: {lag_order}")

    print("Fitting VAR model...")
    fitted = fit_var_model(
        data=var_data,
        lag_order=lag_order,
    )

    print("Running Granger-style causality tests...")
    granger_btc = run_granger_tests(fitted, target="btc_ret_1d")
    granger_eth = run_granger_tests(fitted, target="eth_ret_1d")

    granger_btc_path = outputs_dir / "var_granger_btc.csv"
    granger_eth_path = outputs_dir / "var_granger_eth.csv"

    granger_btc.to_csv(granger_btc_path, index=False)
    granger_eth.to_csv(granger_eth_path, index=False)

    print("Extracting impulse responses...")
    irf_df = extract_irf(
        fitted,
        steps=config.irf_steps,
    )

    irf_path = outputs_dir / "var_impulse_responses.csv"
    irf_df.to_csv(irf_path, index=False)

    print("Making VAR forecast...")
    forecast_df = make_forecast(
        fitted,
        data=var_data,
        steps=config.forecast_steps,
    )

    forecast_path = outputs_dir / "var_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    report = summarize_var_results(
        lag_order=lag_order,
        fitted_model=fitted,
        granger_btc=granger_btc,
        granger_eth=granger_eth,
        irf_df=irf_df,
        forecast_df=forecast_df,
    )

    report_path = outputs_dir / "var_lite_report.md"
    report_path.write_text(report)

    print(report)

    print(f"\nSaved raw VAR dataset to: {var_raw_path}")
    print(f"Saved standardized VAR dataset to: {var_data_path}")
    print(f"Saved scaling table to: {scaling_path}")
    print(f"Saved lag selection to: {lag_selection_path}")
    print(f"Saved BTC Granger tests to: {granger_btc_path}")
    print(f"Saved ETH Granger tests to: {granger_eth_path}")
    print(f"Saved impulse responses to: {irf_path}")
    print(f"Saved VAR forecast to: {forecast_path}")
    print(f"Saved VAR report to: {report_path}")


if __name__ == "__main__":
    main()