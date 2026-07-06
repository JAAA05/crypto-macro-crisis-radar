from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.econometrics.var_robustness import (
    VarRobustnessConfig,
    run_var_robustness,
)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VAR robustness analysis.")
    parser.add_argument(
        "--input",
        default="data/processed/scored_regime_history.csv",
        help="Path to scored regime history CSV.",
    )
    parser.add_argument(
        "--maxlags-list",
        default="14,7,5",
        help="Comma-separated maxlags values to test.",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=14,
        help="Forecast steps for each VAR run.",
    )
    parser.add_argument(
        "--irf-steps",
        type=int,
        default=14,
        help="Impulse response steps for each VAR run.",
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

    outputs_dir = ROOT / "outputs/econometrics/var_robustness"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    config = VarRobustnessConfig(
        maxlags_list=parse_int_tuple(args.maxlags_list),
        forecast_steps=int(args.forecast_steps),
        irf_steps=int(args.irf_steps),
    )

    print(f"Running VAR robustness for maxlags: {config.maxlags_list}")

    run_summary, btc_table, eth_table, irf_table, report = run_var_robustness(
        df=df,
        outputs_dir=outputs_dir,
        config=config,
    )

    run_summary_path = outputs_dir / "var_robustness_run_summary.csv"
    btc_path = outputs_dir / "var_robustness_btc_granger.csv"
    eth_path = outputs_dir / "var_robustness_eth_granger.csv"
    irf_path = outputs_dir / "var_robustness_irf.csv"
    report_path = outputs_dir / "var_robustness_report.md"

    run_summary.to_csv(run_summary_path, index=False)
    btc_table.to_csv(btc_path, index=False)
    eth_table.to_csv(eth_path, index=False)
    irf_table.to_csv(irf_path, index=False)
    report_path.write_text(report)

    print(report)

    print(f"\nSaved run summary to: {run_summary_path}")
    print(f"Saved BTC robustness table to: {btc_path}")
    print(f"Saved ETH robustness table to: {eth_path}")
    print(f"Saved IRF robustness table to: {irf_path}")
    print(f"Saved robustness report to: {report_path}")


if __name__ == "__main__":
    main()