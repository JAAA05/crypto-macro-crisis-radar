from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_sources.fred_client import FredClient
from src.utils import ensure_dirs, load_config, save_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Update only Macro Engine FRED data.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Request timeout in seconds for each FRED series.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of retries per FRED series.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Sleep seconds between requests/retries.",
    )

    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"].get("end_date")

    macro_engine_series = cfg["data"].get("macro_engine_series", {})

    if not macro_engine_series:
        raise ValueError(
            "No macro_engine_series found in config.yaml. "
            "Add real_gdp, unemployment_rate, cpi, core_cpi, industrial_production, "
            "retail_sales, private_investment, and federal_surplus_deficit."
        )

    print("Fetching Macro Engine data from FRED only...")
    print(f"Series map: {macro_engine_series}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Timeout: {args.timeout}s | Retries: {args.retries} | Sleep: {args.sleep}s")

    fred = FredClient(
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        sleep_seconds=args.sleep,
    )

    macro_engine = fred.fetch_series(
        macro_engine_series,
        start_date=start_date,
        end_date=end_date,
    )

    save_csv(macro_engine, "data/raw/macro_engine_fred.csv")

    print(f"Saved Macro Engine rows: {len(macro_engine):,}")
    print("Saved to data/raw/macro_engine_fred.csv")


if __name__ == "__main__":
    main()