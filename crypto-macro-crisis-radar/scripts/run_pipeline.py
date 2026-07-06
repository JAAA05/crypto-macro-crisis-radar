from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RAW_REQUIRED_FILES = [
    ROOT / "data/raw/crypto_market.csv",
    ROOT / "data/raw/macro_fred.csv",
    ROOT / "data/raw/cross_asset_fred.csv",
    ROOT / "data/raw/stablecoin_liquidity.csv",
    ROOT / "data/raw/news_gdelt_features.csv",
]


def run_script(script_path: str) -> None:
    print(f"\n=== Running {script_path} ===")
    result = subprocess.run([sys.executable, script_path], cwd=ROOT)

    if result.returncode != 0:
        print(f"\nPipeline stopped because {script_path} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def raw_data_exists() -> bool:
    return all(path.exists() for path in RAW_REQUIRED_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Crypto Macro Crisis Radar pipeline.")

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data updates and use existing files in data/raw.",
    )

    mode.add_argument(
        "--force-data",
        action="store_true",
        help="Force full re-download of raw data even if raw files already exist.",
    )

    mode.add_argument(
        "--incremental-data",
        action="store_true",
        help="Update only dates after the latest date already stored in raw CSV files.",
    )

    args = parser.parse_args()

    if args.force_data:
        run_script("scripts/01_update_data.py")

    elif args.incremental_data:
        run_script("scripts/01_update_data_incremental.py")

        macro_engine_script = ROOT / "scripts/01c_update_macro_engine.py"
        if macro_engine_script.exists():
            print("\nMacro Engine updater found. Running it too because it is small/monthly.")
            run_script("scripts/01c_update_macro_engine.py")

    elif args.skip_data:
        print("Skipping data update because --skip-data was provided.")

    elif raw_data_exists():
        print("Raw data files already exist. Skipping scripts/01_update_data.py.")
        print("Use --incremental-data to update only new dates.")
        print("Use --force-data if you really want to download everything again.")

    else:
        print("Raw data files are missing. Running scripts/01_update_data.py.")
        run_script("scripts/01_update_data.py")

    run_script("scripts/02_build_features.py")
    run_script("scripts/03_train_baseline.py")
    run_script("scripts/04_daily_report.py")

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
