from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_sources.yfinance_client import YFinanceCrossAssetClient
from src.utils import ensure_dirs, load_config, save_csv


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"].get("end_date")

    print("Fetching cross-asset data from yfinance only...")
    print("Assets:")
    print("- sp500: ^GSPC")
    print("- nasdaq: ^IXIC")
    print("- gold: GC=F")
    print("- oil_wti: CL=F")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    client = YFinanceCrossAssetClient()
    cross_assets = client.fetch_cross_assets(
        start_date=start_date,
        end_date=end_date,
    )

    save_csv(cross_assets, "data/raw/cross_asset_fred.csv")

    print(f"Saved cross-asset rows: {len(cross_assets):,}")
    print("Saved to data/raw/cross_asset_fred.csv")


if __name__ == "__main__":
    main()