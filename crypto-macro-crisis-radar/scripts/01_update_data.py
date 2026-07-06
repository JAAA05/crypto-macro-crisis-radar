from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_sources.coinbase_public_client import CoinbasePublicClient
from src.data_sources.coingecko_client import CoinGeckoClient
from src.data_sources.defillama_client import DefiLlamaClient
from src.data_sources.fred_client import FredClient
from src.data_sources.gdelt_client import GdeltClient
from src.utils import ensure_dirs, load_config, save_csv


def _merge_coin_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No crypto frames were downloaded")

    out = frames[0]

    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")

    return out.sort_values("date").reset_index(drop=True)


def _fetch_crypto_with_coingecko(cfg: dict, start_date: str) -> pd.DataFrame:
    cg = CoinGeckoClient()
    frames = []

    for coin in cfg["data"]["crypto"]["coins"]:
        frame = cg.market_chart(
            coin_id=coin,
            vs_currency=cfg["data"]["crypto"]["vs_currency"],
            days="max",
        )
        frame = frame[frame["date"] >= pd.to_datetime(start_date)]
        frames.append(frame)

    return _merge_coin_frames(frames)


def _fetch_crypto_with_coinbase(cfg: dict, start_date: str, end_date: str | None) -> pd.DataFrame:
    cb = CoinbasePublicClient()
    frames = []

    for coin in cfg["data"]["crypto"]["coins"]:
        frame = cb.market_chart(
            coin_id=coin,
            vs_currency=cfg["data"]["crypto"]["vs_currency"],
            start_date=start_date,
            end_date=end_date,
        )
        frames.append(frame)

    return _merge_coin_frames(frames)


def _save_neutral_news_fallback(lookback_days: int) -> None:
    fallback_dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=lookback_days,
        freq="D",
    )

    news = pd.DataFrame(
        {
            "date": fallback_dates,
            "news_article_count": 0,
            "news_negative_hits": 0,
            "news_positive_hits": 0,
            "news_risk_score": 50,
        }
    )

    save_csv(news, "data/raw/news_gdelt_features.csv")
    print(f"Saved fallback news rows: {len(news):,}")


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"].get("end_date")

    print("[1/5] Fetching crypto market data...")

    try:
        print("Trying CoinGecko first...")
        crypto = _fetch_crypto_with_coingecko(cfg, start_date=start_date)
        print("CoinGecko download successful.")

    except Exception as exc:
        print(f"CoinGecko download failed: {exc}")
        print("Falling back to public Coinbase daily candles for BTC/ETH...")
        crypto = _fetch_crypto_with_coinbase(
            cfg,
            start_date=start_date,
            end_date=end_date,
        )
        print("Coinbase fallback download successful.")

    save_csv(crypto, "data/raw/crypto_market.csv")
    print(f"Saved crypto rows: {len(crypto):,}")

    print("[2/5] Fetching macro data from FRED...")
    fred = FredClient()
    macro = fred.fetch_series(
        cfg["data"]["macro_series"],
        start_date=start_date,
        end_date=end_date,
    )
    save_csv(macro, "data/raw/macro_fred.csv")
    print(f"Saved macro rows: {len(macro):,}")

    cross_asset_series = cfg["data"].get("cross_asset_series", {})

    if cross_asset_series:
        print("[3/5] Fetching cross-asset data from FRED...")
        cross_assets = fred.fetch_series(
            cross_asset_series,
            start_date=start_date,
            end_date=end_date,
        )
        save_csv(cross_assets, "data/raw/cross_asset_fred.csv")
        print(f"Saved cross-asset rows: {len(cross_assets):,}")
    else:
        print("[3/5] No cross-asset series configured in config.yaml.")

    print("[4/5] Fetching stablecoin liquidity from DeFiLlama...")
    llama = DefiLlamaClient()
    stable = llama.stablecoin_chart_all()
    stable = stable[stable["date"] >= pd.to_datetime(start_date)]
    save_csv(stable, "data/raw/stablecoin_liquidity.csv")
    print(f"Saved stablecoin rows: {len(stable):,}")

    gdelt_cfg = cfg["data"].get("gdelt", {})

    if gdelt_cfg.get("enabled", True):
        print("[5/5] Fetching recent narrative/news features from GDELT...")

        try:
            gdelt = GdeltClient()
            news = gdelt.fetch_recent_news_features(
                query=gdelt_cfg["query"],
                lookback_days=int(gdelt_cfg.get("lookback_days", 30)),
                max_records_per_day=int(gdelt_cfg.get("max_records_per_day", 25)),
            )
            save_csv(news, "data/raw/news_gdelt_features.csv")
            print(f"Saved news rows: {len(news):,}")

        except Exception as exc:
            print(f"[WARN] GDELT failed completely: {exc}")
            print("[WARN] Saving neutral fallback news features so the pipeline can continue.")
            _save_neutral_news_fallback(
                lookback_days=int(gdelt_cfg.get("lookback_days", 30))
            )

    else:
        print("[5/5] GDELT disabled in config.yaml")

        news_path = ROOT / "data/raw/news_gdelt_features.csv"
        if not news_path.exists():
            print("[WARN] No existing news file found. Creating neutral fallback.")
            _save_neutral_news_fallback(lookback_days=30)


if __name__ == "__main__":
    main()