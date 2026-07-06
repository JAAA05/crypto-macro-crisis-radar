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



def _normalize_fred_series_map(series_map: dict) -> dict:
    """
    FredClient expects:
        alias -> fred_series_id

    Some config files are written as:
        fred_series_id -> alias

    This helper converts the config safely for FredClient.
    """
    if not series_map:
        return {}

    normalized = {}

    for key, value in series_map.items():
        key_s = str(key)
        value_s = str(value)

        # If the value looks like a project alias, invert it.
        # Example: {"VIXCLS": "vix"} -> {"vix": "VIXCLS"}
        if value_s.lower() == value_s or "_" in value_s:
            normalized[value_s] = key_s
        else:
            normalized[key_s] = value_s

    return normalized


def _csv_path(relative_path: str) -> Path:
    return ROOT / relative_path


def _last_available_date(relative_path: str, date_col: str = "date") -> pd.Timestamp | None:
    path = _csv_path(relative_path)

    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=[date_col])

    if df.empty or date_col not in df.columns:
        return None

    dates = df[date_col].dropna()

    if dates.empty:
        return None

    return dates.max().normalize()


def _next_start_date(relative_path: str, default_start_date: str) -> str:
    last_date = _last_available_date(relative_path)

    if last_date is None:
        return default_start_date

    return (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _append_incremental_csv(
    new_df: pd.DataFrame,
    relative_path: str,
    label: str,
    date_col: str = "date",
) -> None:
    path = _csv_path(relative_path)

    if new_df is None or new_df.empty:
        print(f"[{label}] No new rows returned. Keeping existing file unchanged.")
        return

    new_df = new_df.copy()
    new_df[date_col] = pd.to_datetime(new_df[date_col]).dt.normalize()

    if path.exists():
        old_df = pd.read_csv(path, parse_dates=[date_col])
        old_rows = len(old_df)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        old_rows = 0
        combined = new_df

    combined[date_col] = pd.to_datetime(combined[date_col]).dt.normalize()

    combined = (
        combined
        .drop_duplicates(subset=[date_col], keep="last")
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    added_rows = len(combined) - old_rows

    save_csv(combined, relative_path)

    print(f"[{label}] Existing rows: {old_rows:,}")
    print(f"[{label}] Rows after update: {len(combined):,}")
    print(f"[{label}] Net new rows added: {max(added_rows, 0):,}")
    print(f"[{label}] Last date: {combined[date_col].max().date()}")


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


def _fetch_crypto_with_coinbase(
    cfg: dict,
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
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


def _update_crypto(cfg: dict, default_start_date: str, end_date: str | None) -> None:
    relative_path = "data/raw/crypto_market.csv"
    start_date = _next_start_date(relative_path, default_start_date)

    start_ts = pd.to_datetime(start_date).normalize()
    today_ts = pd.Timestamp.today().normalize()

    # Daily crypto candles are usually only complete through yesterday.
    # If the next needed date is today or later, keep the existing CSV unchanged.
    if start_ts >= today_ts:
        print(
            f"[1/5] Crypto already has the latest completed daily row "
            f"(next needed date: {start_date}). Keeping crypto_market.csv unchanged."
        )
        return

    effective_end_date = end_date

    if effective_end_date:
        end_ts = pd.to_datetime(effective_end_date).normalize()

        if end_ts < start_ts:
            print(
                f"[1/5] Crypto is already newer than config end_date "
                f"({start_date} > {effective_end_date}). Keeping crypto_market.csv unchanged."
            )
            return

    print(f"[1/5] Incremental crypto update from {start_date}...")

    try:
        print("Trying CoinGecko first...")
        crypto = _fetch_crypto_with_coingecko(cfg, start_date=start_date)
        print("CoinGecko incremental download successful.")

    except Exception as exc:
        print(f"CoinGecko download failed: {exc}")
        print("Falling back to public Coinbase daily candles for BTC/ETH...")

        try:
            crypto = _fetch_crypto_with_coinbase(
                cfg,
                start_date=start_date,
                end_date=effective_end_date,
            )
            print("Coinbase fallback incremental download successful.")

        except Exception as exc2:
            print(f"[crypto_market] Coinbase incremental update failed: {exc2}")
            print("[crypto_market] Keeping existing crypto_market.csv unchanged.")
            return

    _append_incremental_csv(crypto, relative_path, label="crypto_market")


def _update_macro_fred(cfg: dict, default_start_date: str, end_date: str | None) -> None:
    relative_path = "data/raw/macro_fred.csv"
    start_date = _next_start_date(relative_path, default_start_date)

    print(f"[2/5] Incremental macro FRED update from {start_date}...")

    fred = FredClient()
    macro_series = _normalize_fred_series_map(cfg["data"]["macro_series"])

    try:
        macro = fred.fetch_series(
            macro_series,
            start_date=start_date,
            end_date=end_date,
        )

    except RuntimeError as exc:
        print(f"[macro_fred] No usable incremental FRED data returned: {exc}")
        print("[macro_fred] Keeping existing macro_fred.csv unchanged.")
        return

    except Exception as exc:
        print(f"[macro_fred] Incremental FRED update failed: {exc}")
        print("[macro_fred] Keeping existing macro_fred.csv unchanged.")
        return

    _append_incremental_csv(macro, relative_path, label="macro_fred")


def _update_cross_asset_fred(cfg: dict, default_start_date: str, end_date: str | None) -> None:
    relative_path = "data/raw/cross_asset_fred.csv"
    start_date = _next_start_date(relative_path, default_start_date)

    cross_asset_series = cfg["data"].get("cross_asset_series", {})

    if not cross_asset_series:
        print("[3/5] No cross-asset series configured in config.yaml.")
        return

    print(f"[3/5] Incremental cross-asset FRED update from {start_date}...")

    fred = FredClient()
    cross_asset_series = _normalize_fred_series_map(cross_asset_series)

    try:
        cross_assets = fred.fetch_series(
            cross_asset_series,
            start_date=start_date,
            end_date=end_date,
        )

    except RuntimeError as exc:
        print(f"[cross_asset_fred] No usable incremental FRED data returned: {exc}")
        print("[cross_asset_fred] Keeping existing cross_asset_fred.csv unchanged.")
        return

    except Exception as exc:
        print(f"[cross_asset_fred] Incremental FRED update failed: {exc}")
        print("[cross_asset_fred] Keeping existing cross_asset_fred.csv unchanged.")
        return

    _append_incremental_csv(cross_assets, relative_path, label="cross_asset_fred")


def _update_stablecoin_liquidity(cfg: dict, default_start_date: str) -> None:
    relative_path = "data/raw/stablecoin_liquidity.csv"
    start_date = _next_start_date(relative_path, default_start_date)

    print(f"[4/5] Incremental stablecoin liquidity update from {start_date}...")

    llama = DefiLlamaClient()
    stable = llama.stablecoin_chart_all()
    stable = stable[stable["date"] >= pd.to_datetime(start_date)]

    _append_incremental_csv(stable, relative_path, label="stablecoin_liquidity")


def _save_neutral_news_fallback(lookback_days: int) -> pd.DataFrame:
    fallback_dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=lookback_days,
        freq="D",
    )

    return pd.DataFrame(
        {
            "date": fallback_dates,
            "news_article_count": 0,
            "news_negative_hits": 0,
            "news_positive_hits": 0,
            "news_risk_score": 50,
        }
    )


def _update_gdelt_news(cfg: dict) -> None:
    relative_path = "data/raw/news_gdelt_features.csv"
    gdelt_cfg = cfg["data"].get("gdelt", {})

    if not gdelt_cfg.get("enabled", True):
        print("[5/5] GDELT disabled in config.yaml.")
        return

    last_date = _last_available_date(relative_path)
    today = pd.Timestamp.today().normalize()

    base_lookback = int(gdelt_cfg.get("lookback_days", 30))

    if last_date is None:
        lookback_days = base_lookback
    else:
        lookback_days = max(base_lookback, int((today - last_date).days) + 2)

    print(f"[5/5] Incremental GDELT/news update with lookback_days={lookback_days}...")

    try:
        gdelt = GdeltClient()
        news = gdelt.fetch_recent_news_features(
            query=gdelt_cfg["query"],
            lookback_days=lookback_days,
            max_records_per_day=int(gdelt_cfg.get("max_records_per_day", 25)),
        )

    except Exception as exc:
        print(f"[WARN] GDELT failed completely: {exc}")
        print("[WARN] Using neutral fallback news features for recent dates.")
        news = _save_neutral_news_fallback(lookback_days=lookback_days)

    _append_incremental_csv(news, relative_path, label="news_gdelt_features")


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    default_start_date = cfg["data"]["start_date"]
    end_date = cfg["data"].get("end_date")

    print("Running incremental raw data update...")
    print("This appends only dates after each raw CSV's latest date.\n")

    _update_crypto(cfg, default_start_date=default_start_date, end_date=end_date)
    _update_macro_fred(cfg, default_start_date=default_start_date, end_date=end_date)
    _update_cross_asset_fred(cfg, default_start_date=default_start_date, end_date=end_date)
    _update_stablecoin_liquidity(cfg, default_start_date=default_start_date)
    _update_gdelt_news(cfg)

    print("\nIncremental raw data update finished.")


if __name__ == "__main__":
    main()
