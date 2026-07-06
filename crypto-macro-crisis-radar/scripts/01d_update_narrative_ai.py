from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_sources.gdelt_client import GdeltClient
from src.utils import ensure_dirs, load_config, save_csv


DEFAULT_CATEGORIES = {
    "fed_policy": '"Federal Reserve" OR Fed OR FOMC OR "interest rates" OR "monetary policy"',
    "inflation": 'inflation OR CPI OR "consumer prices" OR disinflation OR "price pressures"',
    "recession": 'recession OR slowdown OR "economic contraction" OR "hard landing" OR "soft landing"',
    "fiscal_policy": '"fiscal policy" OR deficit OR debt OR "government spending" OR Treasury',
    "crypto_regulation": 'crypto regulation OR SEC OR CFTC OR lawsuit OR enforcement OR "digital assets"',
    "exchange_risk": 'crypto exchange OR Binance OR Coinbase OR Kraken OR insolvency OR withdrawals',
    "stablecoin_risk": 'stablecoin OR USDT OR USDC OR depeg OR Tether OR Circle',
    "etf_institutional": 'bitcoin ETF OR spot ETF OR BlackRock OR institutional crypto OR ETF flows',
    "geopolitical": 'geopolitical OR war OR conflict OR sanctions OR Middle East OR China OR Russia',
    "market_sentiment": 'risk assets OR market sentiment OR risk-off OR volatility OR selloff OR rally',
}


def build_neutral_category_frame(category: str, lookback_days: int) -> pd.DataFrame:
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=lookback_days,
        freq="D",
    )

    return pd.DataFrame(
        {
            "date": dates,
            f"{category}_article_count": 0,
            f"{category}_negative_hits": 0,
            f"{category}_positive_hits": 0,
            f"{category}_risk_score": 50.0,
        }
    )


def normalize_gdelt_category_frame(
    frame: pd.DataFrame,
    category: str,
    lookback_days: int,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return build_neutral_category_frame(category, lookback_days)

    out = frame.copy()

    if "date" not in out.columns:
        return build_neutral_category_frame(category, lookback_days)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    rename_map = {
        "news_article_count": f"{category}_article_count",
        "news_negative_hits": f"{category}_negative_hits",
        "news_positive_hits": f"{category}_positive_hits",
        "news_risk_score": f"{category}_risk_score",
    }

    out = out.rename(columns=rename_map)

    required_cols = [
        f"{category}_article_count",
        f"{category}_negative_hits",
        f"{category}_positive_hits",
        f"{category}_risk_score",
    ]

    for col in required_cols:
        if col not in out.columns:
            if col.endswith("_risk_score"):
                out[col] = 50.0
            else:
                out[col] = 0

    out = out[["date"] + required_cols].copy()

    for col in required_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out[f"{category}_article_count"] = out[f"{category}_article_count"].fillna(0)
    out[f"{category}_negative_hits"] = out[f"{category}_negative_hits"].fillna(0)
    out[f"{category}_positive_hits"] = out[f"{category}_positive_hits"].fillna(0)
    out[f"{category}_risk_score"] = out[f"{category}_risk_score"].fillna(50).clip(0, 100)

    out = (
        out.drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    return out


def get_category_queries(cfg: dict) -> dict[str, str]:
    narrative_cfg = cfg["data"].get("narrative_ai", {})
    categories_cfg = narrative_cfg.get("categories", {})

    if not categories_cfg:
        return DEFAULT_CATEGORIES

    out: dict[str, str] = {}

    for category, value in categories_cfg.items():
        if isinstance(value, dict):
            query = value.get("query")
        else:
            query = str(value)

        if query:
            out[str(category)] = str(query)

    if not out:
        return DEFAULT_CATEGORIES

    return out


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    narrative_cfg = cfg["data"].get("narrative_ai", {})

    if not narrative_cfg.get("enabled", True):
        print("Narrative AI disabled in config.yaml.")
        return

    lookback_days = int(narrative_cfg.get("lookback_days", 30))
    max_records_per_day = int(narrative_cfg.get("max_records_per_day", 20))

    category_queries = get_category_queries(cfg)

    print("Fetching Narrative AI v2 features from GDELT...")
    print(f"Categories: {list(category_queries.keys())}")
    print(f"Lookback days: {lookback_days}")
    print(f"Max records per day: {max_records_per_day}")

    gdelt = GdeltClient()

    frames: list[pd.DataFrame] = []

    for category, query in category_queries.items():
        print(f"\nFetching category: {category}")
        print(f"Query: {query}")

        try:
            raw = gdelt.fetch_recent_news_features(
                query=query,
                lookback_days=lookback_days,
                max_records_per_day=max_records_per_day,
            )

            frame = normalize_gdelt_category_frame(
                frame=raw,
                category=category,
                lookback_days=lookback_days,
            )

            print(f"Saved category rows in memory: {len(frame):,}")
            frames.append(frame)

        except Exception as exc:
            print(f"[WARN] Narrative category {category} failed: {exc}")
            print("[WARN] Using neutral fallback for this category.")

            frames.append(
                build_neutral_category_frame(
                    category=category,
                    lookback_days=lookback_days,
                )
            )

    if not frames:
        raise RuntimeError("No Narrative AI frames were created.")

    out = frames[0]

    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)

    save_csv(out, "data/raw/narrative_ai_features.csv")

    print(f"\nSaved Narrative AI rows: {len(out):,}")
    print("Saved to data/raw/narrative_ai_features.csv")


if __name__ == "__main__":
    main()