from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features.crypto_features import build_crypto_features
from src.features.cross_asset_features import add_cross_asset_features
from src.features.liquidity_features import build_liquidity_features
from src.features.macro_engine_features import add_macro_engine_features
from src.features.macro_features import build_macro_features
from src.features.narrative_ai_features import add_narrative_ai_features
from src.features.news_features import build_news_features
from src.features.target_builder import add_crisis_targets
from src.utils import load_config, save_csv


def main() -> None:
    cfg = load_config()
    print("Building master feature table...")

    crypto = pd.read_csv(ROOT / "data/raw/crypto_market.csv", parse_dates=["date"])
    macro = pd.read_csv(ROOT / "data/raw/macro_fred.csv", parse_dates=["date"])
    stable = pd.read_csv(ROOT / "data/raw/stablecoin_liquidity.csv", parse_dates=["date"])

    news_path = ROOT / "data/raw/news_gdelt_features.csv"
    if news_path.exists():
        news = pd.read_csv(news_path, parse_dates=["date"])
    else:
        print("[WARN] data/raw/news_gdelt_features.csv not found. News features will be neutral.")
        news = pd.DataFrame(
            {
                "date": crypto["date"],
                "news_article_count": 0,
                "news_negative_hits": 0,
                "news_positive_hits": 0,
                "news_risk_score": 50,
            }
        )

    narrative_ai_path = ROOT / "data/raw/narrative_ai_features.csv"
    if narrative_ai_path.exists():
        narrative_ai = pd.read_csv(narrative_ai_path, parse_dates=["date"])
    else:
        print("[WARN] data/raw/narrative_ai_features.csv not found. Narrative AI features will be neutral.")
        narrative_ai = pd.DataFrame({"date": crypto["date"]})

    cross_asset_path = ROOT / "data/raw/cross_asset_fred.csv"
    if cross_asset_path.exists():
        cross_assets = pd.read_csv(cross_asset_path, parse_dates=["date"])
    else:
        print("[WARN] data/raw/cross_asset_fred.csv not found. Cross-asset features will be neutral.")
        cross_assets = pd.DataFrame({"date": crypto["date"]})

    macro_engine_path = ROOT / "data/raw/macro_engine_fred.csv"
    if macro_engine_path.exists():
        macro_engine = pd.read_csv(macro_engine_path, parse_dates=["date"])
    else:
        print("[WARN] data/raw/macro_engine_fred.csv not found. Macro Engine features will be neutral.")
        macro_engine = pd.DataFrame({"date": crypto["date"]})

    df = crypto.merge(macro, on="date", how="left")
    df = df.merge(cross_assets, on="date", how="left")
    df = df.merge(macro_engine, on="date", how="left")
    df = df.merge(stable, on="date", how="left")
    df = df.merge(news, on="date", how="left")
    df = df.merge(narrative_ai, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)

    news_cols = [
        "news_article_count",
        "news_negative_hits",
        "news_positive_hits",
        "news_risk_score",
    ]

    narrative_cols = [
        col
        for col in df.columns
        if any(
            col.startswith(prefix)
            for prefix in [
                "fed_policy_",
                "inflation_",
                "recession_",
                "fiscal_policy_",
                "crypto_regulation_",
                "exchange_risk_",
                "stablecoin_risk_",
                "etf_institutional_",
                "geopolitical_",
                "market_sentiment_",
            ]
        )
    ]

    no_ffill_cols = ["date"] + news_cols + narrative_cols

    for col in df.columns:
        if col not in no_ffill_cols:
            df[col] = df[col].ffill()

    if "news_article_count" in df.columns:
        df["news_article_count"] = df["news_article_count"].fillna(0)

    if "news_negative_hits" in df.columns:
        df["news_negative_hits"] = df["news_negative_hits"].fillna(0)

    if "news_positive_hits" in df.columns:
        df["news_positive_hits"] = df["news_positive_hits"].fillna(0)

    if "news_risk_score" in df.columns:
        df["news_risk_score"] = df["news_risk_score"].fillna(50)

    df = build_macro_features(df)
    df = add_cross_asset_features(df)
    df = add_macro_engine_features(df)
    df = build_crypto_features(df)
    df = build_liquidity_features(df)
    df = build_news_features(df)
    df = add_narrative_ai_features(df)

    df = add_crisis_targets(
        df,
        mini_crisis_drawdown_7d=float(cfg["targets"]["mini_crisis_drawdown_7d"]),
        crisis_drawdown_14d=float(cfg["targets"]["crisis_drawdown_14d"]),
    )

    df = df.dropna(subset=["bitcoin_price"]).reset_index(drop=True)

    save_csv(df, "data/features/master_features.csv")
    print(f"Saved master feature rows: {len(df):,}")


if __name__ == "__main__":
    main()