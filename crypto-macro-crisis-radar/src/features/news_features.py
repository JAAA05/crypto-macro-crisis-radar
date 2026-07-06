from __future__ import annotations

import pandas as pd


def build_news_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date")
    if "news_risk_score" not in out.columns:
        out["news_risk_score"] = 50
        out["news_article_count"] = 0
        out["news_negative_hits"] = 0
        out["news_positive_hits"] = 0
        return out
    out["news_risk_score"] = out["news_risk_score"].fillna(50).clip(0, 100)
    for col in ["news_article_count", "news_negative_hits", "news_positive_hits"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    return out
