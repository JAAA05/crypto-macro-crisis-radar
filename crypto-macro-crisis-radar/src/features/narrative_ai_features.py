from __future__ import annotations

import numpy as np
import pandas as pd


NARRATIVE_CATEGORIES = [
    "fed_policy",
    "inflation",
    "recession",
    "fiscal_policy",
    "crypto_regulation",
    "exchange_risk",
    "stablecoin_risk",
    "etf_institutional",
    "geopolitical",
    "market_sentiment",
]


CATEGORY_WEIGHTS = {
    "fed_policy": 0.12,
    "inflation": 0.10,
    "recession": 0.12,
    "fiscal_policy": 0.08,
    "crypto_regulation": 0.12,
    "exchange_risk": 0.13,
    "stablecoin_risk": 0.13,
    "etf_institutional": 0.07,
    "geopolitical": 0.08,
    "market_sentiment": 0.05,
}


CRYPTO_NARRATIVE_CATEGORIES = [
    "crypto_regulation",
    "exchange_risk",
    "stablecoin_risk",
    "etf_institutional",
]


MACRO_NARRATIVE_CATEGORIES = [
    "fed_policy",
    "inflation",
    "recession",
    "fiscal_policy",
    "geopolitical",
    "market_sentiment",
]


def _clip_score(x, low: float = 0, high: float = 100):
    return np.clip(x, low, high)


def _ensure_category_columns(df: pd.DataFrame, category: str) -> None:
    article_col = f"{category}_article_count"
    negative_col = f"{category}_negative_hits"
    positive_col = f"{category}_positive_hits"
    risk_col = f"{category}_risk_score"

    if article_col not in df.columns:
        df[article_col] = 0
    if negative_col not in df.columns:
        df[negative_col] = 0
    if positive_col not in df.columns:
        df[positive_col] = 0
    if risk_col not in df.columns:
        df[risk_col] = 50.0

    for col in [article_col, negative_col, positive_col, risk_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[article_col] = df[article_col].fillna(0)
    df[negative_col] = df[negative_col].fillna(0)
    df[positive_col] = df[positive_col].fillna(0)
    df[risk_col] = df[risk_col].fillna(50).clip(0, 100)


def _build_enhanced_category_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for category in NARRATIVE_CATEGORIES:
        _ensure_category_columns(df, category)

        article_col = f"{category}_article_count"
        negative_col = f"{category}_negative_hits"
        positive_col = f"{category}_positive_hits"
        risk_col = f"{category}_risk_score"
        enhanced_col = f"{category}_enhanced_risk_score"

        volume_boost = np.log1p(df[article_col].fillna(0)) * 2.0
        negative_pressure = df[negative_col].fillna(0) * 2.5
        positive_offset = df[positive_col].fillna(0) * 1.2

        df[enhanced_col] = _clip_score(
            df[risk_col].fillna(50)
            + volume_boost
            + negative_pressure
            - positive_offset
        )

    return df


def _weighted_narrative_score(df: pd.DataFrame) -> pd.Series:
    total_weight = sum(CATEGORY_WEIGHTS.values())
    weighted_score = pd.Series(0.0, index=df.index)

    for category, weight in CATEGORY_WEIGHTS.items():
        enhanced_col = f"{category}_enhanced_risk_score"
        weighted_score += (weight / total_weight) * df[enhanced_col].fillna(50)

    return pd.Series(_clip_score(weighted_score), index=df.index)


def _average_category_score(df: pd.DataFrame, categories: list[str]) -> pd.Series:
    scores = []

    for category in categories:
        col = f"{category}_enhanced_risk_score"

        if col in df.columns:
            scores.append(df[col].fillna(50))
        else:
            scores.append(pd.Series(50.0, index=df.index))

    return pd.Series(_clip_score(np.mean(scores, axis=0)), index=df.index)


def _apply_general_news_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Use general GDELT news risk when category-level Narrative AI is empty.

    This prevents the model from treating missing category data as genuinely neutral.
    A flag is added so the report can disclose that category-level data was not available.
    """
    df = df.copy()

    if "news_risk_score" not in df.columns:
        df["news_risk_score"] = 50.0

    df["news_risk_score"] = pd.to_numeric(df["news_risk_score"], errors="coerce").fillna(50).clip(0, 100)

    category_count_cols = [
        f"{category}_article_count"
        for category in NARRATIVE_CATEGORIES
        if f"{category}_article_count" in df.columns
    ]

    if category_count_cols:
        total_category_articles = pd.concat(
            [df[col].fillna(0) for col in category_count_cols],
            axis=1,
        ).sum(axis=1)
    else:
        total_category_articles = pd.Series(0, index=df.index)

    fallback_mask = total_category_articles <= 0

    df["narrative_ai_fallback_used"] = fallback_mask.astype(int)
    df["narrative_ai_category_article_total"] = total_category_articles

    for category in NARRATIVE_CATEGORIES:
        enhanced_col = f"{category}_enhanced_risk_score"

        if enhanced_col not in df.columns:
            df[enhanced_col] = 50.0

        df.loc[fallback_mask, enhanced_col] = df.loc[fallback_mask, "news_risk_score"]

    return df


def _add_breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    elevated_cols = [
        f"{category}_enhanced_risk_score"
        for category in NARRATIVE_CATEGORIES
    ]

    df["narrative_elevated_category_count"] = (
        pd.concat([df[col].fillna(50) >= 65 for col in elevated_cols], axis=1)
        .sum(axis=1)
    )

    df["narrative_high_category_count"] = (
        pd.concat([df[col].fillna(50) >= 75 for col in elevated_cols], axis=1)
        .sum(axis=1)
    )

    return df


def add_narrative_ai_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build Narrative AI v2 scores.

    This layer decomposes narrative/news risk into thematic categories when
    category-level GDELT data is available. If category-level data is empty,
    it falls back to the general GDELT news risk score and marks the fallback.
    """
    df = df.copy()

    df = _build_enhanced_category_scores(df)
    df = _apply_general_news_fallback(df)

    df["narrative_ai_risk_score"] = _weighted_narrative_score(df)

    df["crypto_narrative_risk_score"] = _average_category_score(
        df,
        CRYPTO_NARRATIVE_CATEGORIES,
    )

    df["macro_narrative_risk_score"] = _average_category_score(
        df,
        MACRO_NARRATIVE_CATEGORIES,
    )

    df = _add_breadth_features(df)

    if "news_risk_score" not in df.columns:
        df["news_risk_score"] = 50.0

    df["news_risk_score"] = pd.to_numeric(df["news_risk_score"], errors="coerce").fillna(50).clip(0, 100)

    df["blended_news_narrative_risk_score"] = _clip_score(
        0.50 * df["news_risk_score"]
        + 0.50 * df["narrative_ai_risk_score"]
    )

    return df