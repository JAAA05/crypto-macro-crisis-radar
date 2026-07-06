from __future__ import annotations

import pandas as pd


def compute_final_risk_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    mapping = {
        "macro_risk": "macro_risk_score",
        "news_risk": "news_risk_score",
        "crypto_stress": "crypto_stress_score",
        "liquidity_stress": "liquidity_stress_score",
        "anomaly": "anomaly_score",
    }
    total = 0
    denom = 0
    for weight_key, col in mapping.items():
        w = float(weights.get(weight_key, 0))
        if col in out.columns and w > 0:
            total = total + w * out[col].fillna(50)
            denom += w
    out["final_risk_score"] = (total / denom if denom else 50).clip(0, 100)
    out["market_regime"] = out["final_risk_score"].apply(classify_regime)
    return out


def classify_regime(score: float) -> str:
    score = float(score)
    if score <= 30:
        return "Risk-On"
    if score <= 50:
        return "Neutral"
    if score <= 70:
        return "Risk-Off"
    if score <= 85:
        return "Mini-Crisis Watch"
    return "Crisis"
