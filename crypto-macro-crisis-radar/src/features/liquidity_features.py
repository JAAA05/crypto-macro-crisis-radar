from __future__ import annotations

import pandas as pd

from src.utils import safe_zscore, score_from_z


def build_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date")
    if "stablecoin_mcap" not in out.columns:
        out["liquidity_stress_score"] = 50
        return out

    out["stablecoin_mcap_7d_chg"] = out["stablecoin_mcap"].pct_change(7)
    out["stablecoin_mcap_30d_chg"] = out["stablecoin_mcap"].pct_change(30)
    out["stablecoin_mcap_z_90d"] = safe_zscore(out["stablecoin_mcap"], 90)

    # This is intentionally conservative:
    # falling stablecoin supply implies liquidity contraction => higher stress.
    contraction_score = (50 - 600 * out["stablecoin_mcap_30d_chg"].fillna(0)).clip(0, 100)
    z_score = score_from_z(-out["stablecoin_mcap_z_90d"].fillna(0))
    out["liquidity_stress_score"] = pd.concat([contraction_score, z_score], axis=1).mean(axis=1).clip(0, 100)
    return out
