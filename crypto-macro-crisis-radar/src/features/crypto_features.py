from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import safe_zscore, score_from_z


def build_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date")
    for coin in ["bitcoin", "ethereum"]:
        price = f"{coin}_price"
        volume = f"{coin}_volume"
        if price not in out.columns:
            continue
        out[f"{coin}_ret_1d"] = out[price].pct_change(1)
        out[f"{coin}_ret_3d"] = out[price].pct_change(3)
        out[f"{coin}_ret_7d"] = out[price].pct_change(7)
        out[f"{coin}_vol_7d"] = out[f"{coin}_ret_1d"].rolling(7, min_periods=5).std() * np.sqrt(365)
        out[f"{coin}_vol_30d"] = out[f"{coin}_ret_1d"].rolling(30, min_periods=15).std() * np.sqrt(365)
        out[f"{coin}_drawdown_30d"] = out[price] / out[price].rolling(30, min_periods=10).max() - 1
        out[f"{coin}_drawdown_90d"] = out[price] / out[price].rolling(90, min_periods=30).max() - 1
        if volume in out.columns:
            out[f"{coin}_volume_z_30d"] = safe_zscore(out[volume], 30)

    if "bitcoin_ret_1d" in out.columns and "ethereum_ret_1d" in out.columns:
        out["btc_eth_corr_30d"] = out["bitcoin_ret_1d"].rolling(30, min_periods=15).corr(out["ethereum_ret_1d"])

    stress_components = []
    for col in ["bitcoin_vol_30d", "ethereum_vol_30d", "bitcoin_volume_z_30d", "ethereum_volume_z_30d"]:
        if col in out.columns:
            stress_components.append(score_from_z(safe_zscore(out[col], 90)))
    for col in ["bitcoin_drawdown_30d", "ethereum_drawdown_30d"]:
        if col in out.columns:
            # More negative drawdown => higher stress
            stress_components.append(((-out[col].fillna(0)) * 350).clip(0, 100))

    if stress_components:
        out["crypto_stress_score"] = pd.concat(stress_components, axis=1).mean(axis=1).clip(0, 100)
    else:
        out["crypto_stress_score"] = 50

    return out
