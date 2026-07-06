from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_score(x: pd.Series | float, low: float = 0, high: float = 100):
    return np.clip(x, low, high)


def _rolling_zscore(series: pd.Series, window: int = 90) -> pd.Series:
    mean = series.rolling(window, min_periods=20).mean()
    std = series.rolling(window, min_periods=20).std()
    return (series - mean) / std.replace(0, np.nan)


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-asset market intelligence features.

    Expected optional columns:
    - sp500
    - nasdaq
    - gold
    - oil_wti
    - vix
    - dollar_index_broad

    The function is defensive: if some columns are missing, it creates
    neutral scores instead of breaking the pipeline.
    """
    df = df.copy()

    required_cols = ["sp500", "nasdaq", "gold", "oil_wti"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["sp500_ret_1d"] = df["sp500"].pct_change()
    df["sp500_ret_7d"] = df["sp500"].pct_change(7)
    df["sp500_ret_30d"] = df["sp500"].pct_change(30)

    df["nasdaq_ret_1d"] = df["nasdaq"].pct_change()
    df["nasdaq_ret_7d"] = df["nasdaq"].pct_change(7)
    df["nasdaq_ret_30d"] = df["nasdaq"].pct_change(30)

    df["gold_ret_7d"] = df["gold"].pct_change(7)
    df["gold_ret_30d"] = df["gold"].pct_change(30)

    df["oil_ret_7d"] = df["oil_wti"].pct_change(7)
    df["oil_ret_30d"] = df["oil_wti"].pct_change(30)
    df["oil_vol_30d"] = df["oil_wti"].pct_change().rolling(30, min_periods=10).std() * np.sqrt(252)

    df["sp500_z_90d"] = _rolling_zscore(df["sp500_ret_7d"], 90)
    df["nasdaq_z_90d"] = _rolling_zscore(df["nasdaq_ret_7d"], 90)
    df["gold_z_90d"] = _rolling_zscore(df["gold_ret_7d"], 90)
    df["oil_z_90d"] = _rolling_zscore(df["oil_ret_7d"], 90)
    df["oil_vol_z_90d"] = _rolling_zscore(df["oil_vol_30d"], 90)

    if "vix" in df.columns:
        df["vix_cross_asset_z_90d"] = _rolling_zscore(df["vix"], 90)
    else:
        df["vix_cross_asset_z_90d"] = np.nan

    if "dollar_index_broad" in df.columns:
        df["dollar_cross_asset_ret_20d"] = df["dollar_index_broad"].pct_change(20)
        df["dollar_cross_asset_z_90d"] = _rolling_zscore(df["dollar_cross_asset_ret_20d"], 90)
    else:
        df["dollar_cross_asset_ret_20d"] = np.nan
        df["dollar_cross_asset_z_90d"] = np.nan

    equity_selloff = (
        (-df["sp500_ret_7d"].fillna(0) * 450)
        + (-df["nasdaq_ret_7d"].fillna(0) * 450)
    )

    vix_pressure = df["vix_cross_asset_z_90d"].fillna(0) * 8

    df["equity_risk_score"] = _clip_score(
        35 + equity_selloff + vix_pressure
    )

    gold_strength = df["gold_ret_7d"].fillna(0) * 500
    dollar_strength = df["dollar_cross_asset_ret_20d"].fillna(0) * 300
    equity_weakness = (
        (-df["sp500_ret_7d"].fillna(0) * 250)
        + (-df["nasdaq_ret_7d"].fillna(0) * 200)
    )

    df["safe_haven_demand_score"] = _clip_score(
        35 + gold_strength + dollar_strength + equity_weakness + vix_pressure
    )

    oil_price_pressure = df["oil_ret_30d"].fillna(0) * 250
    oil_vol_pressure = df["oil_vol_z_90d"].fillna(0) * 10

    df["commodity_pressure_score"] = _clip_score(
        35 + oil_price_pressure + oil_vol_pressure
    )

    df["cross_asset_risk_score"] = _clip_score(
        0.45 * df["equity_risk_score"].fillna(50)
        + 0.35 * df["safe_haven_demand_score"].fillna(50)
        + 0.20 * df["commodity_pressure_score"].fillna(50)
    )

    return df