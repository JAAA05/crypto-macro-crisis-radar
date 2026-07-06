from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_score(x, low: float = 0, high: float = 100):
    return np.clip(x, low, high)


def _rolling_zscore(series: pd.Series, window: int = 365, min_periods: int = 60) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def _ensure_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = np.nan


def add_macro_engine_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build advanced macroeconomic features and scores.

    This layer is designed for mixed-frequency macro data.
    The daily crypto calendar is preserved, while monthly/quarterly data
    should already be forward-filled in scripts/02_build_features.py.
    """

    df = df.copy()

    macro_cols = [
        "real_gdp",
        "unemployment_rate",
        "cpi",
        "core_cpi",
        "industrial_production",
        "retail_sales",
        "private_investment",
        "federal_surplus_deficit",
    ]

    for col in macro_cols:
        _ensure_col(df, col)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Growth cycle
    df["real_gdp_yoy"] = df["real_gdp"].pct_change(365)
    df["real_gdp_6m_chg"] = df["real_gdp"].pct_change(182)

    df["industrial_production_yoy"] = df["industrial_production"].pct_change(365)
    df["industrial_production_6m_chg"] = df["industrial_production"].pct_change(182)

    df["retail_sales_yoy"] = df["retail_sales"].pct_change(365)
    df["retail_sales_6m_chg"] = df["retail_sales"].pct_change(182)

    df["private_investment_yoy"] = df["private_investment"].pct_change(365)
    df["private_investment_6m_chg"] = df["private_investment"].pct_change(182)

    # Labor market
    df["unemployment_change_3m"] = df["unemployment_rate"].diff(90)
    df["unemployment_change_6m"] = df["unemployment_rate"].diff(182)
    df["unemployment_z_3y"] = _rolling_zscore(df["unemployment_rate"], window=756, min_periods=252)

    # Inflation
    df["cpi_yoy"] = df["cpi"].pct_change(365)
    df["core_cpi_yoy"] = df["core_cpi"].pct_change(365)
    df["cpi_3m_chg"] = df["cpi"].pct_change(90)
    df["core_cpi_3m_chg"] = df["core_cpi"].pct_change(90)

    # Monetary conditions, using existing macro columns if present
    if "fed_funds_rate" not in df.columns:
        df["fed_funds_rate"] = np.nan
    if "treasury_2y" not in df.columns:
        df["treasury_2y"] = np.nan
    if "treasury_10y" not in df.columns:
        df["treasury_10y"] = np.nan
    if "yield_curve_inversion_flag" not in df.columns:
        df["yield_curve_inversion_flag"] = 0
    if "dollar_z_90d" not in df.columns:
        df["dollar_z_90d"] = 0
    if "m2_90d_chg" not in df.columns:
        df["m2_90d_chg"] = 0
    if "fed_balance_sheet_90d_chg" not in df.columns:
        df["fed_balance_sheet_90d_chg"] = 0

    df["fed_funds_change_6m"] = df["fed_funds_rate"].diff(182)
    df["treasury_2y_change_6m"] = df["treasury_2y"].diff(182)
    df["treasury_10y_change_6m"] = df["treasury_10y"].diff(182)

    # Fiscal pressure
    # MTSDS133FMS can be positive or negative depending on surplus/deficit.
    # Negative values are treated as deficit pressure.
    df["federal_deficit_pressure_raw"] = (-df["federal_surplus_deficit"]).clip(lower=0)
    df["federal_deficit_pressure_z_3y"] = _rolling_zscore(
        df["federal_deficit_pressure_raw"],
        window=756,
        min_periods=252,
    )

    # Scores
    # 1. Growth slowdown: higher risk when GDP/production/retail/investment weaken.
    growth_weakness = (
        (-df["real_gdp_yoy"].fillna(0) * 550)
        + (-df["industrial_production_yoy"].fillna(0) * 350)
        + (-df["retail_sales_yoy"].fillna(0) * 150)
        + (-df["private_investment_yoy"].fillna(0) * 150)
    )

    df["growth_slowdown_score"] = _clip_score(35 + growth_weakness)

    # 2. Labor stress: higher when unemployment rises or is elevated vs recent history.
    labor_pressure = (
        df["unemployment_change_3m"].fillna(0) * 22
        + df["unemployment_change_6m"].fillna(0) * 12
        + df["unemployment_z_3y"].fillna(0) * 8
    )

    df["labor_stress_score"] = _clip_score(35 + labor_pressure)

    # 3. Inflation pressure: higher when CPI/Core CPI are above a 2% anchor.
    cpi_excess = (df["cpi_yoy"].fillna(0.02) - 0.02).clip(lower=0)
    core_cpi_excess = (df["core_cpi_yoy"].fillna(0.02) - 0.02).clip(lower=0)

    df["inflation_pressure_score"] = _clip_score(
        30
        + cpi_excess * 850
        + core_cpi_excess * 900
        + df["cpi_3m_chg"].fillna(0) * 250
        + df["core_cpi_3m_chg"].fillna(0) * 250
    )

    # 4. Monetary tightening: higher when rates rise, curve is inverted, dollar is strong,
    # M2 contracts, or the Fed balance sheet shrinks.
    rate_pressure = (
        df["fed_funds_change_6m"].fillna(0) * 8
        + df["treasury_2y_change_6m"].fillna(0) * 6
        + df["treasury_10y_change_6m"].fillna(0) * 4
        + df["yield_curve_inversion_flag"].fillna(0) * 12
        + df["dollar_z_90d"].fillna(0) * 5
        + (-df["m2_90d_chg"].fillna(0)).clip(lower=0) * 900
        + (-df["fed_balance_sheet_90d_chg"].fillna(0)).clip(lower=0) * 500
    )

    df["monetary_tightening_score"] = _clip_score(35 + rate_pressure)

    # 5. Fiscal stress: higher when deficit pressure is unusual.
    df["fiscal_stress_score"] = _clip_score(
        35 + df["federal_deficit_pressure_z_3y"].fillna(0) * 10
    )

    # 6. Aggregate Macro Engine Stress Index
    df["macro_engine_stress_index"] = _clip_score(
        0.25 * df["growth_slowdown_score"].fillna(50)
        + 0.15 * df["labor_stress_score"].fillna(50)
        + 0.20 * df["inflation_pressure_score"].fillna(50)
        + 0.25 * df["monetary_tightening_score"].fillna(50)
        + 0.15 * df["fiscal_stress_score"].fillna(50)
    )

    return df