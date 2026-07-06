from __future__ import annotations

import pandas as pd

from src.utils import safe_zscore, score_from_z


def build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date")

    # Forward-fill macro data to daily frequency after joining with crypto dates.
    macro_cols = [c for c in out.columns if c != "date"]
    out[macro_cols] = out[macro_cols].ffill()

    if "treasury_10y" in out.columns and "treasury_2y" in out.columns:
        out["yield_curve_10y_2y"] = out["treasury_10y"] - out["treasury_2y"]
        out["yield_curve_inversion_flag"] = (out["yield_curve_10y_2y"] < 0).astype(int)

    if "vix" in out.columns:
        out["vix_change_5d"] = out["vix"].diff(5)
        out["vix_z_90d"] = safe_zscore(out["vix"], 90)

    if "dollar_index_broad" in out.columns:
        out["dollar_ret_20d"] = out["dollar_index_broad"].pct_change(20)
        out["dollar_z_90d"] = safe_zscore(out["dollar_index_broad"], 90)

    if "fed_balance_sheet" in out.columns:
        out["fed_balance_sheet_90d_chg"] = out["fed_balance_sheet"].pct_change(90)

    if "m2_money_supply" in out.columns:
        out["m2_90d_chg"] = out["m2_money_supply"].pct_change(90)

    if "economic_policy_uncertainty" in out.columns:
        out["epu_z_90d"] = safe_zscore(out["economic_policy_uncertainty"], 90)

    components = []
    for col in ["vix_z_90d", "dollar_z_90d", "epu_z_90d"]:
        if col in out.columns:
            components.append(score_from_z(out[col]))
    if "yield_curve_inversion_flag" in out.columns:
        components.append(50 + 25 * out["yield_curve_inversion_flag"])
    if "fed_balance_sheet_90d_chg" in out.columns:
        # Falling Fed balance sheet is treated as tighter liquidity.
        components.append((50 - 250 * out["fed_balance_sheet_90d_chg"].fillna(0)).clip(0, 100))
    if "m2_90d_chg" in out.columns:
        # Falling/slow M2 growth is treated as tighter liquidity.
        components.append((50 - 200 * out["m2_90d_chg"].fillna(0)).clip(0, 100))

    out["macro_risk_score"] = pd.concat(components, axis=1).mean(axis=1).clip(0, 100) if components else 50
    return out
