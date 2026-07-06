from __future__ import annotations

import pandas as pd


def add_crisis_targets(df: pd.DataFrame, mini_crisis_drawdown_7d: float = -0.07, crisis_drawdown_14d: float = -0.15) -> pd.DataFrame:
    out = df.copy().sort_values("date")
    if "bitcoin_price" not in out.columns:
        raise ValueError("bitcoin_price is required to build crisis targets.")

    price = out["bitcoin_price"]
    future_min_7d = pd.concat([price.shift(-i) / price - 1 for i in range(1, 8)], axis=1).min(axis=1)
    future_min_14d = pd.concat([price.shift(-i) / price - 1 for i in range(1, 15)], axis=1).min(axis=1)

    out["btc_forward_min_return_7d"] = future_min_7d
    out["btc_forward_min_return_14d"] = future_min_14d
    out["mini_crisis_next_7d"] = (future_min_7d <= mini_crisis_drawdown_7d).astype(int)
    out["crisis_next_14d"] = (future_min_14d <= crisis_drawdown_14d).astype(int)
    return out
