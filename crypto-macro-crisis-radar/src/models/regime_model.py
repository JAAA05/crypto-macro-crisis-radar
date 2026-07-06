from __future__ import annotations

import pandas as pd


def markov_switching_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for v0.4.

    Next step: use statsmodels MarkovRegression on BTC returns/volatility to estimate latent risk regimes.
    This file exists so the project architecture is ready for the pro regime layer.
    """
    out = df.copy()
    out["markov_regime_placeholder"] = None
    return out
