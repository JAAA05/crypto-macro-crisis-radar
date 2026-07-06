from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


@dataclass
class VarLiteConfig:
    maxlags: int = 14
    forecast_steps: int = 14
    irf_steps: int = 14
    min_rows: int = 500


RAW_VARIABLES = [
    "bitcoin_ret_1d",
    "ethereum_ret_1d",
    "crypto_stress_score",
    "final_risk_score_with_model",
    "macro_engine_stress_index",
    "cross_asset_risk_score",
    "liquidity_stress_score",
    "blended_news_narrative_risk_score",
]


VAR_VARIABLES = [
    "btc_ret_1d",
    "eth_ret_1d",
    "d_crypto_stress",
    "d_final_risk",
    "d_macro_engine",
    "d_cross_asset",
    "d_liquidity_stress",
    "d_blended_narrative_risk",
]

def safe_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def prepare_var_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a stationary-ish VAR dataset.

    Returns are already closer to stationary.
    Scores are converted into daily changes to reduce trend/persistence.
    """
    required = ["date"] + RAW_VARIABLES
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for VAR Lite: {missing}")

    out = df[required].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    for col in RAW_VARIABLES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    var_df = pd.DataFrame()
    var_df["date"] = out["date"]

    var_df["btc_ret_1d"] = out["bitcoin_ret_1d"]
    var_df["eth_ret_1d"] = out["ethereum_ret_1d"]

    var_df["d_crypto_stress"] = out["crypto_stress_score"].diff()
    var_df["d_final_risk"] = out["final_risk_score_with_model"].diff()
    var_df["d_macro_engine"] = out["macro_engine_stress_index"].diff()
    var_df["d_cross_asset"] = out["cross_asset_risk_score"].diff()
    var_df["d_liquidity_stress"] = out["liquidity_stress_score"].diff()
    var_df["d_blended_narrative_risk"] = out["blended_news_narrative_risk_score"].diff()

    var_df = var_df.replace([np.inf, -np.inf], np.nan)
    var_df = var_df.dropna(subset=VAR_VARIABLES).reset_index(drop=True)

    return var_df


def standardize_var_data(var_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize VAR variables and return standardized data + scaling table."""
    data = var_df[VAR_VARIABLES].copy()

    means = data.mean()
    stds = data.std().replace(0, np.nan)

    standardized = (data - means) / stds
    standardized = standardized.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Align dates after dropna.
    aligned_dates = var_df.loc[standardized.index, "date"].reset_index(drop=True)
    standardized.insert(0, "date", aligned_dates)

    scaling = pd.DataFrame(
        {
            "variable": VAR_VARIABLES,
            "mean": [means[v] for v in VAR_VARIABLES],
            "std": [stds[v] for v in VAR_VARIABLES],
        }
    )

    return standardized, scaling


def select_lag_order(data: pd.DataFrame, maxlags: int) -> tuple[int, pd.DataFrame]:
    model = VAR(data[VAR_VARIABLES])

    maxlags = min(maxlags, max(1, len(data) // 20))

    order_selection = model.select_order(maxlags=maxlags)

    rows = []

    for lag in range(1, maxlags + 1):
        try:
            fitted = model.fit(lag)
            rows.append(
                {
                    "lag": lag,
                    "aic": fitted.aic,
                    "bic": fitted.bic,
                    "hqic": fitted.hqic,
                    "fpe": fitted.fpe,
                }
            )
        except Exception:
            continue

    selection_df = pd.DataFrame(rows)

    selected = order_selection.aic

    if selected is None or pd.isna(selected) or int(selected) < 1:
        if not selection_df.empty:
            selected = int(selection_df.sort_values("aic").iloc[0]["lag"])
        else:
            selected = 1

    selected = int(selected)
    selected = max(1, selected)

    return selected, selection_df


def fit_var_model(data: pd.DataFrame, lag_order: int):
    model = VAR(data[VAR_VARIABLES])
    fitted = model.fit(lag_order)
    return fitted


def run_granger_tests(fitted_model, target: str = "btc_ret_1d") -> pd.DataFrame:
    rows = []

    for causing in VAR_VARIABLES:
        if causing == target:
            continue

        try:
            test = fitted_model.test_causality(
                caused=target,
                causing=[causing],
                kind="f",
            )

            rows.append(
                {
                    "target": target,
                    "causing_variable": causing,
                    "test_statistic": float(test.test_statistic),
                    "p_value": float(test.pvalue),
                    "df": str(test.df),
                    "conclusion_5pct": "reject_no_causality" if float(test.pvalue) < 0.05 else "fail_to_reject",
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "target": target,
                    "causing_variable": causing,
                    "test_statistic": np.nan,
                    "p_value": np.nan,
                    "df": "",
                    "conclusion_5pct": f"error: {exc}",
                }
            )

    return pd.DataFrame(rows).sort_values("p_value", na_position="last").reset_index(drop=True)


def extract_irf(fitted_model, steps: int) -> pd.DataFrame:
    irf = fitted_model.irf(steps)
    values = irf.irfs

    names = list(fitted_model.names)

    rows = []

    for step in range(values.shape[0]):
        for response_idx, response in enumerate(names):
            for impulse_idx, impulse in enumerate(names):
                rows.append(
                    {
                        "step": step,
                        "response": response,
                        "impulse": impulse,
                        "irf_value": values[step, response_idx, impulse_idx],
                    }
                )

    return pd.DataFrame(rows)


def make_forecast(fitted_model, data: pd.DataFrame, steps: int) -> pd.DataFrame:
    lag_order = fitted_model.k_ar
    last_values = data[VAR_VARIABLES].values[-lag_order:]

    forecast_values = fitted_model.forecast(last_values, steps=steps)

    forecast = pd.DataFrame(forecast_values, columns=VAR_VARIABLES)
    forecast.insert(0, "step", range(1, steps + 1))

    return forecast


def summarize_var_results(
    lag_order: int,
    fitted_model,
    granger_btc: pd.DataFrame,
    granger_eth: pd.DataFrame,
    irf_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> str:
    lines = [
        "# VAR Lite Report",
        "",
        "This report fits a lightweight Vector Autoregression model using standardized stationary-style variables.",
        "",
        "The model is designed for research diagnostics: lag interactions, Granger-style causality tests, impulse responses, and short-horizon forecasts.",
        "",
        "## Model Setup",
        "",
        f"- Selected lag order: **{lag_order}**",
        f"- Number of observations used: **{int(fitted_model.nobs)}**",
        f"- Number of equations: **{len(fitted_model.names)}**",
        "",
        "## Variables",
        "",
        "| Variable | Meaning |",
        "|---|---|",
        "| btc_ret_1d | BTC daily return |",
        "| eth_ret_1d | ETH daily return |",
        "| d_crypto_stress | Daily change in crypto stress score |",
        "| d_final_risk | Daily change in model-adjusted final risk score |",
        "| d_macro_engine | Daily change in macro engine stress index |",
        "| d_cross_asset | Daily change in cross-asset risk score |",
        "| d_liquidity_stress | Daily change in liquidity stress score |",
        "| d_blended_narrative_risk | Daily change in blended news/narrative risk score |",
        "",
    ]

    lines.extend(
        [
            "## Granger-Style Causality Tests — Target: BTC Return",
            "",
            "Lower p-values suggest the causing variable adds lagged information for the BTC return equation. This is not proof of true causality.",
            "",
            "| Causing Variable | Test Statistic | P-Value | 5% Conclusion |",
            "|---|---:|---:|---|",
        ]
    )

    for _, row in granger_btc.iterrows():
        p = row["p_value"]
        stat = row["test_statistic"]

        lines.append(
            f"| {row['causing_variable']} | "
            f"{'' if pd.isna(stat) else f'{stat:.4f}'} | "
            f"{'' if pd.isna(p) else f'{p:.4f}'} | "
            f"{row['conclusion_5pct']} |"
        )

    lines.extend(
        [
            "",
            "## Granger-Style Causality Tests — Target: ETH Return",
            "",
            "| Causing Variable | Test Statistic | P-Value | 5% Conclusion |",
            "|---|---:|---:|---|",
        ]
    )

    for _, row in granger_eth.iterrows():
        p = row["p_value"]
        stat = row["test_statistic"]

        lines.append(
            f"| {row['causing_variable']} | "
            f"{'' if pd.isna(stat) else f'{stat:.4f}'} | "
            f"{'' if pd.isna(p) else f'{p:.4f}'} | "
            f"{row['conclusion_5pct']} |"
        )

    focused_irf = irf_df[
        (irf_df["response"].isin(["btc_ret_1d", "eth_ret_1d"]))
        & (irf_df["impulse"] != irf_df["response"])
        & (irf_df["step"].isin([1, 3, 7, 14]))
    ].copy()

    focused_irf["abs_irf"] = focused_irf["irf_value"].abs()

    top_irf = focused_irf.sort_values("abs_irf", ascending=False).head(20)

    lines.extend(
        [
            "",
            "## Largest Impulse Responses",
            "",
            "Values are in standardized units because the VAR input variables were standardized.",
            "",
            "| Step | Response | Impulse | IRF Value |",
            "|---:|---|---|---:|",
        ]
    )

    for _, row in top_irf.iterrows():
        lines.append(
            f"| {int(row['step'])} | "
            f"{row['response']} | "
            f"{row['impulse']} | "
            f"{row['irf_value']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## VAR Forecast Path",
            "",
            "Forecast values are standardized expected changes/returns from the VAR model.",
            "",
            "| Step | BTC Ret | ETH Ret | Δ Final Risk | Δ Crypto Stress | Δ Liquidity Stress | Δ Blended Narrative |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in forecast_df.iterrows():
        lines.append(
            f"| {int(row['step'])} | "
            f"{row['btc_ret_1d']:.4f} | "
            f"{row['eth_ret_1d']:.4f} | "
            f"{row['d_final_risk']:.4f} | "
            f"{row['d_crypto_stress']:.4f} | "
            f"{row['d_liquidity_stress']:.4f} | "
            f"{row['d_blended_narrative_risk']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- VAR models require careful interpretation because crypto and macro variables can be non-stationary, noisy, and regime-dependent.",
            "- This VAR Lite version uses returns and score changes to reduce non-stationarity.",
            "- Granger-style tests show whether lagged values add information inside this model, not true economic causality.",
            "- Impulse responses are based on standardized variables, so they show relative dynamic effects rather than direct dollar predictions.",
            "- This is a research diagnostic, not a trading recommendation.",
        ]
    )

    return "\n".join(lines)