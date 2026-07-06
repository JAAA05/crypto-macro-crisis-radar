from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.econometrics.var_model import (
    VarLiteConfig,
    extract_irf,
    fit_var_model,
    make_forecast,
    prepare_var_dataset,
    run_granger_tests,
    select_lag_order,
    standardize_var_data,
)


@dataclass
class VarRobustnessConfig:
    maxlags_list: tuple[int, ...] = (14, 7, 5)
    forecast_steps: int = 14
    irf_steps: int = 14
    significance_level: float = 0.05
    borderline_level: float = 0.10


CORE_BTC_VARIABLES = [
    "d_final_risk",
    "eth_ret_1d",
    "d_crypto_stress",
    "d_cross_asset",
    "d_macro_engine",
    "d_liquidity_stress",
    "d_blended_narrative_risk",
]


CORE_ETH_VARIABLES = [
    "d_final_risk",
    "btc_ret_1d",
    "d_crypto_stress",
    "d_cross_asset",
    "d_macro_engine",
    "d_liquidity_stress",
    "d_blended_narrative_risk",
]


def _fmt_p(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def _fmt_num(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def classify_robustness(
    p_values: list[float],
    significance_level: float = 0.05,
    borderline_level: float = 0.10,
) -> str:
    valid = [p for p in p_values if not pd.isna(p)]

    if not valid:
        return "Insufficient data"

    sig_count = sum(p < significance_level for p in valid)
    border_count = sum(p < borderline_level for p in valid)

    if sig_count == len(valid):
        return "Strong"
    if sig_count >= 2:
        return "Moderate"
    if sig_count >= 1 and border_count >= 2:
        return "Moderate / borderline"
    if border_count >= 1:
        return "Borderline"
    return "Weak"


def run_single_var_robustness(
    df: pd.DataFrame,
    maxlags: int,
    outputs_dir: Path,
    forecast_steps: int = 14,
    irf_steps: int = 14,
) -> dict:
    run_dir = outputs_dir / f"maxlags_{maxlags}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== VAR robustness run: maxlags={maxlags} ===")

    config = VarLiteConfig(
        maxlags=maxlags,
        forecast_steps=forecast_steps,
        irf_steps=irf_steps,
    )

    print("Preparing VAR dataset...")
    var_raw = prepare_var_dataset(df)

    print("Standardizing VAR dataset...")
    var_data, scaling = standardize_var_data(var_raw)

    print("Selecting lag order...")
    selected_lag_order, lag_selection = select_lag_order(
        data=var_data,
        maxlags=config.maxlags,
    )

    print(f"Selected lag order for maxlags={maxlags}: {selected_lag_order}")

    print("Fitting VAR model...")
    fitted = fit_var_model(
        data=var_data,
        lag_order=selected_lag_order,
    )

    print("Running Granger-style tests...")
    granger_btc = run_granger_tests(fitted, target="btc_ret_1d")
    granger_eth = run_granger_tests(fitted, target="eth_ret_1d")

    print("Extracting impulse responses...")
    irf_df = extract_irf(
        fitted,
        steps=config.irf_steps,
    )

    print("Making forecast...")
    forecast_df = make_forecast(
        fitted,
        data=var_data,
        steps=config.forecast_steps,
    )

    var_raw.to_csv(run_dir / "var_lite_dataset_raw.csv", index=False)
    var_data.to_csv(run_dir / "var_lite_dataset_standardized.csv", index=False)
    scaling.to_csv(run_dir / "var_lite_scaling.csv", index=False)
    lag_selection.to_csv(run_dir / "var_lag_order_selection.csv", index=False)
    granger_btc.to_csv(run_dir / "var_granger_btc.csv", index=False)
    granger_eth.to_csv(run_dir / "var_granger_eth.csv", index=False)
    irf_df.to_csv(run_dir / "var_impulse_responses.csv", index=False)
    forecast_df.to_csv(run_dir / "var_forecast.csv", index=False)

    return {
        "maxlags_requested": maxlags,
        "selected_lag_order": selected_lag_order,
        "nobs": int(fitted.nobs),
        "neqs": len(fitted.names),
        "run_dir": str(run_dir),
        "granger_btc": granger_btc,
        "granger_eth": granger_eth,
        "irf": irf_df,
        "forecast": forecast_df,
    }


def build_run_summary(runs: list[dict]) -> pd.DataFrame:
    rows = []

    for run in runs:
        rows.append(
            {
                "maxlags_requested": run["maxlags_requested"],
                "selected_lag_order": run["selected_lag_order"],
                "nobs": run["nobs"],
                "neqs": run["neqs"],
                "run_dir": run["run_dir"],
            }
        )

    return pd.DataFrame(rows)


def build_granger_robustness_table(
    runs: list[dict],
    target: str,
    variables: list[str],
    significance_level: float = 0.05,
    borderline_level: float = 0.10,
) -> pd.DataFrame:
    rows = []

    for variable in variables:
        row = {
            "target": target,
            "causing_variable": variable,
        }

        p_values = []

        for run in runs:
            maxlags = run["maxlags_requested"]
            granger = run["granger_btc"] if target == "btc_ret_1d" else run["granger_eth"]

            match = granger[granger["causing_variable"] == variable]

            if match.empty:
                p_value = np.nan
                conclusion = "missing"
                test_statistic = np.nan
            else:
                p_value = float(match.iloc[0]["p_value"])
                conclusion = str(match.iloc[0]["conclusion_5pct"])
                test_statistic = float(match.iloc[0]["test_statistic"])

            p_values.append(p_value)

            row[f"p_value_maxlags_{maxlags}"] = p_value
            row[f"test_stat_maxlags_{maxlags}"] = test_statistic
            row[f"conclusion_maxlags_{maxlags}"] = conclusion

        row["significant_count"] = sum(
            (not pd.isna(p)) and p < significance_level for p in p_values
        )
        row["borderline_count"] = sum(
            (not pd.isna(p)) and p < borderline_level for p in p_values
        )
        row["robustness"] = classify_robustness(
            p_values=p_values,
            significance_level=significance_level,
            borderline_level=borderline_level,
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    robustness_order = {
        "Strong": 0,
        "Moderate": 1,
        "Moderate / borderline": 2,
        "Borderline": 3,
        "Weak": 4,
        "Insufficient data": 5,
    }

    out["_order"] = out["robustness"].map(robustness_order).fillna(99)

    out = (
        out.sort_values(["_order", "significant_count", "borderline_count"], ascending=[True, False, False])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    return out


def build_irf_robustness_table(
    runs: list[dict],
    responses: tuple[str, ...] = ("btc_ret_1d", "eth_ret_1d"),
    steps: tuple[int, ...] = (1, 3, 7),
    top_n: int = 30,
) -> pd.DataFrame:
    frames = []

    for run in runs:
        maxlags = run["maxlags_requested"]
        irf = run["irf"].copy()

        focused = irf[
            (irf["response"].isin(responses))
            & (irf["impulse"] != irf["response"])
            & (irf["step"].isin(steps))
        ].copy()

        focused["maxlags_requested"] = maxlags
        focused["abs_irf_value"] = focused["irf_value"].abs()

        frames.append(focused)

    if not frames:
        return pd.DataFrame()

    all_irf = pd.concat(frames, ignore_index=True)

    grouped = (
        all_irf.groupby(["response", "impulse", "step"])
        .agg(
            runs_count=("maxlags_requested", "nunique"),
            avg_irf=("irf_value", "mean"),
            median_irf=("irf_value", "median"),
            avg_abs_irf=("abs_irf_value", "mean"),
            min_irf=("irf_value", "min"),
            max_irf=("irf_value", "max"),
        )
        .reset_index()
    )

    grouped["direction_consistent"] = (
        (grouped["min_irf"] > 0) | (grouped["max_irf"] < 0)
    )

    grouped = grouped.sort_values("avg_abs_irf", ascending=False).head(top_n).reset_index(drop=True)

    return grouped


def build_var_robustness_markdown(
    run_summary: pd.DataFrame,
    btc_table: pd.DataFrame,
    eth_table: pd.DataFrame,
    irf_table: pd.DataFrame,
    output_dir_name: str,
) -> str:
    lines = [
        "# VAR Robustness Report",
        "",
        "This report compares VAR Lite results across multiple maximum lag settings.",
        "",
        "The goal is to check whether the main conclusions survive changes in lag specification.",
        "",
        "## Files Generated",
        "",
        f"- Robustness outputs folder: `{output_dir_name}`",
        f"- Run summary: `{output_dir_name}/var_robustness_run_summary.csv`",
        f"- BTC Granger robustness: `{output_dir_name}/var_robustness_btc_granger.csv`",
        f"- ETH Granger robustness: `{output_dir_name}/var_robustness_eth_granger.csv`",
        f"- IRF robustness: `{output_dir_name}/var_robustness_irf.csv`",
        "",
        "## Run Summary",
        "",
        "| Maxlags Requested | Selected Lag Order | Observations | Equations |",
        "|---:|---:|---:|---:|",
    ]

    for _, row in run_summary.iterrows():
        lines.append(
            f"| {int(row['maxlags_requested'])} | "
            f"{int(row['selected_lag_order'])} | "
            f"{int(row['nobs'])} | "
            f"{int(row['neqs'])} |"
        )

    lines.extend(
        [
            "",
            "## BTC Return Equation — Granger Robustness",
            "",
            "| Causing Variable | Robustness | Significant Count | Borderline Count | P-Values by Maxlags |",
            "|---|---|---:|---:|---|",
        ]
    )

    for _, row in btc_table.iterrows():
        p_cols = [c for c in btc_table.columns if c.startswith("p_value_maxlags_")]
        p_text = ", ".join(
            [
                f"{col.replace('p_value_maxlags_', 'maxlags ')}: {_fmt_p(row[col])}"
                for col in p_cols
            ]
        )

        lines.append(
            f"| {row['causing_variable']} | "
            f"{row['robustness']} | "
            f"{int(row['significant_count'])} | "
            f"{int(row['borderline_count'])} | "
            f"{p_text} |"
        )

    lines.extend(
        [
            "",
            "## ETH Return Equation — Granger Robustness",
            "",
            "| Causing Variable | Robustness | Significant Count | Borderline Count | P-Values by Maxlags |",
            "|---|---|---:|---:|---|",
        ]
    )

    for _, row in eth_table.iterrows():
        p_cols = [c for c in eth_table.columns if c.startswith("p_value_maxlags_")]
        p_text = ", ".join(
            [
                f"{col.replace('p_value_maxlags_', 'maxlags ')}: {_fmt_p(row[col])}"
                for col in p_cols
            ]
        )

        lines.append(
            f"| {row['causing_variable']} | "
            f"{row['robustness']} | "
            f"{int(row['significant_count'])} | "
            f"{int(row['borderline_count'])} | "
            f"{p_text} |"
        )

    lines.extend(
        [
            "",
            "## Top Robust Impulse Responses",
            "",
            "Impulse responses are averaged across the robustness runs. Values are in standardized units.",
            "",
            "| Response | Impulse | Step | Avg IRF | Median IRF | Avg Abs IRF | Direction Consistent? |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )

    if irf_table.empty:
        lines.append("|  |  |  |  |  |  |  |")
    else:
        for _, row in irf_table.iterrows():
            lines.append(
                f"| {row['response']} | "
                f"{row['impulse']} | "
                f"{int(row['step'])} | "
                f"{_fmt_num(row['avg_irf'])} | "
                f"{_fmt_num(row['median_irf'])} | "
                f"{_fmt_num(row['avg_abs_irf'])} | "
                f"{bool(row['direction_consistent'])} |"
            )

    lines.extend(
        [
            "",
            "## Main Robustness Interpretation",
            "",
            "- **Strong** means the variable was significant at the 5% level across all VAR specifications.",
            "- **Moderate** means the variable was significant in most specifications.",
            "- **Borderline** means the variable had at least one result below the 10% level but was not consistently significant.",
            "- Stable results across lag settings are more credible than results that only appear in one VAR specification.",
            "",
            "This is a research diagnostic, not a trading recommendation.",
        ]
    )

    return "\n".join(lines)


def run_var_robustness(
    df: pd.DataFrame,
    outputs_dir: Path,
    config: VarRobustnessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    runs = []

    for maxlags in config.maxlags_list:
        run = run_single_var_robustness(
            df=df,
            maxlags=maxlags,
            outputs_dir=outputs_dir,
            forecast_steps=config.forecast_steps,
            irf_steps=config.irf_steps,
        )
        runs.append(run)

    run_summary = build_run_summary(runs)

    btc_table = build_granger_robustness_table(
        runs=runs,
        target="btc_ret_1d",
        variables=CORE_BTC_VARIABLES,
        significance_level=config.significance_level,
        borderline_level=config.borderline_level,
    )

    eth_table = build_granger_robustness_table(
        runs=runs,
        target="eth_ret_1d",
        variables=CORE_ETH_VARIABLES,
        significance_level=config.significance_level,
        borderline_level=config.borderline_level,
    )

    irf_table = build_irf_robustness_table(runs=runs)

    report = build_var_robustness_markdown(
        run_summary=run_summary,
        btc_table=btc_table,
        eth_table=eth_table,
        irf_table=irf_table,
        output_dir_name="outputs/econometrics/var_robustness",
    )

    return run_summary, btc_table, eth_table, irf_table, report