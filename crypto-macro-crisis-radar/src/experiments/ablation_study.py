from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


@dataclass
class AblationConfig:
    target_col: str = "mini_crisis_next_7d"
    test_size: float = 0.20
    random_state: int = 42


FEATURE_GROUPS = {
    "macro_financial": [
        "macro_risk_score",
        "yield_curve_10y_2y",
        "yield_curve_inversion_flag",
        "vix",
        "vix_change_5d",
        "vix_z_90d",
        "treasury_10y",
        "treasury_2y",
        "fed_funds_rate",
        "dollar_ret_20d",
        "dollar_z_90d",
        "fed_balance_sheet_90d_chg",
        "m2_90d_chg",
        "epu_z_90d",
    ],
    "macro_engine": [
        "real_gdp_yoy",
        "real_gdp_6m_chg",
        "industrial_production_yoy",
        "industrial_production_6m_chg",
        "retail_sales_yoy",
        "retail_sales_6m_chg",
        "private_investment_yoy",
        "private_investment_6m_chg",
        "unemployment_rate",
        "unemployment_change_3m",
        "unemployment_change_6m",
        "unemployment_z_3y",
        "cpi_yoy",
        "core_cpi_yoy",
        "cpi_3m_chg",
        "core_cpi_3m_chg",
        "fed_funds_change_6m",
        "treasury_2y_change_6m",
        "treasury_10y_change_6m",
        "federal_deficit_pressure_z_3y",
        "growth_slowdown_score",
        "labor_stress_score",
        "inflation_pressure_score",
        "monetary_tightening_score",
        "fiscal_stress_score",
        "macro_engine_stress_index",
    ],
    "cross_asset": [
        "sp500",
        "nasdaq",
        "gold",
        "oil_wti",
        "sp500_ret_1d",
        "sp500_ret_7d",
        "sp500_ret_30d",
        "nasdaq_ret_1d",
        "nasdaq_ret_7d",
        "nasdaq_ret_30d",
        "gold_ret_7d",
        "gold_ret_30d",
        "oil_ret_7d",
        "oil_ret_30d",
        "oil_vol_30d",
        "sp500_z_90d",
        "nasdaq_z_90d",
        "gold_z_90d",
        "oil_z_90d",
        "oil_vol_z_90d",
        "equity_risk_score",
        "safe_haven_demand_score",
        "commodity_pressure_score",
        "cross_asset_risk_score",
    ],
    "crypto_market": [
        "bitcoin_ret_1d",
        "bitcoin_ret_3d",
        "bitcoin_ret_7d",
        "bitcoin_vol_7d",
        "bitcoin_vol_30d",
        "bitcoin_drawdown_30d",
        "bitcoin_drawdown_90d",
        "bitcoin_volume_z_30d",
        "ethereum_ret_1d",
        "ethereum_ret_3d",
        "ethereum_ret_7d",
        "ethereum_vol_7d",
        "ethereum_vol_30d",
        "ethereum_drawdown_30d",
        "ethereum_drawdown_90d",
        "ethereum_volume_z_30d",
        "btc_eth_corr_30d",
        "crypto_stress_score",
    ],
    "liquidity": [
        "stablecoin_mcap_7d_chg",
        "stablecoin_mcap_30d_chg",
        "stablecoin_mcap_z_90d",
        "liquidity_stress_score",
    ],
    "news_narrative": [
        "news_article_count",
        "news_negative_hits",
        "news_positive_hits",
        "news_risk_score",
        "narrative_ai_risk_score",
        "crypto_narrative_risk_score",
        "macro_narrative_risk_score",
        "blended_news_narrative_risk_score",
        "narrative_elevated_category_count",
        "narrative_high_category_count",
        "narrative_ai_fallback_used",
        "narrative_ai_category_article_total",
        "fed_policy_enhanced_risk_score",
        "inflation_enhanced_risk_score",
        "recession_enhanced_risk_score",
        "fiscal_policy_enhanced_risk_score",
        "crypto_regulation_enhanced_risk_score",
        "exchange_risk_enhanced_risk_score",
        "stablecoin_risk_enhanced_risk_score",
        "etf_institutional_enhanced_risk_score",
        "geopolitical_enhanced_risk_score",
        "market_sentiment_enhanced_risk_score",
    ],
}


def existing_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in df.columns]


def get_all_grouped_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []

    for group_features in FEATURE_GROUPS.values():
        for feature in group_features:
            if feature in df.columns and feature not in features:
                features.append(feature)

    return features


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=7,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def safe_metric(func, *args, default: float = np.nan):
    try:
        return float(func(*args))
    except Exception:
        return default


def evaluate_feature_set(
    df: pd.DataFrame,
    features: list[str],
    config: AblationConfig,
    experiment_name: str,
    ablation_type: str,
    removed_group: str | None = None,
    only_group: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    features = existing_features(df, features)

    if not features:
        return {
            "experiment": experiment_name,
            "ablation_type": ablation_type,
            "removed_group": removed_group or "",
            "only_group": only_group or "",
            "n_features": 0,
            "status": "no_features",
        }, pd.DataFrame()

    if config.target_col not in df.columns:
        raise ValueError(f"Target column not found: {config.target_col}")

    data = df.copy()
    data[config.target_col] = pd.to_numeric(data[config.target_col], errors="coerce")

    data = data.dropna(subset=[config.target_col]).reset_index(drop=True)
    data = data.sort_values("date").reset_index(drop=True) if "date" in data.columns else data

    X = data[features].replace([np.inf, -np.inf], np.nan)
    y = data[config.target_col].astype(int)

    if y.nunique() < 2:
        return {
            "experiment": experiment_name,
            "ablation_type": ablation_type,
            "removed_group": removed_group or "",
            "only_group": only_group or "",
            "n_features": len(features),
            "status": "target_one_class",
        }, pd.DataFrame()

    split_idx = int(len(data) * (1 - config.test_size))

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    if y_train.nunique() < 2:
        return {
            "experiment": experiment_name,
            "ablation_type": ablation_type,
            "removed_group": removed_group or "",
            "only_group": only_group or "",
            "n_features": len(features),
            "status": "train_one_class",
        }, pd.DataFrame()

    model = build_model(random_state=config.random_state)
    model.fit(X_train, y_train)

    classes = list(model.named_steps["model"].classes_)

    if 1 not in classes:
        return {
            "experiment": experiment_name,
            "ablation_type": ablation_type,
            "removed_group": removed_group or "",
            "only_group": only_group or "",
            "n_features": len(features),
            "status": "positive_class_missing",
        }, pd.DataFrame()

    positive_idx = classes.index(1)

    y_proba = model.predict_proba(X_test)[:, positive_idx]
    y_pred = (y_proba >= 0.50).astype(int)

    metrics = {
        "experiment": experiment_name,
        "ablation_type": ablation_type,
        "removed_group": removed_group or "",
        "only_group": only_group or "",
        "n_features": len(features),
        "n_rows": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "status": "ok",
        "roc_auc": safe_metric(roc_auc_score, y_test, y_proba),
        "average_precision": safe_metric(average_precision_score, y_test, y_proba),
        "brier_score": safe_metric(brier_score_loss, y_test, y_proba),
        "accuracy": safe_metric(accuracy_score, y_test, y_pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_test, y_pred),
        "precision": safe_metric(precision_score, y_test, y_pred, default=0),
        "recall": safe_metric(recall_score, y_test, y_pred, default=0),
        "f1": safe_metric(f1_score, y_test, y_pred, default=0),
    }

    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.named_steps["model"].feature_importances_,
            "experiment": experiment_name,
        }
    ).sort_values("importance", ascending=False)

    return metrics, feature_importance


def run_ablation_study(
    df: pd.DataFrame,
    config: AblationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_features = get_all_grouped_features(df)

    if not all_features:
        raise ValueError("No grouped features found for ablation study.")

    results = []
    importance_frames = []

    full_metrics, full_importance = evaluate_feature_set(
        df=df,
        features=all_features,
        config=config,
        experiment_name="full_model",
        ablation_type="full",
    )

    results.append(full_metrics)

    if not full_importance.empty:
        importance_frames.append(full_importance)

    for group_name, group_features in FEATURE_GROUPS.items():
        group_existing = existing_features(df, group_features)

        features_without_group = [
            feature for feature in all_features if feature not in group_existing
        ]

        metrics, _ = evaluate_feature_set(
            df=df,
            features=features_without_group,
            config=config,
            experiment_name=f"drop_{group_name}",
            ablation_type="drop_group",
            removed_group=group_name,
        )

        results.append(metrics)

    for group_name, group_features in FEATURE_GROUPS.items():
        group_existing = existing_features(df, group_features)

        metrics, _ = evaluate_feature_set(
            df=df,
            features=group_existing,
            config=config,
            experiment_name=f"only_{group_name}",
            ablation_type="only_group",
            only_group=group_name,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    if "roc_auc" in results_df.columns:
        full_auc = results_df.loc[
            results_df["experiment"] == "full_model",
            "roc_auc",
        ]

        full_auc_value = float(full_auc.iloc[0]) if not full_auc.empty else np.nan

        results_df["roc_auc_delta_vs_full"] = results_df["roc_auc"] - full_auc_value
        results_df["roc_auc_loss_vs_full"] = full_auc_value - results_df["roc_auc"]

    if "average_precision" in results_df.columns:
        full_ap = results_df.loc[
            results_df["experiment"] == "full_model",
            "average_precision",
        ]

        full_ap_value = float(full_ap.iloc[0]) if not full_ap.empty else np.nan

        results_df["average_precision_delta_vs_full"] = (
            results_df["average_precision"] - full_ap_value
        )
        results_df["average_precision_loss_vs_full"] = (
            full_ap_value - results_df["average_precision"]
        )

    feature_importance_df = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame()
    )

    return results_df, feature_importance_df


def fmt_num(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def build_ablation_markdown_report(results: pd.DataFrame, feature_importance: pd.DataFrame) -> str:
    if results.empty:
        return "# Ablation Study Report\n\nNo ablation results were generated.\n"

    full = results[results["experiment"] == "full_model"].iloc[0]

    lines = [
        "# Ablation Study Report",
        "",
        "This report evaluates how much each feature layer contributes to the mini-crisis classifier.",
        "",
        "The ablation uses a time-based train/test split and compares the full model against models where one feature group is removed.",
        "",
        "## Full Model Performance",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| ROC-AUC | {fmt_num(full.get('roc_auc'))} |",
        f"| Average Precision | {fmt_num(full.get('average_precision'))} |",
        f"| Brier Score | {fmt_num(full.get('brier_score'))} |",
        f"| Balanced Accuracy | {fmt_num(full.get('balanced_accuracy'))} |",
        f"| Precision | {fmt_num(full.get('precision'))} |",
        f"| Recall | {fmt_num(full.get('recall'))} |",
        f"| F1 | {fmt_num(full.get('f1'))} |",
        f"| Features Used | {int(full.get('n_features', 0))} |",
        f"| Train Rows | {int(full.get('n_train', 0))} |",
        f"| Test Rows | {int(full.get('n_test', 0))} |",
        "",
    ]

    drop = results[results["ablation_type"] == "drop_group"].copy()

    if not drop.empty:
        drop = drop.sort_values("roc_auc_loss_vs_full", ascending=False)

        lines.extend(
            [
                "## Drop-One-Group Ablation",
                "",
                "A larger ROC-AUC loss means the removed group was more important to the classifier.",
                "",
                "| Removed Group | ROC-AUC | AUC Loss vs Full | Avg Precision | AP Loss vs Full | F1 | Features |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for _, row in drop.iterrows():
            lines.append(
                f"| {row.get('removed_group', '')} | "
                f"{fmt_num(row.get('roc_auc'))} | "
                f"{fmt_num(row.get('roc_auc_loss_vs_full'))} | "
                f"{fmt_num(row.get('average_precision'))} | "
                f"{fmt_num(row.get('average_precision_loss_vs_full'))} | "
                f"{fmt_num(row.get('f1'))} | "
                f"{int(row.get('n_features', 0))} |"
            )

        lines.append("")

    only = results[results["ablation_type"] == "only_group"].copy()

    if not only.empty:
        only = only.sort_values("roc_auc", ascending=False)

        lines.extend(
            [
                "## Only-One-Group Models",
                "",
                "This shows how predictive each layer is by itself.",
                "",
                "| Only Group | ROC-AUC | Avg Precision | Balanced Accuracy | F1 | Features |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )

        for _, row in only.iterrows():
            lines.append(
                f"| {row.get('only_group', '')} | "
                f"{fmt_num(row.get('roc_auc'))} | "
                f"{fmt_num(row.get('average_precision'))} | "
                f"{fmt_num(row.get('balanced_accuracy'))} | "
                f"{fmt_num(row.get('f1'))} | "
                f"{int(row.get('n_features', 0))} |"
            )

        lines.append("")

    if not feature_importance.empty:
        full_importance = feature_importance[
            feature_importance["experiment"] == "full_model"
        ].copy()

        top = full_importance.sort_values("importance", ascending=False).head(25)

        lines.extend(
            [
                "## Top Full-Model Feature Importances",
                "",
                "| Rank | Feature | Importance |",
                "|---:|---|---:|",
            ]
        )

        for idx, (_, row) in enumerate(top.iterrows(), start=1):
            lines.append(
                f"| {idx} | {row['feature']} | {fmt_num(row['importance'])} |"
            )

        lines.append("")

    lines.extend(
        [
            "## Interpretation Guide",
            "",
            "- If dropping a group causes a large ROC-AUC loss, that group is important.",
            "- If an only-group model performs well, that layer has standalone predictive value.",
            "- If a group has low standalone performance but hurts when removed, it may interact with other layers.",
            "- This is model validation, not proof of causality or a trading signal.",
        ]
    )

    return "\n".join(lines)


def save_ablation_outputs(
    root: Path,
    input_path: Path,
    config: AblationConfig,
) -> dict[str, str]:
    df = pd.read_csv(input_path, parse_dates=["date"])

    results, feature_importance = run_ablation_study(df, config=config)

    output_dir = root / "outputs/ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "ablation_results.csv"
    importance_path = output_dir / "feature_importance_full_model.csv"
    report_path = output_dir / "ablation_report.md"

    results.to_csv(results_path, index=False)
    feature_importance.to_csv(importance_path, index=False)

    report = build_ablation_markdown_report(results, feature_importance)
    report_path.write_text(report)

    return {
        "results": str(results_path),
        "importance": str(importance_path),
        "report": str(report_path),
    }