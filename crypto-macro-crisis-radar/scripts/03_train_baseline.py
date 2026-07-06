from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


MACRO_FINANCIAL_FEATURES = [
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
]

MACRO_ENGINE_FEATURES = [
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
]

CROSS_ASSET_FEATURES = [
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
]

CRYPTO_MARKET_FEATURES = [
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
]

LIQUIDITY_FEATURES = [
    "stablecoin_mcap_7d_chg",
    "stablecoin_mcap_30d_chg",
    "stablecoin_mcap_z_90d",
    "liquidity_stress_score",
]

NEWS_NARRATIVE_FEATURES = [
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
]


ALL_FEATURE_COLUMNS = (
    MACRO_FINANCIAL_FEATURES
    + MACRO_ENGINE_FEATURES
    + CROSS_ASSET_FEATURES
    + CRYPTO_MARKET_FEATURES
    + LIQUIDITY_FEATURES
    + NEWS_NARRATIVE_FEATURES
)


RISK_CLASSIFIER_FEATURE_COLUMNS = (
    MACRO_FINANCIAL_FEATURES
    + CROSS_ASSET_FEATURES
    + CRYPTO_MARKET_FEATURES
    + LIQUIDITY_FEATURES
    + NEWS_NARRATIVE_FEATURES
)


SCORE_COLUMNS_WITH_DEFAULTS = {
    "macro_risk_score": 50.0,
    "macro_engine_stress_index": 50.0,
    "growth_slowdown_score": 50.0,
    "labor_stress_score": 50.0,
    "inflation_pressure_score": 50.0,
    "monetary_tightening_score": 50.0,
    "fiscal_stress_score": 50.0,
    "cross_asset_risk_score": 50.0,
    "equity_risk_score": 50.0,
    "safe_haven_demand_score": 50.0,
    "commodity_pressure_score": 50.0,
    "crypto_stress_score": 50.0,
    "liquidity_stress_score": 50.0,
    "news_risk_score": 50.0,
    "narrative_ai_risk_score": 50.0,
    "crypto_narrative_risk_score": 50.0,
    "macro_narrative_risk_score": 50.0,
    "blended_news_narrative_risk_score": 50.0,
    "narrative_elevated_category_count": 0.0,
    "narrative_high_category_count": 0.0,
    "narrative_ai_fallback_used": 0.0,
    "narrative_ai_category_article_total": 0.0,
    "anomaly_score": 0.0,
    "model_crisis_probability": 0.0,
}


def classify_regime(score: float) -> str:
    if score < 30:
        return "Risk-On"
    if score < 50:
        return "Neutral"
    if score < 70:
        return "Risk-Off"
    if score < 85:
        return "Mini-Crisis Watch"
    return "Crisis"


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    output = []

    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)

    return output


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    unique_cols = unique_preserve_order(columns)
    return [col for col in unique_cols if col in df.columns]


def ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, default in SCORE_COLUMNS_WITH_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    existing_features = existing_columns(df, feature_columns)

    if not existing_features:
        raise ValueError("No usable model features found in master_features.csv")

    X = df[existing_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, existing_features


def safe_metric(func, *args, default: float = np.nan, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return default


def build_classifier(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=6,
                    min_samples_leaf=15,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def choose_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> tuple[float, pd.DataFrame]:
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": safe_metric(
                    precision_score,
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "recall": safe_metric(
                    recall_score,
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "f1": safe_metric(
                    f1_score,
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "balanced_accuracy": safe_metric(
                    balanced_accuracy_score,
                    y_true,
                    y_pred,
                ),
                "positive_prediction_rate": float(np.mean(y_pred)),
            }
        )

    threshold_df = pd.DataFrame(rows)

    if metric not in threshold_df.columns:
        metric = "f1"

    best = (
        threshold_df.sort_values(
            [metric, "recall", "precision"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )

    return float(best["threshold"]), threshold_df


def evaluate_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "roc_auc": safe_metric(roc_auc_score, y_true, y_proba),
        "average_precision": safe_metric(average_precision_score, y_true, y_proba),
        "brier_score": safe_metric(brier_score_loss, y_true, y_proba),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
        "precision": safe_metric(
            precision_score,
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": safe_metric(
            recall_score,
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": safe_metric(
            f1_score,
            y_true,
            y_pred,
            zero_division=0,
        ),
        "positive_prediction_rate": float(np.mean(y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def add_anomaly_scores(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    print("Training anomaly model...")

    df = df.copy()

    anomaly_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination=0.05,
                    random_state=42,
                ),
            ),
        ]
    )

    anomaly_pipeline.fit(X)

    model = anomaly_pipeline.named_steps["model"]
    transformed_X = anomaly_pipeline[:-1].transform(X)

    raw_anomaly = -model.decision_function(transformed_X)
    anomaly_score = pd.Series(raw_anomaly, index=df.index).rank(pct=True) * 100

    df["anomaly_score"] = anomaly_score.clip(0, 100).fillna(0)

    return df


def train_ablation_selected_risk_classifier(
    df: pd.DataFrame,
    X: pd.DataFrame,
    features_used: list[str],
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    print("Training Ablation-Selected Risk Classifier v1...")

    df = df.copy()

    target_col = "mini_crisis_next_7d"

    if target_col not in df.columns:
        print(f"[WARN] {target_col} not found. Using zero crisis probabilities.")
        df["model_crisis_probability"] = 0.0
        return df, {"warning": f"{target_col} not found"}, pd.DataFrame(), pd.DataFrame()

    y = pd.to_numeric(df[target_col], errors="coerce")

    train_mask = y.notna()
    X_labeled = X.loc[train_mask].copy()
    y_labeled = y.loc[train_mask].astype(int)

    if y_labeled.nunique() < 2:
        print("[WARN] Target has only one class. Using zero crisis probabilities.")
        df["model_crisis_probability"] = 0.0
        return df, {"warning": "Target has only one class"}, pd.DataFrame(), pd.DataFrame()

    n = len(X_labeled)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    if train_end < 100:
        train_end = int(n * 0.60)
        val_end = int(n * 0.80)

    X_train = X_labeled.iloc[:train_end]
    y_train = y_labeled.iloc[:train_end]

    X_val = X_labeled.iloc[train_end:val_end]
    y_val = y_labeled.iloc[train_end:val_end]

    X_test = X_labeled.iloc[val_end:]
    y_test = y_labeled.iloc[val_end:]

    report: dict = {
        "classifier_name": "Ablation-Selected Risk Classifier v1",
        "classifier_feature_set": "refined_no_macro_engine",
        "target": target_col,
        "n_rows_total": int(len(df)),
        "n_labeled_rows": int(len(X_labeled)),
        "n_train_rows": int(len(X_train)),
        "n_validation_rows": int(len(X_val)),
        "n_test_rows": int(len(X_test)),
        "positive_rate_labeled": float(y_labeled.mean()),
        "positive_rate_train": float(y_train.mean()) if len(y_train) else None,
        "positive_rate_validation": float(y_val.mean()) if len(y_val) else None,
        "positive_rate_test": float(y_test.mean()) if len(y_test) else None,
        "features_used": list(features_used),
        "excluded_from_classifier": "macro_engine feature group",
        "reason_for_exclusion": (
            "Ablation v2 showed that refined_no_macro_engine improved walk-forward "
            "ROC-AUC relative to the full feature set."
        ),
        "macro_engine_usage_note": (
            "Macro Engine remains active in rule-based scoring, daily report, VAR, "
            "dashboard, and interpretation layers."
        ),
    }

    if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
        print("[WARN] Train/validation/test split has insufficient class variation.")
        report["warning"] = "Insufficient class variation in train/validation/test split."

        final_model = build_classifier(random_state=42)
        final_model.fit(X_labeled, y_labeled)

        classes = list(final_model.named_steps["model"].classes_)

        if 1 in classes:
            positive_idx = classes.index(1)
            df["model_crisis_probability"] = final_model.predict_proba(X)[:, positive_idx] * 100
        else:
            df["model_crisis_probability"] = 0.0

        feature_importance = pd.DataFrame(
            {
                "feature": features_used,
                "importance": final_model.named_steps["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return df, report, feature_importance, pd.DataFrame()

    eval_model = build_classifier(random_state=42)
    eval_model.fit(X_train, y_train)

    classes = list(eval_model.named_steps["model"].classes_)

    if 1 not in classes:
        print("[WARN] Positive class missing from evaluation model.")
        report["warning"] = "Positive class missing from evaluation model."
        df["model_crisis_probability"] = 0.0
        return df, report, pd.DataFrame(), pd.DataFrame()

    positive_idx = classes.index(1)

    val_proba = eval_model.predict_proba(X_val)[:, positive_idx]
    test_proba = eval_model.predict_proba(X_test)[:, positive_idx]

    tuned_threshold, threshold_df = choose_threshold(
        y_true=y_val,
        y_proba=val_proba,
        metric="f1",
    )

    report["tuned_threshold"] = tuned_threshold
    report["validation_metrics_tuned_threshold"] = evaluate_predictions(
        y_true=y_val,
        y_proba=val_proba,
        threshold=tuned_threshold,
    )
    report["test_metrics_tuned_threshold"] = evaluate_predictions(
        y_true=y_test,
        y_proba=test_proba,
        threshold=tuned_threshold,
    )
    report["test_metrics_default_threshold_0_50"] = evaluate_predictions(
        y_true=y_test,
        y_proba=test_proba,
        threshold=0.50,
    )

    final_model = build_classifier(random_state=42)
    final_model.fit(X_labeled, y_labeled)

    final_classes = list(final_model.named_steps["model"].classes_)

    if 1 in final_classes:
        final_positive_idx = final_classes.index(1)
        df["model_crisis_probability"] = final_model.predict_proba(X)[:, final_positive_idx] * 100
    else:
        df["model_crisis_probability"] = 0.0

    feature_importance = pd.DataFrame(
        {
            "feature": features_used,
            "importance": final_model.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return df, report, feature_importance, threshold_df


def add_final_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("Scoring final market regime...")

    df = ensure_score_columns(df)

    df["rule_based_risk_score"] = (
        0.11 * df["macro_risk_score"]
        + 0.15 * df["macro_engine_stress_index"]
        + 0.14 * df["cross_asset_risk_score"]
        + 0.19 * df["crypto_stress_score"]
        + 0.11 * df["liquidity_stress_score"]
        + 0.12 * df["blended_news_narrative_risk_score"]
        + 0.03 * (df["narrative_elevated_category_count"].clip(0, 10) * 10)
        + 0.15 * df["anomaly_score"]
    ).clip(0, 100)

    df["final_risk_score"] = df["rule_based_risk_score"]

    df["final_risk_score_with_model"] = (
        0.75 * df["rule_based_risk_score"]
        + 0.25 * df["model_crisis_probability"]
    ).clip(0, 100)

    df["market_regime"] = df["final_risk_score_with_model"].apply(classify_regime)

    return df


def main() -> None:
    features_path = ROOT / "data/features/master_features.csv"

    if not features_path.exists():
        raise FileNotFoundError(
            "data/features/master_features.csv not found. "
            "Run scripts/02_build_features.py first."
        )

    df = pd.read_csv(features_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    X_anomaly, anomaly_features_used = build_feature_matrix(
        df,
        ALL_FEATURE_COLUMNS,
    )

    X_classifier, classifier_features_used = build_feature_matrix(
        df,
        RISK_CLASSIFIER_FEATURE_COLUMNS,
    )

    df = add_anomaly_scores(df, X_anomaly)

    (
        df,
        model_report,
        feature_importance,
        threshold_table,
    ) = train_ablation_selected_risk_classifier(
        df=df,
        X=X_classifier,
        features_used=classifier_features_used,
    )

    df = add_final_scores(df)

    output_path = ROOT / "data/processed/scored_regime_history.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report_path = ROOT / "outputs/scores/baseline_model_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model_report["anomaly_features_used"] = anomaly_features_used
    model_report["classifier_features_used"] = classifier_features_used

    with open(report_path, "w") as f:
        json.dump(model_report, f, indent=2)

    if not feature_importance.empty:
        importance_path = ROOT / "outputs/scores/ablation_selected_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        print(
            "Saved ablation-selected feature importance to "
            "outputs/scores/ablation_selected_feature_importance.csv"
        )

    if not threshold_table.empty:
        threshold_path = ROOT / "outputs/scores/ablation_selected_threshold_table.csv"
        threshold_table.to_csv(threshold_path, index=False)
        print(
            "Saved ablation-selected threshold table to "
            "outputs/scores/ablation_selected_threshold_table.csv"
        )

    print("Saved scored history to data/processed/scored_regime_history.csv")
    print("Saved model report to outputs/scores/baseline_model_report.json")


if __name__ == "__main__":
    main()
