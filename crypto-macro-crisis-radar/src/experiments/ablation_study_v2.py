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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.experiments.ablation_study import FEATURE_GROUPS


@dataclass
class AblationV2Config:
    target_col: str = "mini_crisis_next_7d"
    n_folds: int = 5
    initial_train_size: float = 0.50
    validation_size: float = 0.10
    test_size: float = 0.10
    random_state: int = 42
    min_train_rows: int = 500
    min_validation_rows: int = 120
    min_test_rows: int = 120


def existing_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in df.columns]


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []

    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)

    return out


def group_features(df: pd.DataFrame, group_name: str) -> list[str]:
    return existing_features(df, FEATURE_GROUPS.get(group_name, []))


def combine_groups(df: pd.DataFrame, groups: list[str]) -> list[str]:
    features: list[str] = []

    for group in groups:
        features.extend(group_features(df, group))

    return unique_preserve_order(features)


def get_all_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []

    for group in FEATURE_GROUPS:
        features.extend(group_features(df, group))

    return unique_preserve_order(features)


def build_experiment_sets(df: pd.DataFrame) -> dict[str, dict]:
    all_features = get_all_features(df)

    experiments: dict[str, dict] = {
        "full_model": {
            "ablation_type": "full",
            "features": all_features,
            "description": "All feature groups.",
        },
        "refined_no_macro_engine": {
            "ablation_type": "refined",
            "features": combine_groups(
                df,
                [
                    "macro_financial",
                    "cross_asset",
                    "crypto_market",
                    "liquidity",
                    "news_narrative",
                ],
            ),
            "description": "Full model without Macro Engine features.",
        },
        "refined_core_market_macro_liquidity": {
            "ablation_type": "refined",
            "features": combine_groups(
                df,
                [
                    "macro_financial",
                    "crypto_market",
                    "liquidity",
                ],
            ),
            "description": "Core market + macro financial + liquidity feature set.",
        },
        "refined_market_cross_liquidity_narrative": {
            "ablation_type": "refined",
            "features": combine_groups(
                df,
                [
                    "cross_asset",
                    "crypto_market",
                    "liquidity",
                    "news_narrative",
                ],
            ),
            "description": "Market, cross-asset, liquidity, and narrative only.",
        },
    }

    for group_name in FEATURE_GROUPS:
        group_existing = set(group_features(df, group_name))

        experiments[f"drop_{group_name}"] = {
            "ablation_type": "drop_group",
            "removed_group": group_name,
            "features": [f for f in all_features if f not in group_existing],
            "description": f"All features except {group_name}.",
        }

    for group_name in FEATURE_GROUPS:
        experiments[f"only_{group_name}"] = {
            "ablation_type": "only_group",
            "only_group": group_name,
            "features": group_features(df, group_name),
            "description": f"Only {group_name} features.",
        }

    return experiments


def build_model(random_state: int = 42) -> Pipeline:
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


def safe_metric(func, *args, default: float = np.nan, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return default


def make_walk_forward_folds(n_rows: int, config: AblationV2Config) -> list[dict]:
    val_size = max(
        config.min_validation_rows,
        int(n_rows * config.validation_size),
    )
    test_size = max(
        config.min_test_rows,
        int(n_rows * config.test_size),
    )
    min_train_end = max(
        config.min_train_rows,
        int(n_rows * config.initial_train_size),
    )

    max_train_end = n_rows - val_size - test_size

    if max_train_end <= min_train_end:
        raise ValueError(
            "Not enough rows for requested walk-forward split. "
            f"n_rows={n_rows}, min_train_end={min_train_end}, "
            f"val_size={val_size}, test_size={test_size}"
        )

    train_ends = np.linspace(
        min_train_end,
        max_train_end,
        config.n_folds,
    ).astype(int)

    folds = []

    for fold_id, train_end in enumerate(train_ends, start=1):
        val_start = train_end
        val_end = val_start + val_size
        test_start = val_end
        test_end = test_start + test_size

        if test_end > n_rows:
            continue

        folds.append(
            {
                "fold": fold_id,
                "train_start": 0,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_rows": train_end,
                "validation_rows": val_end - val_start,
                "test_rows": test_end - test_start,
            }
        )

    return folds


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
                "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
                "recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
                "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
                "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
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
        "roc_auc": safe_metric(roc_auc_score, y_true, y_proba),
        "average_precision": safe_metric(average_precision_score, y_true, y_proba),
        "brier_score": safe_metric(brier_score_loss, y_true, y_proba),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
        "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "positive_prediction_rate": float(np.mean(y_pred)),
    }


def evaluate_experiment_fold(
    data: pd.DataFrame,
    features: list[str],
    experiment_name: str,
    experiment_meta: dict,
    fold: dict,
    config: AblationV2Config,
) -> tuple[dict, pd.DataFrame]:
    features = existing_features(data, features)

    base_result = {
        "experiment": experiment_name,
        "ablation_type": experiment_meta.get("ablation_type", ""),
        "removed_group": experiment_meta.get("removed_group", ""),
        "only_group": experiment_meta.get("only_group", ""),
        "description": experiment_meta.get("description", ""),
        "fold": fold["fold"],
        "n_features": len(features),
        "train_rows": fold["train_rows"],
        "validation_rows": fold["validation_rows"],
        "test_rows": fold["test_rows"],
    }

    if not features:
        base_result["status"] = "no_features"
        return base_result, pd.DataFrame()

    train = data.iloc[fold["train_start"]:fold["train_end"]].copy()
    val = data.iloc[fold["val_start"]:fold["val_end"]].copy()
    test = data.iloc[fold["test_start"]:fold["test_end"]].copy()

    X_train = train[features].replace([np.inf, -np.inf], np.nan)
    y_train = train[config.target_col].astype(int)

    X_val = val[features].replace([np.inf, -np.inf], np.nan)
    y_val = val[config.target_col].astype(int)

    X_test = test[features].replace([np.inf, -np.inf], np.nan)
    y_test = test[config.target_col].astype(int)

    base_result["positive_rate_train"] = float(y_train.mean())
    base_result["positive_rate_validation"] = float(y_val.mean())
    base_result["positive_rate_test"] = float(y_test.mean())

    if y_train.nunique() < 2:
        base_result["status"] = "train_one_class"
        return base_result, pd.DataFrame()

    if y_val.nunique() < 2:
        base_result["status"] = "validation_one_class"
        return base_result, pd.DataFrame()

    if y_test.nunique() < 2:
        base_result["status"] = "test_one_class"
        return base_result, pd.DataFrame()

    model = build_model(random_state=config.random_state)
    model.fit(X_train, y_train)

    classes = list(model.named_steps["model"].classes_)

    if 1 not in classes:
        base_result["status"] = "positive_class_missing"
        return base_result, pd.DataFrame()

    positive_idx = classes.index(1)

    val_proba = model.predict_proba(X_val)[:, positive_idx]
    test_proba = model.predict_proba(X_test)[:, positive_idx]

    tuned_threshold, threshold_df = choose_threshold(
        y_true=y_val,
        y_proba=val_proba,
        metric="f1",
    )

    validation_metrics = evaluate_predictions(
        y_true=y_val,
        y_proba=val_proba,
        threshold=tuned_threshold,
    )

    test_metrics_tuned = evaluate_predictions(
        y_true=y_test,
        y_proba=test_proba,
        threshold=tuned_threshold,
    )

    test_metrics_default = evaluate_predictions(
        y_true=y_test,
        y_proba=test_proba,
        threshold=0.50,
    )

    result = {
        **base_result,
        "status": "ok",
        "tuned_threshold": tuned_threshold,
    }

    for key, value in validation_metrics.items():
        result[f"val_{key}"] = value

    for key, value in test_metrics_tuned.items():
        result[f"test_tuned_{key}"] = value

    for key, value in test_metrics_default.items():
        result[f"test_default_{key}"] = value

    feature_importance = pd.DataFrame(
        {
            "experiment": experiment_name,
            "fold": fold["fold"],
            "feature": features,
            "importance": model.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return result, feature_importance


def run_ablation_v2(
    df: pd.DataFrame,
    config: AblationV2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.copy()

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").reset_index(drop=True)

    if config.target_col not in data.columns:
        raise ValueError(f"Target column not found: {config.target_col}")

    data[config.target_col] = pd.to_numeric(data[config.target_col], errors="coerce")
    data = data.dropna(subset=[config.target_col]).reset_index(drop=True)
    data[config.target_col] = data[config.target_col].astype(int)

    experiments = build_experiment_sets(data)
    folds = make_walk_forward_folds(len(data), config=config)

    fold_rows = []
    importance_frames = []

    for experiment_name, meta in experiments.items():
        features = meta["features"]

        print(f"Running experiment: {experiment_name} ({len(features)} features)")

        for fold in folds:
            result, importance = evaluate_experiment_fold(
                data=data,
                features=features,
                experiment_name=experiment_name,
                experiment_meta=meta,
                fold=fold,
                config=config,
            )

            fold_rows.append(result)

            if not importance.empty:
                importance_frames.append(importance)

    fold_results = pd.DataFrame(fold_rows)

    ok_results = fold_results[fold_results["status"] == "ok"].copy()

    metric_cols = [
        col
        for col in ok_results.columns
        if col.startswith("val_")
        or col.startswith("test_tuned_")
        or col.startswith("test_default_")
        or col in [
            "tuned_threshold",
            "n_features",
            "positive_rate_train",
            "positive_rate_validation",
            "positive_rate_test",
        ]
    ]

    agg_map = {col: ["mean", "std"] for col in metric_cols if col in ok_results.columns}

    summary = (
        ok_results.groupby(
            [
                "experiment",
                "ablation_type",
                "removed_group",
                "only_group",
                "description",
            ],
            dropna=False,
        )
        .agg(agg_map)
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    fold_counts = (
        ok_results.groupby("experiment")
        .size()
        .reset_index(name="successful_folds")
    )

    summary = summary.merge(fold_counts, on="experiment", how="left")

    full = summary[summary["experiment"] == "full_model"]

    if not full.empty:
        full_auc = float(full.iloc[0].get("test_tuned_roc_auc_mean", np.nan))
        full_ap = float(full.iloc[0].get("test_tuned_average_precision_mean", np.nan))
        full_f1 = float(full.iloc[0].get("test_tuned_f1_mean", np.nan))

        summary["auc_delta_vs_full"] = summary["test_tuned_roc_auc_mean"] - full_auc
        summary["auc_loss_vs_full"] = full_auc - summary["test_tuned_roc_auc_mean"]

        summary["ap_delta_vs_full"] = summary["test_tuned_average_precision_mean"] - full_ap
        summary["ap_loss_vs_full"] = full_ap - summary["test_tuned_average_precision_mean"]

        summary["f1_delta_vs_full"] = summary["test_tuned_f1_mean"] - full_f1
        summary["f1_loss_vs_full"] = full_f1 - summary["test_tuned_f1_mean"]

    if importance_frames:
        feature_importance = pd.concat(importance_frames, ignore_index=True)
    else:
        feature_importance = pd.DataFrame()

    if not feature_importance.empty:
        importance_summary = (
            feature_importance.groupby(["experiment", "feature"])
            .agg(
                mean_importance=("importance", "mean"),
                std_importance=("importance", "std"),
                folds=("fold", "nunique"),
            )
            .reset_index()
            .sort_values(["experiment", "mean_importance"], ascending=[True, False])
        )
    else:
        importance_summary = pd.DataFrame()

    fold_config_df = pd.DataFrame(folds)

    return fold_results, summary, importance_summary, fold_config_df


def fmt_num(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def build_ablation_v2_report(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
    fold_config: pd.DataFrame,
) -> str:
    if summary.empty:
        return "# Ablation Study v2 Report\n\nNo successful ablation results were generated.\n"

    full = summary[summary["experiment"] == "full_model"]

    lines = [
        "# Ablation Study v2 Report",
        "",
        "This report improves the first ablation by using walk-forward validation and threshold tuning.",
        "",
        "Instead of relying on one final 20% split and a fixed 0.50 threshold, each fold uses train → validation → test windows. The classification threshold is selected on the validation window and evaluated on the next test window.",
        "",
        "## Walk-Forward Fold Design",
        "",
        "| Fold | Train Rows | Validation Rows | Test Rows | Train End | Validation End | Test End |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in fold_config.iterrows():
        lines.append(
            f"| {int(row['fold'])} | "
            f"{int(row['train_rows'])} | "
            f"{int(row['validation_rows'])} | "
            f"{int(row['test_rows'])} | "
            f"{int(row['train_end'])} | "
            f"{int(row['val_end'])} | "
            f"{int(row['test_end'])} |"
        )

    if not full.empty:
        f = full.iloc[0]

        lines.extend(
            [
                "",
                "## Full Model — Walk-Forward Average",
                "",
                "| Metric | Mean | Std |",
                "|---|---:|---:|",
                f"| Tuned Threshold | {fmt_num(f.get('tuned_threshold_mean'))} | {fmt_num(f.get('tuned_threshold_std'))} |",
                f"| Test ROC-AUC | {fmt_num(f.get('test_tuned_roc_auc_mean'))} | {fmt_num(f.get('test_tuned_roc_auc_std'))} |",
                f"| Test Average Precision | {fmt_num(f.get('test_tuned_average_precision_mean'))} | {fmt_num(f.get('test_tuned_average_precision_std'))} |",
                f"| Test Balanced Accuracy | {fmt_num(f.get('test_tuned_balanced_accuracy_mean'))} | {fmt_num(f.get('test_tuned_balanced_accuracy_std'))} |",
                f"| Test Precision | {fmt_num(f.get('test_tuned_precision_mean'))} | {fmt_num(f.get('test_tuned_precision_std'))} |",
                f"| Test Recall | {fmt_num(f.get('test_tuned_recall_mean'))} | {fmt_num(f.get('test_tuned_recall_std'))} |",
                f"| Test F1 | {fmt_num(f.get('test_tuned_f1_mean'))} | {fmt_num(f.get('test_tuned_f1_std'))} |",
                f"| Default 0.50 F1 | {fmt_num(f.get('test_default_f1_mean'))} | {fmt_num(f.get('test_default_f1_std'))} |",
            ]
        )

    refined = summary[summary["ablation_type"] == "refined"].copy()

    if not refined.empty:
        refined = refined.sort_values(
            ["test_tuned_roc_auc_mean", "test_tuned_average_precision_mean"],
            ascending=[False, False],
        )

        lines.extend(
            [
                "",
                "## Refined Model Experiments",
                "",
                "| Experiment | ROC-AUC | Avg Precision | F1 | Recall | AUC Δ vs Full | F1 Δ vs Full |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for _, row in refined.iterrows():
            lines.append(
                f"| {row['experiment']} | "
                f"{fmt_num(row.get('test_tuned_roc_auc_mean'))} | "
                f"{fmt_num(row.get('test_tuned_average_precision_mean'))} | "
                f"{fmt_num(row.get('test_tuned_f1_mean'))} | "
                f"{fmt_num(row.get('test_tuned_recall_mean'))} | "
                f"{fmt_num(row.get('auc_delta_vs_full'))} | "
                f"{fmt_num(row.get('f1_delta_vs_full'))} |"
            )

    drop = summary[summary["ablation_type"] == "drop_group"].copy()

    if not drop.empty:
        drop = drop.sort_values("auc_loss_vs_full", ascending=False)

        lines.extend(
            [
                "",
                "## Drop-One-Group Ablation — Walk-Forward Average",
                "",
                "A larger AUC loss means the removed group was more useful to the classifier.",
                "",
                "| Removed Group | ROC-AUC | AUC Loss vs Full | Avg Precision | AP Loss vs Full | F1 | F1 Loss vs Full |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for _, row in drop.iterrows():
            lines.append(
                f"| {row.get('removed_group', '')} | "
                f"{fmt_num(row.get('test_tuned_roc_auc_mean'))} | "
                f"{fmt_num(row.get('auc_loss_vs_full'))} | "
                f"{fmt_num(row.get('test_tuned_average_precision_mean'))} | "
                f"{fmt_num(row.get('ap_loss_vs_full'))} | "
                f"{fmt_num(row.get('test_tuned_f1_mean'))} | "
                f"{fmt_num(row.get('f1_loss_vs_full'))} |"
            )

    only = summary[summary["ablation_type"] == "only_group"].copy()

    if not only.empty:
        only = only.sort_values(
            ["test_tuned_roc_auc_mean", "test_tuned_average_precision_mean"],
            ascending=[False, False],
        )

        lines.extend(
            [
                "",
                "## Only-One-Group Models — Walk-Forward Average",
                "",
                "This shows standalone predictive value by layer.",
                "",
                "| Only Group | ROC-AUC | Avg Precision | Balanced Accuracy | F1 | Recall |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )

        for _, row in only.iterrows():
            lines.append(
                f"| {row.get('only_group', '')} | "
                f"{fmt_num(row.get('test_tuned_roc_auc_mean'))} | "
                f"{fmt_num(row.get('test_tuned_average_precision_mean'))} | "
                f"{fmt_num(row.get('test_tuned_balanced_accuracy_mean'))} | "
                f"{fmt_num(row.get('test_tuned_f1_mean'))} | "
                f"{fmt_num(row.get('test_tuned_recall_mean'))} |"
            )

    if not importance_summary.empty:
        full_importance = importance_summary[
            importance_summary["experiment"] == "full_model"
        ].copy()

        top = full_importance.sort_values("mean_importance", ascending=False).head(25)

        lines.extend(
            [
                "",
                "## Top Full-Model Feature Importances Across Walk-Forward Folds",
                "",
                "| Rank | Feature | Mean Importance | Std | Folds |",
                "|---:|---|---:|---:|---:|",
            ]
        )

        for idx, (_, row) in enumerate(top.iterrows(), start=1):
            lines.append(
                f"| {idx} | {row['feature']} | "
                f"{fmt_num(row.get('mean_importance'))} | "
                f"{fmt_num(row.get('std_importance'))} | "
                f"{int(row.get('folds', 0))} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Walk-forward validation is more realistic than a single train/test split for market data.",
            "- Threshold tuning prevents the model from being judged only at the default 0.50 probability cutoff.",
            "- ROC-AUC and Average Precision evaluate ranking quality; F1/recall/precision evaluate classification behavior after thresholding.",
            "- If refined models beat the full model, the full model may have noisy or redundant feature groups.",
            "- This is research validation, not a trading signal.",
        ]
    )

    return "\n".join(lines)


def save_ablation_v2_outputs(
    root: Path,
    input_path: Path,
    config: AblationV2Config,
) -> dict[str, str]:
    df = pd.read_csv(input_path, parse_dates=["date"])

    fold_results, summary, importance_summary, fold_config = run_ablation_v2(
        df=df,
        config=config,
    )

    output_dir = root / "outputs/ablation_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results_path = output_dir / "ablation_v2_fold_results.csv"
    summary_path = output_dir / "ablation_v2_summary.csv"
    importance_path = output_dir / "ablation_v2_feature_importance.csv"
    fold_config_path = output_dir / "ablation_v2_fold_config.csv"
    report_path = output_dir / "ablation_v2_report.md"

    fold_results.to_csv(fold_results_path, index=False)
    summary.to_csv(summary_path, index=False)
    importance_summary.to_csv(importance_path, index=False)
    fold_config.to_csv(fold_config_path, index=False)

    report = build_ablation_v2_report(
        fold_results=fold_results,
        summary=summary,
        importance_summary=importance_summary,
        fold_config=fold_config,
    )

    report_path.write_text(report)

    return {
        "fold_results": str(fold_results_path),
        "summary": str(summary_path),
        "importance": str(importance_path),
        "fold_config": str(fold_config_path),
        "report": str(report_path),
    }