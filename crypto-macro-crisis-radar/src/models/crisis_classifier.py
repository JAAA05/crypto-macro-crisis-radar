from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_crisis_classifier(df: pd.DataFrame, feature_cols: list[str], target_col: str = "mini_crisis_next_7d") -> tuple[Pipeline, dict]:
    clean = df.dropna(subset=[target_col]).copy()
    clean = clean.iloc[:-14].copy()  
    X = clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = clean[target_col].astype(int)

    split_idx = int(len(clean) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=6,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if len(set(y_train)) > 1 else np.zeros(len(y_test))

    report = {
        "n_rows": int(len(clean)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_rate": float(y.mean()),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    try:
        report["roc_auc"] = float(roc_auc_score(y_test, proba))
    except Exception:
        report["roc_auc"] = None
    return pipe, report


def predict_crisis_probability(model: Pipeline, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    if hasattr(model.named_steps["model"], "predict_proba"):
        return pd.Series(model.predict_proba(X)[:, 1], index=df.index)
    return pd.Series(model.predict(X), index=df.index)
