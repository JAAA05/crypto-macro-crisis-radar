from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_anomaly_model(df: pd.DataFrame, feature_cols: list[str]) -> tuple[Pipeline, pd.Series]:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(n_estimators=300, contamination="auto", random_state=42)),
        ]
    )
    pipe.fit(X)
    raw = pipe.named_steps["model"].decision_function(pipe[:-1].transform(X))
    anomaly_score = pd.Series(100 * (1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)), index=df.index)
    return pipe, anomaly_score.clip(0, 100)
