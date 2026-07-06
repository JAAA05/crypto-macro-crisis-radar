from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv(project_root() / ".env")

    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: Dict[str, Any]) -> None:
    root = project_root()
    for key, rel in config["paths"].items():
        (root / rel).mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, rel_path: str) -> Path:
    path = project_root() / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_csv(rel_path: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(project_root() / rel_path, parse_dates=parse_dates)


def save_json(obj: Any, rel_path: str) -> Path:
    path = project_root() / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def safe_zscore(series: pd.Series, window: int = 90) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=max(10, window // 4)).mean()
    rolling_std = series.rolling(window, min_periods=max(10, window // 4)).std()
    return (series - rolling_mean) / rolling_std.replace(0, pd.NA)


def clip_0_100(x: pd.Series | float) -> pd.Series | float:
    if isinstance(x, pd.Series):
        return x.clip(lower=0, upper=100)
    return max(0, min(100, float(x)))


def score_from_z(z: pd.Series, center: float = 50, scale: float = 12) -> pd.Series:
    return clip_0_100(center + scale * z.fillna(0))
