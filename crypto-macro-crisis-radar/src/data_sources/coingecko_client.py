from __future__ import annotations

import os
import time
from typing import Any, Dict

import pandas as pd
import requests


class CoinGeckoClient:
    """Small CoinGecko REST client for research market data only."""

    def __init__(self, base_url: str = "https://api.coingecko.com/api/v3", sleep_seconds: float = 1.5):
        self.base_url = base_url.rstrip("/")
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        api_key = os.getenv("COINGECKO_API_KEY")
        if api_key:
            self.session.headers.update({"x-cg-demo-api-key": api_key})

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(8)
            resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.sleep_seconds)
        return resp.json()

    def market_chart(self, coin_id: str, vs_currency: str = "usd", days: str = "max") -> pd.DataFrame:
        data = self._get(
            f"coins/{coin_id}/market_chart",
            params={"vs_currency": vs_currency, "days": days, "interval": "daily"},
        )

        prices = pd.DataFrame(data.get("prices", []), columns=["timestamp_ms", f"{coin_id}_price"])
        market_caps = pd.DataFrame(data.get("market_caps", []), columns=["timestamp_ms", f"{coin_id}_market_cap"])
        volumes = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp_ms", f"{coin_id}_volume"])

        if prices.empty:
            raise ValueError(f"No price data returned for {coin_id}")

        df = prices.merge(market_caps, on="timestamp_ms", how="left").merge(volumes, on="timestamp_ms", how="left")
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
        df = df.drop(columns=["timestamp_ms"]).drop_duplicates("date")
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
