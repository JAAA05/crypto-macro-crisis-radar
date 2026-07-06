from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


class CoinbasePublicClient:
    """Public Coinbase candle client for research market data only.

    This client:
    - does not authenticate
    - does not read balances
    - does not place trades
    - only downloads historical daily OHLCV candles
    """

    COIN_TO_PRODUCT = {
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
    }

    def __init__(self, base_url: str = "https://api.exchange.coinbase.com", sleep_seconds: float = 0.35):
        self.base_url = base_url.rstrip("/")
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "crypto-macro-crisis-radar research-only",
            }
        )

    @staticmethod
    def _to_unix_seconds(ts: pd.Timestamp) -> int:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)

        if resp.status_code == 429:
            time.sleep(5)
            resp = self.session.get(url, params=params, timeout=30)

        resp.raise_for_status()
        time.sleep(self.sleep_seconds)
        return resp.json()

    def market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        start_date: str = "2017-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return a CoinGecko-like daily dataframe.

        Columns:
        - date
        - <coin>_price
        - <coin>_market_cap
        - <coin>_volume
        """
        if vs_currency.lower() != "usd":
            raise ValueError("Coinbase fallback currently supports USD products only.")

        product_id = self.COIN_TO_PRODUCT.get(coin_id)
        if not product_id:
            raise ValueError(
                f"Coinbase fallback does not support coin_id={coin_id!r}. "
                f"Supported coins: {sorted(self.COIN_TO_PRODUCT)}"
            )

        start = pd.to_datetime(start_date, utc=True).normalize()
        end = pd.to_datetime(end_date, utc=True).normalize() if end_date else pd.Timestamp.now(tz="UTC").normalize()

        if end <= start:
            raise ValueError("end_date must be after start_date")

        all_rows: list[list[Any]] = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=280), end)

            params = {
                "start": datetime.fromtimestamp(self._to_unix_seconds(chunk_start), tz=timezone.utc).isoformat(),
                "end": datetime.fromtimestamp(self._to_unix_seconds(chunk_end), tz=timezone.utc).isoformat(),
                "granularity": 86400,
            }

            rows = self._get(f"products/{product_id}/candles", params=params)

            if isinstance(rows, list):
                all_rows.extend(rows)

            chunk_start = chunk_end

        if not all_rows:
            raise ValueError(f"No candle data returned for {coin_id} via Coinbase fallback")

        df = pd.DataFrame(all_rows, columns=["time", "low", "high", "open", "close", "volume"])

        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
        df[f"{coin_id}_price"] = pd.to_numeric(df["close"], errors="coerce")
        df[f"{coin_id}_volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df[f"{coin_id}_market_cap"] = pd.NA

        out = (
            df[["date", f"{coin_id}_price", f"{coin_id}_market_cap", f"{coin_id}_volume"]]
            .dropna(subset=[f"{coin_id}_price"])
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

        return out