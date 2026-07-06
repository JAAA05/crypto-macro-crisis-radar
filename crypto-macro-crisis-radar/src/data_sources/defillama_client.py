from __future__ import annotations

from typing import Any

import pandas as pd
import requests


def _extract_total_stablecoin_mcap(row: dict[str, Any]) -> float | None:
    """Handle slight schema variations in DeFiLlama stablecoin chart responses."""
    candidates = [
        row.get("totalCirculatingUSD"),
        row.get("totalCirculating"),
        row.get("totalMcap"),
        row.get("mcap"),
    ]
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return float(candidate)
        if isinstance(candidate, dict):
            for key in ["peggedUSD", "usd", "USD"]:
                val = candidate.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
    return None


class DefiLlamaClient:
    """Small DeFiLlama client for stablecoin and TVL research data."""

    def __init__(self):
        self.session = requests.Session()

    def stablecoin_chart_all(self) -> pd.DataFrame:
        urls = [
            "https://api.llama.fi/stablecoincharts/all",
            "https://stablecoins.llama.fi/stablecoincharts/all",
        ]
        last_error = None
        data = None
        for url in urls:
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                last_error = exc
        if data is None:
            raise RuntimeError(f"Could not fetch DeFiLlama stablecoin chart: {last_error}")

        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError("Unexpected DeFiLlama stablecoin chart response shape.")

        rows = []
        for item in data:
            if not isinstance(item, dict):
                continue
            date_raw = item.get("date") or item.get("timestamp")
            total_mcap = _extract_total_stablecoin_mcap(item)
            if date_raw is None or total_mcap is None:
                continue
            date = pd.to_datetime(int(date_raw), unit="s", utc=True).date()
            rows.append({"date": date, "stablecoin_mcap": total_mcap})

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No stablecoin rows could be parsed from DeFiLlama response.")
        df["date"] = pd.to_datetime(df["date"])
        return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
