from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
from requests.exceptions import RequestException


class FredClient:
    """FRED API client using the official series/observations endpoint.

    Expected input:
        {
            "real_gdp": "GDPC1",
            "unemployment_rate": "UNRATE",
            "cpi": "CPIAUCSL",
        }

    Output:
        date, real_gdp, unemployment_rate, cpi
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        sleep_seconds: float = 1.0,
    ):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY is missing. Set it with:\n"
                'export FRED_API_KEY="your_key_here"'
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "crypto-macro-crisis-radar research-only",
            }
        )

    def fetch_one(
        self,
        alias: str,
        fred_series_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {
            "series_id": fred_series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
        }

        if end_date:
            params["observation_end"] = end_date

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                print(
                    f"Fetching FRED API series {alias} ({fred_series_id}) "
                    f"attempt {attempt}/{self.max_retries}..."
                )

                resp = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()

                payload = resp.json()
                observations = payload.get("observations", [])

                if not observations:
                    raise ValueError(
                        f"FRED API returned no observations for {alias} ({fred_series_id})"
                    )

                df = pd.DataFrame(observations)

                if "date" not in df.columns or "value" not in df.columns:
                    raise ValueError(
                        f"Unexpected FRED API response for {alias} ({fred_series_id}). "
                        f"Columns: {list(df.columns)}"
                    )

                out = df[["date", "value"]].copy()
                out.columns = ["date", alias]

                out["date"] = pd.to_datetime(out["date"], errors="coerce")
                out[alias] = (
                    out[alias]
                    .replace(".", pd.NA)
                    .pipe(pd.to_numeric, errors="coerce")
                )

                out = (
                    out.dropna(subset=["date"])
                    .drop_duplicates("date", keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )

                time.sleep(self.sleep_seconds)
                return out

            except (RequestException, ValueError) as exc:
                last_error = exc
                wait = self.sleep_seconds * attempt

                print(
                    f"[WARN] FRED API series {alias} ({fred_series_id}) failed "
                    f"on attempt {attempt}/{self.max_retries}: {exc}"
                )

                if attempt < self.max_retries:
                    print(f"Waiting {wait:.1f}s before retry...")
                    time.sleep(wait)

        raise RuntimeError(
            f"Could not fetch FRED API series {alias} ({fred_series_id}) "
            f"after {self.max_retries} attempts. Last error: {last_error}"
        )

    def fetch_series(
        self,
        series_map: dict[str, str],
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for alias, fred_series_id in series_map.items():
            try:
                frame = self.fetch_one(
                    alias=alias,
                    fred_series_id=fred_series_id,
                    start_date=start_date,
                    end_date=end_date,
                )
                frames.append(frame)

            except Exception as exc:
                print(
                    f"[WARN] Could not fetch FRED API series {alias} "
                    f"({fred_series_id}): {exc}"
                )

        if not frames:
            raise RuntimeError("No FRED API data could be downloaded.")

        out = frames[0]

        for frame in frames[1:]:
            out = out.merge(frame, on="date", how="outer")

        out = out.sort_values("date").reset_index(drop=True)

        return out
