from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests
from requests.exceptions import RequestException
from tqdm import tqdm


NEGATIVE_TERMS = [
    "crash", "collapse", "hack", "exploit", "lawsuit", "sec", "investigation", "ban",
    "fraud", "bankruptcy", "insolvency", "depeg", "liquidation", "selloff", "outflow",
    "panic", "sanction", "probe", "default", "contagion", "risk", "warning",
]

POSITIVE_TERMS = [
    "approval", "approved", "etf", "inflows", "adoption", "institutional", "partnership",
    "clarity", "rally", "recovery", "upgrade", "record inflow", "accumulation",
]


class GdeltClient:
    """Lightweight and fault-tolerant client for GDELT DOC 2.0 article-list queries.

    Important:
    - GDELT can rate-limit or timeout.
    - For this project, failed news days should become empty rows, not fatal errors.
    """

    DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        timeout_seconds: int = 15,
        max_retries: int = 2,
        sleep_seconds: float = 0.75,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "crypto-macro-crisis-radar research-only",
            }
        )

    @staticmethod
    def _fmt(dt: datetime) -> str:
        return dt.strftime("%Y%m%d%H%M%S")

    def fetch_daily_articles(
        self,
        query: str,
        day: datetime,
        max_records: int = 25,
    ) -> list[dict[str, Any]]:
        start = datetime(day.year, day.month, day.day)
        end = start + timedelta(days=1)

        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_records,
            "sort": "datedesc",
            "startdatetime": self._fmt(start),
            "enddatetime": self._fmt(end),
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(
                    self.DOC_URL,
                    params=params,
                    timeout=self.timeout_seconds,
                )

                if resp.status_code == 200:
                    try:
                        payload = resp.json()
                    except Exception:
                        print(f"[WARN] GDELT invalid JSON for {start.date()}")
                        return []

                    time.sleep(self.sleep_seconds)
                    return payload.get("articles", []) or []

                if resp.status_code == 429:
                    wait = 5 * attempt
                    print(f"[WARN] GDELT rate limited for {start.date()}. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                print(f"[WARN] GDELT returned status {resp.status_code} for {start.date()}")
                time.sleep(self.sleep_seconds)
                return []

            except RequestException as exc:
                wait = 3 * attempt
                print(f"[WARN] GDELT request failed for {start.date()} on attempt {attempt}: {exc}")
                time.sleep(wait)

        print(f"[WARN] GDELT skipped {start.date()} after retries.")
        return []

    def fetch_recent_news_features(
        self,
        query: str,
        lookback_days: int = 30,
        max_records_per_day: int = 25,
    ) -> pd.DataFrame:
        today = datetime.utcnow().date()
        rows = []

        for i in tqdm(range(lookback_days), desc="GDELT daily news"):
            day = datetime.combine(
                today - timedelta(days=lookback_days - 1 - i),
                datetime.min.time(),
            )

            articles = self.fetch_daily_articles(
                query=query,
                day=day,
                max_records=max_records_per_day,
            )

            titles = [str(a.get("title", "")) for a in articles]
            title_blob = " ".join(titles).lower()

            neg = sum(title_blob.count(term) for term in NEGATIVE_TERMS)
            pos = sum(title_blob.count(term) for term in POSITIVE_TERMS)
            article_count = len(articles)

            raw_score = 50 + 5.5 * neg - 2.5 * pos + 0.08 * article_count
            news_risk_score = max(0, min(100, raw_score))

            rows.append(
                {
                    "date": pd.to_datetime(day.date()),
                    "news_article_count": article_count,
                    "news_negative_hits": neg,
                    "news_positive_hits": pos,
                    "news_risk_score": news_risk_score,
                }
            )

        return pd.DataFrame(rows)
