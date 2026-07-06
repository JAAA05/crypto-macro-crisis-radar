from __future__ import annotations

import time

import pandas as pd
import yfinance as yf


class YFinanceCrossAssetClient:
    """Cross-asset market data client using yfinance.

    Main tickers:
    - S&P 500: ^GSPC
    - Nasdaq Composite: ^IXIC
    - Gold futures: GC=F
    - WTI crude oil futures: CL=F

    Fallback proxies:
    - S&P 500 ETF: SPY
    - Nasdaq ETF: QQQ
    - Gold ETF: GLD
    - Oil ETF: USO
    """

    TICKER_CANDIDATES = {
        "sp500": ["^GSPC", "SPY"],
        "nasdaq": ["^IXIC", "QQQ"],
        "gold": ["GC=F", "GLD"],
        "oil_wti": ["CL=F", "USO"],
    }

    def __init__(self, sleep_seconds: float = 1.0):
        self.sleep_seconds = sleep_seconds

    def _download_single(
        self,
        alias: str,
        ticker: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        print(f"Downloading {alias} from yfinance ticker {ticker}...")

        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        time.sleep(self.sleep_seconds)

        if data.empty:
            raise RuntimeError(f"yfinance returned empty data for {ticker}")

        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(ticker, level=1, axis=1)
            except Exception:
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

        price_col = None

        for candidate in ["Adj Close", "Close"]:
            if candidate in data.columns:
                price_col = candidate
                break

        if price_col is None:
            raise RuntimeError(
                f"No usable price column found for {ticker}. "
                f"Columns: {list(data.columns)}"
            )

        frame = data[[price_col]].copy()
        frame = frame.rename(columns={price_col: alias})
        frame.index.name = "date"
        frame = frame.reset_index()

        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        frame[alias] = pd.to_numeric(frame[alias], errors="coerce")

        frame = (
            frame.dropna(subset=[alias])
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

        if frame.empty:
            raise RuntimeError(f"Processed frame is empty for {ticker}")

        return frame

    def fetch_one_asset(
        self,
        alias: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        tickers = self.TICKER_CANDIDATES[alias]
        last_error: Exception | None = None

        for ticker in tickers:
            try:
                frame = self._download_single(
                    alias=alias,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                )
                print(f"Successfully downloaded {alias} using {ticker}.")
                return frame

            except Exception as exc:
                last_error = exc
                print(f"[WARN] Failed to download {alias} using {ticker}: {exc}")

        raise RuntimeError(
            f"Could not download {alias} from any yfinance ticker. "
            f"Last error: {last_error}"
        )

    def fetch_cross_assets(
        self,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        frames = []

        for alias in self.TICKER_CANDIDATES:
            try:
                frame = self.fetch_one_asset(
                    alias=alias,
                    start_date=start_date,
                    end_date=end_date,
                )
                frames.append(frame)

            except Exception as exc:
                print(f"[WARN] Could not fetch cross-asset series {alias}: {exc}")

        if not frames:
            raise RuntimeError("No yfinance cross-asset frames could be processed.")

        out = frames[0]

        for frame in frames[1:]:
            out = out.merge(frame, on="date", how="outer")

        out = out.sort_values("date").reset_index(drop=True)

        return out