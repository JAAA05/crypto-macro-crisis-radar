from __future__ import annotations

import pandas as pd


def safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)

    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _fmt_score(value: float) -> str:
    return f"{value:.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _score_band(score: float) -> str:
    if score < 30:
        return "low"
    if score < 50:
        return "moderate"
    if score < 70:
        return "elevated"
    if score < 85:
        return "high"
    return "extreme"


def build_market_analyst_summary(latest: pd.Series, previous: pd.Series | None = None) -> str:
    regime = str(latest.get("market_regime", "Unknown"))

    btc_drawdown_30d = safe_float(latest, "bitcoin_drawdown_30d")
    btc_ret_7d = safe_float(latest, "bitcoin_ret_7d")

    macro_engine = safe_float(latest, "macro_engine_stress_index", 50)
    macro_risk = safe_float(latest, "macro_risk_score", 50)
    cross_asset = safe_float(latest, "cross_asset_risk_score", 50)
    crypto = safe_float(latest, "crypto_stress_score", 50)
    liquidity = safe_float(latest, "liquidity_stress_score", 50)
    news = safe_float(latest, "news_risk_score", 50)
    anomaly = safe_float(latest, "anomaly_score", 0)
    final_score = safe_float(latest, "final_risk_score_with_model", 50)

    pieces: list[str] = []

    pieces.append(
        f"The system currently classifies the market regime as **{regime}** "
        f"with a model-adjusted final risk score of **{_fmt_score(final_score)}**."
    )

    if btc_drawdown_30d <= -0.10:
        pieces.append(
            f"BTC remains under pressure with a 30-day drawdown of **{_fmt_pct(btc_drawdown_30d)}**, "
            "but this weakness is not automatically treated as a systemic crisis unless confirmed by macro, liquidity, narrative, or cross-asset stress."
        )
    elif btc_ret_7d > 0.03:
        pieces.append(
            f"BTC has positive short-term momentum with a 7-day return of **{_fmt_pct(btc_ret_7d)}**."
        )
    else:
        pieces.append(
            f"BTC short-term momentum is not strongly directional, with a 7-day return of **{_fmt_pct(btc_ret_7d)}**."
        )

    pieces.append(
        f"The broader macroeconomic engine is **{_score_band(macro_engine)}** "
        f"at **{_fmt_score(macro_engine)}**, while traditional macro financial risk is "
        f"**{_score_band(macro_risk)}** at **{_fmt_score(macro_risk)}**."
    )

    pieces.append(
        f"Cross-asset confirmation is **{_score_band(cross_asset)}** "
        f"at **{_fmt_score(cross_asset)}**, meaning equities, gold, oil, and other traditional market signals "
        "are not strongly confirming a broad risk-off move unless this score rises."
    )

    pieces.append(
        f"Crypto market stress is **{_score_band(crypto)}** at **{_fmt_score(crypto)}**, "
        f"liquidity stress is **{_score_band(liquidity)}** at **{_fmt_score(liquidity)}**, "
        f"and narrative/news risk is **{_score_band(news)}** at **{_fmt_score(news)}**."
    )

    if anomaly >= 70:
        pieces.append(
            f"The anomaly score is high at **{_fmt_score(anomaly)}**, so the current market configuration deserves closer monitoring."
        )
    elif anomaly >= 50:
        pieces.append(
            f"The anomaly score is moderate at **{_fmt_score(anomaly)}**, suggesting some unusual behavior, but not enough by itself to confirm crisis conditions."
        )
    else:
        pieces.append(
            f"The anomaly score is low at **{_fmt_score(anomaly)}**, so the current setup does not look statistically extreme."
        )

    if previous is not None:
        previous_regime = str(previous.get("market_regime", "Unknown"))
        if previous_regime != regime:
            pieces.append(f"The regime changed from **{previous_regime}** to **{regime}** compared with the previous observation.")
        else:
            pieces.append(f"The regime remained **{regime}** compared with the previous observation.")

    return " ".join(pieces)


def build_scenario_watch(latest: pd.Series) -> dict[str, str]:
    regime = str(latest.get("market_regime", "Unknown"))

    btc_drawdown_30d = safe_float(latest, "bitcoin_drawdown_30d")
    btc_ret_7d = safe_float(latest, "bitcoin_ret_7d")

    macro_engine = safe_float(latest, "macro_engine_stress_index", 50)
    cross_asset = safe_float(latest, "cross_asset_risk_score", 50)
    crypto = safe_float(latest, "crypto_stress_score", 50)
    liquidity = safe_float(latest, "liquidity_stress_score", 50)
    news = safe_float(latest, "news_risk_score", 50)
    anomaly = safe_float(latest, "anomaly_score", 0)
    model_prob = safe_float(latest, "model_crisis_probability", 0)

    if regime in {"Risk-On", "Neutral"}:
        base_case = (
            "The base case is that the market remains in a non-crisis regime because macroeconomic stress, "
            "cross-asset risk, and crypto stress are not jointly elevated."
        )
    elif regime == "Risk-Off":
        base_case = (
            "The base case is defensive. The system detects enough stress to reduce simulated exposure, "
            "but not enough for a full crisis classification."
        )
    elif regime == "Mini-Crisis Watch":
        base_case = (
            "The base case is caution. The system detects conditions consistent with a potential mini-crisis, "
            "so new exposure should be evaluated only in simulation/research mode."
        )
    else:
        base_case = (
            "The base case is crisis monitoring. Multiple risk layers are elevated, so the system prioritizes observation, "
            "driver documentation, and post-event analysis."
        )

    bullish_confirmation = (
        "A more constructive regime would require BTC to recover from its current drawdown, crypto stress to fall, "
        "liquidity stress to stabilize, and equities/cross-assets to remain supportive. "
        f"Current BTC 7D return is {_fmt_pct(btc_ret_7d)} and current cross-asset risk is {_fmt_score(cross_asset)}."
    )

    risk_off_trigger = (
        "A shift toward Risk-Off would become more likely if BTC weakness deepens, crypto stress rises above 60, "
        "liquidity stress rises above 60, cross-asset risk rises above 50, or news risk remains above 70. "
        f"Current BTC 30D drawdown is {_fmt_pct(btc_drawdown_30d)}, crypto stress is {_fmt_score(crypto)}, "
        f"liquidity stress is {_fmt_score(liquidity)}, cross-asset risk is {_fmt_score(cross_asset)}, "
        f"and news risk is {_fmt_score(news)}."
    )

    crisis_trigger = (
        "A Mini-Crisis Watch or Crisis signal would require confirmation across multiple layers: elevated crypto stress, "
        "rising liquidity stress, high anomaly score, negative narrative risk, and confirmation from macro or cross-asset data. "
        f"Current anomaly score is {_fmt_score(anomaly)} and model crisis probability is {_fmt_score(model_prob)}%."
    )

    recovery_signal = (
        "A recovery signal would be stronger if BTC drawdown improves, volatility cools, news risk normalizes, "
        "stablecoin/liquidity stress declines, and cross-asset risk remains low. "
        f"Current Macro Engine stress is {_fmt_score(macro_engine)} and cross-asset risk is {_fmt_score(cross_asset)}."
    )

    return {
        "Base Case": base_case,
        "Bullish Confirmation": bullish_confirmation,
        "Risk-Off Trigger": risk_off_trigger,
        "Mini-Crisis / Crisis Trigger": crisis_trigger,
        "Recovery Signal": recovery_signal,
    }


def build_regime_change_triggers(latest: pd.Series) -> list[str]:
    regime = str(latest.get("market_regime", "Unknown"))

    btc_drawdown_30d = safe_float(latest, "bitcoin_drawdown_30d")
    crypto = safe_float(latest, "crypto_stress_score", 50)
    liquidity = safe_float(latest, "liquidity_stress_score", 50)
    cross_asset = safe_float(latest, "cross_asset_risk_score", 50)
    macro_engine = safe_float(latest, "macro_engine_stress_index", 50)
    news = safe_float(latest, "news_risk_score", 50)
    anomaly = safe_float(latest, "anomaly_score", 0)

    triggers: list[str] = []

    if regime in {"Risk-On", "Neutral"}:
        triggers.extend(
            [
                f"BTC 30D drawdown worsens below -15% while currently at {_fmt_pct(btc_drawdown_30d)}.",
                f"Crypto Market Stress rises above 60 while currently at {_fmt_score(crypto)}.",
                f"Liquidity Stress rises above 60 while currently at {_fmt_score(liquidity)}.",
                f"Cross-Asset Risk rises above 50 while currently at {_fmt_score(cross_asset)}.",
                f"News / Narrative Risk remains above 70 while currently at {_fmt_score(news)}.",
                f"Anomaly Score rises above 75 while currently at {_fmt_score(anomaly)}.",
            ]
        )
    elif regime == "Risk-Off":
        triggers.extend(
            [
                "The system would move toward Mini-Crisis Watch if crypto stress, liquidity stress, anomaly score, and news risk rise together.",
                f"Macro Engine Stress rising above 60 would add confirmation from the broader economy. Current value: {_fmt_score(macro_engine)}.",
                f"Cross-Asset Risk above 65 would confirm broader traditional-market stress. Current value: {_fmt_score(cross_asset)}.",
            ]
        )
    else:
        triggers.extend(
            [
                "A return toward Neutral would require crypto stress, liquidity stress, news risk, and anomaly score to normalize together.",
                f"Cross-Asset Risk falling below 40 would help remove broad risk-off confirmation. Current value: {_fmt_score(cross_asset)}.",
                f"Macro Engine Stress falling below 40 would reduce broader economic-cycle pressure. Current value: {_fmt_score(macro_engine)}.",
            ]
        )

    return triggers


def build_risk_flags(latest: pd.Series) -> list[str]:
    flags: list[str] = []

    btc_drawdown_30d = safe_float(latest, "bitcoin_drawdown_30d")
    macro_engine = safe_float(latest, "macro_engine_stress_index", 50)
    cross_asset = safe_float(latest, "cross_asset_risk_score", 50)
    crypto = safe_float(latest, "crypto_stress_score", 50)
    liquidity = safe_float(latest, "liquidity_stress_score", 50)
    news = safe_float(latest, "news_risk_score", 50)
    anomaly = safe_float(latest, "anomaly_score", 0)
    model_prob = safe_float(latest, "model_crisis_probability", 0)

    if btc_drawdown_30d <= -0.10:
        flags.append(f"BTC drawdown remains meaningful at {_fmt_pct(btc_drawdown_30d)}.")

    if news >= 60:
        flags.append(f"Narrative/news risk is above neutral at {_fmt_score(news)}.")

    if anomaly >= 50:
        flags.append(f"Anomaly score is above neutral at {_fmt_score(anomaly)}.")

    if liquidity >= 50:
        flags.append(f"Liquidity stress should be monitored because it is near or above neutral at {_fmt_score(liquidity)}.")

    if crypto >= 50:
        flags.append(f"Crypto market stress is near or above neutral at {_fmt_score(crypto)}.")

    if cross_asset >= 50:
        flags.append(f"Cross-asset risk is confirming broader stress at {_fmt_score(cross_asset)}.")
    else:
        flags.append(f"Cross-asset risk is not confirming broad market stress yet at {_fmt_score(cross_asset)}.")

    if macro_engine >= 50:
        flags.append(f"Macro Engine stress is near or above neutral at {_fmt_score(macro_engine)}.")
    else:
        flags.append(f"Macro Engine stress is not confirming major economic-cycle stress yet at {_fmt_score(macro_engine)}.")

    if model_prob >= 40:
        flags.append(f"Model crisis probability is approaching caution territory at {_fmt_score(model_prob)}%.")

    if not flags:
        flags.append("No major risk flags are currently active.")

    return flags


def build_scenario_analysis(latest: pd.Series, previous: pd.Series | None = None) -> dict[str, object]:
    return {
        "summary": build_market_analyst_summary(latest, previous),
        "scenario_watch": build_scenario_watch(latest),
        "regime_change_triggers": build_regime_change_triggers(latest),
        "risk_flags": build_risk_flags(latest),
    }