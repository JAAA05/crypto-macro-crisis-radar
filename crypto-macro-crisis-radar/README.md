# Crypto Macro Crisis Radar

A research pipeline for detecting **stress regimes in crypto markets** by combining crypto market behavior, macroeconomic indicators, cross-asset signals, stablecoin liquidity, news/narrative risk, anomaly detection, and baseline machine learning.

> This is a research and portfolio project. It does **not** place trades, connect to exchange trading APIs, or provide financial advice.

## Project goal

Crypto selloffs are rarely explained by one variable. A drawdown can come from crypto-specific stress, liquidity changes, macro pressure, equity-market risk-off behavior, or narrative shocks. This project builds a transparent early-warning framework that turns those signals into a daily market-regime report.

The main output is a Markdown report similar to an analyst note:

```text
outputs/reports/latest_market_regime_report.md
```

A sample report is included here:

```text
examples/reports/latest_market_regime_report.md
```

## What the system does

The pipeline:

1. Loads or downloads BTC/ETH market data.
2. Loads or downloads macro data from FRED.
3. Loads or downloads cross-asset data such as S&P 500, Nasdaq, gold, and oil.
4. Loads stablecoin liquidity data.
5. Adds general news and category-level narrative risk features.
6. Builds a master feature table.
7. Creates forward-looking stress labels.
8. Trains an anomaly model and a baseline crisis classifier.
9. Scores the latest market regime.
10. Generates a daily Markdown report.

The project also includes optional research modules for event studies, lag/shock analysis, VAR-style econometric checks, and ablation studies.

## Methodology

### Target definition

The primary target is a forward-looking mini-crisis label:

```text
mini_crisis_next_7d = 1 if BTC's minimum forward 7-day return <= -7%
```

This frames the task as **risk detection**, not simple price prediction.

### Feature groups

- **Crypto stress:** BTC/ETH returns, volatility, drawdowns, volume shocks, BTC-ETH correlation.
- **Macro risk:** VIX, Treasury yields, yield curve, Fed Funds, dollar index proxy, money supply, policy uncertainty.
- **Macro engine:** GDP, unemployment, CPI/core CPI, industrial production, retail sales, private investment, fiscal balance.
- **Cross-asset confirmation:** S&P 500, Nasdaq, gold, oil, safe-haven demand, commodity pressure.
- **Liquidity:** stablecoin market-cap growth and liquidity stress.
- **News / narrative risk:** GDELT-based news and category-level narrative signals.
- **Model signals:** anomaly score, model-estimated mini-crisis probability, rule-based risk score, final regime score.

## Repository structure

```text
crypto-macro-crisis-radar/
├── config.yaml
├── requirements.txt
├── Makefile
├── .env.example
├── data/
│   ├── raw/                  # small public-source snapshot for reproducible demo
│   ├── features/             # generated, ignored by Git
│   ├── processed/            # generated, ignored by Git
│   └── events/               # manual event template
├── examples/                 # lightweight sample reports for portfolio review
├── models/                   # generated model artifacts, ignored by Git
├── outputs/                  # generated reports/scores, ignored by Git
├── scripts/                  # runnable pipeline and research scripts
└── src/                      # reusable project modules
```

## Setup

Recommended: Python 3.11 or 3.12.

```bash
git clone https://github.com/YOUR-USERNAME/crypto-macro-crisis-radar.git
cd crypto-macro-crisis-radar
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You can also use the Makefile:

```bash
make setup
```

## Quick demo using included data

This repository includes a small public-source raw-data snapshot so the project can be run without downloading fresh data first.

```bash
python scripts/run_pipeline.py --skip-data
```

Or:

```bash
make demo
```

This generates:

```text
data/features/master_features.csv
data/processed/scored_regime_history.csv
models/anomaly_model.joblib
models/crisis_classifier.joblib
outputs/scores/baseline_model_report.json
outputs/reports/latest_market_regime_report.md
```

## Run with fresh or updated data

For fresh macro downloads, create a `.env` file from the example:

```bash
cp .env.example .env
```

Then add your FRED key:

```text
FRED_API_KEY=your_key_here
```

Run an incremental update:

```bash
python scripts/run_pipeline.py --incremental-data
```

Or force a full data refresh:

```bash
python scripts/run_pipeline.py --force-data
```

Optional: add a CoinGecko API key in `.env` if you hit public rate limits.

## Optional research suite

After running the main pipeline, run the research modules:

```bash
python scripts/run_research_suite.py
```

Or:

```bash
make research
```

This runs:

```text
scripts/05_event_study.py
scripts/06_lag_shock_analysis.py
scripts/07_var_analysis.py
scripts/07b_var_robustness.py
scripts/09_ablation_study.py
scripts/09b_ablation_study_v2.py
scripts/08_build_research_dashboard.py
```

Sample research outputs are included under:

```text
examples/research/
```

## Main commands

```bash
# Reproduce the demo from included raw data
python scripts/run_pipeline.py --skip-data

# Update only new raw data rows, then rebuild features/models/report
python scripts/run_pipeline.py --incremental-data

# Force full data refresh
python scripts/run_pipeline.py --force-data

# Run extra research modules after the main pipeline
python scripts/run_research_suite.py

# Remove generated artifacts
make clean
```

## Example output snapshot

The included sample report classifies the latest snapshot as **Neutral**, with a model-adjusted final risk score around **36.9**. The report also explains the main drivers, including BTC drawdown, macro risk, cross-asset confirmation, liquidity stress, narrative risk, anomaly score, and model crisis probability.

See:

```text
examples/reports/latest_market_regime_report.md
```

## What is intentionally excluded from Git

The original working folder contained local virtual environments, cache files, backup data, generated feature tables, generated model files, and output artifacts. Those are intentionally excluded or ignored here so the repository stays clean and professional.

Ignored/generated files include:

```text
.venv/
__pycache__/
models/*.joblib
data/features/*
data/processed/*
outputs/*
```

## Limitations

- The model is a research prototype, not a production trading system.
- Data availability and API rate limits can affect fresh downloads.
- GDELT narrative features are simple keyword-based features, not a full institutional NLP system.
- The classifier is intentionally baseline/interpretable before moving to heavier modeling.
- Signals should be interpreted as research context, not trading recommendations.

## Disclaimer

This project is for education, research, and portfolio demonstration only. It is not investment advice, financial advice, or a recommendation to buy or sell any asset.
