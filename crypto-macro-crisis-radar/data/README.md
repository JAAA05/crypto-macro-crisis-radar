# Data Notes

`data/raw/` contains a small public-source snapshot used to make the demo reproducible with:

```bash
python scripts/run_pipeline.py --skip-data
```

Generated feature tables, scored history, reports, model files, and research outputs are intentionally ignored by Git. They can be regenerated from the pipeline.

For a fresh data update, set `FRED_API_KEY` in `.env` and run:

```bash
python scripts/run_pipeline.py --incremental-data
```
