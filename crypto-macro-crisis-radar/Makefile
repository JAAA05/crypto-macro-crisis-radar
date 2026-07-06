.PHONY: setup run demo update research clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	python scripts/run_pipeline.py

demo:
	python scripts/run_pipeline.py --skip-data

update:
	python scripts/run_pipeline.py --incremental-data

research:
	python scripts/run_research_suite.py

clean:
	rm -rf data/features/* data/processed/* outputs/* models/*.joblib
	touch data/features/.gitkeep data/processed/.gitkeep outputs/.gitkeep models/.gitkeep
