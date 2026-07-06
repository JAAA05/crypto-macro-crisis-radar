from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESEARCH_STEPS = [
    "scripts/05_event_study.py",
    "scripts/06_lag_shock_analysis.py",
    "scripts/07_var_analysis.py",
    "scripts/07b_var_robustness.py",
    "scripts/09_ablation_study.py",
    "scripts/09b_ablation_study_v2.py",
    "scripts/08_build_research_dashboard.py",
]


def run_step(script_path: str) -> None:
    print(f"\n=== Running {script_path} ===")
    result = subprocess.run([sys.executable, script_path], cwd=ROOT)

    if result.returncode != 0:
        print(
            f"\nResearch suite stopped because {script_path} "
            f"failed with exit code {result.returncode}."
        )
        sys.exit(result.returncode)


def main() -> None:
    scored_history = ROOT / "data/processed/scored_regime_history.csv"

    if not scored_history.exists():
        raise FileNotFoundError(
            "data/processed/scored_regime_history.csv was not found. "
            "Run `python scripts/run_pipeline.py --skip-data` first."
        )

    for step in RESEARCH_STEPS:
        run_step(step)

    print("\nResearch suite finished successfully.")


if __name__ == "__main__":
    main()
