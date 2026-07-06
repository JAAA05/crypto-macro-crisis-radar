from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.experiments.ablation_study_v2 import (
    AblationV2Config,
    save_ablation_v2_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ablation Study v2.")
    parser.add_argument(
        "--input",
        default="data/features/master_features.csv",
        help="Path to master feature table.",
    )
    parser.add_argument(
        "--target",
        default="mini_crisis_next_7d",
        help="Target column.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of walk-forward folds.",
    )
    parser.add_argument(
        "--initial-train-size",
        type=float,
        default=0.50,
        help="Initial train size as fraction of dataset.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.10,
        help="Validation window size as fraction of dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.10,
        help="Test window size as fraction of dataset.",
    )

    args = parser.parse_args()

    input_path = ROOT / args.input

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run scripts/02_build_features.py first."
        )

    print("Running Ablation Study v2...")

    config = AblationV2Config(
        target_col=args.target,
        n_folds=int(args.n_folds),
        initial_train_size=float(args.initial_train_size),
        validation_size=float(args.validation_size),
        test_size=float(args.test_size),
    )

    outputs = save_ablation_v2_outputs(
        root=ROOT,
        input_path=input_path,
        config=config,
    )

    print("\nAblation Study v2 completed.")
    print(f"Fold results: {outputs['fold_results']}")
    print(f"Summary: {outputs['summary']}")
    print(f"Feature importance: {outputs['importance']}")
    print(f"Fold config: {outputs['fold_config']}")
    print(f"Report: {outputs['report']}")

    print("\n--- Ablation v2 Report Preview ---\n")
    print(Path(outputs["report"]).read_text())


if __name__ == "__main__":
    main()