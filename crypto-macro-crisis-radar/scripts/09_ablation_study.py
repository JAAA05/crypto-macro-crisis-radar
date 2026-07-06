from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.experiments.ablation_study import AblationConfig, save_ablation_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature-group ablation study.")
    parser.add_argument(
        "--input",
        default="data/features/master_features.csv",
        help="Path to master feature table.",
    )
    parser.add_argument(
        "--target",
        default="mini_crisis_next_7d",
        help="Target column for ablation study.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Time-based test size fraction.",
    )

    args = parser.parse_args()

    input_path = ROOT / args.input

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run scripts/02_build_features.py first."
        )

    print("Running ablation study...")

    config = AblationConfig(
        target_col=args.target,
        test_size=float(args.test_size),
    )

    outputs = save_ablation_outputs(
        root=ROOT,
        input_path=input_path,
        config=config,
    )

    print("\nAblation study completed.")
    print(f"Results: {outputs['results']}")
    print(f"Feature importance: {outputs['importance']}")
    print(f"Report: {outputs['report']}")

    print("\n--- Ablation Report Preview ---\n")
    print(Path(outputs["report"]).read_text())


if __name__ == "__main__":
    main()