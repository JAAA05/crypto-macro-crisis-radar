from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.reporting.research_dashboard import save_dashboard_outputs


def main() -> None:
    print("Building research dashboard...")

    outputs = save_dashboard_outputs(ROOT)

    print("\nResearch dashboard generated successfully.")
    print(f"Dashboard: {outputs['dashboard']}")
    print(f"Latest snapshot: {outputs['snapshot']}")
    print(f"Key findings: {outputs['findings']}")

    dashboard_path = Path(outputs["dashboard"])

    print("\n--- Dashboard Preview ---\n")
    print(dashboard_path.read_text())


if __name__ == "__main__":
    main()