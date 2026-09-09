"""Run the disparate-impact audit on the shipping model.

    python scripts/audit_fairness.py

Trains on a stratified split and audits High-tier flag rates on the full
dataset (the population a manager rollout would actually flag across).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retention_risk.data import build_dataset  # noqa: E402
from retention_risk.fairness import audit_fairness  # noqa: E402
from retention_risk.training import run_training  # noqa: E402


def main() -> int:
    ds = build_dataset()
    result = run_training()
    report = audit_fairness(result.best, ds.X, ds.protected)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nshipping model: {result.best.kind}   passes fairness audit: {report.passes}")
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
