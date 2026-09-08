"""Train, evaluate, and persist the risk model.

    python scripts/train.py [--out models/risk_model.joblib]

Prints the baseline and XGBoost reports side by side and saves the shipping
model (best held-out AUC) to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retention_risk.training import run_training  # noqa: E402  (after sys.path bootstrap)

DEFAULT_OUT = Path("models/risk_model.joblib")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true", help="evaluate only")
    args = parser.parse_args()

    result = run_training(data_seed=args.seed, split_seed=args.seed)

    for report in (result.baseline_report, result.xgboost_report):
        print(f"\n=== {report.model_kind} ===")
        print(json.dumps(report.to_dict(), indent=2))

    shipping = result.best
    print(f"\nshipping model: {shipping.kind}")
    if not args.no_save:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shipping.save(args.out)
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
