"""Show the explanation + narrative for the highest- and lowest-risk employees.

    python scripts/explain_demo.py [--live]

--live uses the Claude narrator if ANTHROPIC_API_KEY is set (.env is read);
otherwise the deterministic template narrator is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retention_risk.data import build_dataset  # noqa: E402
from retention_risk.explain import explain_prediction  # noqa: E402
from retention_risk.narrative import TemplateNarrator, get_narrator  # noqa: E402
from retention_risk.training import run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use Claude if a key is available")
    args = parser.parse_args()

    ds = build_dataset()
    model = run_training().best
    narrator = get_narrator() if args.live else TemplateNarrator()
    proba = model.predict_proba(ds.X)

    for label, idx in [
        ("HIGHEST RISK", int(np.argmax(proba))),
        ("LOWEST RISK", int(np.argmin(proba))),
    ]:
        row = ds.X.iloc[[idx]]
        exp = explain_prediction(model, row)
        nar = narrator.narrate(exp, employee_ref="This report")
        rule = "=" * 70
        print(f"\n{rule}\n{label}  |  score={exp.risk_score:.2f}  tier={exp.risk_tier}\n{rule}")
        print("Top drivers (SHAP):")
        for c in exp.top_contributions:
            print(f"  {c.feature:22s} = {str(c.value):>10}   {c.direction} risk")
        print(f"\nNarrative ({nar.source}):")
        print(f"  {nar.summary}")
        for d in nar.drivers:
            print(f"   - {d}")
        print(f"  Suggested: {nar.suggested_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
