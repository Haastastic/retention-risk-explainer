"""Fetch the raw IBM HR Analytics Attrition dataset.

The CSV is committed under ``data/raw/`` so this script is only needed to
refresh it or to re-provision a clean checkout. Run from the repo root:

    python scripts/fetch_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# IBM Watson Analytics sample data, redistributed via the IBM AIF360 examples repo.
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
)
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "ibm_hr_attrition.csv"
EXPECTED_ROWS = 1470


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as resp:  # noqa: S310 (trusted URL)
        body = resp.read()
    DEST.write_bytes(body)

    n_lines = body.decode("utf-8-sig").strip().count("\n")
    if n_lines != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} data rows, got {n_lines}", file=sys.stderr)
        return 1
    print(f"Wrote {DEST} ({len(body):,} bytes, {n_lines} data rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
