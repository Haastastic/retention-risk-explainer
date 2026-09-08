"""Disparate-impact audit of the High-risk flag across protected groups.

What is audited: the rate at which each protected group lands in the **High**
risk tier (`flag_rate`), and the group's mean predicted risk (`mean_score`).

The High flag is not itself an adverse action — the discovery brief is explicit
that the score is a manager conversation prompt, never an input to pay,
promotion, or performance. But a flag that concentrates on a protected group
would still steer manager attention unevenly and erode trust in the tool, so it
is held to the four-fifths rule anyway.

Method (see `docs/fairness-audit.md`):
  * For each protected attribute, groups smaller than ``min_group_size`` are
    reported but excluded from the pass/fail decision — a thin slice is
    "insufficient data", not "fair".
  * Reference group = the sufficient group with the **lowest** flag rate (the
    least-flagged, most-favoured group).
  * Per-group ``di_ratio`` = group flag rate / reference flag rate (>= 1).
  * Attribute ``di_ratio`` = reference rate / highest group rate, i.e.
    min-rate / max-rate over sufficient groups (<= 1). Passes when it is
    >= 0.8, equivalently every group within [0.8, 1.25] of the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from retention_risk.model import RiskModel

DEFAULT_MIN_GROUP_SIZE = 30
FOUR_FIFTHS = 0.8
AUDIT_ATTRIBUTES: tuple[str, ...] = ("Gender", "MaritalStatus", "AgeBand")


@dataclass
class GroupResult:
    attribute: str
    group: str
    n: int
    flag_rate: float
    mean_score: float
    di_ratio: float  # vs the reference group; nan if this group is the reference or data is thin
    sufficient: bool


@dataclass
class AttributeResult:
    attribute: str
    reference_group: str | None
    di_ratio: float  # min-rate / max-rate over sufficient groups; nan if < 2 sufficient groups
    passes: bool | None  # None => not enough data to decide
    groups: list[GroupResult]

    @property
    def insufficient_groups(self) -> list[str]:
        return [g.group for g in self.groups if not g.sufficient]


@dataclass
class FairnessReport:
    tier: str
    min_group_size: int
    n_rows: int
    attributes: list[AttributeResult] = field(default_factory=list)

    @property
    def decided_attributes(self) -> list[AttributeResult]:
        return [a for a in self.attributes if a.passes is not None]

    @property
    def passes(self) -> bool:
        """True only if every attribute we could decide on passes the four-fifths rule."""
        decided = self.decided_attributes
        return bool(decided) and all(a.passes for a in decided)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "min_group_size": self.min_group_size,
            "n_rows": self.n_rows,
            "passes": self.passes,
            "attributes": {
                a.attribute: {
                    "reference_group": a.reference_group,
                    "di_ratio": None if np.isnan(a.di_ratio) else round(a.di_ratio, 4),
                    "passes": a.passes,
                    "insufficient_groups": a.insufficient_groups,
                    "groups": {
                        g.group: {
                            "n": g.n,
                            "flag_rate": round(g.flag_rate, 4),
                            "mean_score": round(g.mean_score, 4),
                            "di_ratio": None if np.isnan(g.di_ratio) else round(g.di_ratio, 4),
                            "sufficient": g.sufficient,
                        }
                        for g in a.groups
                    },
                }
                for a in self.attributes
            },
        }


def _audit_attribute(
    attribute: str,
    groups: pd.Series,
    is_high: np.ndarray,
    score: np.ndarray,
    min_group_size: int,
) -> AttributeResult:
    # (n, flag_rate, mean_score) per group value
    stats: dict[str, tuple[int, float, float]] = {}
    for value in sorted(groups.unique()):
        mask = (groups == value).to_numpy()
        n = int(mask.sum())
        stats[str(value)] = (
            n,
            float(is_high[mask].mean()),
            float(score[mask].mean()),
        )

    sufficient = {g: s for g, s in stats.items() if s[0] >= min_group_size}

    reference_group: str | None = None
    attr_di = float("nan")
    passes: bool | None = None
    if len(sufficient) >= 2:
        reference_group = min(sufficient, key=lambda g: sufficient[g][1])
        ref_rate = sufficient[reference_group][1]
        max_rate = max(s[1] for s in sufficient.values())
        attr_di = (ref_rate / max_rate) if max_rate > 0 else 1.0
        passes = attr_di >= FOUR_FIFTHS

    rows: list[GroupResult] = []
    for group, (n, flag_rate, mean_score) in stats.items():
        is_suff = n >= min_group_size
        if reference_group is None or not is_suff or group == reference_group:
            di = float("nan")
        else:
            ref_rate = sufficient[reference_group][1]
            di = flag_rate / ref_rate if ref_rate > 0 else float("inf")
        rows.append(GroupResult(attribute, group, n, flag_rate, mean_score, di, is_suff))

    return AttributeResult(attribute, reference_group, attr_di, passes, rows)


def audit_fairness(
    model: RiskModel,
    X: pd.DataFrame,
    protected: pd.DataFrame,
    *,
    tier: str = "High",
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    attributes: tuple[str, ...] = AUDIT_ATTRIBUTES,
) -> FairnessReport:
    """Score ``X`` with ``model`` and audit ``tier``-flag rates across ``protected``."""
    if len(X) != len(protected):
        raise ValueError("X and protected must have the same number of rows")

    score = model.predict_proba(X)
    is_high = model.assign_tier(score) == tier

    report = FairnessReport(tier=tier, min_group_size=min_group_size, n_rows=len(X))
    for attribute in attributes:
        if attribute not in protected.columns:
            raise KeyError(f"protected frame has no column {attribute!r}")
        groups = protected[attribute].astype("string").fillna("Unknown")
        groups.index = pd.RangeIndex(len(groups))
        report.attributes.append(
            _audit_attribute(
                attribute,
                groups,
                np.asarray(is_high),
                np.asarray(score),
                min_group_size,
            )
        )
    return report
