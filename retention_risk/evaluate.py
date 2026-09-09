"""Evaluation against the discovery-brief success metrics.

Brief targets (docs/discovery-brief.md):
  * AUC-ROC >= 0.75 on a held-out set
  * precision >= 0.30 in the "High" tier, and that tier <= 15% of the population
  * recall >= 0.50 across the top two tiers ("High" + "Medium") combined
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from retention_risk.model import TIERS, RiskModel, build_pipeline

BRIEF_MIN_AUC = 0.75
BRIEF_MIN_HIGH_PRECISION = 0.30
BRIEF_MAX_HIGH_SHARE = 0.15
BRIEF_MIN_TOP2_RECALL = 0.50


@dataclass
class TierStats:
    tier: str
    n: int
    share: float
    precision: float
    recall: float


@dataclass
class EvalReport:
    model_kind: str
    n_test: int
    positives_test: int
    auc_roc: float
    brier: float
    cv_auc_mean: float
    cv_auc_std: float
    tiers: list[TierStats]
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passes_brief(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict:
        return {
            "model_kind": self.model_kind,
            "n_test": self.n_test,
            "positives_test": self.positives_test,
            "auc_roc": round(self.auc_roc, 4),
            "brier": round(self.brier, 4),
            "cv_auc": f"{self.cv_auc_mean:.4f} +/- {self.cv_auc_std:.4f}",
            "tiers": {
                t.tier: {
                    "n": t.n,
                    "share": round(t.share, 4),
                    "precision": round(t.precision, 4),
                    "recall": round(t.recall, 4),
                }
                for t in self.tiers
            },
            "checks": self.checks,
            "passes_brief": self.passes_brief,
        }


def _tier_stats(y_true: np.ndarray, tiers: np.ndarray) -> list[TierStats]:
    total = len(y_true)
    total_pos = int(y_true.sum())
    out: list[TierStats] = []
    for name in TIERS:
        mask = tiers == name
        n = int(mask.sum())
        tp = int(y_true[mask].sum())
        precision = tp / n if n else 0.0
        recall = tp / total_pos if total_pos else 0.0
        out.append(TierStats(name, n, n / total if total else 0.0, precision, recall))
    return out


def evaluate_model(
    model: RiskModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    X_cv: pd.DataFrame | None = None,
    y_cv: pd.Series | None = None,
    cv_splits: int = 5,
    seed: int = 42,
) -> EvalReport:
    """Score ``model`` on the held-out set and check it against the brief."""
    y_true = np.asarray(y_test, dtype=int)
    proba = model.predict_proba(X_test)
    tiers = model.assign_tier(proba)
    tier_stats = _tier_stats(y_true, tiers)

    auc = float(roc_auc_score(y_true, proba))
    brier = float(brier_score_loss(y_true, proba))

    cv_mean = cv_std = float("nan")
    if X_cv is not None and y_cv is not None:
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        est = build_pipeline(model.kind, y_cv, seed)
        scores = cross_val_score(est, X_cv, y_cv, cv=skf, scoring="roc_auc")
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

    high = next(t for t in tier_stats if t.tier == "High")
    top2_recall = sum(t.recall for t in tier_stats if t.tier in ("High", "Medium"))
    checks = {
        "auc_roc>=0.75": auc >= BRIEF_MIN_AUC,
        "high_precision>=0.30": high.precision >= BRIEF_MIN_HIGH_PRECISION,
        "high_share<=0.15": high.share <= BRIEF_MAX_HIGH_SHARE,
        "top2_recall>=0.50": top2_recall >= BRIEF_MIN_TOP2_RECALL,
    }

    return EvalReport(
        model_kind=model.kind,
        n_test=len(y_true),
        positives_test=int(y_true.sum()),
        auc_roc=auc,
        brier=brier,
        cv_auc_mean=cv_mean,
        cv_auc_std=cv_std,
        tiers=tier_stats,
        checks=checks,
    )
