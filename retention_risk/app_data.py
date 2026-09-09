"""Pure data assembly for the Streamlit dashboard.

Kept separate from ``app.py`` so it can be unit-tested without a Streamlit
runtime. Nothing here imports streamlit.

Two views, matching the discovery brief:

* **Manager view** — one report at a time: risk tier, the plain-language
  narrative, and the SHAP drivers behind it.
* **HRBP view** — a team/org rollup that *surfaces the fairness picture at the
  point of use*: flag rate and group size per protected group.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from retention_risk.data import Dataset, build_dataset
from retention_risk.explain import Explanation, explain_prediction
from retention_risk.fairness import FairnessReport, audit_fairness
from retention_risk.narrative import Narrative, Narrator, get_narrator
from retention_risk.training import TrainingResult, run_training


def report_label(i: int) -> str:
    """Stable pseudonym so a manager sees "Report 07", never an employee number."""
    return f"Report {i + 1:02d}"


@dataclass
class AppBundle:
    """Everything the dashboard needs, built once."""

    dataset: Dataset
    training: TrainingResult
    scores: pd.Series  # index-aligned to dataset.X
    tiers: pd.Series

    @property
    def model(self):
        return self.training.best

    def roster(self) -> pd.DataFrame:
        """One row per report: label, tier, score — for the manager's pick list."""
        return pd.DataFrame(
            {
                "label": [report_label(i) for i in range(len(self.dataset.X))],
                "tier": self.tiers.to_numpy(),
                "score": self.scores.to_numpy().round(3),
                "row": range(len(self.dataset.X)),
            }
        )


def build_bundle(*, seed: int = 42) -> AppBundle:
    ds = build_dataset(seed=seed)
    training = run_training(data_seed=seed, split_seed=seed)
    scores = pd.Series(training.best.predict_proba(ds.X), index=ds.X.index, name="score")
    tiers = pd.Series(training.best.assign_tier(scores.to_numpy()), index=ds.X.index, name="tier")
    return AppBundle(dataset=ds, training=training, scores=scores, tiers=tiers)


@dataclass
class ManagerCard:
    label: str
    tier: str
    score: float
    narrative: Narrative
    explanation: Explanation

    def driver_chart_frame(self) -> pd.DataFrame:
        """Top contributions as a tidy frame for a diverging bar chart (probability pts)."""
        rows = [
            {
                "feature": c.feature,
                "contribution_pp": round(c.shap * 100, 1),
                "direction": c.direction,
            }
            for c in self.explanation.top_contributions
        ]
        return pd.DataFrame(rows)


def manager_card(
    bundle: AppBundle, row: int, *, narrator: Narrator | None = None, top_k: int = 5
) -> ManagerCard:
    """Assemble the single-report manager view."""
    if not 0 <= row < len(bundle.dataset.X):
        raise IndexError(f"row {row} out of range")
    narrator = narrator or get_narrator()
    x = bundle.dataset.X.iloc[[row]]
    explanation = explain_prediction(bundle.model, x, top_k=top_k)
    label = report_label(row)
    narrative = narrator.narrate(explanation, employee_ref=label)
    return ManagerCard(
        label=label,
        tier=explanation.risk_tier,
        score=explanation.risk_score,
        narrative=narrative,
        explanation=explanation,
    )


@dataclass
class HrbpView:
    tier_counts: pd.Series  # index: tier, value: count
    fairness: FairnessReport

    def flag_rate_table(self) -> pd.DataFrame:
        """Per protected group: flag rate + group size, the numbers the brief
        requires the HRBP surface to show."""
        rows = []
        for attr in self.fairness.attributes:
            for g in attr.groups:
                rows.append(
                    {
                        "attribute": attr.attribute,
                        "group": g.group,
                        "n": g.n,
                        "high_flag_rate": round(g.flag_rate, 3),
                        "mean_score": round(g.mean_score, 3),
                        "sufficient_n": g.sufficient,
                    }
                )
        return pd.DataFrame(rows)

    def fairness_failures(self) -> list[tuple[str, float]]:
        """(attribute, di_ratio) for every protected attribute that fails the
        four-fifths rule on the High flag. Empty if all pass or are undecidable."""
        return [
            (attr.attribute, attr.di_ratio)
            for attr in self.fairness.attributes
            if attr.passes is False
        ]


def hrbp_view(bundle: AppBundle) -> HrbpView:
    counts = bundle.tiers.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
    fairness = audit_fairness(bundle.model, bundle.dataset.X, bundle.dataset.protected)
    return HrbpView(tier_counts=counts, fairness=fairness)
