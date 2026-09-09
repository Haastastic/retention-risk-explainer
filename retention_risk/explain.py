"""Per-prediction explanations via SHAP.

This sits *below* the narrative layer: it turns a fitted :class:`RiskModel` and a
single employee row into a structured :class:`Explanation` — which features moved
the risk up or down, and by how much. The LLM in :mod:`retention_risk.narrative`
consumes this; it never runs the model or SHAP itself.

One-hot columns are folded back to their source feature so a manager sees
"Department", not "Department_Sales".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from retention_risk import schema
from retention_risk.model import RiskModel

DEFAULT_TOP_K = 5


@dataclass
class FeatureContribution:
    feature: str
    value: object  # the employee's actual value for this feature
    shap: float  # signed contribution in probability space (see Explanation)
    direction: str  # "increases" | "decreases"


@dataclass
class Explanation:
    """Structured explanation of one prediction. ``risk_score`` and ``risk_tier``
    are copied from the model output — nothing here recomputes them.

    SHAP values (``base_value`` and every ``shap`` / ``all_contributions``) are in
    **probability space** regardless of model kind, so
    ``base_value + sum(all_contributions.values()) ≈ risk_score``."""

    risk_score: float
    risk_tier: str
    base_value: float
    top_contributions: list[FeatureContribution]
    all_contributions: dict[str, float]

    def as_prompt_facts(self) -> str:
        """Compact, number-light rendering for the narrative layer."""
        lines = [f"Risk tier: {self.risk_tier}"]
        for c in self.top_contributions:
            lines.append(f"- {c.feature} = {c.value} ({c.direction} risk)")
        return "\n".join(lines)


def _onehot_source(transformed_name: str) -> str:
    """Map a transformed column name back to its original model feature."""
    if transformed_name in set(schema.MODEL_FEATURES):
        return transformed_name
    for cat in schema.CATEGORICAL_FEATURES:
        if transformed_name.startswith(f"{cat}_"):
            return cat
    return transformed_name


def _fold_onehot(trans_names: list[str], values: np.ndarray) -> dict[str, float]:
    """Sum transformed-column SHAP values back onto their source model feature."""
    folded: dict[str, float] = {}
    for name, contribution in zip(trans_names, values, strict=True):
        src = _onehot_source(name)
        folded[src] = folded.get(src, 0.0) + float(contribution)
    return folded


def _require_background(model: RiskModel) -> pd.DataFrame:
    if model.background_ is None:
        raise RuntimeError("model has no background sample; refit with the current code")
    return model.background_


def _contributions(model: RiskModel, row: pd.DataFrame) -> tuple[dict[str, float], float]:
    """SHAP contributions for one row, in probability space, folded to source features.

    One code path for every estimator kind: a model-agnostic explainer over the
    fitted estimator's ``predict_proba`` in the preprocessed numeric space,
    against the background sample captured at ``fit`` time. Probability space
    means ``base_value + sum(contributions) ≈ P(leave)`` regardless of whether
    the shipping model is XGBoost or the logistic baseline.
    """
    prep = model.pipeline_.named_steps["prep"]
    estimator = model.pipeline_.named_steps["est"]
    trans_names = list(prep.get_feature_names_out())

    background = np.asarray(prep.transform(_require_background(model)))
    x_trans = np.asarray(prep.transform(row))
    explainer = shap.Explainer(lambda a: estimator.predict_proba(a)[:, 1], background)
    result = explainer(x_trans)

    values = np.asarray(result.values)[0]
    base_value = float(np.ravel(result.base_values)[0])
    return _fold_onehot(trans_names, values), base_value


def explain_prediction(
    model: RiskModel, row: pd.DataFrame, *, top_k: int = DEFAULT_TOP_K
) -> Explanation:
    """Explain a single-row prediction.

    ``row`` must be a 1-row DataFrame with the model feature columns. Works for
    any estimator kind; SHAP values come back in probability space.
    """
    if len(row) != 1:
        raise ValueError("explain_prediction expects exactly one row")

    score = float(model.predict_proba(row)[0])
    tier = str(model.assign_tier(np.array([score]))[0])

    folded, base_value = _contributions(model, row)

    ordered = sorted(folded.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = []
    for feature, contribution in ordered[:top_k]:
        raw_value = row.iloc[0][feature] if feature in row.columns else None
        if isinstance(raw_value, float):
            raw_value = round(raw_value, 2)
        top.append(
            FeatureContribution(
                feature=feature,
                value=raw_value,
                shap=contribution,
                direction="increases" if contribution > 0 else "decreases",
            )
        )

    return Explanation(
        risk_score=score,
        risk_tier=tier,
        base_value=base_value,
        top_contributions=top,
        all_contributions=folded,
    )
