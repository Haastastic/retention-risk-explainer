"""SHAP explanations: structure, one-hot folding, and score pass-through."""

import numpy as np
import pytest

from retention_risk import schema
from retention_risk.explain import DEFAULT_TOP_K, explain_prediction
from retention_risk.model import train_baseline
from retention_risk.training import run_training


@pytest.fixture(scope="session")
def model(dataset):
    return run_training(split_seed=42).best


@pytest.fixture(scope="session")
def high_risk_row(dataset, model):
    proba = model.predict_proba(dataset.X)
    return dataset.X.iloc[[int(np.argmax(proba))]]


def test_explanation_has_top_k_contributions(model, high_risk_row):
    exp = explain_prediction(model, high_risk_row)
    assert len(exp.top_contributions) == DEFAULT_TOP_K
    assert exp.top_contributions == sorted(
        exp.top_contributions, key=lambda c: abs(c.shap), reverse=True
    )


def test_score_and_tier_are_copied_from_the_model_not_recomputed(model, high_risk_row):
    exp = explain_prediction(model, high_risk_row)
    assert exp.risk_score == pytest.approx(float(model.predict_proba(high_risk_row)[0]))
    assert exp.risk_tier == str(model.assign_tier(np.array([exp.risk_score]))[0])


def test_contributions_are_folded_onto_source_features_only(model, high_risk_row):
    exp = explain_prediction(model, high_risk_row)
    assert set(exp.all_contributions).issubset(set(schema.MODEL_FEATURES))
    # no one-hot leakage like "Department_Sales"
    assert not any("_" in k and k not in schema.MODEL_FEATURES for k in exp.all_contributions)


def test_direction_matches_sign(model, high_risk_row):
    exp = explain_prediction(model, high_risk_row)
    for c in exp.top_contributions:
        assert c.direction == ("increases" if c.shap > 0 else "decreases")


def test_contributions_are_additive_in_probability_space(model, high_risk_row):
    """base_value + sum(contributions) reconstructs P(leave) — same unit both kinds."""
    exp = explain_prediction(model, high_risk_row)
    recon = exp.base_value + sum(exp.all_contributions.values())
    assert recon == pytest.approx(exp.risk_score, abs=1e-6)
    assert 0.0 <= exp.base_value <= 1.0


def test_top_k_is_configurable(model, high_risk_row):
    assert len(explain_prediction(model, high_risk_row, top_k=3).top_contributions) == 3


def test_rejects_multi_row_input(model, dataset):
    with pytest.raises(ValueError, match="one row"):
        explain_prediction(model, dataset.X.head(2))


def test_prompt_facts_omit_the_numeric_score(model, high_risk_row):
    exp = explain_prediction(model, high_risk_row)
    facts = exp.as_prompt_facts()
    assert exp.risk_tier in facts
    assert f"{exp.risk_score:.2f}" not in facts
    assert str(exp.risk_score) not in facts


def test_non_tree_model_is_explainable(dataset):
    """TrainingResult.best can ship the logistic baseline; it must still explain."""
    baseline = train_baseline(dataset.X, dataset.y, seed=1)
    row = dataset.X.iloc[[int(np.argmax(baseline.predict_proba(dataset.X)))]]
    exp = explain_prediction(baseline, row)
    assert len(exp.top_contributions) == DEFAULT_TOP_K
    assert set(exp.all_contributions).issubset(set(schema.MODEL_FEATURES))
    assert exp.risk_score == pytest.approx(float(baseline.predict_proba(row)[0]))
    recon = exp.base_value + sum(exp.all_contributions.values())
    assert recon == pytest.approx(exp.risk_score, abs=1e-6)
