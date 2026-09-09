"""Narrative layer: template output, Claude narrator (mocked), and the layer boundary."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from retention_risk.explain import explain_prediction
from retention_risk.narrative import (
    ClaudeNarrator,
    Narrative,
    TemplateNarrator,
    get_narrator,
)
from retention_risk.training import run_training


@pytest.fixture(scope="session")
def model(dataset):
    return run_training(split_seed=42).best


@pytest.fixture(scope="session")
def explanation(dataset, model):
    proba = model.predict_proba(dataset.X)
    return explain_prediction(model, dataset.X.iloc[[int(np.argmax(proba))]])


@pytest.fixture(scope="session")
def low_explanation(dataset, model):
    proba = model.predict_proba(dataset.X)
    return explain_prediction(model, dataset.X.iloc[[int(np.argmin(proba))]])


class _FakeMessages:
    def __init__(self, payload: str):
        self.payload = payload
        self.captured: dict | None = None

    def create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text=self.payload)])


class _FakeClient:
    def __init__(self, payload: str):
        self.messages = _FakeMessages(payload)


# --- template narrator -------------------------------------------------------
def test_template_narrative_shape(explanation):
    n = TemplateNarrator().narrate(explanation, employee_ref="Sam")
    assert n.source == "template"
    assert n.summary.startswith("Sam")
    assert 1 <= len(n.drivers) <= 4
    assert n.suggested_action


def test_template_is_deterministic(explanation):
    a = TemplateNarrator().narrate(explanation)
    b = TemplateNarrator().narrate(explanation)
    assert a == b


def test_low_risk_reads_as_reassuring(low_explanation):
    n = TemplateNarrator().narrate(low_explanation)
    assert "not currently showing elevated" in n.summary


# --- claude narrator (mocked) ---------------------------------------------
def test_claude_narrator_parses_json(explanation):
    payload = json.dumps(
        {
            "summary": "Sam has a few signals worth a conversation.",
            "drivers": ["engagement dipped", "workload is heavy"],
            "suggested_action": "Book a relaxed 1:1 this week.",
        }
    )
    n = ClaudeNarrator(client=_FakeClient(payload)).narrate(explanation, employee_ref="Sam")
    assert n.source == "claude"
    assert n.drivers == ["engagement dipped", "workload is heavy"]


def test_claude_narrator_tolerates_code_fences(explanation):
    payload = (
        "```json\n"
        + json.dumps({"summary": "s", "drivers": ["a", "b"], "suggested_action": "act"})
        + "\n```"
    )
    n = ClaudeNarrator(client=_FakeClient(payload)).narrate(explanation)
    assert n.summary == "s"


class _RaisingClient:
    class messages:  # noqa: N801
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("network is down")


def test_claude_narrator_falls_back_to_template_on_api_error(explanation):
    n = ClaudeNarrator(client=_RaisingClient()).narrate(explanation, employee_ref="Sam")
    assert n.source == "template"
    assert n.summary.startswith("Sam")


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        "{ truncated",
        json.dumps({"summary": "s"}),  # missing drivers / suggested_action
        json.dumps(
            {"summary": "s", "drivers": "a, b, c", "suggested_action": "x"}
        ),  # drivers not a list
        json.dumps(["summary", "drivers"]),  # not even an object
    ],
)
def test_claude_narrator_falls_back_on_bad_payload(explanation, payload):
    n = ClaudeNarrator(client=_FakeClient(payload)).narrate(explanation)
    assert n.source == "template"


# --- layer boundary: the LLM never sees or sets the score --------------
def test_narrative_type_has_no_score_field():
    fields = {f.name for f in dataclasses.fields(Narrative)}
    assert "risk_score" not in fields and "risk_tier" not in fields
    assert not any("score" in f or "prob" in f for f in fields)


def test_prompt_sent_to_claude_contains_no_score(explanation):
    client = _FakeClient(json.dumps({"summary": "s", "drivers": ["a"], "suggested_action": "act"}))
    ClaudeNarrator(client=client).narrate(explanation, employee_ref="Sam")
    sent = json.dumps(client.messages.captured)
    assert str(explanation.risk_score) not in sent
    assert f"{explanation.risk_score:.2f}" not in sent
    assert f"{explanation.risk_score:.4f}" not in sent
    assert explanation.risk_tier in sent  # the tier is fine to share


def test_narrate_does_not_call_the_model(explanation, monkeypatch):
    import retention_risk.model as model_mod

    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("narrator touched the model")

    monkeypatch.setattr(model_mod.RiskModel, "predict_proba", _boom)
    monkeypatch.setattr(model_mod.RiskModel, "assign_tier", _boom)
    TemplateNarrator().narrate(explanation)
    ClaudeNarrator(
        client=_FakeClient(json.dumps({"summary": "s", "drivers": ["a"], "suggested_action": "x"}))
    ).narrate(explanation)


def test_narrator_cannot_mutate_the_explanation_score(explanation):
    before = explanation.risk_score
    TemplateNarrator().narrate(explanation)
    assert explanation.risk_score == before


# --- factory --------------------------------------------------------------
def test_factory_returns_template_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(get_narrator(), TemplateNarrator)


def test_factory_prefers_claude_when_asked(monkeypatch):
    monkeypatch.setattr(
        "retention_risk.narrative.ClaudeNarrator",
        lambda *a, **k: SimpleNamespace(source="claude"),
    )
    assert get_narrator(prefer_claude=True).source == "claude"


def test_factory_falls_back_if_claude_init_fails(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("no key")

    monkeypatch.setattr("retention_risk.narrative.ClaudeNarrator", _raise)
    assert isinstance(get_narrator(prefer_claude=True), TemplateNarrator)
