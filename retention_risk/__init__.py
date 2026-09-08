"""Retention Risk Explainer — explainable, fairness-audited attrition risk.

Layer boundary (enforced by tests): the classifier in :mod:`retention_risk.model`
produces the risk score; the LLM in :mod:`retention_risk.narrative` only narrates
an explanation that is already computed. The LLM never sees or alters the score.
"""

__version__ = "0.2.0"
