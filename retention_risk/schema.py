"""Column policy for the attrition model — the Phase 2 exclusion list, as code.

This module is the single source of truth for which raw columns may become model
features, which are withheld as protected attributes, and which are excluded as
age proxies. ``docs/data-framing.md`` is the prose companion; the two must agree,
and ``tests/test_schema.py`` checks that the built feature matrix honours this.
"""

from __future__ import annotations

# --- Withheld: protected attributes -----------------------------------------
# Never features. Retained separately, for the fairness audit only (Phase 4).
# The IBM dataset has no race or disability columns; if it did they would list here.
PROTECTED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "Age",
        "Gender",
        "MaritalStatus",
    }
)

# --- Withheld: age-correlated proxies --------------------------------------
# Excluded as features because they function as stand-ins for age (discovery
# brief, Constraints). Excluded outright — NOT re-encoded into an engineered
# composite, since a near-recoverable form would defeat the exclusion. The
# overlap with age is mostly also carried by allowed features (YearsAtCompany,
# YearsInCurrentRole, TrainingTimesLastYear). See docs/data-framing.md §4.
#   TotalWorkingYears       -> total career tenure, near-linear in age
#   YearsSinceLastPromotion -> age/seniority proxy; long tail is older employees
AGE_PROXY_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "TotalWorkingYears",
        "YearsSinceLastPromotion",
    }
)

# --- Withheld: identifiers and constants ---------------------------------
# No signal. Dropped on load.
ID_OR_CONSTANT: frozenset[str] = frozenset(
    {
        "EmployeeNumber",
        "EmployeeCount",
        "StandardHours",
        "Over18",
    }
)

# --- Kept, but watched ---------------------------------------------------
# Company-tenure columns correlate with age but are legitimate retention signals
# (tenure milestones, manager-relationship depth). Kept as features AND reported
# in the fairness audit so the age correlation stays visible.
WATCHED_TENURE_FEATURES: frozenset[str] = frozenset(
    {
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsWithCurrManager",
    }
)

# --- Engineered engagement-survey overlay ------------------------------
# Added by data.py :: engineer_engagement_features. Every entry documents the
# real IBM columns it is derived from in docs/data-framing.md.
ENGINEERED_ENGAGEMENT_FEATURES: tuple[str, ...] = (
    "engagement_score",
    "manager_relationship",
    "growth_opportunity",
    "workload_strain",
    "recognition",
    "enps",
    "comp_ratio",
)

# --- Raw IBM columns that survive as model features -------------------
# Everything in the raw file that is not withheld above and not the label.
RAW_MODEL_FEATURES: tuple[str, ...] = (
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsWithCurrManager",
)

MODEL_FEATURES: tuple[str, ...] = RAW_MODEL_FEATURES + ENGINEERED_ENGAGEMENT_FEATURES

RAW_TARGET_COLUMN = "Attrition"
TARGET_COLUMN = "left_within_horizon"
DEFAULT_HORIZON_MONTHS = 12

# Categorical model features (the rest are numeric).
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "OverTime",
)


def withheld_columns() -> frozenset[str]:
    """Every column that must never appear in the model feature matrix.

    Includes the label columns: ``Attrition`` deliberately survives ``load_raw``
    (it feeds target construction and the ``enps`` term), so the leakage guard
    has to reject it explicitly rather than rely on the feature allowlist.
    """
    return (
        PROTECTED_ATTRIBUTES
        | AGE_PROXY_EXCLUSIONS
        | ID_OR_CONSTANT
        | {RAW_TARGET_COLUMN, TARGET_COLUMN}
    )
