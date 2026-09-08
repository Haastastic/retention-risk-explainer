"""Load the IBM base data, add the engagement-survey overlay, define the target.

Everything here is deterministic given ``seed``. The label stays real
(``Attrition == "Yes"``); only the engagement-survey features and the
per-leaver ``months_to_departure`` are synthesised, and both are seeded.

See ``docs/data-framing.md`` for the target definition and the derivation of
each engineered column, and ``retention_risk/schema.py`` for the column policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from retention_risk import schema

DEFAULT_RAW_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "ibm_hr_attrition.csv"

_TRAVEL_STRAIN = {
    "Non-Travel": 0.0,
    "Travel_Rarely": 0.35,
    "Travel_Frequently": 1.0,
}


@dataclass
class Dataset:
    """The Phase 2 output the rest of the pipeline consumes.

    ``X`` holds model features only — no protected attributes, no age proxies.
    ``protected`` is kept beside it strictly for the Phase 4 fairness audit.
    """

    X: pd.DataFrame
    y: pd.Series
    protected: pd.DataFrame
    months_to_departure: pd.Series
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        leaked = set(self.X.columns) & schema.withheld_columns()
        if leaked:
            raise AssertionError(f"withheld columns leaked into feature matrix: {sorted(leaked)}")


def load_raw(csv_path: str | Path = DEFAULT_RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV and drop identifier / constant columns."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in schema.ID_OR_CONSTANT if c in df.columns])
    return df


def _scale_1_4(series: pd.Series) -> pd.Series:
    """Map a 1..4 Likert column onto 0..100."""
    return (series.astype(float) - 1.0) / 3.0 * 100.0


def _saturating(x: pd.Series, half: float) -> pd.Series:
    """0..1, reaching 0.5 at ``x == half``. Used for 'years with X' effects."""
    return x.astype(float) / (x.astype(float) + half)


def engineer_engagement_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Derive the engagement-survey overlay from real IBM columns + seeded noise.

    Returns a frame with exactly ``schema.ENGINEERED_ENGAGEMENT_FEATURES``.
    """
    n = len(df)

    def noise(scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, n)

    engagement_score = (
        _scale_1_4(df["JobSatisfaction"])
        + _scale_1_4(df["JobInvolvement"])
        + _scale_1_4(df["EnvironmentSatisfaction"])
    ) / 3.0 + noise(4.0)

    manager_relationship = (
        0.6 * _scale_1_4(df["RelationshipSatisfaction"])
        + 40.0 * _saturating(df["YearsWithCurrManager"], half=4.0)
        + noise(5.0)
    )

    # Built only from allowed features: training volume (+), role stagnation (−),
    # and career velocity = JobLevel relative to company tenure (+). The excluded
    # age proxy YearsSinceLastPromotion is NOT used here — folding in a near-
    # recoverable form of it would defeat the exclusion (see docs/data-framing.md §4).
    career_velocity = df["JobLevel"].astype(float) / (df["YearsAtCompany"].astype(float) + 3.0)
    growth_opportunity = (
        45.0
        + 5.0 * df["TrainingTimesLastYear"].astype(float)
        - 3.0 * df["YearsInCurrentRole"].astype(float)
        + 25.0 * career_velocity
        + noise(6.0)
    )

    workload_strain = (
        45.0 * (df["OverTime"] == "Yes").astype(float)
        + 25.0 * df["BusinessTravel"].map(_TRAVEL_STRAIN).fillna(0.0)
        + (4.0 - df["WorkLifeBalance"].astype(float)) * 10.0
        + noise(6.0)
    )

    recognition = (
        2.2 * df["PercentSalaryHike"].astype(float)
        + 8.0 * (df["PerformanceRating"].astype(float) - 3.0)
        + 6.0 * df["StockOptionLevel"].astype(float)
        + noise(6.0)
    )

    # eNPS carries a deliberate, mild pull from the real label — a real eNPS item
    # correlates with attrition. Coefficient kept small so it is signal, not leak.
    leaver = (df[schema.RAW_TARGET_COLUMN] == "Yes").astype(float)
    enps = (
        0.9 * (engagement_score - 50.0)
        + 0.5 * (manager_relationship - 50.0)
        - 0.4 * (workload_strain - 40.0)
        + 0.3 * (recognition - 45.0)
        - 12.0 * leaver
        + noise(12.0)
    )

    level_median_income = df.groupby("JobLevel")["MonthlyIncome"].transform("median")
    comp_ratio = df["MonthlyIncome"].astype(float) / level_median_income

    out = pd.DataFrame(
        {
            "engagement_score": engagement_score.clip(0, 100),
            "manager_relationship": manager_relationship.clip(0, 100),
            "growth_opportunity": growth_opportunity.clip(0, 100),
            "workload_strain": workload_strain.clip(0, 100),
            "recognition": recognition.clip(0, 100),
            "enps": enps.clip(-100, 100),
            "comp_ratio": comp_ratio,
        },
        index=df.index,
    )
    assert list(out.columns) == list(schema.ENGINEERED_ENGAGEMENT_FEATURES)
    return out.round(2)


def synthesize_time_to_departure(leaver_mask: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Months from the observation snapshot to departure, for leavers only.

    Log-normal, median ~8 months, clipped to [1, 36]. Stayers get ``NaN``.
    """
    months = pd.Series(np.nan, index=leaver_mask.index, dtype=float)
    k = int(leaver_mask.sum())
    if k:
        draws = rng.lognormal(mean=np.log(8.0), sigma=0.6, size=k)
        months.loc[leaver_mask] = np.clip(np.round(draws, 1), 1.0, 36.0)
    return months


def build_dataset(
    csv_path: str | Path = DEFAULT_RAW_CSV,
    *,
    horizon_months: int = schema.DEFAULT_HORIZON_MONTHS,
    seed: int = 42,
) -> Dataset:
    """Assemble the modelling dataset.

    Target: an employee who left (`Attrition == "Yes"`) **within
    ``horizon_months``** of the snapshot. Leavers projected to depart after the
    horizon are censored to a negative label — they are not at-risk *within the
    window the manager can act on*.
    """
    rng = np.random.default_rng(seed)
    raw = load_raw(csv_path)

    leaver = raw[schema.RAW_TARGET_COLUMN] == "Yes"
    months_to_departure = synthesize_time_to_departure(leaver, rng)
    y = (leaver & (months_to_departure <= horizon_months)).astype(int)
    y.name = schema.TARGET_COLUMN

    engagement = engineer_engagement_features(raw, rng)
    X = pd.concat([raw[list(schema.RAW_MODEL_FEATURES)], engagement], axis=1)
    X = X[list(schema.MODEL_FEATURES)]

    protected = raw[list(schema.PROTECTED_ATTRIBUTES)].copy()
    protected["AgeBand"] = pd.cut(
        raw["Age"],
        bins=[17, 29, 39, 49, 60],
        labels=["18-29", "30-39", "40-49", "50-60"],
    )

    meta = {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "horizon_months": horizon_months,
        "seed": seed,
        "raw_attrition_rate": round(float(leaver.mean()), 4),
        "target_positive_rate": round(float(y.mean()), 4),
        "n_positive": int(y.sum()),
        "censored_leavers": int((leaver & (months_to_departure > horizon_months)).sum()),
        "source_csv": str(csv_path),
    }
    return Dataset(
        X=X, y=y, protected=protected, months_to_departure=months_to_departure, meta=meta
    )
