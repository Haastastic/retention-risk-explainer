"""The risk model: preprocessing + estimator + risk-tier thresholds, in one object.

This is the *decision layer*. It owns the probability and the tier. The LLM
explanation layer (:mod:`retention_risk.narrative`, Phase 5) consumes this
object's output and never feeds back into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from retention_risk import schema
from retention_risk.pipeline import build_preprocessor

ModelKind = Literal["baseline", "xgboost"]
Tier = Literal["High", "Medium", "Low"]
TIERS: tuple[Tier, ...] = ("High", "Medium", "Low")

# Default tier cut-points, as quantiles of the training-set score distribution.
# "High" ~ top 12% (brief: high-risk tier <= 15% of a team);
# "High"+"Medium" ~ top 35% (brief: recall measured across the top two tiers).
DEFAULT_HIGH_QUANTILE = 0.88
DEFAULT_MEDIUM_QUANTILE = 0.65

# Rows in the SHAP background reference (see synthesize_background).
BACKGROUND_ROWS = 100


def synthesize_background(
    X: pd.DataFrame, *, n: int = BACKGROUND_ROWS, seed: int = 42
) -> pd.DataFrame:
    """A synthetic SHAP background sample — no intact employee record is kept.

    Each column is resampled independently from its own training values, so every
    marginal distribution is preserved but the joint combination in any output
    row belongs to no real person. This is what gets persisted with the model
    (``RiskModel.save``) and shipped on deploy, so it must carry no PII.
    """
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    return pd.DataFrame(
        {col: rng.choice(X[col].to_numpy(), size=n, replace=True) for col in X.columns},
        index=idx,
    ).astype(X.dtypes.to_dict())


@dataclass
class TierThresholds:
    """Score cut-points learned from a reference (training) distribution."""

    high: float
    medium: float

    def assign(self, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, dtype=float)
        out = np.full(proba.shape, "Low", dtype=object)
        out[proba >= self.medium] = "Medium"
        out[proba >= self.high] = "High"
        return out


def build_pipeline(kind: ModelKind, y: pd.Series, seed: int = 42) -> Pipeline:
    """A fresh, unfitted preprocessing + estimator pipeline.

    Shared by :meth:`RiskModel.fit` and the cross-validation in
    :mod:`retention_risk.evaluate` so both score the identical model.
    """
    return Pipeline([("prep", build_preprocessor()), ("est", _make_estimator(kind, y, seed))])


def _make_estimator(kind: ModelKind, y: pd.Series, seed: int):
    if kind == "baseline":
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
                ),
            ]
        )
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=(neg / pos) if pos else 1.0,
        eval_metric="auc",
        random_state=seed,
        n_jobs=2,
    )


class RiskModel:
    """Fit / score wrapper around a preprocessing + estimator pipeline."""

    def __init__(
        self,
        kind: ModelKind = "xgboost",
        *,
        seed: int = 42,
        high_quantile: float = DEFAULT_HIGH_QUANTILE,
        medium_quantile: float = DEFAULT_MEDIUM_QUANTILE,
    ) -> None:
        if not 0.0 < medium_quantile < high_quantile < 1.0:
            raise ValueError("require 0 < medium_quantile < high_quantile < 1")
        self.kind = kind
        self.seed = seed
        self.high_quantile = high_quantile
        self.medium_quantile = medium_quantile
        self.pipeline_: Pipeline | None = None
        self.thresholds_: TierThresholds | None = None
        self.feature_names_: list[str] = []
        self.background_: pd.DataFrame | None = None  # synthetic SHAP reference; no PII

    # -- fit / predict -----------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> RiskModel:
        self._check_columns(X)
        self.feature_names_ = list(X.columns)
        self.pipeline_ = build_pipeline(self.kind, y, self.seed)
        self.pipeline_.fit(X, y)
        self.background_ = synthesize_background(X, seed=self.seed)
        train_scores = self._raw_proba(X)
        self.thresholds_ = TierThresholds(
            high=float(np.quantile(train_scores, self.high_quantile)),
            medium=float(np.quantile(train_scores, self.medium_quantile)),
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(leaves within the horizon), as a 1-D array."""
        self._check_fitted()
        self._check_columns(X)
        return self._raw_proba(X)

    def assign_tier(self, proba: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.thresholds_.assign(proba)

    def predict_tier(self, X: pd.DataFrame) -> np.ndarray:
        return self.assign_tier(self.predict_proba(X))

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        self._check_fitted()
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: str | Path) -> RiskModel:
        obj = joblib.load(Path(path))
        if not isinstance(obj, RiskModel):
            raise TypeError(f"{path} is not a RiskModel")
        return obj

    # -- internals -----------------------------------------------------
    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X)[:, 1]

    def _check_fitted(self) -> None:
        if self.pipeline_ is None or self.thresholds_ is None:
            raise RuntimeError("RiskModel is not fitted; call fit() first")

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = [c for c in schema.MODEL_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"feature frame missing columns: {missing}")
        leaked = set(X.columns) & schema.withheld_columns()
        if leaked:
            raise ValueError(f"withheld columns present in feature frame: {sorted(leaked)}")


def train_baseline(X: pd.DataFrame, y: pd.Series, *, seed: int = 42) -> RiskModel:
    return RiskModel("baseline", seed=seed).fit(X, y)


def train_xgboost(X: pd.DataFrame, y: pd.Series, *, seed: int = 42) -> RiskModel:
    return RiskModel("xgboost", seed=seed).fit(X, y)
