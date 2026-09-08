"""Train the baseline and XGBoost risk models and evaluate them against the brief."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from retention_risk.data import Dataset, build_dataset
from retention_risk.evaluate import EvalReport, evaluate_model
from retention_risk.model import RiskModel, train_baseline, train_xgboost

DEFAULT_TEST_SIZE = 0.25


@dataclass
class TrainingResult:
    baseline: RiskModel
    xgboost: RiskModel
    baseline_report: EvalReport
    xgboost_report: EvalReport
    split_seed: int

    #: Ship XGBoost unless the baseline beats it on held-out AUC by more than this
    #: (a fraction of a point of AUC on ~370 test rows is noise, not a reason to
    #: prefer the linear model).
    SHIP_BASELINE_AUC_MARGIN = 0.02

    @property
    def best(self) -> RiskModel:
        """The model that ships: XGBoost unless the baseline clearly out-AUCs it."""
        margin = self.baseline_report.auc_roc - self.xgboost_report.auc_roc
        return self.baseline if margin > self.SHIP_BASELINE_AUC_MARGIN else self.xgboost


def split_dataset(
    ds: Dataset, *, test_size: float = DEFAULT_TEST_SIZE, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(ds.X, ds.y, test_size=test_size, stratify=ds.y, random_state=seed)


def run_training(
    *, data_seed: int = 42, split_seed: int = 42, test_size: float = DEFAULT_TEST_SIZE
) -> TrainingResult:
    ds = build_dataset(seed=data_seed)
    X_train, X_test, y_train, y_test = split_dataset(ds, test_size=test_size, seed=split_seed)

    baseline = train_baseline(X_train, y_train, seed=split_seed)
    xgb = train_xgboost(X_train, y_train, seed=split_seed)

    common = {"X_cv": X_train, "y_cv": y_train, "seed": split_seed}
    return TrainingResult(
        baseline=baseline,
        xgboost=xgb,
        baseline_report=evaluate_model(baseline, X_test, y_test, **common),
        xgboost_report=evaluate_model(xgb, X_test, y_test, **common),
        split_seed=split_seed,
    )
