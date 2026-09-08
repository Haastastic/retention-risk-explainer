"""Preprocessing: a ``ColumnTransformer`` shared by the baseline and XGBoost models.

Categoricals are one-hot encoded; numerics pass through (tree models don't need
scaling, and the baseline gets a scaler bolted on in :mod:`retention_risk.model`).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from retention_risk import schema


def build_preprocessor() -> ColumnTransformer:
    """Map the raw feature frame to a numeric matrix.

    ``remainder="passthrough"`` keeps every numeric model feature; the explicit
    ``numeric`` selector below is only for column-name bookkeeping in tests.
    """
    categorical = list(schema.CATEGORICAL_FEATURES)
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False),
                categorical,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def numeric_feature_names() -> list[str]:
    """Model features that are not categorical."""
    return [c for c in schema.MODEL_FEATURES if c not in set(schema.CATEGORICAL_FEATURES)]
