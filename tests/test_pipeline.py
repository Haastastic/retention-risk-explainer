"""Preprocessing: shape, column coverage, unseen-category handling."""

from retention_risk import schema
from retention_risk.pipeline import build_preprocessor, numeric_feature_names


def test_numeric_and_categorical_cover_all_model_features():
    assert set(numeric_feature_names()) | set(schema.CATEGORICAL_FEATURES) == set(
        schema.MODEL_FEATURES
    )
    assert set(numeric_feature_names()).isdisjoint(schema.CATEGORICAL_FEATURES)


def test_transform_produces_all_numeric_matrix(dataset):
    prep = build_preprocessor()
    out = prep.fit_transform(dataset.X)
    assert out.shape[0] == len(dataset.X)
    # one-hot expansion means at least as many columns as raw features
    assert out.shape[1] >= len(schema.MODEL_FEATURES)
    assert out.dtype.kind == "f"


def test_unseen_category_does_not_blow_up(dataset):
    prep = build_preprocessor()
    prep.fit(dataset.X)
    novel = dataset.X.head(5).copy()
    novel.loc[:, "Department"] = "Time Travel"
    transformed = prep.transform(novel)
    assert transformed.shape == (5, prep.transform(dataset.X.head(5)).shape[1])
