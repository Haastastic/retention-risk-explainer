"""The column policy must stay internally consistent and match docs/data-framing.md."""

from pathlib import Path

from retention_risk import schema

DOC = Path(__file__).resolve().parents[1] / "docs" / "data-framing.md"


def test_withheld_and_feature_sets_are_disjoint():
    assert set(schema.MODEL_FEATURES).isdisjoint(schema.withheld_columns())


def test_protected_attributes_are_the_brief_list():
    assert sorted(schema.PROTECTED_ATTRIBUTES) == ["Age", "Gender", "MaritalStatus"]


def test_age_proxies_are_withheld_not_featured():
    assert schema.AGE_PROXY_EXCLUSIONS.issubset(schema.withheld_columns())
    assert schema.AGE_PROXY_EXCLUSIONS.isdisjoint(schema.MODEL_FEATURES)


def test_label_columns_are_withheld_from_features():
    assert {schema.RAW_TARGET_COLUMN, schema.TARGET_COLUMN}.issubset(schema.withheld_columns())
    assert schema.RAW_TARGET_COLUMN not in schema.MODEL_FEATURES
    assert schema.TARGET_COLUMN not in schema.MODEL_FEATURES


def test_model_features_have_no_duplicates():
    assert len(schema.MODEL_FEATURES) == len(set(schema.MODEL_FEATURES))


def test_engineered_features_are_appended_to_raw_features():
    assert schema.MODEL_FEATURES[-len(schema.ENGINEERED_ENGAGEMENT_FEATURES) :] == tuple(
        schema.ENGINEERED_ENGAGEMENT_FEATURES
    )


def test_categorical_features_are_a_subset_of_model_features():
    assert set(schema.CATEGORICAL_FEATURES) <= set(schema.MODEL_FEATURES)


def test_every_withheld_and_engineered_column_is_named_in_the_doc():
    text = DOC.read_text(encoding="utf-8")
    for col in schema.withheld_columns() | set(schema.ENGINEERED_ENGAGEMENT_FEATURES):
        assert col in text, f"{col} is not documented in data-framing.md"
