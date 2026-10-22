from __future__ import annotations

import pandas as pd
import pytest

from credit_intelligence.constants import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from credit_intelligence.data import validate_dataset


def source_frame() -> pd.DataFrame:
    row = {
        IDENTIFIER_COLUMN: 1,
        **{name: 0 for name in CATEGORICAL_FEATURES},
        **{name: 100.0 for name in NUMERIC_FEATURES},
        TARGET_COLUMN: 0,
    }
    second = {**row, IDENTIFIER_COLUMN: 2, "AGE": 44, TARGET_COLUMN: 1}
    return pd.DataFrame([row, second])


def test_schema_has_the_documented_23_predictors() -> None:
    assert len(FEATURE_COLUMNS) == 23
    assert len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES) == 23


def test_validation_reports_quality_and_excludes_identifier_from_features() -> None:
    validated, profile = validate_dataset(source_frame())

    assert len(validated) == 2
    assert profile["rows"] == 2
    assert profile["features"] == 23
    assert profile["default_rate"] == 0.5
    assert profile["feature_names"] == FEATURE_COLUMNS
    assert profile["missing_values"] == 0


def test_validation_fails_explicitly_when_source_schema_is_incomplete() -> None:
    invalid = source_frame().drop(columns=["PAY_AMT6"])
    with pytest.raises(ValueError, match="does not match the documented schema"):
        validate_dataset(invalid)


def test_validation_rejects_a_non_binary_target() -> None:
    invalid = source_frame()
    invalid.loc[0, TARGET_COLUMN] = 3
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_dataset(invalid)
