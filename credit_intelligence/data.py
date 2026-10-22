from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from credit_intelligence.constants import (
    IDENTIFIER_COLUMN,
    RAW_DATA_DIR,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    UCI_DATASET_NAME,
    UCI_DATASET_URL,
)

DATASET_FILE = "default of credit card clients.xls"
CACHED_DATASET_PATH = RAW_DATA_DIR / "default_of_credit_card_clients.xls"
SOURCE_TARGET_COLUMN = "default payment next month"


def download_dataset(destination: Path = CACHED_DATASET_PATH) -> Path:
    """Download the original UCI archive and cache its spreadsheet locally."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    request = Request(
        UCI_DATASET_URL,
        headers={"User-Agent": "Credit-Intelligence-Portfolio/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if DATASET_FILE not in archive.namelist():
            raise ValueError(
                f"The UCI archive did not contain the expected file "
                f"{DATASET_FILE!r}; found {archive.namelist()}."
            )
        with archive.open(DATASET_FILE) as source, destination.open("wb") as target:
            target.write(source.read())
    return destination


def validate_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate the observed UCI schema and report data quality without fabricating values."""
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    source_column_mapping: dict[str, str] = {}
    if TARGET_COLUMN not in normalized.columns and SOURCE_TARGET_COLUMN in normalized.columns:
        normalized = normalized.rename(columns={SOURCE_TARGET_COLUMN: TARGET_COLUMN})
        source_column_mapping[SOURCE_TARGET_COLUMN] = TARGET_COLUMN
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in normalized]
    if missing_columns:
        raise ValueError(
            "The downloaded UCI file does not match the documented schema. "
            f"Missing columns: {missing_columns}; observed columns: "
            f"{normalized.columns.tolist()}."
        )

    normalized = normalized[REQUIRED_COLUMNS]
    for column in REQUIRED_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")

    if normalized[TARGET_COLUMN].isna().any():
        raise ValueError(f"The target {TARGET_COLUMN!r} contains missing values.")
    target_values = set(normalized[TARGET_COLUMN].dropna().unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"The target must contain only 0 and 1; observed {sorted(target_values)}."
        )
    if not normalized[IDENTIFIER_COLUMN].is_unique:
        raise ValueError("The source identifier column ID is not unique.")

    features = normalized.drop(columns=[IDENTIFIER_COLUMN, TARGET_COLUMN])
    numeric_features = features.select_dtypes(include="number").to_numpy(dtype=float)
    if np.isinf(numeric_features).any():
        raise ValueError("The feature data contains infinite numeric values.")

    profile: dict[str, object] = {
        "name": UCI_DATASET_NAME,
        "source": UCI_DATASET_URL,
        "target": TARGET_COLUMN,
        "source_column_mapping": source_column_mapping,
        "rows": int(len(normalized)),
        "features": int(features.shape[1]),
        "feature_names": features.columns.tolist(),
        "missing_values": int(normalized.isna().sum().sum()),
        "duplicate_rows": int(normalized.duplicated().sum()),
        "duplicate_identifiers": int(normalized[IDENTIFIER_COLUMN].duplicated().sum()),
        "default_rate": float(normalized[TARGET_COLUMN].mean()),
        "target_distribution": {
            str(int(label)): int(count)
            for label, count in normalized[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "retained_duplicate_rows": True,
        "cleaning_note": (
            "The source schema and numeric values are validated. ID is retained for "
            "quality checks but excluded from predictors. Missing feature values, if "
            "present, are imputed inside each training pipeline. Exact duplicate rows "
            "are reported and retained; no records are silently removed."
        ),
    }
    return normalized, profile


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    dataset_path = path or download_dataset()
    # The official XLS has a descriptive row before its actual column labels.
    source = pd.read_excel(dataset_path, engine="xlrd", header=1)
    return validate_dataset(source)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
