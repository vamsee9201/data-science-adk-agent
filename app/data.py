from __future__ import annotations

import csv
import io
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn import datasets


SAMPLES = {
    "iris": ("Iris", "classification", datasets.load_iris),
    "wine": ("Wine", "classification", datasets.load_wine),
    "breast-cancer": ("Breast Cancer", "classification", datasets.load_breast_cancer),
    "diabetes": ("Diabetes", "regression", datasets.load_diabetes),
}


class DatasetValidationError(ValueError):
    pass


def json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) or math.isinf(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def records(dataframe: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    frame = dataframe.head(limit).copy()
    return [
        {str(column): json_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def parse_csv(content: bytes, max_bytes: int) -> pd.DataFrame:
    if not content:
        raise DatasetValidationError("The uploaded CSV is empty.")
    if len(content) > max_bytes:
        raise DatasetValidationError(
            f"The CSV exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
        )
    if b"\x00" in content:
        raise DatasetValidationError("The file contains binary data and is not a valid CSV.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("CSV files must use UTF-8 or UTF-8-SIG encoding.") from exc

    try:
        header = next(csv.reader(io.StringIO(text)))
    except (csv.Error, StopIteration) as exc:
        raise DatasetValidationError("The CSV header could not be read.") from exc
    normalized = [name.strip() for name in header]
    if not normalized or all(not name for name in normalized):
        raise DatasetValidationError("The CSV must contain at least one named column.")
    if any(not name for name in normalized):
        raise DatasetValidationError("Every CSV column must have a name.")
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise DatasetValidationError(f"Duplicate column names are not allowed: {', '.join(duplicates)}")

    try:
        dataframe = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise DatasetValidationError(f"The CSV could not be parsed: {exc}") from exc
    if dataframe.columns.empty:
        raise DatasetValidationError("The CSV must contain at least one column.")
    if dataframe.empty:
        raise DatasetValidationError("The CSV contains column names but no data rows.")
    dataframe.columns = normalized
    return dataframe


def sample_catalog() -> list[dict[str, str]]:
    return [
        {"id": sample_id, "name": name, "task_type": task_type}
        for sample_id, (name, task_type, _) in SAMPLES.items()
    ]


def load_sample(sample_id: str) -> tuple[str, pd.DataFrame]:
    if sample_id not in SAMPLES:
        raise DatasetValidationError(f"Unknown sample dataset: {sample_id}")
    name, _, loader = SAMPLES[sample_id]
    bunch = loader(as_frame=True)
    dataframe = bunch.frame.copy()
    if "target" in dataframe.columns and getattr(bunch, "target_names", None) is not None:
        target_names = list(bunch.target_names)
        dataframe["target_label"] = dataframe["target"].map(
            lambda value: str(target_names[int(value)]) if int(value) < len(target_names) else str(value)
        )
    return name, dataframe


def potential_identifiers(dataframe: pd.DataFrame) -> list[str]:
    identifiers: list[str] = []
    rows = len(dataframe)
    for column in dataframe.columns:
        series = dataframe[column]
        name = str(column).lower()
        unique_ratio = series.nunique(dropna=True) / max(rows, 1)
        if name in {"id", "uuid", "identifier", "index"} or name.endswith("_id"):
            identifiers.append(str(column))
        elif unique_ratio > 0.98 and (
            pd.api.types.is_string_dtype(series) or pd.api.types.is_integer_dtype(series)
        ):
            identifiers.append(str(column))
    return identifiers


def infer_schema(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    schema = []
    identifiers = set(potential_identifiers(dataframe))
    for column in dataframe.columns:
        series = dataframe[column]
        if pd.api.types.is_bool_dtype(series):
            semantic_type = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            semantic_type = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            semantic_type = "numeric"
        else:
            semantic_type = "categorical" if series.nunique(dropna=True) <= 50 else "text"
        schema.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "semantic_type": semantic_type,
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "potential_identifier": str(column) in identifiers,
            }
        )
    return schema


def dataset_metadata(name: str, dataframe: pd.DataFrame, preview_rows: int = 10) -> dict:
    return {
        "name": name,
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "schema": infer_schema(dataframe),
        "preview": records(dataframe, preview_rows),
    }


def profile_dataset(dataframe: pd.DataFrame) -> dict:
    numeric = dataframe.select_dtypes(include=np.number)
    descriptions: dict[str, dict[str, Any]] = {}
    for column in dataframe.columns:
        series = dataframe[column]
        item: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 2),
            "unique": int(series.nunique(dropna=True)),
        }
        if column in numeric.columns and series.notna().any():
            item.update(
                mean=json_value(series.mean()),
                median=json_value(series.median()),
                std=json_value(series.std()),
                min=json_value(series.min()),
                max=json_value(series.max()),
            )
        else:
            modes = series.mode(dropna=True)
            item["most_common"] = json_value(modes.iloc[0]) if not modes.empty else None
        descriptions[str(column)] = item
    return {
        "shape": {"rows": int(len(dataframe)), "columns": int(len(dataframe.columns))},
        "duplicates": int(dataframe.duplicated().sum()),
        "potential_identifiers": potential_identifiers(dataframe),
        "columns": descriptions,
    }
