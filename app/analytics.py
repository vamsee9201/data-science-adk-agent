from __future__ import annotations

import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

_matplotlib_cache = Path(tempfile.gettempdir()) / "data-science-agent-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.data import json_value, potential_identifiers, records
from app.sessions import TransformationProposal, WorkspaceSession


RANDOM_SEED = 42
MODEL_ROW_LIMIT = 50_000
CHART_BACKGROUND = "#18191b"
CHART_PANEL = "#1c1d1f"
CHART_TEXT = "#e7e9ee"
CHART_MUTED = "#a0a4ad"
CHART_GRID = "#34363a"
CHART_ACCENT = "#4f8cff"


def _column(dataframe: pd.DataFrame, name: str) -> pd.Series:
    if name not in dataframe.columns:
        raise ValueError(f"Column '{name}' does not exist.")
    return dataframe[name]


def missing_values(dataframe: pd.DataFrame) -> dict:
    rows = max(len(dataframe), 1)
    values = [
        {
            "column": str(column),
            "missing": int(dataframe[column].isna().sum()),
            "percent": round(float(dataframe[column].isna().sum() / rows * 100), 2),
        }
        for column in dataframe.columns
    ]
    return {"rows": values, "total_missing": int(dataframe.isna().sum().sum())}


def outlier_summary(dataframe: pd.DataFrame, column: str | None = None) -> dict:
    columns = [column] if column else list(dataframe.select_dtypes(include=np.number).columns)
    output = []
    for name in columns:
        series = _column(dataframe, name)
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column '{name}' must be numeric for outlier analysis.")
        clean = series.dropna()
        if clean.empty:
            output.append({"column": name, "count": 0, "lower": None, "upper": None})
            continue
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((clean < lower) | (clean > upper)).sum())
        output.append(
            {
                "column": name,
                "count": count,
                "percent": round(count / len(clean) * 100, 2),
                "lower": json_value(lower),
                "upper": json_value(upper),
            }
        )
    return {"rows": output}


def correlation_analysis(dataframe: pd.DataFrame, columns: list[str] | None = None) -> dict:
    frame = dataframe[columns] if columns else dataframe.select_dtypes(include=np.number)
    numeric = frame.select_dtypes(include=np.number)
    if len(numeric.columns) < 2:
        raise ValueError("At least two numeric columns are required for correlation analysis.")
    matrix = numeric.corr(method="pearson")
    pairs = []
    for left_index, left in enumerate(matrix.columns):
        for right in matrix.columns[left_index + 1 :]:
            value = matrix.loc[left, right]
            if pd.notna(value):
                pairs.append(
                    {"column_a": str(left), "column_b": str(right), "correlation": round(float(value), 4)}
                )
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return {"pairs": pairs, "matrix": matrix.round(4).fillna(0).to_dict()}


def distribution_analysis(dataframe: pd.DataFrame, column: str) -> dict:
    series = _column(dataframe, column)
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if clean.empty:
            return {"column": column, "kind": "numeric", "count": 0}
        return {
            "column": column,
            "kind": "numeric",
            "count": int(clean.size),
            "mean": json_value(clean.mean()),
            "median": json_value(clean.median()),
            "std": json_value(clean.std()),
            "skew": json_value(clean.skew()),
            "quantiles": {str(q): json_value(clean.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        }
    counts = series.fillna("(missing)").astype(str).value_counts().head(20)
    return {
        "column": column,
        "kind": "categorical",
        "count": int(series.notna().sum()),
        "levels": int(series.nunique(dropna=True)),
        "top_values": [{"value": key, "count": int(value)} for key, value in counts.items()],
    }


def aggregate_data(
    dataframe: pd.DataFrame,
    group_by: str,
    value_column: str | None = None,
    aggregation: str = "count",
    limit: int = 25,
) -> dict:
    _column(dataframe, group_by)
    allowed = {"count", "sum", "mean", "median", "min", "max", "nunique"}
    if aggregation not in allowed:
        raise ValueError(f"Aggregation must be one of: {', '.join(sorted(allowed))}.")
    if aggregation == "count":
        result = dataframe.groupby(group_by, dropna=False).size().rename("count").reset_index()
    else:
        if not value_column:
            raise ValueError(f"A value column is required for the '{aggregation}' aggregation.")
        _column(dataframe, value_column)
        result = (
            dataframe.groupby(group_by, dropna=False)[value_column]
            .agg(aggregation)
            .rename(aggregation)
            .reset_index()
        )
    result = result.sort_values(result.columns[-1], ascending=False).head(min(max(limit, 1), 100))
    return {"columns": [str(column) for column in result.columns], "rows": records(result, len(result))}


def filter_data(
    dataframe: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
    limit: int = 50,
) -> dict:
    series = _column(dataframe, column)
    operators = {
        "eq": lambda: series == value,
        "ne": lambda: series != value,
        "gt": lambda: series > value,
        "gte": lambda: series >= value,
        "lt": lambda: series < value,
        "lte": lambda: series <= value,
        "contains": lambda: series.astype(str).str.contains(str(value), case=False, na=False, regex=False),
    }
    if operator not in operators:
        raise ValueError(f"Operator must be one of: {', '.join(operators)}.")
    try:
        filtered = dataframe.loc[operators[operator]()]
    except TypeError as exc:
        raise ValueError(f"Value '{value}' cannot be compared with column '{column}'.") from exc
    return {"matched_rows": int(len(filtered)), "rows": records(filtered, min(max(limit, 1), 100))}


def sort_data(
    dataframe: pd.DataFrame, column: str, descending: bool = False, limit: int = 50
) -> dict:
    _column(dataframe, column)
    sorted_frame = dataframe.sort_values(column, ascending=not descending, na_position="last")
    return {
        "sorted_by": column,
        "direction": "descending" if descending else "ascending",
        "rows": records(sorted_frame, min(max(limit, 1), 100)),
    }


def statistical_test(dataframe: pd.DataFrame, column_a: str, column_b: str) -> dict:
    left, right = _column(dataframe, column_a), _column(dataframe, column_b)
    left_numeric = pd.api.types.is_numeric_dtype(left)
    right_numeric = pd.api.types.is_numeric_dtype(right)
    if left_numeric and right_numeric:
        aligned = dataframe[[column_a, column_b]].dropna()
        if len(aligned) < 3:
            raise ValueError("At least three complete pairs are required.")
        statistic, p_value = stats.pearsonr(aligned[column_a], aligned[column_b])
        return {
            "test": "Pearson correlation",
            "statistic": json_value(statistic),
            "p_value": json_value(p_value),
            "sample_size": int(len(aligned)),
            "interpretation": "association, not evidence of causation",
        }
    if not left_numeric and not right_numeric:
        contingency = pd.crosstab(left.fillna("(missing)"), right.fillna("(missing)"))
        if contingency.size == 0 or min(contingency.shape) < 2:
            raise ValueError("Both categorical columns need at least two observed levels.")
        statistic, p_value, dof, _ = stats.chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        phi2 = statistic / n
        cramers_v = math.sqrt(phi2 / max(min(contingency.shape) - 1, 1))
        return {
            "test": "Chi-square independence",
            "statistic": json_value(statistic),
            "p_value": json_value(p_value),
            "degrees_of_freedom": int(dof),
            "cramers_v": json_value(cramers_v),
            "sample_size": int(n),
            "interpretation": "association, not evidence of causation",
        }
    numeric_name, category_name = (column_a, column_b) if left_numeric else (column_b, column_a)
    pairs = dataframe[[numeric_name, category_name]].dropna()
    groups = [group[numeric_name].to_numpy() for _, group in pairs.groupby(category_name) if len(group) >= 2]
    if len(groups) < 2:
        raise ValueError("At least two groups with two observations each are required.")
    if len(groups) == 2:
        statistic, p_value = stats.ttest_ind(*groups, equal_var=False)
        test_name = "Welch's t-test"
    else:
        statistic, p_value = stats.f_oneway(*groups)
        test_name = "One-way ANOVA"
    return {
        "test": test_name,
        "statistic": json_value(statistic),
        "p_value": json_value(p_value),
        "groups": len(groups),
        "sample_size": int(sum(len(group) for group in groups)),
        "interpretation": "group association, not evidence of causation",
    }


def _style_dark_figure(figure: plt.Figure) -> None:
    figure.patch.set_facecolor(CHART_BACKGROUND)
    for axis in figure.axes:
        axis.set_facecolor(CHART_PANEL)
        axis.tick_params(colors=CHART_MUTED, labelsize=9)
        axis.xaxis.label.set_color(CHART_TEXT)
        axis.yaxis.label.set_color(CHART_TEXT)
        axis.title.set_color(CHART_TEXT)
        for spine in axis.spines.values():
            spine.set_color(CHART_GRID)
        axis.grid(color=CHART_GRID, linewidth=.7, alpha=.42)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(CHART_PANEL)
            legend.get_frame().set_edgecolor(CHART_GRID)
            for text in legend.get_texts():
                text.set_color(CHART_TEXT)


def _save_figure(session: WorkspaceSession, figure: plt.Figure, title: str) -> dict:
    filename = f"chart-{uuid.uuid4().hex}.png"
    path = session.directory / filename
    _style_dark_figure(figure)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=CHART_BACKGROUND)
    plt.close(figure)
    artifact_id = session.add_artifact(path)
    return {
        "artifact_id": artifact_id,
        "title": title,
        "url": f"/api/sessions/{session.id}/artifacts/{artifact_id}",
    }


def create_chart(
    session: WorkspaceSession,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    title: str | None = None,
) -> dict:
    dataframe = session.require_dataframe()
    allowed = {"histogram", "bar", "line", "scatter", "box", "correlation_heatmap"}
    if chart_type not in allowed:
        raise ValueError(f"Chart type must be one of: {', '.join(sorted(allowed))}.")
    figure, axis = plt.subplots(figsize=(8, 4.8))
    chart_title = title or chart_type.replace("_", " ").title()
    if chart_type == "correlation_heatmap":
        numeric = dataframe.select_dtypes(include=np.number)
        if len(numeric.columns) < 2:
            plt.close(figure)
            raise ValueError("At least two numeric columns are required for a heatmap.")
        sns.heatmap(numeric.corr(), cmap="icefire", center=0, ax=axis)
    elif chart_type == "histogram":
        if not x:
            raise ValueError("A numeric x column is required for a histogram.")
        sns.histplot(data=dataframe, x=x, kde=True, color=CHART_ACCENT, ax=axis)
    elif chart_type == "bar":
        if not x:
            raise ValueError("An x column is required for a bar chart.")
        if y:
            sns.barplot(data=dataframe, x=x, y=y, errorbar=None, color=CHART_ACCENT, ax=axis)
        else:
            counts = dataframe[x].fillna("(missing)").astype(str).value_counts().head(20)
            sns.barplot(x=counts.index, y=counts.values, color=CHART_ACCENT, ax=axis)
            axis.set_ylabel("Count")
        axis.tick_params(axis="x", rotation=35)
    elif chart_type == "line":
        if not x or not y:
            raise ValueError("Both x and y columns are required for a line chart.")
        sns.lineplot(data=dataframe.sort_values(x), x=x, y=y, errorbar=None, color=CHART_ACCENT, ax=axis)
    elif chart_type == "scatter":
        if not x or not y:
            raise ValueError("Both x and y columns are required for a scatter plot.")
        sns.scatterplot(data=dataframe, x=x, y=y, color=CHART_ACCENT, ax=axis)
    elif chart_type == "box":
        if not y:
            y = x
            x = None
        if not y:
            raise ValueError("A numeric y column is required for a box plot.")
        sns.boxplot(data=dataframe, x=x, y=y, color=CHART_ACCENT, ax=axis)
        if x:
            axis.tick_params(axis="x", rotation=35)
    axis.set_title(chart_title)
    return _save_figure(session, figure, chart_title)


def propose_cleaning(
    session: WorkspaceSession,
    operation: str,
    columns: list[str] | None = None,
    strategy: str | None = None,
) -> dict:
    dataframe = session.require_dataframe()
    transformed = dataframe.copy(deep=True)
    selected = columns or list(transformed.columns)
    missing_columns = [column for column in selected if column not in transformed.columns]
    if missing_columns:
        raise ValueError(f"Unknown columns: {', '.join(missing_columns)}")
    warnings: list[str] = []

    if operation == "drop_duplicates":
        transformed = transformed.drop_duplicates()
    elif operation == "drop_missing":
        transformed = transformed.dropna(subset=selected)
    elif operation == "fill_missing":
        strategy = strategy or "auto"
        for column in selected:
            series = transformed[column]
            if strategy == "median" or (strategy == "auto" and pd.api.types.is_numeric_dtype(series)):
                fill_value = series.median()
            elif strategy in {"mode", "auto"}:
                modes = series.mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else "unknown"
            else:
                raise ValueError("Fill strategy must be auto, median, or mode.")
            transformed[column] = series.fillna(fill_value)
    elif operation == "convert_numeric":
        for column in selected:
            before_non_null = int(transformed[column].notna().sum())
            transformed[column] = pd.to_numeric(transformed[column], errors="coerce")
            introduced = before_non_null - int(transformed[column].notna().sum())
            if introduced:
                warnings.append(f"{introduced} values in '{column}' would become missing.")
    elif operation in {"clip_outliers", "drop_outliers"}:
        mask = pd.Series(True, index=transformed.index)
        for column in selected:
            if not pd.api.types.is_numeric_dtype(transformed[column]):
                raise ValueError(f"Column '{column}' must be numeric for outlier handling.")
            clean = transformed[column].dropna()
            q1, q3 = clean.quantile([0.25, 0.75])
            lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            if operation == "clip_outliers":
                transformed[column] = transformed[column].clip(lower, upper)
            else:
                mask &= transformed[column].isna() | transformed[column].between(lower, upper)
        if operation == "drop_outliers":
            transformed = transformed.loc[mask]
    else:
        raise ValueError(
            "Operation must be drop_duplicates, drop_missing, fill_missing, convert_numeric, "
            "clip_outliers, or drop_outliers."
        )

    proposal_id = uuid.uuid4().hex
    summary = {
        "proposal_id": proposal_id,
        "operation": operation,
        "columns": selected,
        "before_rows": int(len(dataframe)),
        "after_rows": int(len(transformed)),
        "changed_cells": int((dataframe.astype(str) != transformed.reindex(dataframe.index).astype(str)).sum().sum())
        if dataframe.shape == transformed.reindex(dataframe.index).shape
        else None,
        "before_preview": records(dataframe, 5),
        "after_preview": records(transformed, 5),
        "warnings": warnings,
    }
    session.proposal = TransformationProposal(proposal_id, operation, transformed, summary)
    return summary


def apply_proposal(session: WorkspaceSession, proposal_id: str) -> dict:
    if session.proposal is None or session.proposal.id != proposal_id:
        raise ValueError("The transformation proposal does not exist or has expired.")
    session.dataframe = session.proposal.dataframe.copy(deep=True)
    summary = session.proposal.summary
    session.proposal = None
    session.touch()
    return {"applied": True, "operation": summary["operation"], "rows": len(session.dataframe)}


def reject_proposal(session: WorkspaceSession, proposal_id: str) -> dict:
    if session.proposal is None or session.proposal.id != proposal_id:
        raise ValueError("The transformation proposal does not exist or has expired.")
    session.proposal = None
    session.touch()
    return {"rejected": True}


def reset_dataset(session: WorkspaceSession) -> dict:
    if session.original_dataframe is None:
        raise ValueError("No dataset is active.")
    session.dataframe = session.original_dataframe.copy(deep=True)
    session.proposal = None
    session.model_results.clear()
    session.touch()
    return {"reset": True, "rows": len(session.dataframe)}


def _problem_type(target: pd.Series) -> str:
    unique = target.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(target) or unique <= max(20, int(len(target) * 0.05)):
        return "classification"
    return "regression"


def _metrics(problem_type: str, truth: pd.Series, prediction: np.ndarray, probabilities=None) -> dict:
    if problem_type == "classification":
        output = {
            "accuracy": round(float(accuracy_score(truth, prediction)), 4),
            "macro_f1": round(float(f1_score(truth, prediction, average="macro")), 4),
        }
        if probabilities is not None:
            try:
                if probabilities.shape[1] == 2:
                    output["roc_auc"] = round(float(roc_auc_score(truth, probabilities[:, 1])), 4)
                else:
                    output["roc_auc_ovr"] = round(
                        float(roc_auc_score(truth, probabilities, multi_class="ovr")), 4
                    )
            except ValueError:
                pass
        return output
    return {
        "mae": round(float(mean_absolute_error(truth, prediction)), 4),
        "rmse": round(float(math.sqrt(mean_squared_error(truth, prediction))), 4),
        "r2": round(float(r2_score(truth, prediction)), 4),
    }


def train_baseline_model(session: WorkspaceSession, target: str) -> dict:
    dataframe = session.require_dataframe()
    _column(dataframe, target)
    frame = dataframe.dropna(subset=[target]).copy()
    if len(frame) < 30:
        raise ValueError("At least 30 rows with a non-missing target are required for modeling.")
    sampled = len(frame) > MODEL_ROW_LIMIT
    if sampled:
        frame = frame.sample(MODEL_ROW_LIMIT, random_state=RANDOM_SEED)

    excluded = set(potential_identifiers(frame))
    excluded.add(target)
    excluded.update(column for column in frame.columns if frame[column].nunique(dropna=True) <= 1)
    target_name = target.lower()
    for column in frame.columns:
        if column == target:
            continue
        series = frame[column]
        name = str(column).lower()
        if target_name in name or (
            not pd.api.types.is_numeric_dtype(series)
            and (series.nunique(dropna=True) > 100 or series.nunique(dropna=True) > len(frame) * 0.5)
        ):
            excluded.add(column)
            continue
        paired = frame[[column, target]].dropna()
        if not paired.empty and paired[column].nunique() <= max(frame[target].nunique(), 20):
            if paired.groupby(column)[target].nunique().max() == 1:
                excluded.add(column)
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No usable feature columns remain after excluding the target and identifiers.")

    features, labels = frame[feature_columns], frame[target]
    problem_type = _problem_type(labels)
    if problem_type == "classification" and labels.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    if problem_type == "classification" and labels.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two examples for a stratified split.")

    numeric = list(features.select_dtypes(include=np.number).columns)
    categorical = [column for column in features.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)
    stratify = labels if problem_type == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
    )

    if problem_type == "classification":
        estimators = {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1_000, random_state=RANDOM_SEED),
            "random_forest": RandomForestClassifier(
                n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1, max_depth=12
            ),
        }
        primary_metric = "macro_f1"
    else:
        estimators = {
            "dummy": DummyRegressor(strategy="mean"),
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1, max_depth=12
            ),
        }
        primary_metric = "r2"

    fitted: dict[str, Pipeline] = {}
    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        probabilities = (
            pipeline.predict_proba(x_test)
            if problem_type == "classification" and hasattr(pipeline, "predict_proba")
            else None
        )
        fitted[name] = pipeline
        predictions[name] = prediction
        results[name] = _metrics(problem_type, y_test, prediction, probabilities)

    candidate_names = [name for name in estimators if name != "dummy"]
    best_name = max(candidate_names, key=lambda name: results[name][primary_metric])
    best_pipeline = fitted[best_name]
    feature_names = list(best_pipeline.named_steps["preprocess"].get_feature_names_out())
    estimator = best_pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        importance = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    else:
        importance = np.zeros(len(feature_names))
    ranked = sorted(
        zip(feature_names, importance, strict=False), key=lambda item: float(item[1]), reverse=True
    )[:15]

    if problem_type == "classification":
        labels_for_plot = sorted(pd.Series(y_test).unique(), key=str)
        matrix = confusion_matrix(y_test, predictions[best_name], labels=labels_for_plot)
        figure, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="mako",
            annot_kws={"color": CHART_TEXT},
            ax=axis,
        )
        axis.set_title(f"{best_name.replace('_', ' ').title()} confusion matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    else:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(y_test, predictions[best_name], alpha=0.72, color=CHART_ACCENT)
        low = min(float(np.min(y_test)), float(np.min(predictions[best_name])))
        high = max(float(np.max(y_test)), float(np.max(predictions[best_name])))
        axis.plot([low, high], [low, high], linestyle="--", color="#42b883")
        axis.set_title(f"{best_name.replace('_', ' ').title()}: actual vs predicted")
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
    chart = _save_figure(session, figure, "Model evaluation")

    output = {
        "target": target,
        "problem_type": problem_type,
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "sampled_to_50000_rows": sampled,
        "excluded_columns": sorted(str(column) for column in excluded if column != target),
        "models": results,
        "selected_model": best_name,
        "feature_importance": [
            {"feature": str(name), "importance": round(float(value), 6)} for name, value in ranked
        ],
        "chart": chart,
        "limitations": [
            "This is an exploratory baseline, not a deployment-ready model.",
            "Feature importance describes predictive association, not causation.",
            "Performance is measured on one fixed holdout split.",
        ],
    }
    session.model_results.append(output)
    session.touch()
    return output
