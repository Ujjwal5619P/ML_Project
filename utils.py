from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

TARGET_MAP = {"<=50K": 0, ">50K": 1}
TARGET_LABELS = {0: "<=50K", 1: ">50K"}


@dataclass
class PreparedData:
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    numeric_features: List[str]
    categorical_features: List[str]


def load_raw_data(path: str | Path) -> pd.DataFrame:      #<- Data loading logic with removing extra first space anf making "?" as NaN,
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values=" ?")   # also removing leading/trailing spaces from string columns to ensure clean data for processing.
    """Load Adult Income dataset and normalize missing values/spaces."""
    path = Path(path)

    # Remove leading/trailing spaces from string columns.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"?": np.nan, "nan": np.nan})

    return df


def prepare_dataframe(df: pd.DataFrame) -> PreparedData:
    df = df.copy()

    # Drop duplicated rows to reduce noise.
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert target to numeric label.
    df["income"] = df["income"].map(TARGET_MAP)

    # Drop rows with missing target if any.
    df = df.dropna(subset=["income"])

    # Feature engineering.
    df["capital_balance"] = df["capital_gain"].fillna(0) - df["capital_loss"].fillna(0)
    df["has_capital_gain"] = (df["capital_gain"].fillna(0) > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"].fillna(0) > 0).astype(int)
    df["is_long_workweek"] = (df["hours_per_week"].fillna(0) > 40).astype(int)

    # Remove one redundant educational column after keeping more informative numeric form.
    df = df.drop(columns=["education"], errors="ignore")

    X = df.drop(columns=["income"])
    y = df["income"].astype(int)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return PreparedData(
        df=df,
        X=X,
        y=y,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def get_model_catalog() -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    }


def train_and_evaluate_models(
    prepared: PreparedData,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=test_size,
        random_state=random_state,
        stratify=prepared.y,
    )

    models: Dict[str, Pipeline] = {}
    rows = []

    for name, estimator in get_model_catalog().items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(prepared.numeric_features, prepared.categorical_features),
                ),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            }
        )
        models[name] = pipeline

    results = pd.DataFrame(rows).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
    return models, results, (X_train, X_test, y_train, y_test)


def get_confusion_and_report(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[np.ndarray, str]:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
    return cm, report


def save_artifacts(model: Pipeline, results: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "best_model.joblib")
    results.to_csv(output_dir / "model_results.csv", index=False)


def load_model(model_path: str | Path) -> Pipeline:
    return joblib.load(model_path)


def create_sample_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
                "capital_balance": 2174,
                "has_capital_gain": 1,
                "has_capital_loss": 0,
                "is_long_workweek": 0,
            },
            {
                "age": 52,
                "workclass": "Self-emp-not-inc",
                "fnlwgt": 209642,
                "education_num": 9,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 45,
                "native_country": "United-States",
                "capital_balance": 0,
                "has_capital_gain": 0,
                "has_capital_loss": 0,
                "is_long_workweek": 1,
            },
            {
                "age": 28,
                "workclass": "Private",
                "fnlwgt": 338409,
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Wife",
                "race": "Black",
                "sex": "Female",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "Cuba",
                "capital_balance": 0,
                "has_capital_gain": 0,
                "has_capital_loss": 0,
                "is_long_workweek": 0,
            },
        ]
    )


def plot_income_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = df["income"].map(TARGET_LABELS)
    labels.value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Income Class Distribution")
    ax.set_xlabel("Income")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_age_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["age"], bins=30, kde=True, ax=ax)
    ax.set_title("Age Distribution")
    plt.tight_layout()
    return fig


def plot_workclass_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    df["workclass"].fillna("Missing").value_counts().head(10).plot(kind="bar", ax=ax)
    ax.set_title("Top Workclass Categories")
    ax.set_xlabel("Workclass")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_numeric_correlation(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["<=50K", ">50K"])
    ax.set_yticklabels(["<=50K", ">50K"], rotation=0)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig
