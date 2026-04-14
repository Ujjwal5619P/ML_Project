
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import (
    TARGET_LABELS,
    create_sample_inputs,
    get_confusion_and_report,
    load_raw_data,
    plot_age_distribution,
    plot_confusion_matrix,
    plot_income_distribution,
    plot_numeric_correlation,
    plot_workclass_distribution,
    prepare_dataframe,
    train_and_evaluate_models,
)

st.set_page_config(page_title="Income Classification Dashboard", layout="wide")

st.markdown("""
<style>
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 18px 20px;
    min-height: 110px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.metric-label {
    font-size: 15px;
    color: #9ca3af;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: white;
}
.workflow-card {
    background: #12263f;
    border: 1px solid #1e3a5f;
    border-radius: 18px;
    padding: 22px;
    min-height: 220px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.workflow-title {
    font-size: 21px;
    font-weight: 700;
    color: #4ea1ff;
    margin-bottom: 14px;
}
.workflow-list {
    color: #d1d5db;
    font-size: 16px;
    line-height: 1.8;
    padding-left: 20px;
    margin: 0;
}
.pipeline-box {
    background: #0f172a;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 18px 20px;
    color: #e5e7eb;
    font-size: 17px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "adult.data"


@st.cache_data
def get_raw_data():
    return load_raw_data(DATA_PATH)


@st.cache_data
def get_prepared_data():
    return prepare_dataframe(get_raw_data())


@st.cache_resource
def get_training_outputs():
    return train_and_evaluate_models(get_prepared_data())


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_percent = (df.isna().mean() * 100).round(2)
    summary = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": missing_count.values,
            "Missing %": missing_percent.values,
        }
    )
    summary = summary[summary["Missing Count"] > 0].sort_values(
        by="Missing %", ascending=False
    ).reset_index(drop=True)
    return summary


def drop_rows_by_columns(df: pd.DataFrame, selected_columns):
    before_rows = df.shape[0]
    new_df = df.dropna(subset=selected_columns).reset_index(drop=True)
    removed_rows = before_rows - new_df.shape[0]
    return new_df, removed_rows


def apply_manual_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

    return df


def get_iqr_candidate_columns(df: pd.DataFrame):
    excluded_cols = {"income", "has_capital_gain", "has_capital_loss", "is_long_workweek"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in excluded_cols]


def detect_outliers_iqr_for_column(df: pd.DataFrame, column: str):
    series = df[column].dropna()

    if series.empty:
        return {
            "column": column,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "mask": pd.Series([False] * len(df), index=df.index),
            "note": "No valid values available.",
        }

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    if column == "age":
        lower_bound = max(0, lower_bound)

    mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_count = int(mask.sum())

    note = ""
    if iqr == 0 and column in ["capital_gain", "capital_loss"]:
        note = "Highly zero-inflated column: Q1 and Q3 are both 0, so non-zero values may be flagged."

    return {
        "column": column,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "mask": mask,
        "note": note,
    }


def get_outlier_summary(df: pd.DataFrame):
    candidate_cols = get_iqr_candidate_columns(df)
    results = []
    combined_mask = pd.Series([False] * len(df), index=df.index)

    for col in candidate_cols:
        result = detect_outliers_iqr_for_column(df, col)
        results.append(
            {
                "Column": result["column"],
                "Lower Bound": None if result["lower_bound"] is None else round(result["lower_bound"], 2),
                "Upper Bound": None if result["upper_bound"] is None else round(result["upper_bound"], 2),
                "Outliers Detected": result["outlier_count"],
                "Note": result["note"],
            }
        )
        combined_mask = combined_mask | result["mask"]

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="Outliers Detected", ascending=False).reset_index(drop=True)

    total_rows_with_any_outlier = int(combined_mask.sum())
    return summary_df, combined_mask, total_rows_with_any_outlier


def remove_all_detected_outliers(df: pd.DataFrame):
    summary_df, combined_mask, total_rows_with_any_outlier = get_outlier_summary(df)
    new_df = df[~combined_mask].reset_index(drop=True)
    removed_rows = df.shape[0] - new_df.shape[0]
    return new_df, summary_df, removed_rows, total_rows_with_any_outlier


def cap_outliers_iqr_for_column(df: pd.DataFrame, column: str):
    df = df.copy()
    result = detect_outliers_iqr_for_column(df, column)
    lower_bound = result["lower_bound"]
    upper_bound = result["upper_bound"]

    if lower_bound is None or upper_bound is None:
        return df, 0

    original_series = df[column].copy()
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    capped_count = int((original_series != df[column]).sum())
    return df, capped_count


def cap_all_detected_outliers(df: pd.DataFrame):
    df = df.copy()
    candidate_cols = get_iqr_candidate_columns(df)
    capped_summary = []
    total_capped_values = 0

    for col in candidate_cols:
        df, capped_count = cap_outliers_iqr_for_column(df, col)
        capped_summary.append({"Column": col, "Capped Values": capped_count})
        total_capped_values += capped_count

    capped_summary_df = pd.DataFrame(capped_summary).sort_values(
        by="Capped Values", ascending=False
    ).reset_index(drop=True)

    return df, capped_summary_df, total_capped_values


def detect_isolation_forest_outliers(df: pd.DataFrame, contamination: float = 0.05):
    numeric_cols = get_iqr_candidate_columns(df)
    if not numeric_cols:
        return pd.DataFrame(), pd.Series(False, index=df.index), 0

    temp = df[numeric_cols].copy()
    temp = temp.fillna(temp.median(numeric_only=True))

    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(temp)
    mask = pd.Series(labels == -1, index=df.index)

    summary = pd.DataFrame(
        {
            "Method": ["Isolation Forest"],
            "Numeric Columns Used": [len(numeric_cols)],
            "Estimated Outlier Rows": [int(mask.sum())],
            "Contamination": [contamination],
        }
    )
    return summary, mask, int(mask.sum())


def get_feature_selection_summary(df: pd.DataFrame):
    if "income" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    temp = df.copy()
    temp = apply_manual_imputation(temp)
    temp["income_num"] = temp["income"].map({"<=50K": 0, ">50K": 1})

    numeric_cols = [
        c for c in temp.select_dtypes(include=[np.number]).columns.tolist()
        if c != "income_num"
    ]

    if not numeric_cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X_num = temp[numeric_cols]
    y = temp["income_num"]

    selector = VarianceThreshold()
    selector.fit(X_num)
    variance_df = pd.DataFrame(
        {"Feature": numeric_cols, "Variance": selector.variances_}
    ).sort_values("Variance", ascending=False).reset_index(drop=True)

    corr_rows = []
    for col in numeric_cols:
        corr_rows.append({"Feature": col, "Correlation with Target": X_num[col].corr(y)})
    corr_df = pd.DataFrame(corr_rows).sort_values(
        "Correlation with Target",
        key=lambda s: s.abs(),
        ascending=False
    ).reset_index(drop=True)

    mi_scores = mutual_info_classif(X_num, y, random_state=42)
    mi_df = pd.DataFrame(
        {"Feature": numeric_cols, "Mutual Information": mi_scores}
    ).sort_values("Mutual Information", ascending=False).reset_index(drop=True)

    return variance_df, corr_df, mi_df


@st.cache_data
def get_cv_scores():
    prepared = get_prepared_data()
    models, _, _ = get_training_outputs()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for model_name, pipeline in models.items():
        scores = cross_val_score(
            pipeline,
            prepared.X,
            prepared.y,
            cv=cv,
            scoring="f1",
            n_jobs=None,
        )
        rows.append(
            {
                "Model": model_name,
                "Mean CV F1": round(float(scores.mean()), 4),
                "Std CV F1": round(float(scores.std()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("Mean CV F1", ascending=False).reset_index(drop=True)


st.title("Income Classification Dashboard")
st.caption("Income prediction using demographic and work-related attributes")

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Dataset Preview",
        "EDA",
        "Preprocessing",
        "Model Training & Evaluation",
        "Single Prediction",
        "Batch Prediction Demo",
    ],
)

raw_df = get_raw_data()
prepared = get_prepared_data()
models, results, split_data = get_training_outputs()
_, X_test, _, y_test = split_data
best_model_name = results.iloc[0]["Model"]
best_model = models[best_model_name]
cm, report = get_confusion_and_report(best_model, X_test, y_test)

if page == "Project Overview":
    st.markdown("## Problem Statement")
    st.write(
        "This project predicts whether a person earns more than 50K per year using demographic and employment-related attributes such as age, workclass, occupation, education, marital status, and hours worked per week."
    )

    st.markdown("## Project Snapshot")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset</div>
            <div class="metric-value">Adult Income</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Models Used</div>
            <div class="metric-value">3</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Prediction Modes</div>
            <div class="metric-value">Single + Batch</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Workflow Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="workflow-card">
            <div class="workflow-title">1. Data Understanding</div>
            <ul class="workflow-list">
                <li>Load Adult Income dataset</li>
                <li>Understand demographic and work-related features</li>
                <li>Define target variable: income class</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="workflow-card">
            <div class="workflow-title">2. Data Preparation</div>
            <ul class="workflow-list">
                <li>Handle missing values</li>
                <li>Remove inconsistent entries</li>
                <li>Detect outliers and cap extremes</li>
                <li>Create engineered features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="workflow-card">
            <div class="workflow-title">3. Model Building</div>
            <ul class="workflow-list">
                <li>Train Logistic Regression</li>
                <li>Train KNN</li>
                <li>Train Decision Tree</li>
                <li>Compare performance metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="workflow-card">
            <div class="workflow-title">4. Prediction System</div>
            <ul class="workflow-list">
                <li>Evaluate models using confusion matrix</li>
                <li>Predict single record income</li>
                <li>Predict batch records</li>
                <li>Display results in dashboard</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## End-to-End Pipeline")
    st.markdown("""
    <div class="pipeline-box">
        Dataset Loading → Cleaning & Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction Dashboard
    </div>
    """, unsafe_allow_html=True)

elif page == "Dataset Preview":
    st.subheader("Raw Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", raw_df.shape[0])
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Missing Cells", int(raw_df.isna().sum().sum()))
    st.dataframe(raw_df.head(15), use_container_width=True)

    st.subheader("Missing Values by Column")
    missing = raw_df.isna().sum().sort_values(ascending=False)
    st.dataframe(missing[missing > 0].rename("missing_count"))

elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_income_distribution(prepared.df))
        st.pyplot(plot_workclass_distribution(raw_df))
    with col2:
        st.pyplot(plot_age_distribution(prepared.df))
        st.pyplot(plot_numeric_correlation(prepared.df))

    st.markdown("---")
    st.subheader("Feature Selection Summary")
    variance_df, corr_df, mi_df = get_feature_selection_summary(raw_df)

    if variance_df.empty:
        st.warning("Feature selection summary could not be generated.")
    else:
        t1, t2, t3 = st.tabs(["Variance", "Correlation", "Mutual Information"])
        with t1:
            st.dataframe(variance_df, use_container_width=True)
        with t2:
            st.dataframe(corr_df, use_container_width=True)
        with t3:
            st.dataframe(mi_df, use_container_width=True)

elif page == "Preprocessing":
    st.subheader("Interactive Data Preprocessing")
    st.caption(
        "Explore missing values, delete rows selectively, apply imputation, detect outliers, and track dataset shape after each step."
    )

    st.info(
        "Note: This interactive preprocessing page is for dataset analysis and demonstration. "
        "The current model training still uses the backend preprocessing pipeline defined in utils.py."
    )

    if "working_df" not in st.session_state:
        st.session_state.working_df = raw_df.copy()

    working_df = st.session_state.working_df.copy()

    st.markdown("### Current Dataset Status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", working_df.shape[0])
    c2.metric("Columns", working_df.shape[1])
    c3.metric("Missing Cells", int(working_df.isna().sum().sum()))

    st.markdown("---")

    st.markdown("### 1. Missing Value Analysis")
    missing_summary = get_missing_summary(working_df)

    if missing_summary.empty:
        st.success("No missing values found in the current dataset.")
    else:
        st.dataframe(missing_summary, use_container_width=True)

    st.markdown("---")

    st.markdown("### 2. Remove Rows with Missing Values")
    st.write("Choose the columns for which rows containing missing values should be removed.")
    removable_columns = missing_summary["Column"].tolist() if not missing_summary.empty else []

    selected_drop_cols = st.multiselect("Select columns for row deletion", removable_columns)

    if st.button("Apply Row Deletion"):
        if selected_drop_cols:
            before_shape = working_df.shape
            working_df, removed_rows = drop_rows_by_columns(working_df, selected_drop_cols)
            st.session_state.working_df = working_df
            st.success(
                f"Row deletion applied successfully. Removed rows: {removed_rows}. Shape changed from {before_shape} to {working_df.shape}."
            )
        else:
            st.warning("Please select at least one column.")

    st.markdown("---")

    st.markdown("### 3. Missing Value Imputation")
    st.write(
        "Numeric columns are filled using median. Categorical columns are filled using mode. This step can be used with or without row deletion."
    )

    if st.button("Apply Imputation"):
        before_missing = int(working_df.isna().sum().sum())
        before_shape = working_df.shape

        working_df = apply_manual_imputation(working_df)
        st.session_state.working_df = working_df

        after_missing = int(working_df.isna().sum().sum())
        st.success(
            f"Imputation completed. Missing cells changed from {before_missing} to {after_missing}. Current shape: {before_shape} to {working_df.shape}."
        )

    st.markdown("---")

    st.markdown("### 4. Outlier Detection")
    method = st.selectbox(
        "Choose outlier detection method",
        ["IQR Summary", "Isolation Forest Summary"],
    )

    if method == "IQR Summary":
        st.warning(
            "IQR-based outlier detection is applied across suitable numeric columns. "
            "Use capping to preserve dataset size."
        )
        outlier_summary_df, _, total_rows_with_any_outlier = get_outlier_summary(working_df)

        if outlier_summary_df.empty:
            st.success("No valid numeric columns available for outlier detection.")
        else:
            st.dataframe(outlier_summary_df, use_container_width=True)
            st.info(f"Rows having at least one detected outlier: {total_rows_with_any_outlier}")
            st.caption("Recommended: Use capping to preserve dataset size.")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Remove All Detected Outliers"):
                    before_shape = working_df.shape
                    working_df, _, removed_rows, _ = remove_all_detected_outliers(working_df)
                    st.session_state.working_df = working_df
                    st.success(
                        f"Outlier removal completed. Removed rows: {removed_rows}. Shape changed from {before_shape} to {working_df.shape}."
                    )

            with b2:
                if st.button("Cap All Detected Outliers"):
                    before_shape = working_df.shape
                    working_df, capped_summary_df, total_capped_values = cap_all_detected_outliers(working_df)
                    st.session_state.working_df = working_df
                    st.success(
                        f"Outlier capping completed. Total capped values: {total_capped_values}. Dataset shape remains {before_shape}."
                    )
                    st.write("Capping Summary")
                    st.dataframe(capped_summary_df, use_container_width=True)
    else:
        contamination = st.slider("Estimated outlier fraction", 0.01, 0.20, 0.05, 0.01)
        iso_summary, iso_mask, iso_count = detect_isolation_forest_outliers(working_df, contamination)
        if iso_summary.empty:
            st.success("No valid numeric columns available for Isolation Forest.")
        else:
            st.dataframe(iso_summary, use_container_width=True)
            st.info(f"Isolation Forest estimated outlier rows: {iso_count}")

            if st.button("Remove Isolation Forest Outliers"):
                before_shape = working_df.shape
                working_df = working_df.loc[~iso_mask].reset_index(drop=True)
                st.session_state.working_df = working_df
                st.success(
                    f"Isolation Forest removal completed. Removed rows: {before_shape[0] - working_df.shape[0]}. Shape changed from {before_shape} to {working_df.shape}."
                )

    st.markdown("---")
    st.markdown("### 5. Updated Dataset Preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", st.session_state.working_df.shape[0])
    c2.metric("Columns", st.session_state.working_df.shape[1])
    c3.metric("Missing Cells", int(st.session_state.working_df.isna().sum().sum()))

    st.dataframe(st.session_state.working_df.head(15), use_container_width=True)

    st.markdown("### 6. Feature Type Summary")
    current_df = st.session_state.working_df
    numeric_features = current_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in current_df.columns if c not in numeric_features]
    st.json({"numeric_features": numeric_features, "categorical_features": categorical_features})

    st.download_button(
        "Download cleaned dataset as CSV",
        data=st.session_state.working_df.to_csv(index=False),
        file_name="cleaned_income_dataset.csv",
        mime="text/csv",
    )

    if st.button("Reset to Original Raw Dataset"):
        st.session_state.working_df = raw_df.copy()
        st.success("Dataset has been reset to original raw data.")

elif page == "Model Training & Evaluation":
    st.subheader("Model Comparison")
    st.success(f"Current best model based on F1 Score: **{best_model_name}**")
    st.dataframe(results, use_container_width=True)

    st.markdown("### 5-Fold Cross Validation Summary")
    cv_df = get_cv_scores()
    st.dataframe(cv_df, use_container_width=True)

    model_name = st.selectbox("Select model to inspect", results["Model"].tolist())
    selected_model = models[model_name]
    cm_selected, report_selected = get_confusion_and_report(selected_model, X_test, y_test)

    st.pyplot(plot_confusion_matrix(cm_selected))
    st.text(report_selected)

elif page == "Single Prediction":
    st.subheader("Predict Income for One Person")
    with st.form("single_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=17, max_value=90, value=39)
            workclass = st.selectbox("Workclass", sorted(raw_df["workclass"].dropna().unique().tolist()))
            fnlwgt = st.number_input("Fnlwgt", min_value=10000, max_value=1500000, value=77516)
            education_num = st.slider("Education Number", min_value=1, max_value=16, value=13)
            marital_status = st.selectbox("Marital Status", sorted(raw_df["marital_status"].dropna().unique().tolist()))
        with c2:
            occupation = st.selectbox("Occupation", sorted(raw_df["occupation"].dropna().unique().tolist()))
            relationship = st.selectbox("Relationship", sorted(raw_df["relationship"].dropna().unique().tolist()))
            race = st.selectbox("Race", sorted(raw_df["race"].dropna().unique().tolist()))
            sex = st.selectbox("Sex", sorted(raw_df["sex"].dropna().unique().tolist()))
        with c3:
            capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
            capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=0)
            hours_per_week = st.slider("Hours Per Week", min_value=1, max_value=99, value=40)
            native_country = st.selectbox("Native Country", sorted(raw_df["native_country"].dropna().unique().tolist()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "workclass": workclass,
                    "fnlwgt": fnlwgt,
                    "education_num": education_num,
                    "marital_status": marital_status,
                    "occupation": occupation,
                    "relationship": relationship,
                    "race": race,
                    "sex": sex,
                    "capital_gain": capital_gain,
                    "capital_loss": capital_loss,
                    "hours_per_week": hours_per_week,
                    "native_country": native_country,
                    "capital_balance": capital_gain - capital_loss,
                    "has_capital_gain": int(capital_gain > 0),
                    "has_capital_loss": int(capital_loss > 0),
                    "is_long_workweek": int(hours_per_week > 40),
                }
            ]
        )
        prediction = best_model.predict(input_df)[0]
        probability = best_model.predict_proba(input_df)[0][1]
        st.success(f"Predicted income class: **{TARGET_LABELS[int(prediction)]}**")
        st.info(f"Probability of >50K: **{probability:.2%}**")
        st.dataframe(input_df, use_container_width=True)

elif page == "Batch Prediction Demo":
    st.subheader("Batch Prediction Demo")
    sample_df = create_sample_inputs()
    predicted = best_model.predict(sample_df)
    prob = best_model.predict_proba(sample_df)[:, 1]
    out = sample_df.copy()
    out["predicted_income"] = [TARGET_LABELS[int(x)] for x in predicted]
    out["probability_gt_50k"] = prob.round(4)
    st.dataframe(out, use_container_width=True)
    st.download_button(
        "Download sample input CSV",
        data=sample_df.to_csv(index=False),
        file_name="demo_input.csv",
        mime="text/csv",
    )
