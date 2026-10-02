# Product Requirements Document (PRD)

## Project Title
Big Data Analytics and Machine Learning Based Airline Flight Delay Rate Prediction Using Hadoop HDFS and Apache PySpark

---

## 1. Executive Summary
Airline flight delays cause inconvenience to passengers and increase operational costs for airlines and airports. This project builds a Big Data Analytics and Machine Learning system for analyzing historical airline delay patterns and predicting future airline-airport delay rates.

The system uses the `Airline_Delay_Cause.csv` dataset containing approximately 171,666 records representing monthly airline and airport performance.

Technologies:
- **Apache Hadoop HDFS**: Distributed data lake storage.
- **Apache Spark / PySpark**: Distributed data processing, cleaning, feature engineering, and ML.
- **Spark SQL**: Analytical aggregation queries.
- **Spark MLlib**: Regression and classification machine learning models.
- **Streamlit**: Interactive visualization dashboard.

---

## 2. Functional Requirements
- **FR-01 (HDFS Ingestion)**: Store raw CSV in HDFS data lake layer.
- **FR-02 (PySpark Processing)**: Read HDFS dataset using explicit schema in PySpark.
- **FR-03 (Data Cleaning)**: Perform duplicate removal, missing value & NaN handling, domain validation, and quality reporting.
- **FR-04 (Distributed Analytics)**: Answer key analytical questions using Spark SQL.
- **FR-05 (Feature Engineering)**: Generate calendar-strict lag features (`lag_1_delay_rate`, `lag_2_delay_rate`, `lag_3_delay_rate`, `lag_12_delay_rate`), rolling features, seasonal features (`month_sin`, `month_cos`), and target variables (`next_month_delay_rate`, `high_delay`).
- **FR-06 (Target Leakage Prevention)**: Exclude same-month delay cause metrics from model predictive inputs.
- **FR-07 (Regression)**: Train Linear Regression, Random Forest Regressor, GBT Regressor to predict `next_month_delay_rate`.
- **FR-08 (Classification)**: Train Logistic Regression, Random Forest Classifier, GBT Classifier to predict `high_delay` (threshold >= 0.20).
- **FR-09 (Model Evaluation)**: Evaluate models using RMSE, MAE, R², Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrices.
- **FR-10 (Model Persistence & Prediction)**: Save trained best models and generate future-period route predictions.
- **FR-11 (Visualization Dashboard)**: Display analytical insights, route predictions, and model performance metrics via Streamlit.
