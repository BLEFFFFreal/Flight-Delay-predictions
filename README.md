# Big Data Analytics & ML-Based Airline Flight Delay Rate Prediction

This project implements an end-to-end Big Data analytics and machine learning pipeline to analyze historical airline operational performance and predict future airline-airport delay rates using **Apache Hadoop HDFS**, **Apache PySpark**, **Spark SQL**, **Spark MLlib**, and **Streamlit**.

## Pipeline Execution Order

```bash
python src/ingestion/load_to_hdfs.py
python src/preprocessing/clean_data.py
python src/analytics/delay_analysis.py
python src/features/create_features.py
python src/models/train_regression.py
python src/models/train_classification.py
python src/models/evaluate_models.py
python src/prediction/predict.py
python -m streamlit run dashboard/app.py
```

## Main Outputs

- `data/processed/airline_clean.parquet`: cleaned dataset
- `data/features/airline_features.parquet`: engineered features
- `data/models/`: persisted Spark ML models
- `output/analytics/`: analytical CSV reports
- `output/reports/`: model metrics and data quality reports
- `output/predictions/`: prediction results
- `output/charts/`: evaluation charts

## Tests

```bash
python -m unittest tests/test_pipeline.py
```

## Local Dashboard

```bash
python -m streamlit run dashboard/app.py
```
