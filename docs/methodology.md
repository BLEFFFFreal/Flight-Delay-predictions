# Machine Learning Methodology

## 1. Feature Engineering & Lag Calculation
To predict future delay rates accurately without data leakage:
- **Absolute Calendar Indexing**: Each row receives `month_index = year * 12 + month`.
- **Lag Features**: Features `lag_1_delay_rate`, `lag_2_delay_rate`, `lag_3_delay_rate`, and `lag_12_delay_rate` verify `month_index - lag_month_index == lag_amount`.
- **Rolling Averages**: Preceding window averages over 3, 6, and 12 months (`rolling_3_month_delay_rate`, etc.).
- **Seasonal Features**: Trigonometric encoding `month_sin = sin(2π * month / 12)` and `month_cos = cos(2π * month / 12)`.

## 2. Target Leakage Prevention
Same-month delay cause metrics (`carrier_delay_percentage`, `weather_delay_percentage`, `arr_delay`, etc.) are omitted from predictive feature vectors for next month prediction.

## 3. Chronological Temporal Splitting
To evaluate model performance realistically on future data:
- **Training Set**: Historical observations up to end year (e.g. 2013–2020).
- **Validation Set**: Mid-period observations (e.g. 2021–2022).
- **Test Set**: Final period observations (e.g. 2023).

## 4. Model Selection & Metrics
- **Regression Evaluation**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Coefficient of Determination (R²).
- **Classification Evaluation**: Accuracy, Weighted Precision, Weighted Recall, F1-Score, Area Under ROC (ROC-AUC), and Confusion Matrix.
