import os
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def create_spark_session():
    """Create and return a SparkSession."""
    java_home_adoptium = r"C:\Program Files\Eclipse Adoptium\jre-25.0.1.8-hotspot"
    if os.path.exists(java_home_adoptium):
        os.environ["JAVA_HOME"] = java_home_adoptium
        os.environ["PATH"] = os.path.join(java_home_adoptium, "bin") + os.pathsep + os.environ.get("PATH", "")

    return SparkSession.builder \
        .appName("AirlineDelayRegressionTraining") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

def train_regression_models(spark, input_path=None, model_output_dir=None):
    """
    Trains and evaluates multiple regression models on chronological splits.
    Outputs evaluation CSV reports and saves the best model.
    """
    if input_path is None:
        if os.path.exists("data/features/airline_features.parquet"):
            input_path = "data/features/airline_features.parquet"
        else:
            input_path = "hdfs:///data/airline/features/airline_features.parquet"

    if model_output_dir is None:
        model_output_dir = "data/models/regression"

    print(f"Reading feature-engineered dataset from: {input_path}")
    df = spark.read.parquet(input_path)

    # Determine chronological splits based on available dataset years
    years = [row["year"] for row in df.select("year").distinct().collect() if row["year"] is not None]
    min_year, max_year = min(years), max(years)
    print(f"Data spans from year {min_year} to {max_year}")

    # Set temporal split bounds
    if max_year >= 2023 and min_year <= 2020:
        train_end_year = 2020
        val_end_year = 2022
    else:
        # Dynamic split for arbitrary year ranges
        span = max_year - min_year + 1
        if span >= 3:
            train_end_year = max_year - 2
            val_end_year = max_year - 1
        else:
            train_end_year = max_year - 1
            val_end_year = max_year - 1

    train_df = df.filter(col("year") <= train_end_year)
    val_df = df.filter((col("year") > train_end_year) & (col("year") <= val_end_year))
    test_df = df.filter(col("year") > val_end_year)

    if test_df.count() == 0:
        # Fallback split if test_df is empty due to small dataset
        test_df = val_df

    print(f"Train set records ({min_year}-{train_end_year}): {train_df.count()}")
    print(f"Validation set records ({train_end_year + 1}-{val_end_year}): {val_df.count()}")
    print(f"Test set records ({val_end_year + 1}-{max_year}): {test_df.count()}")

    categorical_features = ["carrier", "airport"]
    numerical_features = [
        "month", "lag_1_delay_rate", "lag_2_delay_rate", "lag_3_delay_rate", "lag_12_delay_rate",
        "previous_month_arr_flights", "previous_month_cancel_rate", "previous_month_diversion_rate",
        "previous_month_avg_delay", "rolling_3_month_delay_rate", "rolling_6_month_delay_rate",
        "rolling_12_month_delay_rate", "month_sin", "month_cos"
    ]
    label_col = "next_month_delay_rate"

    # Fill remaining feature nulls with 0.0
    for num_col in numerical_features:
        train_df = train_df.withColumn(num_col, coalesce(col(num_col), lit(0.0)))
        val_df = val_df.withColumn(num_col, coalesce(col(num_col), lit(0.0)))
        test_df = test_df.withColumn(num_col, coalesce(col(num_col), lit(0.0)))

    stages = []
    for c in categorical_features:
        indexer = StringIndexer(inputCol=c, outputCol=c + "_indexed", handleInvalid="keep")
        encoder = OneHotEncoder(inputCol=c + "_indexed", outputCol=c + "_encoded")
        stages += [indexer, encoder]

    encoded_features = [c + "_encoded" for c in categorical_features]
    assembler_inputs = numerical_features + encoded_features
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_assembled", handleInvalid="skip")
    scaler = StandardScaler(inputCol="features_assembled", outputCol="features", withStd=True, withMean=False)
    stages += [assembler, scaler]

    models_to_train = {
        "Linear Regression": LinearRegression(featuresCol="features", labelCol=label_col, maxIter=20, regParam=0.01),
        "Random Forest Regressor": RandomForestRegressor(featuresCol="features", labelCol=label_col, numTrees=30, maxDepth=6, seed=42),
        "GBT Regressor": GBTRegressor(featuresCol="features", labelCol=label_col, maxIter=20, maxDepth=5, seed=42)
    }

    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

    results = []
    trained_models = {}
    best_model = None
    best_model_name = None
    best_test_rmse = float("inf")

    print("\n--- Training & Evaluating Regression Models ---")
    for name, estimator in models_to_train.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(stages=stages + [estimator])
        model = pipeline.fit(train_df)
        trained_models[name] = model

        # Save individual model
        model_save_path = os.path.join(model_output_dir, name.lower().replace(" ", "_"))
        model.write().overwrite().save(model_save_path)
        print(f"Saved {name} model to: {model_save_path}")

        val_preds = model.transform(val_df)
        val_rmse = evaluator_rmse.evaluate(val_preds)
        val_mae = evaluator_mae.evaluate(val_preds)
        val_r2 = evaluator_r2.evaluate(val_preds)

        test_preds = model.transform(test_df)
        test_rmse = evaluator_rmse.evaluate(test_preds)
        test_mae = evaluator_mae.evaluate(test_preds)
        test_r2 = evaluator_r2.evaluate(test_preds)

        print(f"Validation Metrics -> RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        print(f"Test Metrics       -> RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

        results.append({
            "Model": name,
            "Validation_RMSE": round(val_rmse, 4),
            "Validation_MAE": round(val_mae, 4),
            "Validation_R2": round(val_r2, 4),
            "Test_RMSE": round(test_rmse, 4),
            "Test_MAE": round(test_mae, 4),
            "Test_R2": round(test_r2, 4)
        })

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_model = model
            best_model_name = name

    # Save Regression Metrics Report
    os.makedirs("output/reports", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_csv_path = "output/reports/regression_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nRegression results saved to: {results_csv_path}")

    # Extract Feature Importances for Tree Models (e.g., Random Forest or GBT)
    rf_model = trained_models.get("Random Forest Regressor")
    if rf_model is not None:
        tree_stage = rf_model.stages[-1]
        importances = tree_stage.featureImportances.toArray()
        # Build feature names list matching assembled vector
        feature_names = list(numerical_features) + [f"cat_{i}" for i in range(len(importances) - len(numerical_features))]
        feat_df = pd.DataFrame({
            "Feature": feature_names[:len(importances)],
            "Importance": [round(float(imp), 6) for imp in importances]
        }).sort_values("Importance", ascending=False)

        feat_imp_path = "output/reports/feature_importance.csv"
        feat_df.to_csv(feat_imp_path, index=False)
        print(f"Feature importances saved to: {feat_imp_path}")

    # Save Best Regression Model & Metadata
    best_model_path = os.path.join(model_output_dir, "best_regression_model")
    best_model.write().overwrite().save(best_model_path)
    metadata = {
        "best_model_name": best_model_name,
        "test_rmse": round(best_test_rmse, 4),
        "target": label_col,
        "train_end_year": train_end_year,
        "val_end_year": val_end_year,
        "max_year": max_year
    }
    with open(os.path.join(model_output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSelected Best Regression Model: {best_model_name} (Test RMSE: {best_test_rmse:.4f})")
    print(f"Best model saved to: {best_model_path}")
    return results_df

if __name__ == "__main__":
    spark = create_spark_session()
    train_regression_models(spark)
    spark.stop()
