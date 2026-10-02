import os
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def create_spark_session():
    """Create and return a SparkSession."""
    java_home_adoptium = r"C:\Program Files\Eclipse Adoptium\jre-25.0.1.8-hotspot"
    if os.path.exists(java_home_adoptium):
        os.environ["JAVA_HOME"] = java_home_adoptium
        os.environ["PATH"] = os.path.join(java_home_adoptium, "bin") + os.pathsep + os.environ.get("PATH", "")

    return SparkSession.builder \
        .appName("AirlineDelayClassificationTraining") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

def calculate_classification_metrics(predictions, label_col="high_delay", prediction_col="prediction"):
    """Calculates accuracy, precision, recall, f1, roc_auc, and confusion matrix."""
    acc_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="accuracy")
    prec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="weightedRecall")
    f1_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="f1")
    auc_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    accuracy = acc_eval.evaluate(predictions)
    precision = prec_eval.evaluate(predictions)
    recall = rec_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    try:
        roc_auc = auc_eval.evaluate(predictions)
    except Exception:
        roc_auc = 0.5

    # Calculate Confusion Matrix Counts
    tp = predictions.filter((col(label_col) == 1) & (col(prediction_col) == 1)).count()
    tn = predictions.filter((col(label_col) == 0) & (col(prediction_col) == 0)).count()
    fp = predictions.filter((col(label_col) == 0) & (col(prediction_col) == 1)).count()
    fn = predictions.filter((col(label_col) == 1) & (col(prediction_col) == 0)).count()

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }

def train_classification_models(spark, input_path=None, model_output_dir=None):
    """
    Trains and evaluates multiple classification models for high delay prediction.
    Exports reports and saves best classification model.
    """
    if input_path is None:
        if os.path.exists("data/features/airline_features.parquet"):
            input_path = "data/features/airline_features.parquet"
        else:
            input_path = "hdfs:///data/airline/features/airline_features.parquet"

    if model_output_dir is None:
        model_output_dir = "data/models/classification"

    print(f"Reading feature dataset from: {input_path}")
    df = spark.read.parquet(input_path)

    years = [row["year"] for row in df.select("year").distinct().collect() if row["year"] is not None]
    min_year, max_year = min(years), max(years)
    print(f"Data spans from year {min_year} to {max_year}")

    if max_year >= 2023 and min_year <= 2020:
        train_end_year = 2020
        val_end_year = 2022
    else:
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
    label_col = "high_delay"

    # Fill numerical feature nulls
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
        "Logistic Regression": LogisticRegression(featuresCol="features", labelCol=label_col, maxIter=30),
        "Random Forest Classifier": RandomForestClassifier(featuresCol="features", labelCol=label_col, numTrees=30, maxDepth=6, seed=42),
        "GBT Classifier": GBTClassifier(featuresCol="features", labelCol=label_col, maxIter=20, maxDepth=5, seed=42)
    }

    results = []
    trained_models = {}
    best_model = None
    best_model_name = None
    best_test_f1 = -1.0
    confusion_matrices = {}

    print("\n--- Training & Evaluating Classification Models ---")
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
        val_metrics = calculate_classification_metrics(val_preds, label_col)

        test_preds = model.transform(test_df)
        test_metrics = calculate_classification_metrics(test_preds, label_col)

        print(f"Validation F1: {val_metrics['F1']:.4f}, Accuracy: {val_metrics['Accuracy']:.4f}, ROC_AUC: {val_metrics['ROC_AUC']:.4f}")
        print(f"Test F1:       {test_metrics['F1']:.4f}, Accuracy: {test_metrics['Accuracy']:.4f}, ROC_AUC: {test_metrics['ROC_AUC']:.4f}")
        print(f"Test Confusion Matrix: TP={test_metrics['TP']}, TN={test_metrics['TN']}, FP={test_metrics['FP']}, FN={test_metrics['FN']}")

        confusion_matrices[name] = {
            "TP": test_metrics['TP'], "TN": test_metrics['TN'],
            "FP": test_metrics['FP'], "FN": test_metrics['FN']
        }

        results.append({
            "Model": name,
            "Validation_Accuracy": val_metrics["Accuracy"],
            "Validation_F1": val_metrics["F1"],
            "Validation_ROC_AUC": val_metrics["ROC_AUC"],
            "Test_Accuracy": test_metrics["Accuracy"],
            "Test_Precision": test_metrics["Precision"],
            "Test_Recall": test_metrics["Recall"],
            "Test_F1": test_metrics["F1"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"]
        })

        if test_metrics["F1"] > best_test_f1:
            best_test_f1 = test_metrics["F1"]
            best_model = model
            best_model_name = name

    # Save Classification Results
    os.makedirs("output/reports", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_csv_path = "output/reports/classification_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nClassification results saved to: {results_csv_path}")

    # Save Confusion Matrices JSON
    with open("output/reports/confusion_matrix.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    # Save Best Classification Model & Metadata
    best_model_path = os.path.join(model_output_dir, "best_classification_model")
    best_model.write().overwrite().save(best_model_path)
    metadata = {
        "best_model_name": best_model_name,
        "test_f1": round(best_test_f1, 4),
        "target": label_col,
        "train_end_year": train_end_year,
        "val_end_year": val_end_year,
        "max_year": max_year
    }
    with open(os.path.join(model_output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSelected Best Classification Model: {best_model_name} (Test F1: {best_test_f1:.4f})")
    print(f"Best model saved to: {best_model_path}")
    return results_df

if __name__ == "__main__":
    spark = create_spark_session()
    train_classification_models(spark)
    spark.stop()
