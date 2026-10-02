import os
import math
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, when, lit, round as spark_round, avg, lag, lead,
    sin, cos, radians, coalesce
)
from pyspark.sql.types import DoubleType, IntegerType

def create_spark_session():
    """Create and return a SparkSession with window optimization."""
    java_home_adoptium = r"C:\Program Files\Eclipse Adoptium\jre-25.0.1.8-hotspot"
    if os.path.exists(java_home_adoptium):
        os.environ["JAVA_HOME"] = java_home_adoptium
        os.environ["PATH"] = os.path.join(java_home_adoptium, "bin") + os.pathsep + os.environ.get("PATH", "")

    return SparkSession.builder \
        .appName("AirlineDelayFeatureEngineering") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

def create_features(spark, input_path=None, output_path=None):
    """
    Generates historical lag features, rolling features, seasonal features,
    and target variables while strictly preventing target leakage.
    Ensures calendar-correct next/lag month matching using absolute month indices.
    """
    if input_path is None:
        if os.path.exists("data/processed/airline_clean.parquet"):
            input_path = "data/processed/airline_clean.parquet"
        else:
            input_path = "hdfs:///data/airline/processed/airline_clean.parquet"

    if output_path is None:
        if input_path.startswith("hdfs://"):
            output_path = "hdfs:///data/airline/features/airline_features.parquet"
        else:
            output_path = "data/features/airline_features.parquet"

    print(f"Reading cleaned data from: {input_path}")
    df = spark.read.parquet(input_path)
    initial_count = df.count()
    print(f"Input records count: {initial_count}")

    # --- 1. Calculate Same-Month Base Ratios & Percentages ---
    df_features = df.withColumn(
        "delay_rate",
        when(col("arr_flights") > 0, col("arr_del15") / col("arr_flights")).otherwise(0.0)
    ).withColumn(
        "cancellation_rate",
        when(col("arr_flights") > 0, col("arr_cancelled") / col("arr_flights")).otherwise(0.0)
    ).withColumn(
        "diversion_rate",
        when(col("arr_flights") > 0, col("arr_diverted") / col("arr_flights")).otherwise(0.0)
    ).withColumn(
        "avg_delay_per_flight",
        when(col("arr_flights") > 0, col("arr_delay") / col("arr_flights")).otherwise(0.0)
    )

    # Delay cause percentages
    total_cause_delay = (
        col("carrier_delay") + col("weather_delay") + col("nas_delay") + 
        col("security_delay") + col("late_aircraft_delay")
    )
    df_features = df_features.withColumn(
        "carrier_delay_percentage",
        when(total_cause_delay > 0, col("carrier_delay") / total_cause_delay).otherwise(0.0)
    ).withColumn(
        "weather_delay_percentage",
        when(total_cause_delay > 0, col("weather_delay") / total_cause_delay).otherwise(0.0)
    ).withColumn(
        "nas_delay_percentage",
        when(total_cause_delay > 0, col("nas_delay") / total_cause_delay).otherwise(0.0)
    ).withColumn(
        "security_delay_percentage",
        when(total_cause_delay > 0, col("security_delay") / total_cause_delay).otherwise(0.0)
    ).withColumn(
        "late_aircraft_delay_percentage",
        when(total_cause_delay > 0, col("late_aircraft_delay") / total_cause_delay).otherwise(0.0)
    )

    # --- 2. Calendar Month Index & Window Definitions ---
    # Construct an absolute month index: month_index = year * 12 + month
    df_features = df_features.withColumn("month_index", col("year") * 12 + col("month"))

    window_spec = Window.partitionBy("carrier", "airport").orderBy("month_index")

    # --- 3. Calendar-Strict Lag Features ---
    # Compute differences in month_index to ensure true calendar adjacency
    df_features = df_features.withColumn("m_lag1", lag("month_index", 1).over(window_spec)) \
                             .withColumn("m_lag2", lag("month_index", 2).over(window_spec)) \
                             .withColumn("m_lag3", lag("month_index", 3).over(window_spec)) \
                             .withColumn("m_lag12", lag("month_index", 12).over(window_spec))

    df_features = df_features.withColumn(
        "lag_1_delay_rate",
        when(col("month_index") - col("m_lag1") == 1, lag("delay_rate", 1).over(window_spec))
    ).withColumn(
        "lag_2_delay_rate",
        when(col("month_index") - col("m_lag2") == 2, lag("delay_rate", 2).over(window_spec))
    ).withColumn(
        "lag_3_delay_rate",
        when(col("month_index") - col("m_lag3") == 3, lag("delay_rate", 3).over(window_spec))
    ).withColumn(
        "lag_12_delay_rate",
        when(col("month_index") - col("m_lag12") == 12, lag("delay_rate", 12).over(window_spec))
    )

    # Historical previous month metrics
    df_features = df_features.withColumn(
        "previous_month_arr_flights",
        when(col("month_index") - col("m_lag1") == 1, lag("arr_flights", 1).over(window_spec))
    ).withColumn(
        "previous_month_cancel_rate",
        when(col("month_index") - col("m_lag1") == 1, lag("cancellation_rate", 1).over(window_spec))
    ).withColumn(
        "previous_month_diversion_rate",
        when(col("month_index") - col("m_lag1") == 1, lag("diversion_rate", 1).over(window_spec))
    ).withColumn(
        "previous_month_avg_delay",
        when(col("month_index") - col("m_lag1") == 1, lag("avg_delay_per_flight", 1).over(window_spec))
    )

    # Clean intermediate lag helper columns
    df_features = df_features.drop("m_lag1", "m_lag2", "m_lag3", "m_lag12")

    # --- 4. Rolling Features ---
    # Window of preceding rows (1 to 3, 1 to 6, 1 to 12)
    window_3_rolling = Window.partitionBy("carrier", "airport").orderBy("month_index").rowsBetween(-3, -1)
    window_6_rolling = Window.partitionBy("carrier", "airport").orderBy("month_index").rowsBetween(-6, -1)
    window_12_rolling = Window.partitionBy("carrier", "airport").orderBy("month_index").rowsBetween(-12, -1)

    df_features = df_features.withColumn("rolling_3_month_delay_rate", spark_round(avg(col("delay_rate")).over(window_3_rolling), 4)) \
                             .withColumn("rolling_6_month_delay_rate", spark_round(avg(col("delay_rate")).over(window_6_rolling), 4)) \
                             .withColumn("rolling_12_month_delay_rate", spark_round(avg(col("delay_rate")).over(window_12_rolling), 4))

    # Fill nulls from rolling or lag features at start of series with reasonable defaults (0.0 or lag_1)
    for c in ["rolling_3_month_delay_rate", "rolling_6_month_delay_rate", "rolling_12_month_delay_rate"]:
        df_features = df_features.withColumn(c, coalesce(col(c), col("lag_1_delay_rate"), lit(0.0)))

    for c in ["previous_month_arr_flights", "previous_month_cancel_rate", "previous_month_diversion_rate", "previous_month_avg_delay"]:
        df_features = df_features.withColumn(c, coalesce(col(c), lit(0.0)))

    # --- 5. Seasonal Features ---
    pi_val = math.pi
    df_features = df_features.withColumn(
        "month_sin", spark_round(sin(col("month") * (2 * pi_val / 12.0)), 4)
    ).withColumn(
        "month_cos", spark_round(cos(col("month") * (2 * pi_val / 12.0)), 4)
    )

    # --- 6. Target Variables ---
    # Target is next calendar month's delay_rate
    df_features = df_features.withColumn("m_lead1", lead("month_index", 1).over(window_spec))
    df_features = df_features.withColumn(
        "next_month_delay_rate",
        when(col("m_lead1") - col("month_index") == 1, lead("delay_rate", 1).over(window_spec))
    ).drop("m_lead1")

    HIGH_DELAY_THRESHOLD = 0.20
    df_features = df_features.withColumn(
        "high_delay",
        when(col("next_month_delay_rate") >= HIGH_DELAY_THRESHOLD, 1).otherwise(0)
    )

    # Filter out rows where the target is null (end of partition or missing next calendar month)
    df_final_features = df_features.dropna(subset=["next_month_delay_rate", "high_delay"])
    final_count = df_final_features.count()
    dropped_target_nulls = initial_count - final_count

    print(f"Dropped {dropped_target_nulls} rows where target variable (next_month_delay_rate) was null.")
    print(f"Final records count with valid features & targets: {final_count}")

    # Ensure output parent directory exists for local paths
    if not output_path.startswith("hdfs://"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing feature-engineered dataset to: {output_path}")
    df_final_features.write.mode("overwrite").parquet(output_path)
    print("Feature engineering complete.")
    return df_final_features

if __name__ == "__main__":
    spark = create_spark_session()
    create_features(spark)
    spark.stop()
