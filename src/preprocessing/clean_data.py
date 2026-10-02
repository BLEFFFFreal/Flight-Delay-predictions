import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count, when, isnan, lit, round, coalesce
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

def create_spark_session():
    """Create and return a SparkSession."""
    java_home_adoptium = r"C:\Program Files\Eclipse Adoptium\jre-25.0.1.8-hotspot"
    if os.path.exists(java_home_adoptium):
        os.environ["JAVA_HOME"] = java_home_adoptium
        os.environ["PATH"] = os.path.join(java_home_adoptium, "bin") + os.pathsep + os.environ.get("PATH", "")

    return SparkSession.builder \
        .appName("AirlineDelayDataCleaning") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

def clean_data(spark, input_path=None, output_path=None, report_output_path=None):
    """
    Performs data cleaning on initial Parquet data:
    - Missing value & NaN analysis
    - Logical key duplicate removal (year, month, carrier, airport)
    - Domain range & logical boundary validation:
        - month between 1 and 12
        - valid year range (1990..2030)
        - arr_cancelled <= arr_flights
        - arr_diverted <= arr_flights
        - arr_del15 <= arr_flights
        - non-negative delay counts and minutes
    - Generates and exports Data Quality Report to file.
    """
    if input_path is None:
        if os.path.exists("data/processed/airline_initial_load.parquet"):
            input_path = "data/processed/airline_initial_load.parquet"
        else:
            input_path = "hdfs:///data/airline/processed/airline_initial_load.parquet"

    if output_path is None:
        if input_path.startswith("hdfs://"):
            output_path = "hdfs:///data/airline/processed/airline_clean.parquet"
        else:
            output_path = "data/processed/airline_clean.parquet"

    if report_output_path is None:
        report_output_path = "output/reports/data_quality_report.json"

    print(f"Reading data for cleaning from: {input_path}")
    df = spark.read.parquet(input_path)

    initial_record_count = df.count()
    print(f"Initial record count: {initial_record_count}")

    # --- 1. Duplicate Detection ---
    key_columns = ["year", "month", "carrier", "airport"]
    df_deduplicated = df.dropDuplicates(subset=key_columns)
    duplicate_records = initial_record_count - df_deduplicated.count()
    print(f"Duplicate records found and removed: {duplicate_records}")

    df_current = df_deduplicated

    # --- 2. Missing Value & NaN Analysis ---
    print("\n--- Missing & NaN Value Analysis ---")
    missing_analysis = {}
    for c in df_current.columns:
        # Check both isNull and isnan for double/float columns
        if isinstance(df_current.schema[c].dataType, DoubleType):
            condition = col(c).isNull() | isnan(col(c))
        else:
            condition = col(c).isNull()
        
        null_nan_count = df_current.filter(condition).count()
        missing_analysis[c] = null_nan_count

    print(f"{'Column':<30} {'Null/NaN Count':<15} {'% Missing':<15}")
    print("-" * 60)
    for col_name, missing_cnt in missing_analysis.items():
        pct = (missing_cnt / df_current.count()) * 100 if df_current.count() > 0 else 0.0
        print(f"{col_name:<30} {missing_cnt:<15} {pct:<14.2f}%")

    # Strategy for missing values:
    # Fill numerical delay/flight metrics with 0.0 if null or NaN
    numeric_cols_to_fill_zero = [
        "arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct", 
        "security_ct", "late_aircraft_ct", "arr_cancelled", "arr_diverted",
        "arr_delay", "carrier_delay", "weather_delay", "nas_delay", 
        "security_delay", "late_aircraft_delay"
    ]
    for c in numeric_cols_to_fill_zero:
        df_current = df_current.withColumn(
            c,
            when(col(c).isNull() | isnan(col(c)), 0.0).otherwise(col(c))
        )

    # Drop rows with null/NaN critical key identifiers
    rows_before_critical_drop = df_current.count()
    critical_id_condition = (
        col("year").isNull() | isnan(col("year")) |
        col("month").isNull() | isnan(col("month")) |
        col("carrier").isNull() |
        col("airport").isNull()
    )
    df_current = df_current.filter(~critical_id_condition)
    dropped_critical_null_rows = rows_before_critical_drop - df_current.count()
    print(f"\nDropped {dropped_critical_null_rows} rows due to critical key nulls.")

    # Explicit data type casting
    df_current = df_current.withColumn("year", col("year").cast(IntegerType())) \
                           .withColumn("month", col("month").cast(IntegerType()))

    # --- 3. Domain Range & Logical Validation ---
    invalid_conditions = [
        col("year") < 1990,
        col("year") > 2030,
        col("month") < 1,
        col("month") > 12,
        col("arr_flights") < 0,
        col("arr_del15") < 0,
        (col("arr_flights") > 0) & (col("arr_del15") > col("arr_flights")),
        col("arr_cancelled") < 0,
        (col("arr_flights") > 0) & (col("arr_cancelled") > col("arr_flights")),
        col("arr_diverted") < 0,
        (col("arr_flights") > 0) & (col("arr_diverted") > col("arr_flights")),
        col("carrier_ct") < 0,
        col("weather_ct") < 0,
        col("nas_ct") < 0,
        col("security_ct") < 0,
        col("late_aircraft_ct") < 0,
        col("arr_delay") < 0,
        col("carrier_delay") < 0,
        col("weather_delay") < 0,
        col("nas_delay") < 0,
        col("security_delay") < 0,
        col("late_aircraft_delay") < 0
    ]

    combined_invalid_condition = invalid_conditions[0]
    for cond in invalid_conditions[1:]:
        combined_invalid_condition = combined_invalid_condition | cond

    invalid_records_count = df_current.filter(combined_invalid_condition).count()
    print(f"\nInvalid records found (range/logic validation): {invalid_records_count}")

    df_cleaned = df_current.filter(~combined_invalid_condition)
    clean_record_count = df_cleaned.count()

    # --- 4. Data Quality Report & Output ---
    quality_report = {
        "input_records": initial_record_count,
        "duplicate_records_removed": duplicate_records,
        "critical_null_rows_dropped": dropped_critical_null_rows,
        "invalid_records_removed": invalid_records_count,
        "clean_records": clean_record_count,
        "missing_analysis": missing_analysis
    }

    print("\n--- DATA QUALITY REPORT SUMMARY ---")
    print(json.dumps(quality_report, indent=2))

    # Ensure output report directory exists if local path
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w") as f:
        json.dump(quality_report, f, indent=2)
    print(f"Data Quality Report saved to: {report_output_path}")

    print(f"Writing cleaned Parquet data to: {output_path}")
    df_cleaned.write.mode("overwrite").parquet(output_path)
    print("Data cleaning complete.")
    return df_cleaned

if __name__ == "__main__":
    spark = create_spark_session()
    clean_data(spark)
    spark.stop()
