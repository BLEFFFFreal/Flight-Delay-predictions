import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import col, count, sum as spark_sum, when

def create_spark_session():
    """Create and return a SparkSession with optimized configuration."""
    java_home_adoptium = r"C:\Program Files\Eclipse Adoptium\jre-25.0.1.8-hotspot"
    if os.path.exists(java_home_adoptium):
        os.environ["JAVA_HOME"] = java_home_adoptium
        os.environ["PATH"] = os.path.join(java_home_adoptium, "bin") + os.pathsep + os.environ.get("PATH", "")

    return SparkSession.builder \
        .appName("AirlineDelayIngestion") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

def resolve_path(primary_path, fallback_path):
    """
    Returns primary_path if HDFS or explicit path exists/accessible,
    otherwise returns fallback local path.
    """
    if primary_path.startswith("hdfs://"):
        # If running in environment without hdfs daemon, we check local fallback
        return primary_path
    return primary_path if os.path.exists(primary_path) else fallback_path

def load_data_to_hdfs(spark, input_path=None, output_path=None):
    """
    Reads the raw CSV, applies explicit schema, performs ingestion validation,
    and writes Parquet storage format.
    """
    schema = StructType([
        StructField("year", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("carrier", StringType(), True),
        StructField("carrier_name", StringType(), True),
        StructField("airport", StringType(), True),
        StructField("airport_name", StringType(), True),
        StructField("arr_flights", DoubleType(), True),
        StructField("arr_del15", DoubleType(), True),
        StructField("carrier_ct", DoubleType(), True),
        StructField("weather_ct", DoubleType(), True),
        StructField("nas_ct", DoubleType(), True),
        StructField("security_ct", DoubleType(), True),
        StructField("late_aircraft_ct", DoubleType(), True),
        StructField("arr_cancelled", DoubleType(), True),
        StructField("arr_diverted", DoubleType(), True),
        StructField("arr_delay", DoubleType(), True),
        StructField("carrier_delay", DoubleType(), True),
        StructField("weather_delay", DoubleType(), True),
        StructField("nas_delay", DoubleType(), True),
        StructField("security_delay", DoubleType(), True),
        StructField("late_aircraft_delay", DoubleType(), True),
    ])

    if input_path is None:
        # Check standard HDFS path vs local dataset path
        if os.path.exists("data/Airline_Delay_Cause.csv"):
            input_path = "data/Airline_Delay_Cause.csv"
        else:
            input_path = "hdfs:///data/airline/raw/Airline_Delay_Cause.csv"

    if output_path is None:
        if input_path.startswith("hdfs://"):
            output_path = "hdfs:///data/airline/processed/airline_initial_load.parquet"
        else:
            output_path = "data/processed/airline_initial_load.parquet"

    print(f"Reading CSV from: {input_path}")
    df = spark.read \
        .option("header", "true") \
        .schema(schema) \
        .csv(input_path)

    record_count = df.count()
    print(f"Ingested Raw Records Count: {record_count}")

    # Schema Validation and Ingestion Quality Reporting
    print("\n--- Ingestion Quality Check ---")
    null_year_count = df.filter(col("year").isNull()).count()
    null_carrier_count = df.filter(col("carrier").isNull()).count()
    null_airport_count = df.filter(col("airport").isNull()).count()

    print(f"Null Years: {null_year_count}")
    print(f"Null Carriers: {null_carrier_count}")
    print(f"Null Airports: {null_airport_count}")

    if null_year_count > 0 or null_carrier_count > 0 or null_airport_count > 0:
        print("Warning: Essential key fields contain null values during ingestion.")

    print(f"\nWriting Parquet to: {output_path}")
    df.write.mode("overwrite").parquet(output_path)
    print("Ingestion pipeline completed successfully.")
    return df

if __name__ == "__main__":
    spark = create_spark_session()
    load_data_to_hdfs(spark)
    spark.stop()
