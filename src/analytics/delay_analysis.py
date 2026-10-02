import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round, sum as spark_sum, avg, desc

def create_spark_session():
    """Create and return a SparkSession."""
    java_home_adoptium = r"C:\Program Files\Eclipse Adoptium\jre-25.0.1.8-hotspot"
    if os.path.exists(java_home_adoptium):
        os.environ["JAVA_HOME"] = java_home_adoptium
        os.environ["PATH"] = os.path.join(java_home_adoptium, "bin") + os.pathsep + os.environ.get("PATH", "")

    return SparkSession.builder \
        .appName("AirlineDelayAnalytics") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

def save_dataframe(df, name, hdfs_base="hdfs:///data/airline/analytics/", local_base="data/analytics/"):
    """Saves DataFrame as CSV and Parquet to output storage."""
    # Write to local CSV for easy dashboard inspection and HDFS Parquet if applicable
    local_csv_path = os.path.join("output/analytics", f"{name}.csv")
    os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)
    df.toPandas().to_csv(local_csv_path, index=False)
    print(f"Exported analytics CSV to: {local_csv_path}")

    local_parquet_path = os.path.join(local_base, f"{name}.parquet")
    df.write.mode("overwrite").parquet(local_parquet_path)
    print(f"Exported analytics Parquet to: {local_parquet_path}")

def run_analytics(spark, input_path=None):
    """
    Performs exploratory data analysis using Spark SQL and exports analytical views.
    """
    if input_path is None:
        if os.path.exists("data/processed/airline_clean.parquet"):
            input_path = "data/processed/airline_clean.parquet"
        else:
            input_path = "hdfs:///data/airline/processed/airline_clean.parquet"

    print(f"Reading cleaned data for analytics from: {input_path}")
    df_cleaned = spark.read.parquet(input_path)
    df_cleaned.createOrReplaceTempView("airline_data")

    # Q1: Airline Rankings by Delay Rate
    print("\n--- Q1: Airline Rankings by Delay Rate ---")
    airline_rankings = spark.sql("""
        SELECT
            carrier,
            carrier_name,
            SUM(arr_flights) AS total_flights,
            SUM(arr_del15) AS total_delayed_flights,
            ROUND(SUM(arr_del15) / SUM(arr_flights), 4) AS delay_rate,
            ROUND(SUM(arr_cancelled) / SUM(arr_flights), 4) AS cancellation_rate
        FROM airline_data
        WHERE arr_flights > 0
        GROUP BY carrier, carrier_name
        ORDER BY delay_rate DESC
    """)
    airline_rankings.show(10, truncate=False)
    save_dataframe(airline_rankings, "airline_rankings")

    # Q2: Airport Rankings by Delay Rate
    print("\n--- Q2: Airport Rankings by Delay Rate ---")
    airport_rankings = spark.sql("""
        SELECT
            airport,
            airport_name,
            SUM(arr_flights) AS total_flights,
            SUM(arr_del15) AS total_delayed_flights,
            ROUND(SUM(arr_del15) / SUM(arr_flights), 4) AS delay_rate
        FROM airline_data
        WHERE arr_flights > 0
        GROUP BY airport, airport_name
        ORDER BY delay_rate DESC
    """)
    airport_rankings.show(10, truncate=False)
    save_dataframe(airport_rankings, "airport_rankings")

    # Q3: Delay Cause Breakdown
    print("\n--- Q3: Dominant Delay Causes ---")
    delay_causes = spark.sql("""
        SELECT
            'Carrier Delay' AS delay_cause,
            SUM(carrier_ct) AS total_delay_count,
            SUM(carrier_delay) / 60 AS total_delay_hours
        FROM airline_data
        UNION ALL
        SELECT
            'Weather Delay' AS delay_cause,
            SUM(weather_ct) AS total_delay_count,
            SUM(weather_delay) / 60 AS total_delay_hours
        FROM airline_data
        UNION ALL
        SELECT
            'NAS Delay' AS delay_cause,
            SUM(nas_ct) AS total_delay_count,
            SUM(nas_delay) / 60 AS total_delay_hours
        FROM airline_data
        UNION ALL
        SELECT
            'Security Delay' AS delay_cause,
            SUM(security_ct) AS total_delay_count,
            SUM(security_delay) / 60 AS total_delay_hours
        FROM airline_data
        UNION ALL
        SELECT
            'Late Arrival Delay' AS delay_cause,
            SUM(late_aircraft_ct) AS total_delay_count,
            SUM(late_aircraft_delay) / 60 AS total_delay_hours
        FROM airline_data
        ORDER BY total_delay_hours DESC
    """)
    delay_causes.show(truncate=False)
    save_dataframe(delay_causes, "delay_causes")

    # Q4: Monthly Delay Trends
    print("\n--- Q4: Monthly Trends ---")
    monthly_trends = spark.sql("""
        SELECT
            month,
            SUM(arr_flights) AS total_flights,
            ROUND(SUM(arr_del15) / SUM(arr_flights), 4) AS delay_rate,
            ROUND(SUM(arr_cancelled) / SUM(arr_flights), 4) AS cancellation_rate
        FROM airline_data
        WHERE arr_flights > 0
        GROUP BY month
        ORDER BY month ASC
    """)
    monthly_trends.show(12, truncate=False)
    save_dataframe(monthly_trends, "monthly_trends")

    # Q5: Annual Delay Trends
    print("\n--- Q5: Annual Delay Trends ---")
    annual_trends = spark.sql("""
        SELECT
            year,
            SUM(arr_flights) AS total_flights,
            ROUND(SUM(arr_del15) / SUM(arr_flights), 4) AS annual_delay_rate,
            ROUND(SUM(arr_cancelled) / SUM(arr_flights), 4) AS annual_cancellation_rate
        FROM airline_data
        WHERE arr_flights > 0
        GROUP BY year
        ORDER BY year ASC
    """)
    annual_trends.show(20, truncate=False)
    save_dataframe(annual_trends, "annual_trends")

    # Q6: Poor-Performing Airline-Airport Combinations
    print("\n--- Q6: Poor-Performing Airline-Airport Combinations ---")
    poor_performing_combinations = spark.sql("""
        SELECT
            carrier,
            carrier_name,
            airport,
            airport_name,
            SUM(arr_flights) AS total_flights,
            ROUND(SUM(arr_del15) / SUM(arr_flights), 4) AS delay_rate
        FROM airline_data
        WHERE arr_flights > 0
        GROUP BY carrier, carrier_name, airport, airport_name
        HAVING SUM(arr_flights) >= 100
        ORDER BY delay_rate DESC
    """)
    poor_performing_combinations.show(10, truncate=False)
    save_dataframe(poor_performing_combinations, "poor_performing_combinations")

    # Q7: Cancellation Analysis
    print("\n--- Q7: Cancellation Analysis ---")
    cancellation_analysis = spark.sql("""
        WITH MonthlyStats AS (
            SELECT
                year,
                month,
                ROUND(SUM(arr_del15) / SUM(arr_flights), 4) AS monthly_delay_rate,
                ROUND(SUM(arr_cancelled) / SUM(arr_flights), 4) AS monthly_cancellation_rate,
                CASE WHEN ROUND(SUM(arr_cancelled) / SUM(arr_flights), 4) >= 0.05 THEN 'High' ELSE 'Low' END AS cancellation_level
            FROM airline_data
            WHERE arr_flights > 0
            GROUP BY year, month
        )
        SELECT
            cancellation_level,
            ROUND(AVG(monthly_delay_rate), 4) AS avg_delay_rate_in_period,
            ROUND(AVG(monthly_cancellation_rate), 4) AS avg_cancellation_rate_in_period,
            COUNT(*) AS num_months
        FROM MonthlyStats
        GROUP BY cancellation_level
        ORDER BY cancellation_level DESC
    """)
    cancellation_analysis.show(truncate=False)
    save_dataframe(cancellation_analysis, "cancellation_analysis")

    print("\nAnalytics job completed successfully.")

if __name__ == "__main__":
    spark = create_spark_session()
    run_analytics(spark)
    spark.stop()
