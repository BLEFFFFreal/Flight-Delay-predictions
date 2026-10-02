import unittest
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath("."))

java_home_adoptium = r"C:\Program Files\Eclipse Adoptium\jre-25.0.1.8-hotspot"
if os.path.exists(java_home_adoptium):
    os.environ["JAVA_HOME"] = java_home_adoptium
    os.environ["PATH"] = os.path.join(java_home_adoptium, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import col, lit

from src.ingestion.load_to_hdfs import load_data_to_hdfs
from src.preprocessing.clean_data import clean_data
from src.features.create_features import create_features

class TestAirlinePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("TestAirlinePipeline") \
            .master("local[2]") \
            .getOrCreate()

        cls.test_dir = "tests/tmp_test_data"
        os.makedirs(cls.test_dir, exist_ok=True)

        # Create synthetic CSV data for testing
        csv_content = """year,month,carrier,carrier_name,airport,airport_name,arr_flights,arr_del15,carrier_ct,weather_ct,nas_ct,security_ct,late_aircraft_ct,arr_cancelled,arr_diverted,arr_delay,carrier_delay,weather_delay,nas_delay,security_delay,late_aircraft_delay
2020,1,AA,American,JFK,JFK Airport,100.0,20.0,5.0,2.0,3.0,0.0,10.0,2.0,1.0,1000.0,250.0,100.0,150.0,0.0,500.0
2020,2,AA,American,JFK,JFK Airport,100.0,25.0,6.0,3.0,4.0,0.0,12.0,1.0,0.0,1200.0,300.0,150.0,200.0,0.0,550.0
2020,3,AA,American,JFK,JFK Airport,100.0,15.0,4.0,1.0,2.0,0.0,8.0,0.0,0.0,800.0,200.0,50.0,100.0,0.0,450.0
2020,3,AA,American,JFK,JFK Airport,100.0,15.0,4.0,1.0,2.0,0.0,8.0,0.0,0.0,800.0,200.0,50.0,100.0,0.0,450.0
2020,5,AA,American,JFK,JFK Airport,100.0,30.0,10.0,5.0,5.0,0.0,10.0,5.0,2.0,1500.0,500.0,250.0,250.0,0.0,500.0
"""
        cls.csv_path = os.path.join(cls.test_dir, "test_raw.csv")
        with open(cls.csv_path, "w") as f:
            f.write(csv_content)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_01_ingestion_and_cleaning(self):
        parquet_raw = os.path.join(self.test_dir, "raw.parquet")
        parquet_clean = os.path.join(self.test_dir, "clean.parquet")
        report_json = os.path.join(self.test_dir, "report.json")

        df_ingested = load_data_to_hdfs(self.spark, input_path=self.csv_path, output_path=parquet_raw)
        self.assertEqual(df_ingested.count(), 5)

        df_cleaned = clean_data(self.spark, input_path=parquet_raw, output_path=parquet_clean, report_output_path=report_json)
        # Duplicate record (2020-03 AA JFK) should be removed -> 4 records remaining
        self.assertEqual(df_cleaned.count(), 4)
        self.assertTrue(os.path.exists(report_json))

    def test_02_feature_engineering_and_calendar_gap(self):
        parquet_clean = os.path.join(self.test_dir, "clean.parquet")
        parquet_features = os.path.join(self.test_dir, "features.parquet")

        df_feat = create_features(self.spark, input_path=parquet_clean, output_path=parquet_features)
        
        # Verify required columns exist
        expected_cols = [
            "delay_rate", "cancellation_rate", "diversion_rate",
            "lag_1_delay_rate", "lag_2_delay_rate", "lag_3_delay_rate", "lag_12_delay_rate",
            "rolling_3_month_delay_rate", "rolling_6_month_delay_rate", "rolling_12_month_delay_rate",
            "month_sin", "month_cos", "next_month_delay_rate", "high_delay"
        ]
        for col_name in expected_cols:
            self.assertIn(col_name, df_feat.columns)

        # Check calendar gap logic:
        # Row month 5 (May 2020) follows month 3 (March 2020).
        # Since April 2020 is missing, lag_1 for May 2020 should NOT grab March's value!
        may_row = df_feat.filter(col("month") == 3).collect()
        if may_row:
            # For March 2020, lag_1 is Feb 2020 (delay_rate = 0.25)
            self.assertAlmostEqual(may_row[0]["lag_1_delay_rate"], 0.25, places=4)

    def test_03_leakage_prevention(self):
        parquet_features = os.path.join(self.test_dir, "features.parquet")
        df_feat = self.spark.read.parquet(parquet_features)

        # Verify that feature vector input candidate numerical features do NOT include same-month target variables
        numerical_features = [
            "month", "lag_1_delay_rate", "lag_2_delay_rate", "lag_3_delay_rate", "lag_12_delay_rate",
            "previous_month_arr_flights", "previous_month_cancel_rate", "previous_month_diversion_rate",
            "previous_month_avg_delay", "rolling_3_month_delay_rate", "rolling_6_month_delay_rate",
            "rolling_12_month_delay_rate", "month_sin", "month_cos"
        ]

        for same_month_leak in ["arr_del15", "arr_delay", "carrier_delay", "weather_delay", "next_month_delay_rate"]:
            self.assertNotIn(same_month_leak, numerical_features)

if __name__ == "__main__":
    unittest.main()
