# Setup and Deployment Guide

## Prerequisites
- Python 3.8+
- Java 8 or 11 (required for PySpark)
- Apache Hadoop (Optional for HDFS cluster; local mode fallback supported automatically)

## Local Environment Installation

```bash
# Clone or open workspace directory
cd "flight - Copy"

# Install Python dependencies
pip install -r requirements.txt
```

## Running the Complete Pipeline

Run each stage using Python or `spark-submit`:

```bash
# Stage 1: Load CSV data to storage
python src/ingestion/load_to_hdfs.py

# Stage 2: Clean data & check quality
python src/preprocessing/clean_data.py

# Stage 3: Run analytics queries
python src/analytics/delay_analysis.py

# Stage 4: Generate ML features
python src/features/create_features.py

# Stage 5: Train regression models
python src/models/train_regression.py

# Stage 6: Train classification models
python src/models/train_classification.py

# Stage 7: Generate model evaluation charts
python src/models/evaluate_models.py

# Stage 8: Generate future route predictions
python src/prediction/predict.py

# Stage 9: Launch Streamlit Dashboard
streamlit run dashboard/app.py
```

## Docker Environment Setup

To run Hadoop and Spark services via Docker Compose:

```bash
docker-compose up -d
```
