# Architecture Specification

## Overview

The system integrates Apache Hadoop HDFS for distributed storage with Apache Spark for scalable processing and machine learning:

```
HADOOP HDFS (Distributed Storage Layer)
  ├── /data/airline/raw/                <- Raw uploaded dataset CSV
  ├── /data/airline/processed/          <- Cleaned Parquet storage format
  ├── /data/airline/features/           <- Engineered feature vector Parquet
  ├── /data/airline/analytics/          <- Analytical SQL output datasets
  ├── /data/airline/predictions/        <- Future route prediction outputs
  └── /data/airline/models/             <- Saved Spark ML models & metadata
```

## Computation Engine
- **PySpark DataFrames**: Distributed transformations, null/NaN cleaning, schema enforcement.
- **Spark SQL**: In-memory analytical aggregations for airline and airport performance rankings.
- **Spark Window Functions**: Grouped chronological windowing partitioned by `(carrier, airport)` and ordered by `month_index` (`year * 12 + month`).
- **Spark MLlib Pipelines**: Standardized ML pipeline chaining `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `StandardScaler`, and ML estimators.
