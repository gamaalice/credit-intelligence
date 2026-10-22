from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from credit_intelligence.constants import FEATURE_COLUMNS, TARGET_COLUMN


def run_spark_analysis(frame: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    """Run actual local PySpark transformations and SQL aggregations on the UCI rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("credit-intelligence-uci-analysis")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        source_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
        spark_frame = spark.createDataFrame(frame[source_columns])
        prepared = spark_frame.withColumnRenamed(
            TARGET_COLUMN, "default_next_month"
        ).withColumn(
            "default_next_month", F.col("default_next_month").cast("integer")
        ).withColumn(
            "age_band",
            F.when(F.col("AGE") < 30, "under_30")
            .when(F.col("AGE") < 40, "30_to_39")
            .when(F.col("AGE") < 50, "40_to_49")
            .when(F.col("AGE") < 60, "50_to_59")
            .otherwise("60_or_older"),
        )
        prepared.createOrReplaceTempView("uci_credit_clients")

        quality_row = spark.sql(
            """
            SELECT
                COUNT(*) AS rows,
                SUM(default_next_month) AS defaults,
                AVG(default_next_month) AS default_rate,
                AVG(LIMIT_BAL) AS average_credit_limit
            FROM uci_credit_clients
            """
        ).first()
        age_rates = spark.sql(
            """
            SELECT
                age_band,
                COUNT(*) AS clients,
                AVG(default_next_month) AS default_rate,
                AVG(LIMIT_BAL) AS average_credit_limit
            FROM uci_credit_clients
            GROUP BY age_band
            ORDER BY age_band
            """
        )
        age_rates.toPandas().to_csv(
            output_dir / "pyspark_age_default_rates.csv", index=False
        )

        payment_rates = spark.sql(
            """
            SELECT
                PAY_0 AS most_recent_repayment_status,
                COUNT(*) AS clients,
                AVG(default_next_month) AS default_rate
            FROM uci_credit_clients
            GROUP BY PAY_0
            ORDER BY PAY_0
            """
        )
        payment_rates.toPandas().to_csv(
            output_dir / "pyspark_repayment_status_rates.csv", index=False
        )

        summary = {
            "rows": int(quality_row["rows"]),
            "defaults": int(quality_row["defaults"]),
            "default_rate": float(quality_row["default_rate"]),
            "average_credit_limit": float(quality_row["average_credit_limit"]),
            "age_segments": int(age_rates.count()),
            "repayment_status_segments": int(payment_rates.count()),
            "engine": "PySpark local mode",
        }
        pd.DataFrame([summary]).to_csv(
            output_dir / "pyspark_sql_summary.csv", index=False
        )
        return summary
    finally:
        spark.stop()
