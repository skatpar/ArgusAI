"""
PySpark Connector for ClickHouse
Handles large-scale data loading and processing
"""

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame
import pandas as pd
from typing import Optional, Dict, Any


class SparkConnector:
    """Spark connector for ClickHouse with optimized settings"""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8123,
        username: str = 'default',
        password: str = '',
        database: str = 'public',
        app_name: str = 'ArgusAI'
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.app_name = app_name
        self.spark = None

    def get_spark_session(self) -> SparkSession:
        """Create or get existing Spark session"""
        if self.spark is None:
            self.spark = (SparkSession.builder
                .appName(self.app_name)
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .getOrCreate())
        return self.spark

    def execute_query(self, query: str) -> SparkDataFrame:
        """Execute SQL query and return Spark DataFrame"""
        spark = self.get_spark_session()

        # Build JDBC URL for ClickHouse
        jdbc_url = f"jdbc:clickhouse://{self.host}:{self.port}/{self.database}"

        # Read data using JDBC
        df = (spark.read
            .format("jdbc")
            .option("url", jdbc_url)
            .option("user", self.username)
            .option("password", self.password)
            .option("query", query)
            .option("driver", "com.clickhouse.jdbc.ClickHouseDriver")
            .load())

        return df

    def execute_query_to_pandas(self, query: str, limit: int = None) -> pd.DataFrame:
        """Execute query and convert to Pandas (with optional limit)"""
        spark_df = self.execute_query(query)

        if limit:
            spark_df = spark_df.limit(limit)

        return spark_df.toPandas()

    def to_pandas_limited(self, spark_df: SparkDataFrame, limit: int = 1000) -> pd.DataFrame:
        """Convert Spark DataFrame to Pandas with limit for display"""
        return spark_df.limit(limit).toPandas()

    def cache_dataframe(self, spark_df: SparkDataFrame) -> SparkDataFrame:
        """Cache Spark DataFrame for faster access"""
        return spark_df.cache()

    def stop(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            self.spark = None


class SparkDataFrameWrapper:
    """Wrapper to handle both Spark and Pandas DataFrames uniformly"""

    def __init__(self, data, is_spark: bool = True):
        self.data = data
        self.is_spark = is_spark
        self._cached_count = None

    def to_pandas(self, limit: int = None) -> pd.DataFrame:
        """Convert to Pandas with optional limit"""
        if self.is_spark:
            if limit:
                return self.data.limit(limit).toPandas()
            return self.data.toPandas()
        else:
            if limit:
                return self.data.head(limit)
            return self.data

    def count(self) -> int:
        """Get row count (cached for Spark)"""
        if self._cached_count is None:
            if self.is_spark:
                self._cached_count = self.data.count()
            else:
                self._cached_count = len(self.data)
        return self._cached_count

    def columns(self):
        """Get column names"""
        if self.is_spark:
            return self.data.columns
        else:
            return self.data.columns.tolist()

    def select(self, columns):
        """Select columns"""
        if self.is_spark:
            return SparkDataFrameWrapper(self.data.select(columns), is_spark=True)
        else:
            return SparkDataFrameWrapper(self.data[columns], is_spark=False)

    def filter(self, condition):
        """Filter rows (condition is a SQL string for Spark, boolean Series for Pandas)"""
        if self.is_spark:
            return SparkDataFrameWrapper(self.data.filter(condition), is_spark=True)
        else:
            return SparkDataFrameWrapper(self.data[condition], is_spark=False)

    def get_native(self):
        """Get native Spark or Pandas DataFrame"""
        return self.data


def create_spark_connector(config: Dict[str, Any]) -> SparkConnector:
    """Factory function to create Spark connector from config"""
    return SparkConnector(
        host=config.get('host', 'localhost'),
        port=config.get('port', 8123),
        username=config.get('username', 'default'),
        password=config.get('password', ''),
        database=config.get('database', 'public')
    )
