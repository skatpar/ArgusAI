"""
ClickHouse Data Connector
Handles connections and queries to ClickHouse database
"""

import clickhouse_connect
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClickHouseConnector:
    """ClickHouse database connector for fraud detection data"""

    def __init__(self, host: str = 'localhost', port: int = 8123,
                 username: str = 'default', password: str = '',
                 database: str = 'default'):
        """
        Initialize ClickHouse connector

        Args:
            host: ClickHouse server host
            port: ClickHouse HTTP port (default 8123)
            username: Database username
            password: Database password
            database: Database name
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.client = None

    def connect(self) -> bool:
        """
        Establish connection to ClickHouse

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database
            )

            # Test connection
            result = self.client.command('SELECT 1')
            logger.info(f"Connected to ClickHouse at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {str(e)}")
            return False

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            pd.DataFrame: Query results
        """
        try:
            if not self.client:
                self.connect()

            result = self.client.query_df(query, parameters=parameters)
            logger.info(f"Query executed successfully, returned {len(result)} rows")
            return result

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def get_transactions(self,
                         table: str = 'transactions',
                         filters: Optional[Dict[str, Any]] = None,
                         limit: int = 10000,
                         order_by: str = 'timestamp DESC') -> pd.DataFrame:
        """
        Get transactions with optional filters

        Args:
            table: Table name
            filters: Dictionary of column:value filters
            limit: Maximum rows to return
            order_by: ORDER BY clause

        Returns:
            pd.DataFrame: Transaction data
        """
        query = f"SELECT * FROM {table}"

        # Build WHERE clause
        if filters:
            conditions = []
            for col, val in filters.items():
                if isinstance(val, str):
                    conditions.append(f"{col} = '{val}'")
                elif isinstance(val, (list, tuple)):
                    # Handle IN clause
                    if isinstance(val[0], str):
                        vals = "', '".join(val)
                        conditions.append(f"{col} IN ('{vals}')")
                    else:
                        vals = ", ".join(map(str, val))
                        conditions.append(f"{col} IN ({vals})")
                else:
                    conditions.append(f"{col} = {val}")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        # Add ORDER BY and LIMIT
        query += f" ORDER BY {order_by} LIMIT {limit}"

        return self.execute_query(query)

    def get_transactions_by_date_range(self,
                                       table: str = 'transactions',
                                       start_date: str = None,
                                       end_date: str = None,
                                       additional_filters: Optional[Dict] = None,
                                       limit: int = 100000) -> pd.DataFrame:
        """
        Get transactions within a date range

        Args:
            table: Table name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            additional_filters: Additional filter conditions
            limit: Maximum rows

        Returns:
            pd.DataFrame: Filtered transactions
        """
        conditions = []

        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")

        if additional_filters:
            for col, val in additional_filters.items():
                if isinstance(val, str):
                    conditions.append(f"{col} = '{val}'")
                else:
                    conditions.append(f"{col} = {val}")

        query = f"SELECT * FROM {table}"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        return self.execute_query(query)

    def get_aggregated_stats(self,
                            table: str = 'transactions',
                            group_by_cols: List[str] = ['merchant_category'],
                            agg_cols: Dict[str, str] = None,
                            filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Get aggregated statistics

        Args:
            table: Table name
            group_by_cols: Columns to group by
            agg_cols: Dictionary of {column: aggregation_function}
            filters: Filter conditions

        Returns:
            pd.DataFrame: Aggregated data
        """
        if agg_cols is None:
            agg_cols = {
                'transaction_amount': 'AVG',
                'is_fraud': 'SUM',
                '*': 'COUNT'
            }

        # Build aggregation expressions
        agg_exprs = []
        for col, func in agg_cols.items():
            if col == '*':
                agg_exprs.append(f"{func}(*) as count")
            else:
                agg_exprs.append(f"{func}({col}) as {col}_{func.lower()}")

        # Build query
        query = f"SELECT {', '.join(group_by_cols)}, {', '.join(agg_exprs)} FROM {table}"

        # Add filters
        if filters:
            conditions = []
            for col, val in filters.items():
                if isinstance(val, str):
                    conditions.append(f"{col} = '{val}'")
                else:
                    conditions.append(f"{col} = {val}")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        # Add GROUP BY
        query += f" GROUP BY {', '.join(group_by_cols)}"

        return self.execute_query(query)

    def get_table_schema(self, table: str) -> pd.DataFrame:
        """
        Get schema information for a table

        Args:
            table: Table name

        Returns:
            pd.DataFrame: Schema information
        """
        query = f"DESCRIBE TABLE {table}"
        return self.execute_query(query)

    def get_table_list(self) -> List[str]:
        """
        Get list of all tables in database

        Returns:
            List[str]: Table names
        """
        query = "SHOW TABLES"
        result = self.execute_query(query)
        return result.iloc[:, 0].tolist()

    def get_row_count(self, table: str, filters: Optional[Dict] = None) -> int:
        """
        Get row count for a table

        Args:
            table: Table name
            filters: Optional filters

        Returns:
            int: Row count
        """
        query = f"SELECT COUNT(*) as count FROM {table}"

        if filters:
            conditions = []
            for col, val in filters.items():
                if isinstance(val, str):
                    conditions.append(f"{col} = '{val}'")
                else:
                    conditions.append(f"{col} = {val}")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        result = self.execute_query(query)
        return int(result.iloc[0, 0])

    def execute_custom_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query

        Args:
            query: Custom SQL query

        Returns:
            pd.DataFrame: Query results
        """
        return self.execute_query(query)

    def close(self):
        """Close the database connection"""
        if self.client:
            self.client.close()
            logger.info("ClickHouse connection closed")


def test_connection(host: str, port: int, username: str,
                   password: str, database: str) -> Dict[str, Any]:
    """
    Test ClickHouse connection and return status

    Args:
        host: ClickHouse host
        port: ClickHouse port
        username: Username
        password: Password
        database: Database name

    Returns:
        Dict with connection status and info
    """
    try:
        connector = ClickHouseConnector(host, port, username, password, database)
        success = connector.connect()

        if success:
            # Get database info
            tables = connector.get_table_list()
            version_result = connector.execute_query("SELECT version()")
            version = version_result.iloc[0, 0] if not version_result.empty else "Unknown"

            connector.close()

            return {
                'success': True,
                'message': 'Connection successful',
                'version': version,
                'tables': tables,
                'table_count': len(tables)
            }
        else:
            return {
                'success': False,
                'message': 'Connection failed',
                'error': 'Could not establish connection'
            }

    except Exception as e:
        return {
            'success': False,
            'message': 'Connection failed',
            'error': str(e)
        }
