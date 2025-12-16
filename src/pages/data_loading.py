"""
Data Loading Module
Load data from ClickHouse, CSV, or generate synthetic data
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('/home/user/ArgusAI')

from src.utils.clickhouse_connector import ClickHouseConnector, test_connection
from src.utils.data_generator import generate_fraud_data


def show():
    st.markdown('<p class="main-header">Data Loading & Management</p>', unsafe_allow_html=True)
    st.markdown("Load data from ClickHouse, CSV files, or generate synthetic data for analysis and model training")

    # Initialize session state
    if 'ch_config' not in st.session_state:
        st.session_state.ch_config = {
            'host': 'localhost',
            'port': 8123,
            'username': 'default',
            'password': '',
            'database': 'public',
            'connected': False
        }

    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None

    if 'data_source' not in st.session_state:
        st.session_state.data_source = None

    # Create tabs
    tabs = st.tabs([
        "ClickHouse Connection",
        "Load from ClickHouse",
        "Load from CSV",
        "Generate Synthetic Data",
        "Data Preview & Export"
    ])

    with tabs[0]:
        show_clickhouse_connection()

    with tabs[1]:
        show_clickhouse_loader()

    with tabs[2]:
        show_csv_loader()

    with tabs[3]:
        show_synthetic_data_generator()

    with tabs[4]:
        show_data_preview()


def show_clickhouse_connection():
    """ClickHouse connection configuration"""
    st.markdown("### ClickHouse Connection")
    st.markdown("Configure connection to your ClickHouse database")

    # Connection status
    if st.session_state.ch_config['connected']:
        st.success(f"Connected to ClickHouse: {st.session_state.ch_config['host']}:{st.session_state.ch_config['port']}")
    else:
        st.warning("Not connected to ClickHouse")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Connection Settings")

        host = st.text_input(
            "Host:",
            value=st.session_state.ch_config['host'],
            help="ClickHouse server host"
        )

        port = st.number_input(
            "Port:",
            min_value=1,
            max_value=65535,
            value=st.session_state.ch_config['port'],
            help="ClickHouse HTTP port (default: 8123)"
        )

        database = st.text_input(
            "Database:",
            value=st.session_state.ch_config['database'],
            help="Database name"
        )

    with col2:
        st.markdown("#### Authentication")

        username = st.text_input(
            "Username:",
            value=st.session_state.ch_config['username']
        )

        password = st.text_input(
            "Password:",
            value=st.session_state.ch_config['password'],
            type="password"
        )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Test Connection", type="secondary"):
            with st.spinner("Testing connection..."):
                result = test_connection(host, port, username, password, database)

                if result['success']:
                    st.success(f"Connection successful!")
                    st.info(f"ClickHouse Version: {result['version']}")
                    st.info(f"Found {result['table_count']} tables")

                    with st.expander("Available Tables"):
                        for table in result['tables']:
                            st.text(f"- {table}")
                else:
                    st.error(f"Connection failed: {result.get('error', 'Unknown error')}")

    with col2:
        if st.button("Save & Connect", type="primary"):
            with st.spinner("Connecting..."):
                result = test_connection(host, port, username, password, database)

                if result['success']:
                    st.session_state.ch_config = {
                        'host': host,
                        'port': port,
                        'username': username,
                        'password': password,
                        'database': database,
                        'connected': True,
                        'tables': result['tables']
                    }
                    st.success("Connected and saved!")
                    st.rerun()
                else:
                    st.error(f"Connection failed: {result.get('error')}")

    with col3:
        if st.button("Disconnect"):
            st.session_state.ch_config['connected'] = False
            st.success("Disconnected!")
            st.rerun()


def show_clickhouse_loader():
    """Load data from ClickHouse"""
    st.markdown("### Load Data from ClickHouse")

    if not st.session_state.ch_config.get('connected'):
        st.warning("Please connect to ClickHouse first (see ClickHouse Connection tab)")
        return

    st.markdown("Select loading method:")

    method = st.radio(
        "Loading Method:",
        ["Custom SQL Query", "Query Builder", "Quick Filters"],
        index=0,
        horizontal=True,
        help="Custom SQL Query is recommended for direct access to stixor_fraud_features_distributed"
    )

    if method == "Custom SQL Query":
        show_custom_query()
    elif method == "Query Builder":
        show_query_builder()
    else:
        show_quick_filters()


def show_query_builder():
    """Visual query builder"""
    st.markdown("#### Query Builder")

    # Get available tables
    tables = st.session_state.ch_config.get('tables', [])

    if not tables:
        st.warning("No tables found in database")
        return

    # Set default table to stixor_fraud_features_distributed if it exists
    default_table = "stixor_fraud_features_distributed"
    default_index = 0
    if default_table in tables:
        default_index = tables.index(default_table)

    col1, col2 = st.columns(2)

    with col1:
        selected_table = st.selectbox("Select Table:", tables, index=default_index)

        # Get table schema
        if st.button("Load Schema"):
            connector = ClickHouseConnector(
                host=st.session_state.ch_config['host'],
                port=st.session_state.ch_config['port'],
                username=st.session_state.ch_config['username'],
                password=st.session_state.ch_config['password'],
                database=st.session_state.ch_config['database']
            )
            connector.connect()

            schema = connector.get_table_schema(selected_table)
            st.session_state.table_schema = schema
            connector.close()

            st.success(f"Loaded schema for {selected_table}")

    with col2:
        limit = st.number_input("Row Limit:", min_value=100, max_value=1000000, value=10000, step=1000)

    # Show schema
    if 'table_schema' in st.session_state:
        st.markdown("**Table Schema:**")
        st.dataframe(st.session_state.table_schema, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Filters (Optional)")

    # Add filters
    num_filters = st.number_input("Number of filters:", 0, 10, 0)

    filters = {}
    for i in range(num_filters):
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_col = st.text_input(f"Column {i+1}:", key=f"filter_col_{i}")

        with col2:
            filter_op = st.selectbox(f"Operator {i+1}:", ["=", ">", "<", ">=", "<=", "IN"], key=f"filter_op_{i}")

        with col3:
            filter_val = st.text_input(f"Value {i+1}:", key=f"filter_val_{i}")

        if filter_col and filter_val:
            if filter_op == "IN":
                # Parse comma-separated values
                vals = [v.strip() for v in filter_val.split(',')]
                filters[filter_col] = vals
            else:
                filters[filter_col] = filter_val

    st.markdown("---")

    if st.button("Load Data", type="primary"):
        with st.spinner("Loading data from ClickHouse..."):
            try:
                connector = ClickHouseConnector(
                    host=st.session_state.ch_config['host'],
                    port=st.session_state.ch_config['port'],
                    username=st.session_state.ch_config['username'],
                    password=st.session_state.ch_config['password'],
                    database=st.session_state.ch_config['database']
                )
                connector.connect()

                df = connector.get_transactions(
                    table=selected_table,
                    filters=filters if filters else None,
                    limit=limit
                )

                connector.close()

                st.session_state.loaded_data = df
                st.session_state.data_source = f"ClickHouse: {selected_table}"

                st.success(f"Loaded {len(df)} rows from {selected_table}")
                st.dataframe(df.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")


def show_custom_query():
    """Custom SQL query interface"""
    st.markdown("#### Custom SQL Query")
    st.markdown("Write SQL queries to load data from **stixor_fraud_features_distributed** or other tables")

    # Sample queries dropdown
    st.markdown("**Sample Queries:**")
    sample_queries = {
        "Load all data (limit 10K)": "SELECT * FROM stixor_fraud_features_distributed LIMIT 10000",
        "Load recent data": "SELECT * FROM stixor_fraud_features_distributed WHERE timestamp >= today() - 30 LIMIT 10000",
        "Load fraud cases only": "SELECT * FROM stixor_fraud_features_distributed WHERE is_fraud = 1 LIMIT 10000",
        "Load by date range": "SELECT * FROM stixor_fraud_features_distributed WHERE timestamp BETWEEN '2025-01-01' AND '2025-12-31' LIMIT 50000",
        "Custom query": ""
    }

    selected_sample = st.selectbox(
        "Select a sample query or write your own:",
        list(sample_queries.keys()),
        index=0
    )

    # Pre-fill query based on selection
    default_query = sample_queries[selected_sample]

    query = st.text_area(
        "SQL Query:",
        value=default_query,
        height=200,
        help="Enter any valid ClickHouse SQL query. Use stixor_fraud_features_distributed table."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("Execute Query", type="primary"):
            if not query.strip():
                st.warning("Please enter a query")
            else:
                with st.spinner("Executing query..."):
                    try:
                        connector = ClickHouseConnector(
                            host=st.session_state.ch_config['host'],
                            port=st.session_state.ch_config['port'],
                            username=st.session_state.ch_config['username'],
                            password=st.session_state.ch_config['password'],
                            database=st.session_state.ch_config['database']
                        )
                        connector.connect()

                        df = connector.execute_custom_query(query)

                        connector.close()

                        st.session_state.loaded_data = df
                        st.session_state.data_source = "ClickHouse: Custom Query"

                        st.success(f"Query executed successfully! Loaded {len(df)} rows")
                        st.dataframe(df.head(10), use_container_width=True)

                    except Exception as e:
                        st.error(f"Query error: {str(e)}")

    with col2:
        if st.button("Clear Query"):
            st.rerun()


def show_quick_filters():
    """Quick filter interface for common queries"""
    st.markdown("#### Quick Filters")

    tables = st.session_state.ch_config.get('tables', [])

    # Set default table to stixor_fraud_features_distributed if it exists
    default_table = "stixor_fraud_features_distributed"
    default_index = 0
    if default_table in tables:
        default_index = tables.index(default_table)

    selected_table = st.selectbox("Select Table:", tables, index=default_index)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Date Range:**")
        start_date = st.date_input("Start Date:", datetime.now() - timedelta(days=30))
        end_date = st.date_input("End Date:", datetime.now())

    with col2:
        st.markdown("**Filters:**")
        fraud_only = st.checkbox("Fraud transactions only")
        limit = st.number_input("Row Limit:", 100, 1000000, 10000, 1000)

    if st.button("Load Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                connector = ClickHouseConnector(
                    host=st.session_state.ch_config['host'],
                    port=st.session_state.ch_config['port'],
                    username=st.session_state.ch_config['username'],
                    password=st.session_state.ch_config['password'],
                    database=st.session_state.ch_config['database']
                )
                connector.connect()

                filters = {}
                if fraud_only:
                    filters['is_fraud'] = 1

                df = connector.get_transactions_by_date_range(
                    table=selected_table,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    additional_filters=filters,
                    limit=limit
                )

                connector.close()

                st.session_state.loaded_data = df
                st.session_state.data_source = f"ClickHouse: {selected_table} ({start_date} to {end_date})"

                st.success(f"Loaded {len(df)} rows")
                st.dataframe(df.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")


def show_csv_loader():
    """Load data from CSV file"""
    st.markdown("### Load Data from CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV file:",
        type=['csv'],
        help="Upload a CSV file containing transaction data"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success(f"Loaded {len(df)} rows from CSV")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Use This Data", type="primary"):
                st.session_state.loaded_data = df
                st.session_state.data_source = f"CSV: {uploaded_file.name}"
                st.success("Data loaded successfully!")
                st.rerun()

        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")


def show_synthetic_data_generator():
    """Generate synthetic transaction data"""
    st.markdown("### Generate Synthetic Data")
    st.markdown("Generate synthetic transaction data for testing and development")

    col1, col2 = st.columns(2)

    with col1:
        num_records = st.number_input("Number of Records:", 100, 100000, 10000, 1000)
        fraud_ratio = st.slider("Fraud Ratio:", 0.0, 0.5, 0.05, 0.01)

    with col2:
        start_date = st.date_input("Start Date:", datetime.now() - timedelta(days=30))
        end_date = st.date_input("End Date:", datetime.now())

    if st.button("Generate Data", type="primary"):
        with st.spinner("Generating synthetic data..."):
            df = generate_fraud_data(
                n_samples=num_records,
                fraud_rate=fraud_ratio,
                random_state=42
            )

            st.session_state.loaded_data = df
            st.session_state.data_source = f"Synthetic Data ({num_records} records)"

            st.success(f"Generated {len(df)} synthetic transactions")
            st.info(f"Fraud rate: {(df['is_fraud'].sum() / len(df) * 100):.2f}%")
            st.dataframe(df.head(10), use_container_width=True)


def show_data_preview():
    """Preview and export loaded data"""
    st.markdown("### Data Preview & Export")

    if st.session_state.loaded_data is None:
        st.info("No data loaded. Please load data from one of the other tabs.")
        return

    df = st.session_state.loaded_data

    # Data info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        st.metric("Data Source", st.session_state.data_source)

    with col4:
        if 'is_fraud' in df.columns:
            fraud_count = df['is_fraud'].sum()
            fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.markdown("---")

    # Data preview
    st.markdown("#### Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("---")

    # Data statistics
    st.markdown("#### Data Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Numeric Columns:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    with col2:
        st.markdown("**Missing Values:**")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': (missing.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]

        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("No missing values!")

    st.markdown("---")

    # Export options
    st.markdown("#### Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"fraud_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        parquet = df.to_parquet()
        st.download_button(
            label="Download Parquet",
            data=parquet,
            file_name=f"fraud_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
            mime="application/octet-stream"
        )

    with col3:
        if st.button("Use for Training"):
            st.success("Data ready for model training! Go to Model Training tab.")
