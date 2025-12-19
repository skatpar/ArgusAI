"""
Global Settings Configuration Page
Manage connection strings, secrets, and global paths
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.settings import SettingsManager


def show():
    """Display settings configuration page"""
    st.markdown("## Global Settings")
    st.markdown("Configure connection strings, secrets, and global paths for the application")

    # Initialize settings manager
    if 'settings_manager' not in st.session_state:
        st.session_state.settings_manager = SettingsManager()

    settings_mgr = st.session_state.settings_manager
    settings = settings_mgr.get_all()

    st.markdown("---")

    # Tabs for different setting categories
    tabs = st.tabs([
        "ClickHouse",
        "File Paths",
        "API Configuration",
        "MLflow",
        "Spark Configuration"
    ])

    # ClickHouse Settings
    with tabs[0]:
        st.markdown("### ClickHouse Connection")

        ch_settings = settings.get('clickhouse', {})

        col1, col2 = st.columns(2)

        with col1:
            ch_host = st.text_input(
                "Host",
                value=ch_settings.get('host', 'localhost'),
                key="ch_host"
            )
            ch_port = st.number_input(
                "Port",
                value=ch_settings.get('port', 9000),
                min_value=1,
                max_value=65535,
                key="ch_port"
            )
            ch_http_port = st.number_input(
                "HTTP Port",
                value=ch_settings.get('http_port', 8123),
                min_value=1,
                max_value=65535,
                key="ch_http_port"
            )

        with col2:
            ch_database = st.text_input(
                "Database",
                value=ch_settings.get('database', 'public'),
                key="ch_database"
            )
            ch_user = st.text_input(
                "User",
                value=ch_settings.get('user', 'default'),
                key="ch_user"
            )
            ch_password = st.text_input(
                "Password",
                value=ch_settings.get('password', ''),
                type="password",
                key="ch_password"
            )

        if st.button("Save ClickHouse Settings", key="save_ch"):
            settings['clickhouse'] = {
                'host': ch_host,
                'port': ch_port,
                'http_port': ch_http_port,
                'database': ch_database,
                'user': ch_user,
                'password': ch_password
            }
            settings_mgr.save_settings(settings)
            st.success("ClickHouse settings saved successfully")
            st.rerun()

    # File Paths Settings
    with tabs[1]:
        st.markdown("### Global File Paths")

        paths = settings.get('paths', {})

        models_dir = st.text_input(
            "Models Directory",
            value=paths.get('models_dir', ''),
            key="models_dir"
        )
        rules_dir = st.text_input(
            "Rules Directory",
            value=paths.get('rules_dir', ''),
            key="rules_dir"
        )
        analysis_dir = st.text_input(
            "Analysis Directory",
            value=paths.get('analysis_dir', ''),
            key="analysis_dir"
        )
        logs_dir = st.text_input(
            "Logs Directory",
            value=paths.get('logs_dir', ''),
            key="logs_dir"
        )

        if st.button("Save Paths", key="save_paths"):
            settings['paths'] = {
                'models_dir': models_dir,
                'rules_dir': rules_dir,
                'analysis_dir': analysis_dir,
                'logs_dir': logs_dir
            }
            settings_mgr.save_settings(settings)
            st.success("Path settings saved successfully")
            st.rerun()

    # API Configuration
    with tabs[2]:
        st.markdown("### Default API Configuration")

        api_settings = settings.get('api', {})

        api_endpoint = st.text_input(
            "API Endpoint URL",
            value=api_settings.get('endpoint_url', ''),
            key="api_endpoint"
        )
        api_auth_type = st.selectbox(
            "Authentication Type",
            options=["None", "API Key", "Bearer Token", "Basic Auth"],
            index=["None", "API Key", "Bearer Token", "Basic Auth"].index(
                api_settings.get('auth_type', 'None')
            ),
            key="api_auth_type"
        )
        api_key = st.text_input(
            "API Key / Token",
            value=api_settings.get('api_key', ''),
            type="password",
            key="api_key"
        )
        api_timeout = st.number_input(
            "Request Timeout (seconds)",
            value=api_settings.get('timeout', 30),
            min_value=1,
            max_value=300,
            key="api_timeout"
        )

        if st.button("Save API Settings", key="save_api"):
            settings['api'] = {
                'endpoint_url': api_endpoint,
                'auth_type': api_auth_type,
                'api_key': api_key,
                'timeout': api_timeout
            }
            settings_mgr.save_settings(settings)
            st.success("API settings saved successfully")
            st.rerun()

    # MLflow Settings
    with tabs[3]:
        st.markdown("### MLflow Configuration")

        mlflow_settings = settings.get('mlflow', {})

        mlflow_uri = st.text_input(
            "Tracking URI",
            value=mlflow_settings.get('tracking_uri', ''),
            key="mlflow_uri",
            help="MLflow tracking server URI (e.g., http://localhost:5000)"
        )
        mlflow_experiment = st.text_input(
            "Experiment Name",
            value=mlflow_settings.get('experiment_name', 'fraud_detection_pipeline'),
            key="mlflow_experiment"
        )
        mlflow_artifact_location = st.text_input(
            "Artifact Location",
            value=mlflow_settings.get('artifact_location', ''),
            key="mlflow_artifact",
            help="Optional: Custom artifact storage location"
        )

        if st.button("Save MLflow Settings", key="save_mlflow"):
            settings['mlflow'] = {
                'tracking_uri': mlflow_uri,
                'experiment_name': mlflow_experiment,
                'artifact_location': mlflow_artifact_location
            }
            settings_mgr.save_settings(settings)
            st.success("MLflow settings saved successfully")
            st.rerun()

    # Spark Configuration
    with tabs[4]:
        st.markdown("### Spark Configuration")

        spark_settings = settings.get('spark', {})

        col1, col2 = st.columns(2)

        with col1:
            spark_exec_memory = st.text_input(
                "Executor Memory",
                value=spark_settings.get('executor_memory', '150g'),
                key="spark_exec_mem",
                help="e.g., 150g, 8g"
            )
            spark_driver_memory = st.text_input(
                "Driver Memory",
                value=spark_settings.get('driver_memory', '8g'),
                key="spark_driver_mem",
                help="e.g., 8g, 4g"
            )
            spark_exec_cores = st.number_input(
                "Executor Cores",
                value=spark_settings.get('executor_cores', 32),
                min_value=1,
                max_value=128,
                key="spark_exec_cores"
            )

        with col2:
            spark_exec_instances = st.number_input(
                "Executor Instances",
                value=spark_settings.get('executor_instances', 2),
                min_value=1,
                max_value=100,
                key="spark_exec_instances"
            )
            spark_shuffle_partitions = st.number_input(
                "Shuffle Partitions",
                value=spark_settings.get('shuffle_partitions', 200),
                min_value=1,
                max_value=10000,
                key="spark_shuffle"
            )

        if st.button("Save Spark Settings", key="save_spark"):
            settings['spark'] = {
                'executor_memory': spark_exec_memory,
                'driver_memory': spark_driver_memory,
                'executor_cores': spark_exec_cores,
                'executor_instances': spark_exec_instances,
                'shuffle_partitions': spark_shuffle_partitions
            }
            settings_mgr.save_settings(settings)
            st.success("Spark settings saved successfully")
            st.rerun()

    # Reset to defaults
    st.markdown("---")
    st.markdown("### Reset Settings")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Reset to Defaults", type="secondary"):
            settings_mgr.reset_to_defaults()
            st.success("Settings reset to defaults")
            st.rerun()

    with col2:
        if st.button("Reload Settings"):
            settings_mgr.settings = settings_mgr.load_settings()
            st.success("Settings reloaded")
            st.rerun()

    # Show current settings file location
    st.markdown("---")
    st.caption(f"Settings file: `{settings_mgr.settings_file}`")
