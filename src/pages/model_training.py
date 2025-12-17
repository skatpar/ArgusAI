"""
Model Training Module
Train fraud detection models using custom training scripts or built-in algorithms
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import json
import sys
import os
import yaml
sys.path.append('/home/user/ArgusAI')

from src.utils.config_manager import config_manager
from src.utils.mlflow_tracker import MLflowTracker


def show():
    st.markdown('<p class="main-header">Model Training & Experimentation</p>', unsafe_allow_html=True)
    st.markdown("Train fraud detection models using your custom training scripts or built-in algorithms")

    # Initialize session state
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []

    if 'current_model' not in st.session_state:
        st.session_state.current_model = None

    if 'training_config' not in st.session_state:
        st.session_state.training_config = config_manager.load_config()

    # Create tabs
    tabs = st.tabs([
        "Configuration",
        "Custom Training Script",
        "Built-in Training",
        "Model Selection & Management",
        "Training History",
        "Model Comparison"
    ])

    with tabs[0]:
        show_configuration()

    with tabs[1]:
        show_custom_script_training()

    with tabs[2]:
        show_builtin_training()

    with tabs[3]:
        show_model_selection()

    with tabs[4]:
        show_training_history()

    with tabs[5]:
        show_model_comparison()


def show_configuration():
    """Global training configuration management"""
    st.markdown("### Training Configuration")
    st.markdown("Manage global configuration for data loading, model training, and artifact saving")

    config = st.session_state.training_config

    # Configuration sections
    config_section = st.radio(
        "Configuration Section:",
        ["Data Source", "Model Settings", "Training Scripts", "Artifacts & Saving", "View/Edit YAML"],
        horizontal=True
    )

    st.markdown("---")

    if config_section == "Data Source":
        show_data_source_config(config)
    elif config_section == "Model Settings":
        show_model_settings_config(config)
    elif config_section == "Training Scripts":
        show_scripts_config(config)
    elif config_section == "Artifacts & Saving":
        show_artifacts_config(config)
    else:
        show_yaml_config(config)


def show_data_source_config(config):
    """Configure data source settings"""
    st.markdown("#### Data Source Configuration")

    data_source = config.get('data_source', {})

    # Data source type
    source_type = st.selectbox(
        "Data Source Type:",
        ["clickhouse", "csv", "synthetic"],
        index=["clickhouse", "csv", "synthetic"].index(data_source.get('type', 'clickhouse'))
    )

    if source_type == "clickhouse":
        st.markdown("**ClickHouse Settings:**")

        col1, col2 = st.columns(2)

        with col1:
            database = st.text_input(
                "Database:",
                value=data_source.get('clickhouse', {}).get('database', 'public')
            )

            table = st.text_input(
                "Table:",
                value=data_source.get('clickhouse', {}).get('table', 'stixor_fraud_features_distributed')
            )

        with col2:
            limit = st.number_input(
                "Row Limit:",
                min_value=1000,
                max_value=10000000,
                value=data_source.get('clickhouse', {}).get('limit', 100000),
                step=10000
            )

        st.markdown("**Data Filters:**")

        # Date range filter
        date_filter = data_source.get('clickhouse', {}).get('filters', {}).get('date_range', {})
        date_enabled = st.checkbox("Enable Date Range Filter", value=date_filter.get('enabled', True))

        if date_enabled:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.text_input(
                    "Start Date (YYYY-MM-DD):",
                    value=date_filter.get('start_date', '2025-03-01')
                )
            with col2:
                end_date = st.text_input(
                    "End Date (YYYY-MM-DD):",
                    value=date_filter.get('end_date', '2025-06-30')
                )

        # Fraud filter
        fraud_filter = data_source.get('clickhouse', {}).get('filters', {}).get('fraud_filter', {})
        fraud_enabled = st.checkbox("Enable Fraud Filter", value=fraud_filter.get('enabled', False))

        if fraud_enabled:
            fraud_only = st.checkbox("Load Fraud Cases Only", value=fraud_filter.get('fraud_only', False))

        # Custom WHERE clause
        custom_filter = data_source.get('clickhouse', {}).get('filters', {}).get('custom_where', {})
        custom_enabled = st.checkbox("Enable Custom WHERE Clause", value=custom_filter.get('enabled', False))

        if custom_enabled:
            custom_clause = st.text_area(
                "Custom WHERE Clause:",
                value=custom_filter.get('clause', ''),
                help="Add custom SQL WHERE conditions"
            )

        if st.button("Save Data Source Config", type="primary"):
            # Update config
            config['data_source']['type'] = source_type
            config['data_source']['clickhouse']['database'] = database
            config['data_source']['clickhouse']['table'] = table
            config['data_source']['clickhouse']['limit'] = limit

            if date_enabled:
                config['data_source']['clickhouse']['filters']['date_range'] = {
                    'enabled': True,
                    'start_date': start_date,
                    'end_date': end_date
                }

            if fraud_enabled:
                config['data_source']['clickhouse']['filters']['fraud_filter'] = {
                    'enabled': True,
                    'fraud_only': fraud_only
                }

            if custom_enabled:
                config['data_source']['clickhouse']['filters']['custom_where'] = {
                    'enabled': True,
                    'clause': custom_clause
                }

            config_manager.save_config(config)
            st.session_state.training_config = config
            st.success("Data source configuration saved!")


def show_model_settings_config(config):
    """Configure model settings"""
    st.markdown("#### Model Settings Configuration")

    model_config = config.get('model', {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Type:**")
        model_type = st.selectbox(
            "Algorithm:",
            ["random_forest", "gradient_boosting", "logistic_regression", "xgboost", "lightgbm", "neural_network"],
            index=["random_forest", "gradient_boosting", "logistic_regression", "xgboost", "lightgbm", "neural_network"].index(
                model_config.get('type', 'random_forest')
            )
        )

    with col2:
        st.markdown("**Training Parameters:**")
        test_size = st.slider(
            "Test Size:",
            0.1, 0.5,
            model_config.get('training', {}).get('test_size', 0.2),
            0.05,
            key="config_test_size"
        )

        random_state = st.number_input(
            "Random State:",
            value=model_config.get('training', {}).get('random_state', 42)
        )

    st.markdown("---")
    st.markdown(f"**Hyperparameters for {model_type}:**")

    hyperparams = model_config.get('hyperparameters', {}).get(model_type, {})

    if model_type == "random_forest":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.number_input("n_estimators:", value=hyperparams.get('n_estimators', 100))
        with col2:
            max_depth = st.number_input("max_depth:", value=hyperparams.get('max_depth', 10))
        with col3:
            min_samples_split = st.number_input("min_samples_split:", value=hyperparams.get('min_samples_split', 2))

    if st.button("Save Model Config", type="primary"):
        config['model']['type'] = model_type
        config['model']['training']['test_size'] = test_size
        config['model']['training']['random_state'] = random_state

        if model_type == "random_forest":
            config['model']['hyperparameters']['random_forest']['n_estimators'] = n_estimators
            config['model']['hyperparameters']['random_forest']['max_depth'] = max_depth
            config['model']['hyperparameters']['random_forest']['min_samples_split'] = min_samples_split

        config_manager.save_config(config)
        st.session_state.training_config = config
        st.success("Model configuration saved!")


def show_scripts_config(config):
    """Configure training scripts"""
    st.markdown("#### Training Scripts Configuration")

    scripts_config = config.get('scripts', {})
    scripts_dir = scripts_config.get('directory', 'training_scripts')

    st.info(f"Training scripts directory: **{scripts_dir}**")

    # List available scripts
    available_scripts = config_manager.list_training_scripts()

    if available_scripts:
        st.markdown("**Available Training Scripts:**")

        for script in available_scripts:
            with st.expander(f"📄 {script}"):
                script_path = os.path.join(scripts_dir, script)

                # Show script info
                st.text(f"Path: {script_path}")

                # Check if custom config exists
                custom_scripts = scripts_config.get('custom_scripts', {})
                if script in custom_scripts:
                    st.text(f"Description: {custom_scripts[script].get('description', 'N/A')}")
                    st.text(f"Arguments: {', '.join(custom_scripts[script].get('args', []))}")
                else:
                    st.warning("No custom configuration. Using default arguments.")

    else:
        st.warning(f"No training scripts found in {scripts_dir}")
        st.info("Add your training scripts (.py files) to the training_scripts directory")

    st.markdown("---")
    st.markdown("**Default Script Arguments:**")

    default_args = scripts_config.get('default_args', {})

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.text_input("Model Type:", value=default_args.get('model_type', 'random_forest'))
        test_size = st.number_input("Test Size:", value=default_args.get('test_size', 0.2))

    with col2:
        epochs = st.number_input("Epochs:", value=default_args.get('epochs', 100))
        save_model = st.checkbox("Save Model:", value=default_args.get('save_model', True))

    if st.button("Save Scripts Config", type="primary"):
        config['scripts']['default_args']['model_type'] = model_type
        config['scripts']['default_args']['test_size'] = test_size
        config['scripts']['default_args']['epochs'] = epochs
        config['scripts']['default_args']['save_model'] = save_model

        config_manager.save_config(config)
        st.session_state.training_config = config
        st.success("Scripts configuration saved!")


def show_artifacts_config(config):
    """Configure artifacts and saving"""
    st.markdown("#### Artifacts & Saving Configuration")

    artifacts_config = config.get('artifacts', {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Saving:**")

        models_dir = st.text_input(
            "Models Directory:",
            value=artifacts_config.get('models', {}).get('save_dir', 'artifacts/models')
        )

        versioning = st.checkbox(
            "Enable Versioning:",
            value=artifacts_config.get('models', {}).get('versioning', True)
        )

        version_format = st.selectbox(
            "Version Format:",
            ["v{timestamp}", "v{date}", "v{counter}"],
            index=0
        )

        save_format = st.selectbox(
            "Save Format:",
            ["joblib", "pickle", "onnx"],
            index=0
        )

    with col2:
        st.markdown("**Logs & Metrics:**")

        logs_dir = st.text_input(
            "Logs Directory:",
            value=artifacts_config.get('logs', {}).get('save_dir', 'artifacts/logs')
        )

        save_logs = st.checkbox(
            "Save Training Logs:",
            value=artifacts_config.get('logs', {}).get('save_training_logs', True)
        )

        save_metrics = st.checkbox(
            "Save Metrics:",
            value=artifacts_config.get('logs', {}).get('save_metrics', True)
        )

    st.markdown("---")
    st.markdown("**Metadata:**")

    metadata_config = artifacts_config.get('metadata', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        save_config = st.checkbox("Save Config", value=metadata_config.get('save_config', True))
    with col2:
        save_features = st.checkbox("Save Features", value=metadata_config.get('save_features', True))
    with col3:
        save_schema = st.checkbox("Save Schema", value=metadata_config.get('save_schema', True))
    with col4:
        save_performance = st.checkbox("Save Performance", value=metadata_config.get('save_performance', True))

    if st.button("Save Artifacts Config", type="primary"):
        config['artifacts']['models']['save_dir'] = models_dir
        config['artifacts']['models']['versioning'] = versioning
        config['artifacts']['models']['version_format'] = version_format
        config['artifacts']['models']['save_format'] = save_format

        config['artifacts']['logs']['save_dir'] = logs_dir
        config['artifacts']['logs']['save_training_logs'] = save_logs
        config['artifacts']['logs']['save_metrics'] = save_metrics

        config['artifacts']['metadata']['save_config'] = save_config
        config['artifacts']['metadata']['save_features'] = save_features
        config['artifacts']['metadata']['save_schema'] = save_schema
        config['artifacts']['metadata']['save_performance'] = save_performance

        config_manager.save_config(config)
        st.session_state.training_config = config
        st.success("Artifacts configuration saved!")


def show_yaml_config(config):
    """View and edit raw YAML configuration"""
    st.markdown("#### View/Edit Configuration (YAML)")

    # Convert config to YAML string
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)

    # Editable text area
    edited_yaml = st.text_area(
        "Configuration YAML:",
        value=yaml_str,
        height=600,
        help="Edit the configuration directly in YAML format"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save Changes", type="primary"):
            try:
                new_config = yaml.safe_load(edited_yaml)
                config_manager.save_config(new_config)
                st.session_state.training_config = new_config
                st.success("Configuration saved successfully!")
            except Exception as e:
                st.error(f"Error parsing YAML: {str(e)}")

    with col2:
        if st.button("Reload from File"):
            st.session_state.training_config = config_manager.load_config()
            st.success("Configuration reloaded!")
            st.rerun()

    with col3:
        if st.button("Export to JSON"):
            json_str = config_manager.export_config_to_json()
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def show_custom_script_training():
    """Interface for running custom training scripts like train_all.py"""
    st.markdown("### Custom Training Script")
    st.markdown("Run your custom training scripts with configurable arguments from the **training_scripts** directory")

    # Get config
    config = st.session_state.training_config
    scripts_dir = config.get('scripts', {}).get('directory', 'training_scripts')

    # List available scripts
    available_scripts = config_manager.list_training_scripts()

    if not available_scripts:
        st.warning(f"No training scripts found in **{scripts_dir}** directory")
        st.info("Add your Python training scripts (.py files) to the training_scripts directory")

        # Option to upload script
        uploaded_script = st.file_uploader("Upload Training Script:", type=['py'])
        if uploaded_script:
            script_name = uploaded_script.name
            script_content = uploaded_script.read().decode('utf-8')

            st.code(script_content, language='python')

            if st.button("Save Script to training_scripts/"):
                os.makedirs(scripts_dir, exist_ok=True)
                script_path = os.path.join(scripts_dir, script_name)

                with open(script_path, 'w') as f:
                    f.write(script_content)

                st.success(f"Script saved to {script_path}")
                st.rerun()

        return

    # Select script
    selected_script = st.selectbox(
        "Select Training Script:",
        available_scripts,
        help="Choose a training script from the training_scripts directory"
    )

    script_path = os.path.join(scripts_dir, selected_script)
    script_exists = os.path.exists(script_path)

    if not script_exists:
        st.error(f"Script path invalid: {script_path}")
        return

    st.success(f"Using script: **{selected_script}**")

    st.markdown("---")
    st.markdown("#### Training Configuration")
    st.info("Using configuration from **Configuration** tab. Modify there to change defaults.")

    # Get config values
    model_config = config.get('model', {})
    default_args = config.get('scripts', {}).get('default_args', {})

    # Data source selection
    data_source = st.radio(
        "Data Source:",
        ["Use Loaded Data (from Data Loading module)", "Use Config Settings", "Specify Custom Path"],
        help="Choose where to load training data from"
    )

    data_path = None

    if data_source == "Use Config Settings":
        data_config = config.get('data_source', {})
        st.info(f"Using data from: {data_config.get('type', 'clickhouse')} - configured in Configuration tab")

        if data_config.get('type') == 'clickhouse':
            table = data_config.get('clickhouse', {}).get('table', 'stixor_fraud_features_distributed')
            st.text(f"Table: {table}")

    elif data_source == "Specify Custom Path":
        data_path = st.text_input("Data Path:", placeholder="/path/to/training_data.csv")

    elif data_source == "Use Loaded Data (from Data Loading module)":
        if st.session_state.get('loaded_data') is not None:
            st.info(f"Using data from: {st.session_state.get('data_source', 'Unknown')}")
            st.info(f"Rows: {len(st.session_state.loaded_data):,}")

            # Option to save loaded data
            if st.button("Save Loaded Data for Training"):
                # Use configured artifacts directory
                artifacts_dir = config.get('artifacts', {}).get('models', {}).get('save_dir', 'artifacts/models')
                os.makedirs(artifacts_dir, exist_ok=True)

                temp_path = os.path.join(artifacts_dir, f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                st.session_state.loaded_data.to_csv(temp_path, index=False)
                data_path = temp_path
                st.success(f"Data saved to: {temp_path}")
        else:
            st.warning("No data loaded. Please go to Data Loading module first.")

    st.markdown("---")
    st.markdown("#### Script Arguments")
    st.info("Default values loaded from Configuration. Customize here for this run only.")

    # Common training arguments
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Parameters:**")
        model_type = st.selectbox(
            "Model Type:",
            ["random_forest", "gradient_boosting", "logistic_regression",
             "xgboost", "lightgbm", "neural_network"],
            index=["random_forest", "gradient_boosting", "logistic_regression",
                   "xgboost", "lightgbm", "neural_network"].index(
                       model_config.get('type', 'random_forest')
                   ),
            help="Type of model to train"
        )

        test_size = st.slider(
            "Test Size:",
            0.1, 0.5,
            model_config.get('training', {}).get('test_size', 0.2),
            0.05,
            key="script_test_size"
        )

        random_state = st.number_input(
            "Random State:",
            0, 9999,
            model_config.get('training', {}).get('random_state', 42)
        )

    with col2:
        st.markdown("**Training Options:**")
        epochs = st.number_input(
            "Epochs/Iterations:",
            1, 1000,
            default_args.get('epochs', 100)
        )

        early_stopping = st.checkbox("Early Stopping", value=True)

        save_model = st.checkbox("Save Model", value=True)

    # Feature Selection
    st.markdown("---")
    st.markdown("#### Feature Selection")

    if st.session_state.get('data_features') and len(st.session_state.data_features) > 0:
        available_features = st.session_state.data_features

        st.info(f"Found {len(available_features)} features in loaded data")

        # Auto-detect target column
        possible_targets = ['fraud_flag', 'is_fraud', 'fraud', 'label', 'target', 'y']
        detected_target = None
        for target in possible_targets:
            if target in available_features:
                detected_target = target
                break

        col1, col2 = st.columns([3, 1])

        with col1:
            # Exclude columns that shouldn't be features
            exclude_cols = ['transaction_id', 'id', 'timestamp', 'date', 'cutoff_date',
                           'mbar_account_type_name', 'ac_to', 'ac_from', 'end_balance']
            if detected_target:
                exclude_cols.append(detected_target)

            feature_options = [f for f in available_features if f not in exclude_cols]

            selected_features = st.multiselect(
                "Select Features for Training:",
                options=feature_options,
                default=feature_options[:min(10, len(feature_options))],  # Select first 10 by default
                help="Choose which features to use for model training"
            )

        with col2:
            st.markdown("**Quick Actions:**")
            if st.button("Select All"):
                st.session_state.temp_selected_features = feature_options
                st.rerun()

            if st.button("Clear All"):
                st.session_state.temp_selected_features = []
                st.rerun()

        if detected_target:
            st.success(f"Target column detected: **{detected_target}**")
        else:
            st.warning("Target column not automatically detected. Make sure your data has 'is_fraud' or similar column.")

        if len(selected_features) == 0:
            st.error("Please select at least one feature")

        # Store selected features
        st.session_state.selected_training_features = selected_features

    else:
        st.warning("No feature information available. Load data first from Data Loading module.")
        selected_features = []

    # Custom arguments
    st.markdown("---")
    st.markdown("#### Additional Arguments")

    num_custom_args = st.number_input("Number of custom arguments:", 0, 20, 0)

    custom_args = {}
    for i in range(num_custom_args):
        col1, col2 = st.columns(2)
        with col1:
            arg_name = st.text_input(f"Argument {i+1} name:", key=f"arg_name_{i}",
                                     placeholder="--learning_rate")
        with col2:
            arg_value = st.text_input(f"Argument {i+1} value:", key=f"arg_value_{i}",
                                      placeholder="0.01")

        if arg_name and arg_value:
            custom_args[arg_name] = arg_value

    st.markdown("---")

    # Build command
    st.markdown("#### Command Preview")

    command_parts = ["python", script_path]

    if data_path:
        command_parts.extend(["--data_path", data_path])

    command_parts.extend([
        "--model_type", model_type,
        "--test_size", str(test_size),
        "--random_state", str(random_state),
        "--epochs", str(epochs)
    ])

    if early_stopping:
        command_parts.append("--early_stopping")

    if save_model:
        command_parts.append("--save_model")

    for arg_name, arg_value in custom_args.items():
        command_parts.extend([arg_name, arg_value])

    command = " ".join(command_parts)
    st.code(command, language='bash')

    st.markdown("---")

    # Run training
    col1, col2 = st.columns([1, 3])

    with col1:
        run_button = st.button("Start Training", type="primary", disabled=not script_exists)

    if run_button and script_exists:
        st.markdown("---")
        st.markdown("#### Training Output")

        output_container = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Run the training script
            process = subprocess.Popen(
                command_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                output_container.text_area(
                    "Training Log:",
                    value="".join(output_lines[-50:]),  # Show last 50 lines
                    height=400
                )

                # Update status
                if "epoch" in line.lower() or "iteration" in line.lower():
                    status_text.text(f"Training: {line.strip()}")

            process.wait()

            if process.returncode == 0:
                st.success("Training completed successfully!")

                # Save to history
                st.session_state.training_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'script': script_path,
                    'model_type': model_type,
                    'command': command,
                    'status': 'Success',
                    'output': "".join(output_lines)
                })

            else:
                st.error(f"Training failed with return code: {process.returncode}")

                st.session_state.training_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'script': script_path,
                    'model_type': model_type,
                    'command': command,
                    'status': 'Failed',
                    'output': "".join(output_lines)
                })

        except Exception as e:
            st.error(f"Error running training script: {str(e)}")


def show_builtin_training():
    """Built-in training with sklearn models"""
    st.markdown("### Built-in Model Training")
    st.markdown("Train models using built-in scikit-learn algorithms")

    # Check if data is available
    if st.session_state.get('loaded_data') is None:
        st.warning("No data loaded. Please load data from the Data Loading module first.")
        return

    df = st.session_state.loaded_data

    # Check required columns - support both fraud_flag and is_fraud
    target_col = None
    if 'fraud_flag' in df.columns:
        target_col = 'fraud_flag'
    elif 'is_fraud' in df.columns:
        target_col = 'is_fraud'
    else:
        st.error("Dataset must contain 'fraud_flag' or 'is_fraud' column")
        return

    st.success(f"Using data: {st.session_state.get('data_source')} ({len(df):,} rows)")
    st.info(f"Target column: **{target_col}**")

    # Check if cutoff_date exists for time-based split
    if 'cutoff_date' not in df.columns:
        st.error("Dataset must contain 'cutoff_date' column for time-based splitting")
        st.info("Please load data with cutoff_date column from ClickHouse")
        return

    st.markdown("---")

    # Time-based split configuration
    st.markdown("#### Time-Based Data Split")
    st.info("Specify date ranges for training and evaluation periods")

    # Get date range from data
    df['cutoff_date'] = pd.to_datetime(df['cutoff_date'])
    min_date = df['cutoff_date'].min().date()
    max_date = df['cutoff_date'].max().date()
    date_range_days = (max_date - min_date).days

    st.text(f"Available date range: {min_date} to {max_date} ({date_range_days} days)")

    # Calculate smart default values within the valid range
    if date_range_days == 0:
        # All data on same day - use same date for all
        default_train_end = max_date
        default_eval_start = max_date
        default_eval_end = max_date
        st.warning("⚠️ All data is on the same date. Time-based splitting not possible. Consider loading data with a wider date range.")
    elif date_range_days < 7:
        # Less than a week - split 70/30
        split_point = min_date + pd.Timedelta(days=int(date_range_days * 0.7))
        default_train_end = split_point
        default_eval_start = split_point + pd.Timedelta(days=1) if split_point < max_date else max_date
        default_eval_end = max_date
    elif date_range_days < 30:
        # Less than a month - split 80/20
        split_point = min_date + pd.Timedelta(days=int(date_range_days * 0.8))
        default_train_end = split_point
        default_eval_start = split_point + pd.Timedelta(days=1) if split_point < max_date else max_date
        default_eval_end = max_date
    else:
        # Sufficient data - use 90 days for training or 80% of range
        train_days = min(90, int(date_range_days * 0.8))
        default_train_end = min_date + pd.Timedelta(days=train_days)
        default_eval_start = default_train_end + pd.Timedelta(days=1) if default_train_end < max_date else max_date
        default_eval_end = max_date

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Training Period:**")
        train_start = st.date_input(
            "Training Start Date:",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key='train_start'
        )
        train_end = st.date_input(
            "Training End Date:",
            value=default_train_end,
            min_value=min_date,
            max_value=max_date,
            key='train_end'
        )

    with col2:
        st.markdown("**Evaluation Period:**")
        eval_start = st.date_input(
            "Evaluation Start Date:",
            value=default_eval_start,
            min_value=min_date,
            max_value=max_date,
            key='eval_start'
        )
        eval_end = st.date_input(
            "Evaluation End Date:",
            value=default_eval_end,
            min_value=min_date,
            max_value=max_date,
            key='eval_end'
        )

    # Validate date ranges
    if train_start >= train_end:
        st.error("Training end date must be after training start date")
        return

    if eval_start >= eval_end:
        st.error("Evaluation end date must be after evaluation start date")
        return

    if eval_start <= train_end:
        st.warning("⚠️ Warning: Evaluation period overlaps with training period")

    # Split data by time
    train_mask = (df['cutoff_date'].dt.date >= train_start) & (df['cutoff_date'].dt.date <= train_end)
    eval_mask = (df['cutoff_date'].dt.date >= eval_start) & (df['cutoff_date'].dt.date <= eval_end)

    df_train = df[train_mask]
    df_eval = df[eval_mask]

    st.text(f"Training samples: {len(df_train):,} | Evaluation samples: {len(df_eval):,}")

    if len(df_train) == 0:
        st.error("No training data in selected date range")
        return

    if len(df_eval) == 0:
        st.error("No evaluation data in selected date range")
        return

    st.markdown("---")

    # Feature selection
    st.markdown("#### Feature Selection")

    # Exclude non-feature columns
    exclude_cols = ['transaction_id', 'id', 'timestamp', 'date', 'cutoff_date',
                   'mbar_account_type_name', 'ac_to', 'ac_from', 'end_balance', target_col]

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_options = [col for col in numeric_cols if col not in exclude_cols]

    # Enhanced feature selection UI
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**Available Features:** {len(feature_options)} features")

        # Search/filter features
        search_term = st.text_input(
            "🔍 Search features:",
            placeholder="e.g., txn_, user_, channel_",
            help="Filter features by name",
            key='feature_search'
        )

        if search_term:
            filtered_features = [f for f in feature_options if search_term.lower() in f.lower()]
            st.caption(f"Showing {len(filtered_features)} features matching '{search_term}'")
        else:
            filtered_features = feature_options

        # Group features by prefix
        feature_groups = {}
        for feat in filtered_features:
            prefix = feat.split('_')[0] if '_' in feat else 'other'
            if prefix not in feature_groups:
                feature_groups[prefix] = []
            feature_groups[prefix].append(feat)

        # Show feature groups
        st.markdown("**Feature Groups:**")
        for prefix, features in sorted(feature_groups.items()):
            st.text(f"  {prefix}_*: {len(features)} features")

    with col2:
        st.markdown("**Quick Actions:**")
        if st.button("✓ Select All", use_container_width=True, key='select_all_builtin'):
            st.session_state.selected_features_builtin = filtered_features
            st.rerun()

        if st.button("✗ Clear All", use_container_width=True, key='clear_all_builtin'):
            st.session_state.selected_features_builtin = []
            st.rerun()

        # Group selection
        st.markdown("**Select by Group:**")
        for prefix in sorted(feature_groups.keys()):
            if st.button(f"+ {prefix}_*", use_container_width=True, key=f"group_{prefix}_builtin"):
                current = st.session_state.get('selected_features_builtin', [])
                st.session_state.selected_features_builtin = list(set(current + feature_groups[prefix]))
                st.rerun()

    # Main feature multiselect
    selected_features = st.multiselect(
        "Selected Features:",
        options=filtered_features,
        default=st.session_state.get('selected_features_builtin', []),
        help="Select features to use for training",
        key='feature_multiselect_builtin'
    )

    # Store selection
    st.session_state.selected_features_builtin = selected_features

    if not selected_features:
        st.warning("Please select at least one feature")
        return

    st.info(f"**{len(selected_features)}** features selected for training")

    st.markdown("---")

    # Model configuration
    st.markdown("#### Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        algorithm = st.selectbox(
            "Algorithm:",
            ["Random Forest", "Gradient Boosting", "Logistic Regression"]
        )

    with col2:
        st.markdown("**Hyperparameters:**")
        # NO DEFAULTS - user must provide values
        if algorithm == "Random Forest":
            n_estimators = st.number_input("Number of Trees:", min_value=10, max_value=1000, value=None, placeholder="e.g., 100", key='rf_n_est')
            max_depth = st.number_input("Max Depth:", min_value=1, max_value=50, value=None, placeholder="e.g., 10", key='rf_max_depth')

            if n_estimators is None or max_depth is None:
                st.warning("⚠️ Please provide all hyperparameters")

        elif algorithm == "Gradient Boosting":
            n_estimators = st.number_input("Number of Trees:", min_value=10, max_value=1000, value=None, placeholder="e.g., 100", key='gb_n_est')
            learning_rate = st.number_input("Learning Rate:", min_value=0.001, max_value=1.0, value=None, placeholder="e.g., 0.1", format="%.3f", key='gb_lr')

            if n_estimators is None or learning_rate is None:
                st.warning("⚠️ Please provide all hyperparameters")

        elif algorithm == "Logistic Regression":
            C = st.number_input("Regularization (C):", min_value=0.001, max_value=100.0, value=None, placeholder="e.g., 1.0", format="%.3f", key='lr_C')
            max_iter = st.number_input("Max Iterations:", min_value=100, max_value=10000, value=None, placeholder="e.g., 1000", key='lr_max_iter')

            if C is None or max_iter is None:
                st.warning("⚠️ Please provide all hyperparameters")

    st.markdown("---")

    # Validate hyperparameters before training
    params_valid = False
    if algorithm == "Random Forest":
        params_valid = (n_estimators is not None and max_depth is not None)
    elif algorithm == "Gradient Boosting":
        params_valid = (n_estimators is not None and learning_rate is not None)
    elif algorithm == "Logistic Regression":
        params_valid = (C is not None and max_iter is not None)

    if not params_valid:
        st.info("👆 Please provide all required hyperparameters to enable training")

    if st.button("Train Model", type="primary", disabled=not params_valid):
        with st.spinner("Training model..."):
            try:
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

                # Prepare training data
                X_train = df_train[selected_features]
                y_train = df_train[target_col]

                # Prepare evaluation data
                X_eval = df_eval[selected_features]
                y_eval = df_eval[target_col]

                # Train model
                if algorithm == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        random_state=42
                    )
                elif algorithm == "Gradient Boosting":
                    model = GradientBoostingClassifier(
                        n_estimators=int(n_estimators),
                        learning_rate=float(learning_rate),
                        random_state=42
                    )
                elif algorithm == "Logistic Regression":
                    model = LogisticRegression(
                        C=float(C),
                        max_iter=int(max_iter),
                        random_state=42
                    )

                model.fit(X_train, y_train)

                # Evaluate ONLY on evaluation data
                y_pred = model.predict(X_eval)
                y_pred_proba = model.predict_proba(X_eval)[:, 1]

                st.success("Model trained successfully!")

                # Display results
                st.markdown("---")
                st.markdown("#### Training & Evaluation Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Training Samples", f"{len(X_train):,}")

                with col2:
                    st.metric("Evaluation Samples", f"{len(X_eval):,}")

                with col3:
                    roc_auc = roc_auc_score(y_eval, y_pred_proba)
                    st.metric("ROC-AUC (Eval)", f"{roc_auc:.4f}")

                with col4:
                    fraud_rate = (y_eval.sum() / len(y_eval) * 100)
                    st.metric("Fraud Rate (Eval)", f"{fraud_rate:.2f}%")

                # Show period information
                st.markdown("**Training Period:** {} to {}".format(train_start, train_end))
                st.markdown("**Evaluation Period:** {} to {}".format(eval_start, eval_end))

                # Classification report (ONLY on evaluation data)
                st.markdown("---")
                st.markdown("**Classification Report (Evaluation Data):**")
                report = classification_report(y_eval, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)

                # Feature Importance
                st.markdown("---")
                st.markdown("#### Feature Importance")

                if hasattr(model, 'feature_importances_'):
                    import plotly.express as px

                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    fig = px.bar(importance_df.head(20), x='Importance', y='Feature',
                                orientation='h', title="Top 20 Features by Importance")
                    st.plotly_chart(fig, use_container_width=True)

                    # Save feature importance
                    feature_importance_data = importance_df
                else:
                    feature_importance_data = None

                # SHAP Analysis
                st.markdown("---")
                st.markdown("#### SHAP Analysis")
                st.info("Computing SHAP values for model explainability...")

                shap_values = None
                shap_sample_data = None

                try:
                    import shap

                    # Sample data for SHAP (use max 100 samples for performance from evaluation data)
                    shap_sample = X_eval.sample(n=min(100, len(X_eval)), random_state=42)

                    # Create explainer based on model type
                    if algorithm in ["Random Forest", "Gradient Boosting"]:
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(shap_sample)

                        # For binary classification, take the positive class
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[1]

                    else:  # Logistic Regression
                        explainer = shap.LinearExplainer(model, X_train)
                        shap_vals = explainer.shap_values(shap_sample)

                    # Summary plot
                    st.markdown("**SHAP Summary Plot:**")
                    fig_summary = shap.summary_plot(shap_vals, shap_sample, plot_type="bar",
                                                     show=False, max_display=15)
                    st.pyplot(fig_summary.figure, clear_figure=True)

                    # Detailed SHAP plot
                    st.markdown("**SHAP Detailed Plot:**")
                    fig_detailed = shap.summary_plot(shap_vals, shap_sample, show=False, max_display=15)
                    st.pyplot(fig_detailed.figure, clear_figure=True)

                    st.success("SHAP analysis completed!")

                    shap_values = shap_vals
                    shap_sample_data = shap_sample

                except Exception as e:
                    st.warning(f"SHAP analysis failed: {str(e)}. Continuing without SHAP...")

                # Save artifacts
                st.markdown("---")
                st.markdown("#### Save Artifacts")

                config = st.session_state.training_config
                artifacts_dir = config.get('artifacts', {}).get('models', {}).get('save_dir', 'artifacts/models')
                logs_dir = config.get('artifacts', {}).get('logs', {}).get('save_dir', 'artifacts/logs')

                import os
                import joblib

                os.makedirs(artifacts_dir, exist_ok=True)
                os.makedirs(logs_dir, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_name = f"{algorithm.lower().replace(' ', '_')}_{timestamp}"

                # Save model
                model_path = os.path.join(artifacts_dir, f"{model_name}.joblib")
                joblib.dump(model, model_path)

                # Save feature importance
                if feature_importance_data is not None:
                    importance_path = os.path.join(artifacts_dir, f"{model_name}_feature_importance.csv")
                    feature_importance_data.to_csv(importance_path, index=False)

                # Save metrics
                metrics_path = os.path.join(logs_dir, f"{model_name}_metrics.json")
                import json

                # Build hyperparameters dict
                hyperparams = {}
                if algorithm == "Random Forest":
                    hyperparams = {'n_estimators': int(n_estimators), 'max_depth': int(max_depth)}
                elif algorithm == "Gradient Boosting":
                    hyperparams = {'n_estimators': int(n_estimators), 'learning_rate': float(learning_rate)}
                elif algorithm == "Logistic Regression":
                    hyperparams = {'C': float(C), 'max_iter': int(max_iter)}

                with open(metrics_path, 'w') as f:
                    json.dump({
                        'algorithm': algorithm,
                        'features': selected_features,
                        'roc_auc': roc_auc,
                        'training_period': {
                            'start': str(train_start),
                            'end': str(train_end),
                            'samples': len(X_train)
                        },
                        'evaluation_period': {
                            'start': str(eval_start),
                            'end': str(eval_end),
                            'samples': len(X_eval)
                        },
                        'fraud_rate_eval': float(fraud_rate),
                        'trained_at': timestamp,
                        'classification_report': report,
                        'hyperparameters': hyperparams
                    }, f, indent=2)

                # Save SHAP values
                if shap_values is not None:
                    shap_path = os.path.join(artifacts_dir, f"{model_name}_shap_values.npy")
                    np.save(shap_path, shap_values)

                st.success(f"✓ Artifacts saved successfully!")
                st.info(f"Model: {model_path}")
                st.info(f"Metrics: {metrics_path}")
                if feature_importance_data is not None:
                    st.info(f"Feature Importance: {importance_path}")
                if shap_values is not None:
                    st.info(f"SHAP Values: {shap_path}")

                # Log to MLflow
                st.markdown("---")
                st.markdown("#### MLflow Experiment Tracking")
                with st.spinner("Logging experiment to MLflow..."):
                    try:
                        tracker = MLflowTracker()
                        run_name = f"{algorithm.lower().replace(' ', '_')}_{timestamp}"
                        tracker.start_run(run_name=run_name)

                        # Log parameters
                        params = {
                            'algorithm': algorithm,
                            'train_samples': len(X_train),
                            'eval_samples': len(X_eval),
                            'n_features': len(selected_features),
                            'train_period_start': str(train_start),
                            'train_period_end': str(train_end),
                            'eval_period_start': str(eval_start),
                            'eval_period_end': str(eval_end),
                            **hyperparams
                        }
                        tracker.log_params(params)

                        # Log metrics
                        metrics = {
                            'roc_auc': roc_auc,
                            'precision': report['weighted avg']['precision'],
                            'recall': report['weighted avg']['recall'],
                            'f1_score': report['weighted avg']['f1-score'],
                            'accuracy': report['accuracy'],
                            'fraud_rate': float(fraud_rate)
                        }
                        tracker.log_metrics(metrics)

                        # Log model
                        model_type_map = {
                            'Random Forest': 'random_forest',
                            'Gradient Boosting': 'gradient_boosting',
                            'Logistic Regression': 'logistic_regression'
                        }
                        tracker.log_model(model, model_type_map.get(algorithm, 'sklearn'))

                        # Log artifacts
                        tracker.log_artifact(metrics_path)
                        if feature_importance_data is not None:
                            tracker.log_artifact(importance_path)

                        # Set tags
                        tracker.set_tags({
                            'model_name': model_name,
                            'data_source': 'clickhouse',
                            'training_type': 'time_based_split'
                        })

                        tracker.end_run()
                        st.success("✓ Experiment logged to MLflow successfully!")

                    except Exception as e:
                        st.warning(f"MLflow logging failed: {str(e)}. Model saved locally.")

                # Save model to session
                st.session_state.current_model = {
                    'model': model,
                    'features': selected_features,
                    'algorithm': algorithm,
                    'metrics': {
                        'roc_auc': roc_auc,
                        'report': report
                    },
                    'trained_at': datetime.now(),
                    'model_path': model_path,
                    'feature_importance': feature_importance_data,
                    'shap_values': shap_values
                }

                # Store feature importance in session for feature monitoring
                if feature_importance_data is not None:
                    st.session_state.feature_importance = dict(zip(
                        feature_importance_data['Feature'],
                        feature_importance_data['Importance']
                    ))
                    st.session_state.trained_model = model

                st.info("Model saved to session state and artifacts directory!")

            except Exception as e:
                st.error(f"Training error: {str(e)}")


def show_model_selection():
    """Select and load trained models"""
    st.markdown("### Model Selection & Management")
    st.markdown("Load and manage trained models from artifacts directory or custom paths")

    # Get config
    config = st.session_state.training_config
    default_models_dir = config.get('artifacts', {}).get('models', {}).get('save_dir', 'artifacts/models')

    # Model source selection
    st.markdown("---")
    st.markdown("#### Model Source")

    col1, col2 = st.columns([2, 1])

    with col1:
        source_option = st.radio(
            "Select model source:",
            ["Default Artifacts Directory", "Custom Path"],
            horizontal=True,
            help="Choose where to load models from"
        )

    if source_option == "Default Artifacts Directory":
        models_dir = default_models_dir
    else:
        models_dir = st.text_input(
            "Custom Models Directory:",
            value="/root/research-dir/dev/jazzcash-fraud-detection/models",
            help="Enter path to models directory"
        )

    st.info(f"📁 Current directory: `{models_dir}`")

    # Scan for models
    st.markdown("---")
    st.markdown("#### Available Models")

    try:
        if not os.path.exists(models_dir):
            st.warning(f"Directory `{models_dir}` does not exist")
            return

        # Find all model files
        import glob
        import joblib

        model_files = []
        for ext in ['*.joblib', '*.pkl', '*.pickle']:
            model_files.extend(glob.glob(os.path.join(models_dir, ext)))

        if len(model_files) == 0:
            st.info("No models found in the specified directory")
            st.markdown("**Tip:** Train a model using the 'Built-in Training' tab first")
            return

        st.success(f"Found {len(model_files)} model(s)")

        # Display models with metadata
        for model_file in sorted(model_files, reverse=True):
            model_name = os.path.basename(model_file)
            model_name_no_ext = os.path.splitext(model_name)[0]

            with st.expander(f"📊 {model_name}", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.markdown("**Model Information:**")
                    st.text(f"File: {model_name}")

                    # Get file size
                    file_size = os.path.getsize(model_file)
                    size_mb = file_size / (1024 * 1024)
                    st.text(f"Size: {size_mb:.2f} MB")

                    # Get modification time
                    mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                    st.text(f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

                with col2:
                    # Try to load metadata
                    st.markdown("**Metadata:**")

                    # Check for accompanying metrics file
                    logs_dir = config.get('artifacts', {}).get('logs', {}).get('save_dir', 'artifacts/logs')
                    metrics_file = os.path.join(logs_dir, f"{model_name_no_ext}_metrics.json")

                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                metadata = json.load(f)

                            st.text(f"Algorithm: {metadata.get('algorithm', 'Unknown')}")
                            st.text(f"ROC-AUC: {metadata.get('roc_auc', 'N/A'):.4f}")
                            st.text(f"Features: {len(metadata.get('features', []))}")

                            # Training period
                            train_period = metadata.get('training_period', {})
                            if train_period:
                                st.text(f"Train samples: {train_period.get('samples', 'N/A')}")

                            # Hyperparameters
                            hyperparams = metadata.get('hyperparameters', {})
                            if hyperparams:
                                st.markdown("**Hyperparameters:**")
                                for key, value in hyperparams.items():
                                    st.text(f"  {key}: {value}")

                        except Exception as e:
                            st.warning(f"Could not load metadata: {str(e)}")
                    else:
                        st.text("No metadata file found")

                with col3:
                    st.markdown("**Actions:**")

                    if st.button("Load Model", key=f"load_{model_name}"):
                        try:
                            # Load the model
                            with st.spinner("Loading model..."):
                                model = joblib.load(model_file)

                            # Load metadata if available
                            metadata = {}
                            if os.path.exists(metrics_file):
                                with open(metrics_file, 'r') as f:
                                    metadata = json.load(f)

                            # Load feature importance if available
                            importance_file = os.path.join(models_dir, f"{model_name_no_ext}_feature_importance.csv")
                            feature_importance = None
                            if os.path.exists(importance_file):
                                feature_importance = pd.read_csv(importance_file)

                            # Save to session state
                            st.session_state.current_model = {
                                'model': model,
                                'features': metadata.get('features', []),
                                'algorithm': metadata.get('algorithm', 'Unknown'),
                                'metrics': {
                                    'roc_auc': metadata.get('roc_auc', None),
                                    'report': metadata.get('classification_report', {})
                                },
                                'trained_at': metadata.get('trained_at', 'Unknown'),
                                'model_path': model_file,
                                'feature_importance': feature_importance,
                                'shap_values': None
                            }

                            # Store feature importance for monitoring
                            if feature_importance is not None:
                                st.session_state.feature_importance = dict(zip(
                                    feature_importance['Feature'],
                                    feature_importance['Importance']
                                ))
                                st.session_state.trained_model = model

                            st.success(f"✓ Model loaded successfully: {model_name}")
                            st.info("Model is now available in session state for inference")

                        except Exception as e:
                            st.error(f"Failed to load model: {str(e)}")

                    if st.button("View Details", key=f"view_{model_name}"):
                        if os.path.exists(metrics_file):
                            with open(metrics_file, 'r') as f:
                                metadata = json.load(f)

                            st.markdown("---")
                            st.markdown("##### Full Metadata:")
                            st.json(metadata)
                        else:
                            st.warning("No detailed metadata available")

        # MLflow Experiments Section
        st.markdown("---")
        st.markdown("#### MLflow Experiment Runs")

        try:
            runs_df = MLflowTracker.get_experiment_runs()

            if runs_df is not None and len(runs_df) > 0:
                st.success(f"Found {len(runs_df)} MLflow experiment run(s)")

                # Display key columns
                display_cols = ['run_id', 'start_time', 'metrics.roc_auc', 'metrics.f1_score',
                               'params.algorithm', 'tags.model_name']
                available_cols = [col for col in display_cols if col in runs_df.columns]

                if available_cols:
                    st.dataframe(runs_df[available_cols].head(10), use_container_width=True)

                with st.expander("View all runs"):
                    st.dataframe(runs_df, use_container_width=True)
            else:
                st.info("No MLflow experiments found. Train a model to create experiment runs.")

        except Exception as e:
            st.warning(f"Could not load MLflow experiments: {str(e)}")

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")


def show_training_history():
    """Display training history"""
    st.markdown("### Training History")

    if not st.session_state.training_history:
        st.info("No training history yet. Train a model to see history here.")
        return

    history_df = pd.DataFrame(st.session_state.training_history)

    st.dataframe(history_df[['timestamp', 'model_type', 'status']], use_container_width=True)

    st.markdown("---")

    # View details
    selected_idx = st.selectbox(
        "Select training run to view details:",
        range(len(st.session_state.training_history)),
        format_func=lambda x: f"{st.session_state.training_history[x]['timestamp']} - {st.session_state.training_history[x]['model_type']}"
    )

    if selected_idx is not None:
        run = st.session_state.training_history[selected_idx]

        st.markdown("#### Training Details")

        col1, col2 = st.columns(2)

        with col1:
            st.text(f"Timestamp: {run['timestamp']}")
            st.text(f"Model Type: {run['model_type']}")
            st.text(f"Status: {run['status']}")

        with col2:
            st.text(f"Script: {run['script']}")

        st.markdown("**Command:**")
        st.code(run['command'], language='bash')

        st.markdown("**Output:**")
        st.text_area("Training Output:", value=run['output'], height=400)


def show_model_comparison():
    """Compare multiple trained models"""
    st.markdown("### Model Comparison")

    st.info("Feature coming soon: Compare performance of multiple trained models")
