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
sys.path.append('/home/user/ArgusAI')


def show():
    st.markdown('<p class="main-header">Model Training & Experimentation</p>', unsafe_allow_html=True)
    st.markdown("Train fraud detection models using your custom training scripts or built-in algorithms")

    # Initialize session state
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []

    if 'current_model' not in st.session_state:
        st.session_state.current_model = None

    # Create tabs
    tabs = st.tabs([
        "Custom Training Script",
        "Built-in Training",
        "Training History",
        "Model Comparison"
    ])

    with tabs[0]:
        show_custom_script_training()

    with tabs[1]:
        show_builtin_training()

    with tabs[2]:
        show_training_history()

    with tabs[3]:
        show_model_comparison()


def show_custom_script_training():
    """Interface for running custom training scripts like train_all.py"""
    st.markdown("### Custom Training Script")
    st.markdown("Run your custom training script (e.g., train_all.py) with configurable arguments")

    # Check if training script exists
    script_path = st.text_input(
        "Training Script Path:",
        value="train_all.py",
        help="Path to your training script (e.g., train_all.py)"
    )

    script_exists = os.path.exists(script_path)

    if script_exists:
        st.success(f"Script found: {script_path}")
    else:
        st.warning(f"Script not found: {script_path}")
        st.info("Please provide the correct path to your training script or upload it below")

        uploaded_script = st.file_uploader("Upload Training Script:", type=['py'])
        if uploaded_script:
            script_content = uploaded_script.read().decode('utf-8')
            st.code(script_content, language='python')

            if st.button("Save Script"):
                with open(script_path, 'w') as f:
                    f.write(script_content)
                st.success(f"Script saved to {script_path}")
                st.rerun()

    st.markdown("---")
    st.markdown("#### Training Configuration")

    # Data source selection
    data_source = st.radio(
        "Data Source:",
        ["Use Loaded Data (from Data Loading module)", "Specify Data Path", "Use ClickHouse Query"],
        help="Choose where to load training data from"
    )

    data_path = None
    if data_source == "Specify Data Path":
        data_path = st.text_input("Data Path:", placeholder="/path/to/training_data.csv")
    elif data_source == "Use Loaded Data (from Data Loading module)":
        if st.session_state.get('loaded_data') is not None:
            st.info(f"Using data from: {st.session_state.get('data_source', 'Unknown')}")
            st.info(f"Rows: {len(st.session_state.loaded_data):,}")

            # Option to save loaded data
            if st.button("Save Loaded Data for Training"):
                temp_path = f"/tmp/training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.session_state.loaded_data.to_csv(temp_path, index=False)
                data_path = temp_path
                st.success(f"Data saved to: {temp_path}")
        else:
            st.warning("No data loaded. Please go to Data Loading module first.")

    st.markdown("---")
    st.markdown("#### Script Arguments")

    # Common training arguments
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Parameters:**")
        model_type = st.selectbox(
            "Model Type:",
            ["random_forest", "gradient_boosting", "logistic_regression",
             "xgboost", "lightgbm", "neural_network"],
            help="Type of model to train"
        )

        test_size = st.slider("Test Size:", 0.1, 0.5, 0.2, 0.05)

        random_state = st.number_input("Random State:", 0, 9999, 42)

    with col2:
        st.markdown("**Training Options:**")
        epochs = st.number_input("Epochs/Iterations:", 1, 1000, 100)

        early_stopping = st.checkbox("Early Stopping", value=True)

        save_model = st.checkbox("Save Model", value=True)

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

    # Check required columns
    if 'is_fraud' not in df.columns:
        st.error("Dataset must contain 'is_fraud' column")
        return

    st.success(f"Using data: {st.session_state.get('data_source')} ({len(df):,} rows)")

    st.markdown("---")

    # Feature selection
    st.markdown("#### Feature Selection")

    all_cols = df.columns.tolist()
    all_cols.remove('is_fraud')

    # Remove non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_fraud' in numeric_cols:
        numeric_cols.remove('is_fraud')

    selected_features = st.multiselect(
        "Select Features:",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))]
    )

    if not selected_features:
        st.warning("Please select at least one feature")
        return

    st.markdown("---")

    # Model configuration
    st.markdown("#### Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        algorithm = st.selectbox(
            "Algorithm:",
            ["Random Forest", "Gradient Boosting", "Logistic Regression", "XGBoost"]
        )

        test_size = st.slider("Test Split:", 0.1, 0.5, 0.2, 0.05)

    with col2:
        # Algorithm-specific parameters
        if algorithm == "Random Forest":
            n_estimators = st.number_input("Number of Trees:", 10, 1000, 100)
            max_depth = st.number_input("Max Depth:", 1, 50, 10)
        elif algorithm == "Gradient Boosting":
            n_estimators = st.number_input("Number of Trees:", 10, 1000, 100)
            learning_rate = st.number_input("Learning Rate:", 0.001, 1.0, 0.1, 0.01)
        elif algorithm == "Logistic Regression":
            C = st.number_input("Regularization (C):", 0.001, 100.0, 1.0, 0.1)
            max_iter = st.number_input("Max Iterations:", 100, 10000, 1000)

    st.markdown("---")

    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

                # Prepare data
                X = df[selected_features]
                y = df['is_fraud']

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                # Train model
                if algorithm == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                  random_state=42)
                elif algorithm == "Gradient Boosting":
                    model = GradientBoostingClassifier(n_estimators=n_estimators,
                                                       learning_rate=learning_rate,
                                                       random_state=42)
                elif algorithm == "Logistic Regression":
                    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                st.success("Model trained successfully!")

                # Display results
                st.markdown("---")
                st.markdown("#### Training Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Training Samples", len(X_train))

                with col2:
                    st.metric("Test Samples", len(X_test))

                with col3:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    st.metric("ROC-AUC", f"{roc_auc:.4f}")

                # Classification report
                st.markdown("**Classification Report:**")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)

                # Save model
                st.session_state.current_model = {
                    'model': model,
                    'features': selected_features,
                    'algorithm': algorithm,
                    'metrics': {
                        'roc_auc': roc_auc,
                        'report': report
                    },
                    'trained_at': datetime.now()
                }

                st.info("Model saved to session state. You can use it for inference!")

            except Exception as e:
                st.error(f"Training error: {str(e)}")


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
