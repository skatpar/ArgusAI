"""
Enhanced Real-time Inference with Model Selection and Explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
sys.path.append('/home/user/ArgusAI')

from src.utils.api_client import call_api_inference
from src.utils.model_artifacts import ModelArtifactsLoader
from src.utils.mlflow_tracker import MLflowTracker
from src.utils.settings import SettingsManager


def show_single_transaction_enhanced():
    """Enhanced single transaction inference with model selection and explainability"""
    st.markdown("### Single Transaction Inference")
    st.markdown("Score individual transactions with model selection and explainability")

    # Model Selection Section
    st.markdown("---")
    st.markdown("#### Model Selection")

    # Model source selection
    model_source = st.radio(
        "Model Source:",
        options=["Local Files", "MLflow Runs"],
        horizontal=True,
        help="Choose to load models from local files or MLflow experiment runs"
    )

    feature_importance = None
    shap_values = None
    selected_model_id = None
    selected_run_id = None
    selected_model_name = None
    model_metadata = {}

    if model_source == "Local Files":
        # Initialize model artifacts loader
        artifacts_loader = ModelArtifactsLoader()

        col1, col2 = st.columns([2, 1])

        with col1:
            # Get available models
            available_models = artifacts_loader.list_available_models()

            if not available_models:
                st.warning("No trained models found in models directory")
                st.info("Train models using the Model Training page first")
                return

            model_options = {m['model_name']: m['model_id'] for m in available_models}

            selected_model_name = st.selectbox(
                "Select Model:",
                options=list(model_options.keys()),
                help="Choose the model to use for inference",
                key="local_model_select"
            )

            selected_model_id = model_options[selected_model_name]

        with col2:
            # Model metadata
            model_metadata = artifacts_loader.get_model_metadata(selected_model_id)
            if model_metadata:
                st.metric("Model Type", model_metadata.get('model_type', 'Unknown'))
                if model_metadata.get('created_date'):
                    st.caption(f"Created: {model_metadata['created_date']}")

        # Load feature importance for selected model
        feature_importance = artifacts_loader.load_feature_importance(selected_model_id)

    else:  # MLflow Runs
        # Get MLflow settings
        try:
            settings_mgr = SettingsManager()
            mlflow_settings = settings_mgr.get('mlflow', {})

            tracking_uri = mlflow_settings.get('tracking_uri', 'http://localhost:5001')
            experiment_name = mlflow_settings.get('experiment_name', 'fraud_detection_pipeline')

            st.info(f"🔗 MLflow Tracking URI: `{tracking_uri}` | Experiment: `{experiment_name}`")

            # Fetch runs from MLflow
            with st.spinner("Fetching runs from MLflow..."):
                runs_df = MLflowTracker.get_runs_by_experiment(tracking_uri, experiment_name)

            if runs_df is None or runs_df.empty:
                st.warning(f"No runs found in MLflow experiment '{experiment_name}'")
                st.info("Train models using the training script with MLflow integration")
                return

            # Filter for finished runs
            runs_df = runs_df[runs_df['status'] == 'FINISHED']

            if runs_df.empty:
                st.warning("No finished runs found")
                return

            col1, col2 = st.columns([2, 1])

            with col1:
                # Create run options (run_name + timestamp)
                run_options = {}
                for idx, row in runs_df.iterrows():
                    run_name = row.get('tags.mlflow.runName', 'Unknown')
                    run_id = row['run_id']
                    model_type = row.get('params.model_type', 'Unknown')
                    start_time = pd.to_datetime(row['start_time']).strftime('%Y-%m-%d %H:%M')

                    display_name = f"{run_name} ({model_type}) - {start_time}"
                    run_options[display_name] = {
                        'run_id': run_id,
                        'run_name': run_name,
                        'model_type': model_type,
                        'metrics': {
                            'auc': row.get('metrics.auc_roc', 0),
                            'f1': row.get('metrics.f1_score', 0),
                            'precision': row.get('metrics.precision', 0)
                        }
                    }

                selected_run_display = st.selectbox(
                    "Select MLflow Run:",
                    options=list(run_options.keys()),
                    help="Choose an MLflow run to use for inference",
                    key="mlflow_run_select"
                )

                selected_run_info = run_options[selected_run_display]
                selected_run_id = selected_run_info['run_id']
                selected_model_name = selected_run_info['run_name']

            with col2:
                # Display run metrics
                metrics = selected_run_info['metrics']
                st.metric("Model Type", selected_run_info['model_type'])
                st.metric("AUC", f"{metrics['auc']:.4f}")
                st.metric("F1 Score", f"{metrics['f1']:.4f}")

            # Load feature importance from MLflow
            with st.spinner("Loading artifacts from MLflow..."):
                feature_importance = MLflowTracker.load_feature_importance_from_run(
                    tracking_uri, selected_run_id
                )

                # Try to load SHAP values if available
                try:
                    shap_values = MLflowTracker.load_shap_values_from_run(
                        tracking_uri, selected_run_id
                    )
                    if shap_values is not None:
                        st.success("✓ SHAP values loaded")
                except Exception as e:
                    st.info("SHAP values not available for this run")

        except Exception as e:
            st.error(f"Error connecting to MLflow: {str(e)}")
            st.info("Please check your MLflow settings in the Settings page")
            return

    # Display model/run information
    if model_source == "MLflow Runs" and selected_run_id:
        st.caption(f"🔗 Run ID: `{selected_run_id[:8]}...` | Model: {selected_model_name}")
    elif model_source == "Local Files" and selected_model_id:
        st.caption(f"📁 Model ID: `{selected_model_id}`")

    if feature_importance is not None and not feature_importance.empty:
        with st.expander("View Feature Importance for Selected Model", expanded=False):
            st.markdown(f"**Top 20 Important Features for {selected_model_name}**")

            if model_source == "MLflow Runs":
                st.info("✓ Feature importance loaded from MLflow artifacts")

            # Create horizontal bar chart
            top_features = feature_importance.head(20).sort_values('importance', ascending=True)

            fig = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Viridis',
                    showscale=True
                )
            ))

            fig.update_layout(
                title=f"Feature Importance - {selected_model_name}",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=500,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show table
            st.dataframe(
                feature_importance.head(20),
                use_container_width=True,
                hide_index=True
            )

    # Check API configuration
    api_configured = bool(st.session_state.api_config.get('endpoint_url'))

    if not api_configured:
        st.warning("API not configured. Please configure in API Configuration tab.")
        return

    st.markdown("---")
    st.markdown("#### Transaction Scoring")

    # Transaction ID input
    col1, col2 = st.columns([3, 1])

    with col1:
        transaction_id = st.text_input(
            "Transaction ID:",
            placeholder="Enter transaction ID (e.g., 83881056341)",
            key="enhanced_tx_id"
        )

    with col2:
        st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
        score_button = st.button("Score Transaction", type="primary", use_container_width=True)

    if score_button and transaction_id.strip():
        with st.spinner("Calling API..."):
            # Call API
            transaction_data = {"transaction_id": transaction_id.strip()}

            result = call_api_inference(
                endpoint_url=st.session_state.api_config['endpoint_url'],
                transaction_data=transaction_data,
                auth_type=st.session_state.api_config.get('auth_type', 'None'),
                api_key=st.session_state.api_config.get('api_key', ''),
                timeout=st.session_state.api_config.get('timeout', 30)
            )

            # Log request
            st.session_state.api_request_log.append(result)
            if len(st.session_state.api_request_log) > 100:
                st.session_state.api_request_log = st.session_state.api_request_log[-100:]

            # Parse response
            api_data = result.get('data', {})
            if not isinstance(api_data, dict):
                api_data = {}

            if 'data' in api_data and isinstance(api_data['data'], dict):
                api_data = api_data['data']

            if result['success'] and api_data:
                st.markdown("---")
                st.markdown("#### Prediction Results")

                # Extract fraud score
                fraud_score = None
                for field in ['fraud_probability', 'fraud_score', 'probability', 'score']:
                    if field in api_data:
                        try:
                            fraud_score = float(api_data[field])
                            break
                        except:
                            continue

                # Extract prediction
                prediction = api_data.get('prediction', api_data.get('label'))
                if prediction is None and fraud_score is not None:
                    prediction = 1 if fraud_score > 0.5 else 0

                # Display main metrics
                if fraud_score is not None:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Status", "SUCCESS")

                    with col2:
                        score_pct = fraud_score * 100 if fraud_score <= 1 else fraud_score
                        st.metric("Fraud Score", f"{score_pct:.2f}%")

                    with col3:
                        st.metric("Model Used", selected_model_name)

                    with col4:
                        st.metric("Latency", f"{result.get('latency_ms', 0):.0f}ms")

                    st.markdown("---")

                    # Fraud score visualization
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("##### Fraud Score Gauge")

                        # Determine risk level
                        if score_pct >= 70:
                            color = "#dc3545"
                            risk_level = "HIGH RISK"
                        elif score_pct >= 40:
                            color = "#ffc107"
                            risk_level = "MEDIUM RISK"
                        else:
                            color = "#28a745"
                            risk_level = "LOW RISK"

                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score_pct,
                            number={'suffix': "%", 'font': {'size': 40, 'color': color}},
                            title={'text': "Fraud Probability", 'font': {'size': 16}},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                                'bar': {'color': color, 'thickness': 0.75},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgreen"},
                                    {'range': [40, 70], 'color': "lightyellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))

                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=50, b=20),
                            paper_bgcolor="white",
                            font={'color': "darkgray", 'family': "Arial"}
                        )
                        # Use unique key based on model and transaction to force re-render
                        chart_key = f"gauge_{selected_model_name}_{transaction_id}"
                        st.plotly_chart(fig, use_container_width=True, key=chart_key)

                        st.markdown(f"<h3 style='text-align: center; color: {color};'>{risk_level}</h3>",
                                   unsafe_allow_html=True)

                    with col2:
                        st.markdown("##### Prediction Details")

                        if prediction is not None:
                            pred_label = "FRAUD" if prediction == 1 else "LEGITIMATE"
                            pred_color = "#dc3545" if prediction == 1 else "#28a745"

                            st.markdown(f"""
                                <div style='padding: 1.5rem; background-color: {pred_color}22;
                                     border-radius: 8px; border: 2px solid {pred_color}; margin-bottom: 1rem;'>
                                    <h3 style='margin: 0; color: {pred_color}; text-align: center;'>{pred_label}</h3>
                                </div>
                            """, unsafe_allow_html=True)

                        # Show additional response fields
                        if api_data.get('model_used'):
                            st.markdown(f"**Model:** `{api_data['model_used']}`")

                        if api_data.get('risk_level'):
                            st.markdown(f"**Risk Level:** {api_data['risk_level']}")

                        st.markdown(f"**Transaction ID:** {transaction_id}")

                    # Feature Contributions Section
                    st.markdown("---")
                    st.markdown("##### Feature Contributions to Prediction")

                    if feature_importance is not None and not feature_importance.empty:
                        # Get top 10 features
                        top_10_features = feature_importance.head(10)

                        st.markdown("""
                        The prediction was influenced most by these features (based on model's feature importance):
                        """)

                        # Create waterfall-style contribution chart
                        fig = go.Figure(go.Bar(
                            x=top_10_features['importance'],
                            y=top_10_features['feature'],
                            orientation='h',
                            marker=dict(
                                color=top_10_features['importance'],
                                colorscale='RdYlGn_r',
                                showscale=True,
                                colorbar=dict(title="Importance")
                            )
                        ))

                        fig.update_layout(
                            title="Top 10 Features Influencing This Prediction",
                            xaxis_title="Feature Importance",
                            yaxis_title="Feature",
                            height=400,
                            showlegend=False
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Show feature importance table
                        st.dataframe(
                            top_10_features,
                            use_container_width=True,
                            hide_index=True
                        )

                        # SHAP Values Section (if available from MLflow)
                        if shap_values is not None and model_source == "MLflow Runs":
                            st.markdown("---")
                            st.markdown("##### SHAP Values (Explainability)")

                            st.success("✓ SHAP values loaded from MLflow")

                            # Display SHAP values information
                            st.markdown(f"""
                            **SHAP values loaded:** {shap_values.shape}

                            SHAP (SHapley Additive exPlanations) values show the actual contribution of each feature
                            to individual predictions, providing instance-level explainability.
                            """)

                            # If SHAP values have the right shape, display summary
                            if len(shap_values.shape) >= 2:
                                st.metric("Number of Samples", shap_values.shape[0])
                                st.metric("Number of Features", shap_values.shape[1])

                                # Show mean absolute SHAP values as feature importance
                                with st.expander("Mean SHAP Feature Importance"):
                                    mean_shap = np.abs(shap_values).mean(axis=0)

                                    if feature_importance is not None and len(mean_shap) == len(feature_importance):
                                        shap_df = pd.DataFrame({
                                            'feature': feature_importance['feature'],
                                            'mean_shap_value': mean_shap
                                        }).sort_values('mean_shap_value', ascending=False)

                                        fig_shap = go.Figure(go.Bar(
                                            x=shap_df.head(15)['mean_shap_value'],
                                            y=shap_df.head(15)['feature'],
                                            orientation='h',
                                            marker=dict(color='lightblue')
                                        ))

                                        fig_shap.update_layout(
                                            title="Top 15 Features by Mean |SHAP| Value",
                                            xaxis_title="Mean Absolute SHAP Value",
                                            yaxis_title="Feature",
                                            height=400
                                        )

                                        st.plotly_chart(fig_shap, use_container_width=True)
                                        st.dataframe(shap_df.head(15), use_container_width=True, hide_index=True)
                            else:
                                st.info("SHAP values have unusual shape - manual inspection may be needed")

                        st.info("""
                        **Note:** Feature importance shows which features are generally most important for the model's decisions.
                        For transaction-specific explanations, SHAP values show the actual contribution of each feature value
                        to specific predictions.
                        """)
                    else:
                        st.warning("Feature importance not available for selected model")

                    # Raw response
                    with st.expander("View Full API Response"):
                        st.json(api_data)

                else:
                    st.warning("API response received but no fraud score found")
                    st.json(api_data)

            else:
                # Error handling
                error_msg = result.get('error', 'Unknown error')
                st.error(f"API Connection Failed: {error_msg}")

                col1, col2 = st.columns(2)
                with col1:
                    if result.get('status_code'):
                        st.metric("Status Code", result['status_code'])
                with col2:
                    if result.get('latency_ms'):
                        st.metric("Response Time", f"{result['latency_ms']:.0f}ms")

                if result.get('error_data'):
                    with st.expander("Error Response"):
                        st.json(result['error_data'])
