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


def show_single_transaction_enhanced():
    """Enhanced single transaction inference with model selection and explainability"""
    st.markdown("### Single Transaction Inference")
    st.markdown("Score individual transactions with model selection and explainability")

    # Initialize model artifacts loader
    artifacts_loader = ModelArtifactsLoader()

    # Model Selection Section
    st.markdown("---")
    st.markdown("#### Model Selection")

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
            help="Choose the model to use for inference"
        )

        selected_model_id = model_options[selected_model_name]

    with col2:
        # Model metadata
        model_metadata = artifacts_loader.get_model_metadata(selected_model_id)
        if model_metadata:
            st.metric("Model Type", model_metadata['model_type'])
            if model_metadata.get('created_date'):
                st.caption(f"Created: {model_metadata['created_date']}")

    # Load feature importance for selected model
    feature_importance = artifacts_loader.load_feature_importance(selected_model_id)

    if feature_importance is not None and not feature_importance.empty:
        with st.expander("View Feature Importance for Selected Model", expanded=False):
            st.markdown(f"**Top 20 Important Features for {selected_model_name}**")

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
                            title={'text': "Fraud Probability"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': color},
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

                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

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

                        st.info("""
                        **Note:** Feature importance shows which features are generally most important for the model's decisions.
                        For transaction-specific explanations, SHAP values would show the actual contribution of each feature value
                        to this specific prediction.
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
