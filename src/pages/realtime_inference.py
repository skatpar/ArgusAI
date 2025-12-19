"""
Real-time Model Inference Module
Provides real-time fraud detection inference using external APIs (KNIME) or local models
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('/home/user/ArgusAI')

from src.utils.api_client import call_api_inference, test_api_connection
from src.utils.clickhouse_connector import ClickHouseConnector


def show():
    st.markdown('<p class="main-header">Real-time Model Inference</p>', unsafe_allow_html=True)
    st.markdown("Score transactions in real-time using your KNIME API or local models")

    # Initialize session state
    if 'api_config' not in st.session_state:
        st.session_state.api_config = {
            'endpoint_url': '',
            'api_key': '',
            'auth_type': 'None',
            'timeout': 30,
            'content_type': 'application/json',
            'custom_headers': {}
        }

    if 'inference_history' not in st.session_state:
        st.session_state.inference_history = []

    if 'api_request_log' not in st.session_state:
        st.session_state.api_request_log = []

    # Create tabs
    tabs = st.tabs([
        "API Configuration",
        "Single Transaction",
        "Batch Inference",
        "Inference History",
        "API Logs"
    ])

    with tabs[0]:
        show_api_configuration()

    with tabs[1]:
        show_single_transaction_inference()

    with tabs[2]:
        show_batch_inference()

    with tabs[3]:
        show_inference_history()

    with tabs[4]:
        show_api_logs()


def show_api_configuration():
    """Configure external API endpoints for model inference"""
    st.markdown("### API Configuration")
    st.markdown("Configure your KNIME, production model API, or external model API endpoint")

    # Check if API is configured
    api_configured = bool(st.session_state.api_config.get('endpoint_url'))

    if api_configured:
        st.success(f"API Configured: {st.session_state.api_config['endpoint_url']}")
    else:
        st.warning("No API configured. Please configure your API endpoint below.")

    # Quick presets
    st.markdown("---")
    st.markdown("#### Quick Presets")

    preset = st.selectbox(
        "Select Preset:",
        ["Custom", "Production API (localhost:5000)", "KNIME Server"],
        help="Use a preset configuration or create custom"
    )

    if preset == "Production API (localhost:5000)":
        preset_config = {
            'endpoint_url': 'http://localhost:5000/predict',
            'auth_type': 'None',
            'api_key': '',
            'timeout': 30,
            'content_type': 'application/json',
            'custom_headers': {},
            'request_format': 'transaction_id'  # Special format for prod API
        }

        if st.button("Use Production API Preset"):
            st.session_state.api_config.update(preset_config)
            st.success("Production API preset applied!")
            st.rerun()

    elif preset == "KNIME Server":
        st.info("Configure your KNIME server details below")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Endpoint Settings")

        endpoint_url = st.text_input(
            "API Endpoint URL:",
            value=st.session_state.api_config.get('endpoint_url', ''),
            placeholder="https://your-knime-server.com/api/v1/predict",
            help="Full URL of your KNIME model API endpoint"
        )

        auth_type = st.selectbox(
            "Authentication Type:",
            ["None", "API Key", "Bearer Token", "Basic Auth"],
            index=["None", "API Key", "Bearer Token", "Basic Auth"].index(
                st.session_state.api_config.get('auth_type', 'None')
            )
        )

        if auth_type in ["API Key", "Bearer Token"]:
            api_key = st.text_input(
                "API Key / Token:",
                value=st.session_state.api_config.get('api_key', ''),
                type="password",
                help="Your API key or bearer token"
            )
        elif auth_type == "Basic Auth":
            col_a, col_b = st.columns(2)
            with col_a:
                username = st.text_input("Username:", key="api_username")
            with col_b:
                password = st.text_input("Password:", type="password", key="api_password")
            api_key = f"{username}:{password}"
        else:
            api_key = ""

    with col2:
        st.markdown("#### Advanced Settings")

        timeout = st.number_input(
            "Request Timeout (seconds):",
            min_value=5,
            max_value=120,
            value=st.session_state.api_config.get('timeout', 30),
            help="Maximum time to wait for API response"
        )

        content_type = st.selectbox(
            "Content Type:",
            ["application/json", "application/x-www-form-urlencoded"],
            index=0
        )

        # Custom headers
        with st.expander("Custom Headers (Optional)"):
            st.markdown("Add additional headers required by your API")

            num_headers = st.number_input("Number of custom headers:", 0, 10, 0, key="num_custom_headers")
            custom_headers = {}

            for i in range(num_headers):
                col_a, col_b = st.columns(2)
                with col_a:
                    header_key = st.text_input(f"Header {i+1} Key:", key=f"custom_header_key_{i}")
                with col_b:
                    header_value = st.text_input(f"Header {i+1} Value:", key=f"custom_header_val_{i}")

                if header_key and header_value:
                    custom_headers[header_key] = header_value

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save Configuration", type="primary"):
            st.session_state.api_config = {
                'endpoint_url': endpoint_url,
                'api_key': api_key,
                'auth_type': auth_type,
                'timeout': timeout,
                'content_type': content_type,
                'custom_headers': custom_headers
            }
            st.success("Configuration saved successfully!")
            st.rerun()

    with col2:
        if st.button("Test Connection"):
            if endpoint_url:
                with st.spinner("Testing API connection..."):
                    test_result = test_api_connection(
                        endpoint_url=endpoint_url,
                        auth_type=auth_type,
                        api_key=api_key,
                        timeout=timeout,
                        content_type=content_type,
                        custom_headers=custom_headers
                    )

                    if test_result['success']:
                        st.success(f"Connection successful! Latency: {test_result['latency_ms']:.2f}ms")
                    else:
                        st.error(f"Connection failed: {test_result['error']}")
                        if test_result.get('status_code'):
                            st.warning(f"Status Code: {test_result['status_code']}")
            else:
                st.warning("Please enter an endpoint URL first")

    with col3:
        if st.button("Clear Configuration"):
            st.session_state.api_config = {
                'endpoint_url': '',
                'api_key': '',
                'auth_type': 'None',
                'timeout': 30,
                'content_type': 'application/json',
                'custom_headers': {}
            }
            st.success("Configuration cleared!")
            st.rerun()

    # Current configuration display
    if api_configured:
        st.markdown("---")
        st.markdown("#### Current Configuration")

        config_display = pd.DataFrame({
            'Setting': ['Endpoint URL', 'Authentication', 'Timeout', 'Content Type'],
            'Value': [
                st.session_state.api_config.get('endpoint_url', 'Not set'),
                st.session_state.api_config.get('auth_type', 'None'),
                f"{st.session_state.api_config.get('timeout', 30)}s",
                st.session_state.api_config.get('content_type', 'application/json')
            ]
        })

        st.dataframe(config_display, use_container_width=True, hide_index=True, height=180)

    # Sample request/response format
    st.markdown("---")
    st.markdown("#### Expected API Format")

    # Add link icon to expand/collapse
    with st.expander("View API Format Details", expanded=False):
        st.markdown("""
        **Important Notes:**
        - For production API: Use `transaction_id` format (single field)
        - For development/testing: Use full transaction features (JSON format)
        - All numeric values should be valid numbers, not null
        - Categorical fields should match expected values
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Request Format:**")

        # Check if we have loaded data features to show actual schema
        if st.session_state.get('data_features'):
            st.info("Using actual features from loaded data")
            features_list = st.session_state.data_features
            # Create sample request using actual features (show first 10)
            sample_request = {}
            for feat in features_list[:10]:
                if 'amount' in feat.lower() or 'balance' in feat.lower():
                    sample_request[feat] = 1250.0
                elif 'count' in feat.lower() or 'num' in feat.lower():
                    sample_request[feat] = 125
                elif 'hour' in feat.lower():
                    sample_request[feat] = 14
                elif 'age' in feat.lower():
                    sample_request[feat] = 35
                elif 'flag' in feat.lower() or 'is_' in feat.lower():
                    sample_request[feat] = False
                elif 'distance' in feat.lower() or 'ratio' in feat.lower():
                    sample_request[feat] = 5.2
                else:
                    sample_request[feat] = "sample_value"

            st.code(json.dumps(sample_request, indent=2), language="json")

            if len(features_list) > 10:
                with st.expander(f"View all {len(features_list)} features"):
                    st.json(features_list)
        else:
            st.markdown("**Standard Transaction Features:**")
            sample_request = {
                "transaction_amount": 1250.0,
                "merchant_category": "online",
                "transaction_hour": 14,
                "customer_age": 35,
                "account_age_days": 730,
                "previous_transactions": 125,
                "is_foreign": False,
                "distance_from_home": 5.2,
                "device_type": "mobile"
            }
            st.code(json.dumps(sample_request, indent=2), language="json")

    with col2:
        st.markdown("**Expected Response:**")
        sample_response = {
            "fraud_score": 0.75,
            "prediction": "fraud",
            "confidence": 0.85,
            "risk_level": "high"
        }
        st.code(json.dumps(sample_response, indent=2), language="json")

        st.markdown("**Response Fields:**")
        st.markdown("""
        - `fraud_score`: Probability of fraud (0.0 to 1.0)
        - `prediction`: "fraud" or "legitimate"
        - `confidence`: Model confidence (0.0 to 1.0)
        - `risk_level`: "low", "medium", or "high"
        """)


def show_single_transaction_inference():
    """Single transaction inference - simplified transaction ID input"""
    st.markdown("### Single Transaction Inference")
    st.markdown("Score individual transactions by entering transaction ID")

    # Check API configuration
    api_configured = bool(st.session_state.api_config.get('endpoint_url'))

    if api_configured:
        st.success(f"✅ API Configured: {st.session_state.api_config['endpoint_url']}")
    else:
        st.warning("⚠️ API not configured. Please configure in API Configuration tab.")
        st.info("💡 Go to 'API Configuration' tab to set up your model endpoint.")
        return

    st.markdown("---")

    # Simple transaction ID input
    st.markdown("#### Transaction ID Input")

    col1, col2 = st.columns([3, 1])

    with col1:
        transaction_id = st.text_input(
            "Transaction ID:",
            placeholder="Enter transaction ID (e.g., 83881056341)",
            help="Enter the transaction ID to score",
            key="single_tx_id_input"
        )

    with col2:
        st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
        score_button = st.button("🔍 Score Transaction", type="primary", use_container_width=True)

    if score_button:
        if not transaction_id.strip():
            st.error("❌ Please enter a transaction ID")
            return

        st.markdown("---")
        st.markdown("#### Prediction Results")

        with st.spinner("🔄 Calling API..."):
            try:
                # Prepare request data
                transaction_data = {"transaction_id": transaction_id.strip()}

                # Call API
                result = call_api_inference(
                    endpoint_url=st.session_state.api_config['endpoint_url'],
                    transaction_data=transaction_data,
                    auth_type=st.session_state.api_config.get('auth_type', 'None'),
                    api_key=st.session_state.api_config.get('api_key', ''),
                    timeout=st.session_state.api_config.get('timeout', 30),
                    content_type=st.session_state.api_config.get('content_type', 'application/json'),
                    custom_headers=st.session_state.api_config.get('custom_headers', {})
                )

                # Log the request
                st.session_state.api_request_log.append(result)
                if len(st.session_state.api_request_log) > 100:
                    st.session_state.api_request_log = st.session_state.api_request_log[-100:]

                # Parse API response data
                api_data = result.get('data', {})

                # Handle non-dict responses
                if not isinstance(api_data, dict):
                    api_data = {}

                # Check if API returned empty response
                if not api_data and result['success']:
                    st.warning("⚠️ API returned empty response (200 OK but no data)")
                    st.markdown("""
                    **Possible causes:**
                    - Transaction ID not found in database
                    - API processing error without proper error response
                    - API endpoint configuration issue

                    Please check the API logs for more details.
                    """)
                    return

                # Extract fraud score from different possible field names
                fraud_score = None
                for field in ['fraud_score', 'probability', 'score', 'fraud_probability', 'risk_score']:
                    if field in api_data:
                        try:
                            fraud_score = float(api_data[field])
                            break
                        except (ValueError, TypeError):
                            continue

                # Extract prediction (1 = fraud, 0 = legitimate)
                prediction = api_data.get('prediction', api_data.get('label', api_data.get('class')))

                # Determine prediction from fraud score if not provided
                if prediction is None and fraud_score is not None:
                    prediction = 1 if fraud_score > 0.5 else 0

                # Add to inference history
                history_entry = {
                    'Timestamp': result['timestamp'],
                    'Transaction ID': transaction_id,
                    'Prediction': '🚨 FRAUD' if prediction == 1 else '✅ LEGITIMATE' if prediction == 0 else 'Unknown',
                    'Fraud Score': f"{fraud_score*100:.2f}%" if fraud_score is not None else 'N/A',
                    'Status': 'Success' if result['success'] else 'Failed',
                    'Latency (ms)': f"{result.get('latency_ms', 0):.0f}" if result.get('latency_ms') else 'N/A'
                }
                st.session_state.inference_history.append(history_entry)
                if len(st.session_state.inference_history) > 100:
                    st.session_state.inference_history = st.session_state.inference_history[-100:]

                # Display results
                if result['success']:
                    # Success metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Status", "✅ SUCCESS", delta="200 OK")

                    with col2:
                        st.metric("Latency", f"{result['latency_ms']:.0f}ms")

                    with col3:
                        st.metric("Transaction ID", transaction_id)

                    with col4:
                        timestamp = result['timestamp'].split()[1] if ' ' in result['timestamp'] else result['timestamp']
                        st.metric("Time", timestamp)

                    st.markdown("---")

                    if fraud_score is not None:
                        col1, col2 = st.columns(2)

                        with col1:
                            # Fraud score gauge
                            st.markdown("##### Fraud Score")
                            score_pct = fraud_score * 100 if fraud_score <= 1 else fraud_score

                            # Color based on score
                            if score_pct >= 70:
                                color = "#dc3545"  # Red
                                risk_level = "🔴 HIGH RISK"
                            elif score_pct >= 40:
                                color = "#ffc107"  # Yellow
                                risk_level = "🟡 MEDIUM RISK"
                            else:
                                color = "#28a745"  # Green
                                risk_level = "🟢 LOW RISK"

                            st.markdown(f"""
                                <div style='text-align: center; padding: 2rem; background-color: {color}22; border-radius: 10px; border: 2px solid {color};'>
                                    <h1 style='color: {color}; margin: 0; font-size: 3rem;'>{score_pct:.1f}%</h1>
                                    <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: 600;'>{risk_level}</p>
                                </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            # Prediction details
                            st.markdown("##### Prediction Details")

                            if prediction is not None:
                                pred_label = "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE"
                                pred_color = "#dc3545" if prediction == 1 else "#28a745"

                                st.markdown(f"""
                                    <div style='padding: 1rem; background-color: {pred_color}22; border-radius: 8px; border: 1px solid {pred_color};'>
                                        <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: {pred_color};'>Prediction: {pred_label}</p>
                                    </div>
                                """, unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("**Full API Response:**")
                            st.json(api_data)

                    else:
                        st.warning("⚠️ API response received but no fraud score found")
                        st.markdown("**Raw API Response:**")
                        st.json(api_data)
                        st.info("""
                        💡 **Tip:** The API response should contain one of these fields:
                        - `fraud_score` or `probability` (decimal between 0-1)
                        - `score` or `fraud_probability`
                        - `risk_score`

                        Current response structure may not match expected format.
                        """)

                else:
                    # Error display
                    error_msg = result.get('error', 'Unknown error')
                    st.error(f"❌ **API Connection Failed**")

                    st.markdown(f"""
                    <div style='padding: 1rem; background-color: #dc354522; border-radius: 8px; border: 1px solid #dc3545; margin: 1rem 0;'>
                        <h4 style='color: #dc3545; margin-top: 0;'>Error Details</h4>
                        <p style='margin: 0.5rem 0;'><strong>Message:</strong> {error_msg}</p>
                        <p style='margin: 0.5rem 0;'><strong>Endpoint:</strong> <code>{result.get('endpoint', 'N/A')}</code></p>
                        <p style='margin: 0.5rem 0;'><strong>Timestamp:</strong> {result.get('timestamp', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        if result.get('status_code'):
                            st.metric("HTTP Status Code", result['status_code'])
                        else:
                            st.metric("HTTP Status Code", "N/A", help="No response from server")
                    with col2:
                        if result.get('latency_ms'):
                            st.metric("Response Time", f"{result['latency_ms']:.0f}ms")
                        else:
                            st.metric("Response Time", "Timeout/No Response")

                    # Troubleshooting tips
                    with st.expander("🔧 Troubleshooting Tips"):
                        st.markdown("""
                        **Common Causes:**
                        1. **Connection Failed** - The API server is not reachable
                           - Check if the endpoint URL is correct
                           - Verify the server is running and accessible
                           - Check your network connection

                        2. **Timeout** - The API is taking too long to respond
                           - Increase the timeout value in API Configuration
                           - Check if the server is overloaded

                        3. **Authentication Error** (401, 403)
                           - Verify your API key is correct
                           - Check if the authentication type matches server requirements

                        4. **Server Error** (500, 502, 503)
                           - The API server encountered an internal error
                           - Check server logs for details
                           - Contact the API administrator
                        """)

                    if result.get('error_data'):
                        st.markdown("**Server Error Response:**")
                        st.json(result['error_data'])

            except Exception as e:
                st.error(f"❌ Error calling API: {str(e)}")
                st.exception(e)


def show_transaction_id_input(inference_mode, api_configured):
    """Simple transaction ID input for production API"""
    st.markdown("#### Transaction ID Input")
    st.markdown("Enter transaction ID to get fraud score from production model")

    transaction_id = st.text_input(
        "Transaction ID:",
        placeholder="83881056341",
        help="Enter the transaction ID to score"
    )

    if st.button("Score Transaction", type="primary"):
        if not transaction_id.strip():
            st.warning("Please enter a transaction ID")
            return

        st.markdown("---")
        st.markdown("#### Prediction Results")

        with st.spinner("Calling production API..."):
            try:
                # Call production API with transaction_id format
                transaction_data = {"transaction_id": transaction_id}

                result = call_api_inference(
                    endpoint_url=st.session_state.api_config['endpoint_url'],
                    transaction_data=transaction_data,
                    auth_type=st.session_state.api_config['auth_type'],
                    api_key=st.session_state.api_config.get('api_key', ''),
                    timeout=st.session_state.api_config.get('timeout', 30),
                    content_type=st.session_state.api_config.get('content_type', 'application/json'),
                    custom_headers=st.session_state.api_config.get('custom_headers', {})
                )

                # Log the request
                st.session_state.api_request_log.append(result)
                if len(st.session_state.api_request_log) > 100:
                    st.session_state.api_request_log = st.session_state.api_request_log[-100:]

                if result['success']:
                    # Display success metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Status", "SUCCESS", delta="200 OK")

                    with col2:
                        st.metric("Latency", f"{result['latency_ms']:.2f}ms")

                    with col3:
                        st.metric("Transaction ID", transaction_id)

                    with col4:
                        st.metric("Timestamp", result['timestamp'].split()[1])

                    # Display full response
                    st.markdown("---")
                    st.markdown("**API Response:**")
                    st.json(result['data'])

                    # Extract fraud score from response
                    response_data = result['data']

                    # Handle nested "data" field in response
                    if isinstance(response_data, dict) and 'data' in response_data:
                        response_data = response_data['data']

                    if isinstance(response_data, dict):
                        # Extract fraud probability
                        fraud_score = (
                            response_data.get('fraud_probability') or
                            response_data.get('fraud_score') or
                            response_data.get('score') or
                            response_data.get('probability')
                        )

                        # Extract prediction
                        prediction = (
                            response_data.get('prediction') or
                            response_data.get('label') or
                            response_data.get('class')
                        )

                        # Extract risk level
                        risk_level = (
                            response_data.get('risk_level') or
                            response_data.get('risk')
                        )

                        # Extract actual fraud flag if available
                        actual_fraud = response_data.get('actual_fraud_flag')

                        # Display interpreted results
                        if fraud_score is not None or prediction is not None:
                            st.markdown("---")
                            st.markdown("**Fraud Detection Results:**")

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                if fraud_score is not None:
                                    score_val = float(fraud_score)
                                    score_display = f"{score_val:.2%}" if score_val <= 1 else f"{score_val:.4f}"
                                    st.metric("Fraud Probability", score_display)

                            with col2:
                                if prediction is not None:
                                    pred_label = "FRAUD" if int(prediction) == 1 else "LEGITIMATE"
                                    st.metric("Prediction", pred_label)

                            with col3:
                                if risk_level:
                                    st.metric("Risk Level", str(risk_level).upper())
                                elif fraud_score is not None:
                                    score_val = float(fraud_score)
                                    risk = "HIGH RISK" if score_val > 0.5 else "LOW RISK"
                                    st.metric("Risk Level", risk)

                            with col4:
                                st.metric("Latency", f"{result['latency_ms']:.2f}ms")

                            # Show actual fraud flag if available (for validation)
                            if actual_fraud is not None:
                                st.info(f"Actual Fraud Flag: **{actual_fraud}** (for validation)")

                        # Add to history
                        st.session_state.inference_history.append({
                            'Timestamp': result['timestamp'],
                            'Mode': 'Production API',
                            'Transaction ID': transaction_id,
                            'Fraud Probability': f"{fraud_score:.4f}" if fraud_score else 'N/A',
                            'Prediction': str(prediction) if prediction else 'N/A',
                            'Risk Level': str(risk_level) if risk_level else 'N/A',
                            'Latency': f"{result['latency_ms']:.1f}ms",
                            'Status': 'Success'
                        })

                else:
                    st.error(f"API Error (Status {result.get('status_code', 'Unknown')}): {result['error']}")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Status", "FAILED")

                    with col2:
                        if result.get('status_code'):
                            st.metric("Status Code", result['status_code'])

                    with col3:
                        if result.get('latency_ms'):
                            st.metric("Latency", f"{result['latency_ms']:.2f}ms")

                    # Show detailed error data if available
                    if result.get('error_data'):
                        st.markdown("**Error Details:**")
                        st.json(result['error_data'])

                    # Show request details for debugging
                    with st.expander("Request Details (for debugging)"):
                        st.json({
                            "endpoint": result.get('endpoint'),
                            "request_body": {"transaction_id": transaction_id},
                            "timestamp": result.get('timestamp')
                        })

                    # Add failed request to history
                    st.session_state.inference_history.append({
                        'Timestamp': result['timestamp'],
                        'Mode': 'Production API',
                        'Transaction ID': transaction_id,
                        'Fraud Score': 'Error',
                        'Prediction': 'Error',
                        'Latency': 'N/A',
                        'Status': 'Failed'
                    })

            except Exception as e:
                st.error(f"Error: {str(e)}")


def show_json_input(inference_mode, api_configured):
    """JSON input interface"""
    st.markdown("#### Transaction JSON")

    # Sample transaction
    sample_transaction = {
        "transaction_amount": 1250.00,
        "merchant_category": "online",
        "transaction_hour": 14,
        "customer_age": 35,
        "account_age_days": 730,
        "previous_transactions": 125,
        "is_foreign": False,
        "distance_from_home": 5.2,
        "device_type": "mobile",
        "transaction_type": "purchase"
    }

    # JSON editor
    json_input = st.text_area(
        "Enter transaction JSON:",
        value=json.dumps(sample_transaction, indent=2),
        height=300,
        help="Paste your transaction JSON here"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        predict_button = st.button("Score Transaction", type="primary", use_container_width=True)

    with col2:
        if st.button("Use Sample", use_container_width=True):
            st.info("Sample JSON loaded above")

    with col3:
        if st.button("Clear", use_container_width=True):
            st.info("Refresh page to clear")

    # Process prediction
    if predict_button:
        try:
            transaction_data = json.loads(json_input)

            st.markdown("---")
            st.markdown("#### Prediction Results")

            if inference_mode == "External API (KNIME)" and api_configured:
                # Call external API
                with st.spinner("Calling external API..."):
                    result = call_api_inference(
                        endpoint_url=st.session_state.api_config['endpoint_url'],
                        transaction_data=transaction_data,
                        auth_type=st.session_state.api_config['auth_type'],
                        api_key=st.session_state.api_config.get('api_key', ''),
                        timeout=st.session_state.api_config.get('timeout', 30),
                        content_type=st.session_state.api_config.get('content_type', 'application/json'),
                        custom_headers=st.session_state.api_config.get('custom_headers', {})
                    )

                    # Log the request
                    st.session_state.api_request_log.append(result)
                    if len(st.session_state.api_request_log) > 100:
                        st.session_state.api_request_log = st.session_state.api_request_log[-100:]

                    display_api_results(result, transaction_data)

            else:
                # Local simulation
                result = simulate_local_inference(transaction_data)
                display_simulation_results(result, transaction_data)

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


def show_form_input(inference_mode, api_configured):
    """Form-based input interface"""
    st.markdown("#### Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Transaction Info:**")
        transaction_amount = st.number_input("Amount ($):", 0.0, 1000000.0, 1250.0)
        merchant_category = st.selectbox(
            "Merchant Category:",
            ["retail", "online", "groceries", "gas_station", "restaurant",
             "gambling", "crypto", "wire_transfer"]
        )
        transaction_hour = st.slider("Transaction Hour:", 0, 23, 14)
        transaction_type = st.selectbox("Type:", ["purchase", "withdrawal", "transfer"])

    with col2:
        st.markdown("**Customer Info:**")
        customer_age = st.number_input("Customer Age:", 18, 100, 35)
        account_age_days = st.number_input("Account Age (days):", 0, 10000, 730)
        previous_transactions = st.number_input("Previous Transactions:", 0, 100000, 125)

    with col3:
        st.markdown("**Location & Device:**")
        is_foreign = st.checkbox("Foreign Transaction")
        distance_from_home = st.number_input("Distance from Home (km):", 0.0, 10000.0, 5.2)
        device_type = st.selectbox("Device:", ["mobile", "desktop", "tablet"])

    if st.button("Score Transaction", type="primary"):
        transaction_data = {
            "transaction_amount": transaction_amount,
            "merchant_category": merchant_category,
            "transaction_hour": transaction_hour,
            "transaction_type": transaction_type,
            "customer_age": customer_age,
            "account_age_days": account_age_days,
            "previous_transactions": previous_transactions,
            "is_foreign": is_foreign,
            "distance_from_home": distance_from_home,
            "device_type": device_type
        }

        st.markdown("---")
        st.markdown("#### Prediction Results")

        if inference_mode == "External API (KNIME)" and api_configured:
            with st.spinner("Calling external API..."):
                result = call_api_inference(
                    endpoint_url=st.session_state.api_config['endpoint_url'],
                    transaction_data=transaction_data,
                    auth_type=st.session_state.api_config['auth_type'],
                    api_key=st.session_state.api_config.get('api_key', ''),
                    timeout=st.session_state.api_config.get('timeout', 30),
                    content_type=st.session_state.api_config.get('content_type', 'application/json'),
                    custom_headers=st.session_state.api_config.get('custom_headers', {})
                )

                st.session_state.api_request_log.append(result)
                if len(st.session_state.api_request_log) > 100:
                    st.session_state.api_request_log = st.session_state.api_request_log[-100:]

                display_api_results(result, transaction_data)
        else:
            result = simulate_local_inference(transaction_data)
            display_simulation_results(result, transaction_data)


def display_api_results(result, transaction_data):
    """Display results from external API call"""
    if result['success']:
        # Success metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Status", "SUCCESS", delta="200 OK", delta_color="normal")

        with col2:
            st.metric("Latency", f"{result['latency_ms']:.2f}ms")

        with col3:
            st.metric("Endpoint", "Connected", delta_color="normal")

        with col4:
            st.metric("Timestamp", result['timestamp'].split()[1])

        # Display raw response
        st.markdown("---")
        st.markdown("**API Response:**")
        st.json(result['data'])

        # Extract and interpret fraud score
        response_data = result['data']

        if isinstance(response_data, dict):
            fraud_score = (
                response_data.get('fraud_score') or
                response_data.get('score') or
                response_data.get('probability') or
                response_data.get('fraud_probability')
            )

            prediction = (
                response_data.get('prediction') or
                response_data.get('label') or
                response_data.get('class')
            )

            confidence = response_data.get('confidence')
            risk_level = response_data.get('risk_level')

            # Display interpreted results
            if fraud_score is not None or prediction is not None:
                st.markdown("---")
                st.markdown("**Interpreted Results:**")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if fraud_score is not None:
                        score_val = float(fraud_score)
                        score_display = f"{score_val:.2%}" if score_val <= 1 else f"{score_val:.2f}"
                        st.metric("Fraud Score", score_display)

                with col2:
                    if prediction is not None:
                        st.metric("Prediction", str(prediction).upper())

                with col3:
                    if confidence is not None:
                        conf_val = float(confidence)
                        conf_display = f"{conf_val:.2%}" if conf_val <= 1 else f"{conf_val:.2f}"
                        st.metric("Confidence", conf_display)

                with col4:
                    if risk_level is not None:
                        st.metric("Risk Level", str(risk_level).upper())
                    elif fraud_score is not None:
                        score_val = float(fraud_score)
                        risk = "HIGH" if (score_val > 0.5 if score_val <= 1 else score_val > 50) else "LOW"
                        st.metric("Risk Level", risk)

            # Add to history
            st.session_state.inference_history.append({
                'Timestamp': result['timestamp'],
                'Mode': 'External API',
                'Amount': f"${transaction_data.get('transaction_amount', 0):,.2f}",
                'Fraud Score': f"{fraud_score:.2%}" if fraud_score and fraud_score <= 1 else str(fraud_score),
                'Prediction': str(prediction) if prediction else 'N/A',
                'Latency': f"{result['latency_ms']:.1f}ms",
                'Status': 'Success'
            })

    else:
        # Error display
        st.error(f"API Error: {result['error']}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Status", "FAILED", delta_color="inverse")

        with col2:
            if result.get('status_code'):
                st.metric("Status Code", result['status_code'])

        with col3:
            st.metric("Timestamp", result['timestamp'].split()[1])

        # Add failed request to history
        st.session_state.inference_history.append({
            'Timestamp': result['timestamp'],
            'Mode': 'External API',
            'Amount': f"${transaction_data.get('transaction_amount', 0):,.2f}",
            'Fraud Score': 'Error',
            'Prediction': 'Error',
            'Latency': 'N/A',
            'Status': 'Failed'
        })


def display_simulation_results(result, transaction_data):
    """Display results from local simulation"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Fraud Score", f"{result['fraud_score']:.2%}")

    with col2:
        st.metric("Prediction", result['prediction'].upper())

    with col3:
        st.metric("Confidence", f"{result['confidence']:.2%}")

    with col4:
        st.metric("Risk Level", result['risk_level'].upper())

    # Visualization
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['fraud_score'] * 100,
            title={'text': "Fraud Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#744ada"},
                'steps': [
                    {'range': [0, 30], 'color': "#f5f5f5"},
                    {'range': [30, 70], 'color': "#cccccc"},
                    {'range': [70, 100], 'color': "#666666"}
                ],
                'threshold': {
                    'line': {'color': "#000000", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk factors
        st.markdown("**Risk Factors:**")
        for factor in result['risk_factors']:
            st.warning(factor)

    # Add to history
    st.session_state.inference_history.append({
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Mode': 'Local Simulation',
        'Amount': f"${transaction_data.get('transaction_amount', 0):,.2f}",
        'Fraud Score': f"{result['fraud_score']:.2%}",
        'Prediction': result['prediction'],
        'Latency': f"{result['latency_ms']:.1f}ms",
        'Status': 'Success'
    })


def simulate_local_inference(transaction_data):
    """Simulate local model inference"""
    import time
    import random

    time.sleep(random.uniform(0.1, 0.3))  # Simulate processing time

    # Calculate fraud score based on transaction features
    fraud_score = random.uniform(0.1, 0.95)

    # Adjust score based on risk factors
    risk_factors = []

    if transaction_data.get('transaction_amount', 0) > 1000:
        fraud_score = min(fraud_score + 0.1, 0.99)
        risk_factors.append(f"High amount: ${transaction_data['transaction_amount']:,.2f}")

    if transaction_data.get('merchant_category') in ['gambling', 'crypto', 'wire_transfer']:
        fraud_score = min(fraud_score + 0.15, 0.99)
        risk_factors.append(f"High-risk category: {transaction_data.get('merchant_category')}")

    hour = transaction_data.get('transaction_hour', 12)
    if hour < 6 or hour > 22:
        fraud_score = min(fraud_score + 0.1, 0.99)
        risk_factors.append(f"Unusual hour: {hour}:00")

    if transaction_data.get('is_foreign'):
        fraud_score = min(fraud_score + 0.1, 0.99)
        risk_factors.append("Foreign transaction")

    if transaction_data.get('distance_from_home', 0) > 100:
        fraud_score = min(fraud_score + 0.1, 0.99)
        risk_factors.append(f"Large distance: {transaction_data['distance_from_home']:.0f} km")

    if transaction_data.get('account_age_days', 365) < 30:
        fraud_score = min(fraud_score + 0.15, 0.99)
        risk_factors.append(f"New account: {transaction_data['account_age_days']} days")

    is_fraud = fraud_score > 0.5
    confidence = abs(fraud_score - 0.5) * 2

    return {
        'fraud_score': fraud_score,
        'prediction': 'fraud' if is_fraud else 'legitimate',
        'confidence': confidence,
        'risk_level': 'high' if is_fraud else 'low',
        'risk_factors': risk_factors if risk_factors else ['No significant risk factors detected'],
        'latency_ms': random.uniform(15, 50)
    }


def show_batch_inference():
    """Batch inference from CSV upload or ClickHouse"""
    st.markdown("### Batch Inference")
    st.markdown("Load data from CSV or ClickHouse to score multiple transactions")

    # Check API configuration
    api_configured = bool(st.session_state.api_config.get('endpoint_url'))

    if api_configured:
        inference_mode = st.radio(
            "Inference Mode:",
            ["External API (KNIME)", "Local Simulation"],
            horizontal=True
        )
    else:
        st.warning("External API not configured. Using local simulation mode.")
        inference_mode = "Local Simulation"

    st.markdown("---")

    # Data source selector
    st.markdown("**Data Source:**")
    data_source = st.radio(
        "Select data source:",
        ["CSV File", "ClickHouse Query"],
        horizontal=True,
        help="Choose whether to upload a CSV file or query ClickHouse database"
    )

    df = None

    if data_source == "CSV File":
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file:",
            type="csv",
            help="CSV should contain transaction features as columns"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df):,} transactions from CSV")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    else:  # ClickHouse Query
        # Check if ClickHouse is connected
        ch_connected = st.session_state.get('ch_config', {}).get('connected', False)

        if not ch_connected:
            st.warning("⚠️ ClickHouse not connected. Please configure connection in Data Loading module first.")
        else:
            st.markdown("**ClickHouse Query:**")

            # Sample queries
            sample_queries = {
                "Recent transactions (limit 1K)": "SELECT * FROM stixor_fraud_features_distributed LIMIT 1000",
                "By date range": """SELECT * FROM stixor_fraud_features_distributed
WHERE cutoff_date BETWEEN '2025-01-01' AND '2025-12-31'
LIMIT 5000""",
                "Payment gateway transactions": """SELECT * FROM stixor_fraud_features_distributed
WHERE trx_channel='Payment Gateway'
    AND cutoff_date >= today() - 30
LIMIT 5000""",
                "Custom query": ""
            }

            selected_sample = st.selectbox(
                "Select a sample query or write your own:",
                list(sample_queries.keys()),
                index=0
            )

            default_query = sample_queries[selected_sample]

            query = st.text_area(
                "SQL Query:",
                value=default_query,
                height=150,
                help="Enter any valid ClickHouse SQL query"
            )

            col1, col2 = st.columns([1, 4])

            with col1:
                if st.button("Load Data", type="primary"):
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

                                # Store in session state for persistence
                                st.session_state.batch_inference_data = df

                                st.success(f"✅ Loaded {len(df):,} transactions from ClickHouse")
                                st.dataframe(df.head(10), use_container_width=True)

                            except Exception as e:
                                st.error(f"Query execution failed: {str(e)}")

            # Check if data was already loaded in this session
            if 'batch_inference_data' in st.session_state and st.session_state.batch_inference_data is not None:
                df = st.session_state.batch_inference_data
                if df is not None and len(df) > 0:
                    st.info(f"📊 Using previously loaded data: {len(df):,} transactions")
                    with st.expander("Preview loaded data"):
                        st.dataframe(df.head(10), use_container_width=True)

    # Run inference if data is loaded
    if df is not None and len(df) > 0:
        st.markdown("---")

        if st.button("Run Batch Inference", type="primary"):
            with st.spinner(f"Processing {len(df)} transactions..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                results = []

                for idx, row in df.iterrows():
                    status_text.text(f"Processing transaction {idx + 1}/{len(df)}")

                    transaction_data = row.to_dict()

                    if inference_mode == "External API (KNIME)" and api_configured:
                        result = call_api_inference(
                            endpoint_url=st.session_state.api_config['endpoint_url'],
                            transaction_data=transaction_data,
                            auth_type=st.session_state.api_config['auth_type'],
                            api_key=st.session_state.api_config.get('api_key', ''),
                            timeout=st.session_state.api_config.get('timeout', 30),
                            content_type=st.session_state.api_config.get('content_type', 'application/json'),
                            custom_headers=st.session_state.api_config.get('custom_headers', {})
                        )

                        if result['success']:
                            response_data = result['data']
                            fraud_score = (
                                response_data.get('fraud_score') or
                                response_data.get('score') or
                                response_data.get('probability', 0.5)
                            )
                            prediction = response_data.get('prediction', 'unknown')
                        else:
                            fraud_score = None
                            prediction = 'error'
                    else:
                        sim_result = simulate_local_inference(transaction_data)
                        fraud_score = sim_result['fraud_score']
                        prediction = sim_result['prediction']

                    results.append({
                        'fraud_score': fraud_score,
                        'prediction': prediction
                    })

                    progress_bar.progress((idx + 1) / len(df))

                status_text.text("Processing complete!")

                # Add results to dataframe
                results_df = df.copy()
                results_df['fraud_score'] = [r['fraud_score'] for r in results]
                results_df['prediction'] = [r['prediction'] for r in results]

                st.success("Batch inference completed!")

                # Summary metrics
                st.markdown("---")
                st.markdown("#### Batch Results Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total = len(results)
                    st.metric("Total Transactions", total)

                with col2:
                    fraud_count = len([r for r in results if r['prediction'] == 'fraud'])
                    st.metric("Flagged as Fraud", fraud_count)

                with col3:
                    fraud_rate = (fraud_count / total * 100) if total > 0 else 0
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

                with col4:
                    avg_score = np.mean([r['fraud_score'] for r in results if r['fraud_score'] is not None])
                    st.metric("Avg Fraud Score", f"{avg_score:.2%}")

                # Display results
                st.markdown("---")
                st.markdown("#### Detailed Results")
                st.dataframe(results_df, use_container_width=True)

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def show_inference_history():
    """Display inference history"""
    st.markdown("### Inference History")
    st.markdown("Recent inference requests and results")

    if len(st.session_state.inference_history) == 0:
        st.info("No inference history yet. Make some predictions to see history here.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Requests", len(st.session_state.inference_history))

    with col2:
        success_count = len([h for h in st.session_state.inference_history if h.get('Status') == 'Success'])
        st.metric("Successful", success_count)

    with col3:
        # Count entries that have a valid prediction
        valid_count = len([h for h in st.session_state.inference_history if h.get('Prediction') and h.get('Prediction') != 'Unknown'])
        st.metric("Valid Predictions", valid_count)

    with col4:
        fraud_count = len([h for h in st.session_state.inference_history if h.get('Prediction') and 'FRAUD' in h['Prediction']])
        st.metric("Fraud Detected", fraud_count)

    st.markdown("---")

    # Display history table
    history_df = pd.DataFrame(st.session_state.inference_history)
    st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)

    # Download button
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Download History CSV",
        data=csv,
        file_name=f"inference_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # Clear history button
    if st.button("Clear History"):
        st.session_state.inference_history = []
        st.success("History cleared!")
        st.rerun()


def show_api_logs():
    """Display API request/response logs"""
    st.markdown("### API Request Logs")
    st.markdown("Detailed logs of API requests and responses")

    if len(st.session_state.api_request_log) == 0:
        st.info("No API logs yet. Make API calls to see logs here.")
        return

    # Summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total API Calls", len(st.session_state.api_request_log))

    with col2:
        success_count = len([log for log in st.session_state.api_request_log if log['success']])
        st.metric("Successful", success_count)

    with col3:
        failed_count = len([log for log in st.session_state.api_request_log if not log['success']])
        st.metric("Failed", failed_count)

    with col4:
        latencies = [log['latency_ms'] for log in st.session_state.api_request_log if log['latency_ms']]
        avg_latency = np.mean(latencies) if latencies else 0
        st.metric("Avg Latency", f"{avg_latency:.2f}ms")

    st.markdown("---")

    # Detailed logs
    for i, log in enumerate(reversed(st.session_state.api_request_log[-20:])):
        with st.expander(f"Request #{len(st.session_state.api_request_log) - i} - {log['timestamp']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                status = "SUCCESS" if log['success'] else "FAILED"
                st.metric("Status", status)

            with col2:
                if log.get('status_code'):
                    st.metric("Status Code", log['status_code'])

            with col3:
                if log.get('latency_ms'):
                    st.metric("Latency", f"{log['latency_ms']:.2f}ms")

            st.markdown("**Endpoint:**")
            st.code(log.get('endpoint', 'N/A'))

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Request:**")
                if 'request' in log:
                    st.json(log['request'])

            with col_b:
                st.markdown("**Response:**")
                if log['success'] and log.get('data'):
                    st.json(log['data'])
                elif log.get('error'):
                    st.error(log['error'])

    # Clear logs button
    if st.button("Clear Logs"):
        st.session_state.api_request_log = []
        st.success("Logs cleared!")
        st.rerun()
