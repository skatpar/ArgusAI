"""
Model Deployment Platform
Deploy and manage ML models for real-time fraud detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append('/home/user/ArgusAI')


def show():
    st.markdown('<p class="main-header">Model Deployment</p>', unsafe_allow_html=True)
    st.markdown("Deploy and manage models for real-time fraud detection")

    # Initialize session state
    if 'deployed_models' not in st.session_state:
        st.session_state.deployed_models = initialize_deployed_models()
    if 'inference_history' not in st.session_state:
        st.session_state.inference_history = []

    # Create tabs
    tabs = st.tabs(["Model Registry", "Deploy Model", "Real-Time Inference",
                   "Model Monitoring", "Model Management"])

    with tabs[0]:
        show_model_registry()

    with tabs[1]:
        show_model_deployment()

    with tabs[2]:
        show_realtime_inference()

    with tabs[3]:
        show_model_monitoring()

    with tabs[4]:
        show_model_management()


def initialize_deployed_models():
    """Initialize with some deployed models"""
    return [
        {
            'model_id': 'MODEL_001',
            'name': 'Random Forest v2.1',
            'model_type': 'Random Forest',
            'version': '2.1',
            'status': 'Active',
            'accuracy': 0.94,
            'precision': 0.89,
            'recall': 0.91,
            'f1_score': 0.90,
            'deployed_date': datetime.now() - timedelta(days=30),
            'last_updated': datetime.now() - timedelta(hours=2),
            'predictions_today': 15234,
            'avg_latency_ms': 45.3,
            'endpoint': 'https://api.argusai.com/v1/predict/model_001'
        },
        {
            'model_id': 'MODEL_002',
            'name': 'Gradient Boosting v1.5',
            'model_type': 'Gradient Boosting',
            'version': '1.5',
            'status': 'Active',
            'accuracy': 0.96,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90,
            'deployed_date': datetime.now() - timedelta(days=15),
            'last_updated': datetime.now() - timedelta(hours=5),
            'predictions_today': 12456,
            'avg_latency_ms': 62.1,
            'endpoint': 'https://api.argusai.com/v1/predict/model_002'
        },
        {
            'model_id': 'MODEL_003',
            'name': 'Logistic Regression v3.0',
            'model_type': 'Logistic Regression',
            'version': '3.0',
            'status': 'Testing',
            'accuracy': 0.88,
            'precision': 0.85,
            'recall': 0.87,
            'f1_score': 0.86,
            'deployed_date': datetime.now() - timedelta(days=5),
            'last_updated': datetime.now() - timedelta(hours=1),
            'predictions_today': 1234,
            'avg_latency_ms': 28.5,
            'endpoint': 'https://api.argusai.com/v1/predict/model_003'
        }
    ]


def show_model_registry():
    st.markdown("### Model Registry")
    st.markdown("Browse and manage trained models")

    deployed_models = st.session_state.deployed_models

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Models", len(deployed_models))

    with col2:
        active_models = len([m for m in deployed_models if m['status'] == 'Active'])
        st.metric("Active Models", active_models)

    with col3:
        total_predictions = sum([m['predictions_today'] for m in deployed_models])
        st.metric("Predictions Today", f"{total_predictions:,}")

    with col4:
        avg_latency = np.mean([m['avg_latency_ms'] for m in deployed_models])
        st.metric("Avg Latency", f"{avg_latency:.1f}ms")

    st.markdown("---")

    # Model cards
    st.markdown("#### Registered Models")

    for model in deployed_models:
        with st.expander(f"{model['name']} - {model['status']}", expanded=(model['status'] == 'Active')):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**Model ID:** {model['model_id']}")
                st.markdown(f"**Type:** {model['model_type']}")
                st.markdown(f"**Version:** {model['version']}")
                st.markdown(f"**Status:** {model['status']}")

            with col2:
                st.markdown("**Performance Metrics:**")
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Value': [
                        f"{model['accuracy']:.2%}",
                        f"{model['precision']:.2%}",
                        f"{model['recall']:.2%}",
                        f"{model['f1_score']:.2%}"
                    ]
                })
                st.dataframe(metrics_df, hide_index=True)

            with col3:
                st.markdown("**Deployment Info:**")
                st.markdown(f"Deployed: {model['deployed_date'].strftime('%Y-%m-%d')}")
                st.markdown(f"Predictions: {model['predictions_today']:,}")
                st.markdown(f"Latency: {model['avg_latency_ms']:.1f}ms")

            # Performance visualization
            metrics_viz = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [model['accuracy'], model['precision'], model['recall'], model['f1_score']]
            })

            fig = px.bar(metrics_viz, x='Metric', y='Score',
                        title=f'Performance Metrics - {model["name"]}',
                        color='Score', color_continuous_scale='Blues',
                        range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Model actions
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button(f"View Details", key=f"details_{model['model_id']}"):
                    st.info("Model details view")

            with col2:
                if st.button(f"Test Model", key=f"test_{model['model_id']}"):
                    st.info("Model testing interface")

            with col3:
                if model['status'] == 'Active':
                    if st.button(f"Deactivate", key=f"deactivate_{model['model_id']}"):
                        model['status'] = 'Inactive'
                        st.success("Model deactivated")
                else:
                    if st.button(f"Activate", key=f"activate_{model['model_id']}"):
                        model['status'] = 'Active'
                        st.success("Model activated")

            with col4:
                if st.button(f"Remove", key=f"remove_{model['model_id']}"):
                    st.warning("Model removal (requires confirmation)")

    # Upload new model
    st.markdown("---")
    st.markdown("#### Upload New Model")

    with st.expander("Upload Model", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            model_name = st.text_input("Model Name:")
            model_type = st.selectbox("Model Type:",
                                     ["Random Forest", "Gradient Boosting", "Logistic Regression",
                                      "Neural Network", "XGBoost", "Other"])
            model_version = st.text_input("Version:", value="1.0")

        with col2:
            uploaded_file = st.file_uploader("Upload Model File (.pkl, .joblib, .h5):",
                                            type=['pkl', 'joblib', 'h5'])
            description = st.text_area("Description:", height=100)

        if st.button("Upload Model", type="primary"):
            if model_name and uploaded_file:
                st.success(f"Model '{model_name}' uploaded successfully!")
            else:
                st.error("Please provide model name and file")


def show_model_deployment():
    st.markdown("### Deploy Model")
    st.markdown("Deploy models to production or testing environments")

    # Check if models exist in training session
    if 'models' in st.session_state and len(st.session_state.models) > 0:
        st.info("Found trained models from Data Science Workflow")

        st.markdown("#### Available Models for Deployment")

        available_models = list(st.session_state.models.keys())
        selected_model_name = st.selectbox("Select model to deploy:", available_models)

        if selected_model_name:
            model = st.session_state.models[selected_model_name]

            st.markdown("---")
            st.markdown("#### Deployment Configuration")

            col1, col2 = st.columns(2)

            with col1:
                deployment_name = st.text_input("Deployment Name:",
                                               value=f"{selected_model_name}_v1")
                environment = st.selectbox("Environment:", ["Production", "Staging", "Testing"])
                version = st.text_input("Version:", value="1.0")

            with col2:
                auto_scaling = st.checkbox("Enable auto-scaling", value=True)
                if auto_scaling:
                    min_instances = st.number_input("Min instances:", 1, 10, 1)
                    max_instances = st.number_input("Max instances:", 1, 50, 5)

                monitoring = st.checkbox("Enable monitoring", value=True)
                logging = st.checkbox("Enable detailed logging", value=True)

            st.markdown("#### Advanced Settings")

            with st.expander("Advanced Configuration", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    max_latency = st.number_input("Max latency (ms):", 10, 1000, 100)
                    timeout = st.number_input("Request timeout (s):", 1, 60, 30)

                with col2:
                    batch_size = st.number_input("Batch size:", 1, 1000, 32)
                    cache_predictions = st.checkbox("Cache predictions", value=False)

            # Deployment preview
            st.markdown("---")
            st.markdown("#### Deployment Summary")

            deployment_config = {
                "Model": selected_model_name,
                "Deployment Name": deployment_name,
                "Environment": environment,
                "Version": version,
                "Auto-scaling": f"{min_instances}-{max_instances} instances" if auto_scaling else "Disabled",
                "Monitoring": "Enabled" if monitoring else "Disabled",
                "Max Latency": f"{max_latency}ms",
                "Timeout": f"{timeout}s"
            }

            config_df = pd.DataFrame(list(deployment_config.items()), columns=['Setting', 'Value'])
            st.dataframe(config_df, use_container_width=True, hide_index=True)

            # Deploy button
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("Deploy Model", type="primary"):
                    with st.spinner("Deploying model..."):
                        # Simulate deployment
                        import time
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        steps = [
                            "Validating model...",
                            "Creating container...",
                            "Configuring endpoint...",
                            "Running health checks...",
                            "Deployment complete!"
                        ]

                        for i, step in enumerate(steps):
                            status_text.text(step)
                            progress_bar.progress((i + 1) / len(steps))
                            time.sleep(0.5)

                        # Add to deployed models
                        new_model = {
                            'model_id': f'MODEL_{len(st.session_state.deployed_models) + 1:03d}',
                            'name': deployment_name,
                            'model_type': selected_model_name,
                            'version': version,
                            'status': 'Active' if environment == 'Production' else 'Testing',
                            'accuracy': 0.92,
                            'precision': 0.88,
                            'recall': 0.90,
                            'f1_score': 0.89,
                            'deployed_date': datetime.now(),
                            'last_updated': datetime.now(),
                            'predictions_today': 0,
                            'avg_latency_ms': 35.0,
                            'endpoint': f'https://api.argusai.com/v1/predict/{deployment_name.lower().replace(" ", "_")}'
                        }

                        st.session_state.deployed_models.append(new_model)

                        st.success(f"""
                            Model deployed successfully!
                            - Endpoint: {new_model['endpoint']}
                            - Status: {new_model['status']}
                            - Environment: {environment}
                        """)

            with col2:
                if st.button("Test Deployment"):
                    st.info("Running deployment tests...")

    else:
        st.warning("No trained models available. Please train models in the 'Data Science Workflow' tab first.")

        st.markdown("---")
        st.markdown("#### Quick Deploy - Pre-trained Models")

        st.info("You can also deploy pre-trained models from the Model Registry")

        if st.button("Go to Model Registry"):
            st.info("Navigate to Model Registry tab")


def show_realtime_inference():
    st.markdown("### Real-Time Inference")
    st.markdown("Test models with real-time predictions")

    # Check if models are deployed
    active_models = [m for m in st.session_state.deployed_models if m['status'] == 'Active']

    if len(active_models) == 0:
        st.warning("No active models available. Please deploy a model first.")
        return

    # Model selection
    st.markdown("#### 1⃣ Select Model")

    selected_model_name = st.selectbox(
        "Choose model:",
        [f"{m['name']} (v{m['version']})" for m in active_models]
    )

    selected_model = next(m for m in active_models if f"{m['name']} (v{m['version']})" == selected_model_name)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Accuracy", f"{selected_model['accuracy']:.2%}")

    with col2:
        st.metric("Avg Latency", f"{selected_model['avg_latency_ms']:.1f}ms")

    with col3:
        st.metric("Predictions Today", f"{selected_model['predictions_today']:,}")

    # Input method
    st.markdown("---")
    st.markdown("#### 2⃣ Input Transaction Data")

    input_method = st.radio("Input Method:", ["Manual Entry", "JSON Input", "Batch Upload", "Live Stream"])

    if input_method == "Manual Entry":
        show_manual_input(selected_model)

    elif input_method == "JSON Input":
        show_json_input(selected_model)

    elif input_method == "Batch Upload":
        show_batch_upload(selected_model)

    else:  # Live Stream
        show_live_stream(selected_model)

    # Inference history
    if len(st.session_state.inference_history) > 0:
        st.markdown("---")
        st.markdown("#### Recent Predictions")

        history_df = pd.DataFrame(st.session_state.inference_history[-10:][::-1])
        st.dataframe(history_df, use_container_width=True, hide_index=True)


def show_manual_input(model):
    """Manual transaction input for inference"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Transaction Details:**")
        transaction_amount = st.number_input("Transaction Amount ($):", 0.0, 100000.0, 500.0)
        merchant_category = st.selectbox("Merchant Category:",
                                        ["retail", "groceries", "gas_station", "restaurant",
                                         "online", "gambling", "crypto", "wire_transfer"])
        transaction_hour = st.slider("Transaction Hour:", 0, 23, 14)

    with col2:
        st.markdown("**Customer Details:**")
        customer_age = st.number_input("Customer Age:", 18, 100, 35)
        account_age_days = st.number_input("Account Age (days):", 0, 3650, 365)
        previous_transactions = st.number_input("Previous Transactions:", 0, 10000, 50)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Location:**")
        is_foreign = st.checkbox("Foreign Transaction")
        distance_from_home = st.number_input("Distance from Home (km):", 0.0, 10000.0, 5.0)

    with col2:
        st.markdown("**Device:**")
        device_type = st.selectbox("Device Type:", ["Desktop", "Mobile", "Tablet"])
        new_device = st.checkbox("New Device")

    if st.button("Predict", type="primary"):
        with st.spinner("Running prediction..."):
            # Simulate prediction
            import time
            time.sleep(0.5)

            # Generate prediction
            fraud_score = np.random.uniform(0.1, 0.95)
            is_fraud = fraud_score > 0.5

            # Display result
            st.markdown("---")
            st.markdown("#### Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Fraud Probability", f"{fraud_score:.2%}")

            with col2:
                prediction_label = "FRAUD" if is_fraud else "LEGITIMATE"
                st.metric("Prediction", prediction_label)

            with col3:
                confidence = abs(fraud_score - 0.5) * 2
                st.metric("Confidence", f"{confidence:.2%}")

            # Risk factors
            if is_fraud:
                st.markdown("#### Risk Factors")
                risk_factors = []

                if transaction_amount > 1000:
                    risk_factors.append(f"High transaction amount: ${transaction_amount:,.2f}")
                if merchant_category in ['gambling', 'crypto', 'wire_transfer']:
                    risk_factors.append(f"High-risk merchant category: {merchant_category}")
                if transaction_hour < 6 or transaction_hour > 22:
                    risk_factors.append(f"Unusual transaction hour: {transaction_hour}:00")
                if is_foreign:
                    risk_factors.append("Foreign transaction detected")
                if distance_from_home > 100:
                    risk_factors.append(f"Large distance from home: {distance_from_home:.0f} km")

                for factor in risk_factors[:5]:
                    st.warning(f"{factor}")

            # Add to history
            st.session_state.inference_history.append({
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Model': model['name'],
                'Amount': f"${transaction_amount:,.2f}",
                'Fraud Score': f"{fraud_score:.2%}",
                'Prediction': 'Fraud' if is_fraud else 'Legitimate',
                'Latency': f"{np.random.uniform(20, 80):.1f}ms"
            })


def show_json_input(model):
    """JSON input for inference"""
    st.markdown("**Paste transaction JSON:**")

    sample_json = {
        "transaction_amount": 1250.00,
        "merchant_category": "online",
        "transaction_hour": 14,
        "customer_age": 35,
        "account_age_days": 730,
        "previous_transactions": 125,
        "is_foreign": False,
        "distance_from_home": 5.2,
        "device_type": "mobile"
    }

    json_input = st.text_area("Transaction JSON:",
                             value=json.dumps(sample_json, indent=2),
                             height=300)

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Predict", type="primary"):
            try:
                data = json.loads(json_input)
                st.success("Valid JSON - Processing prediction...")

                # Simulate prediction
                fraud_score = np.random.uniform(0.1, 0.95)
                is_fraud = fraud_score > 0.5

                st.markdown("#### Prediction Result")
                st.metric("Fraud Probability", f"{fraud_score:.2%}")
                st.metric("Prediction", "FRAUD" if is_fraud else "LEGITIMATE")

            except json.JSONDecodeError:
                st.error("Invalid JSON format")

    with col2:
        if st.button("Use Sample JSON"):
            st.info("Sample JSON loaded")


def show_batch_upload(model):
    """Batch prediction from CSV upload"""
    st.markdown("**Upload CSV file for batch predictions:**")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown(f"**Loaded {len(df)} transactions**")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run Batch Predictions", type="primary"):
            with st.spinner(f"Processing {len(df)} transactions..."):
                # Simulate batch prediction
                import time
                progress_bar = st.progress(0)

                predictions = []
                for i in range(len(df)):
                    fraud_score = np.random.uniform(0.1, 0.95)
                    predictions.append({
                        'fraud_score': fraud_score,
                        'prediction': 'Fraud' if fraud_score > 0.5 else 'Legitimate'
                    })

                    if i % 10 == 0:
                        progress_bar.progress((i + 1) / len(df))
                        time.sleep(0.01)

                progress_bar.progress(1.0)

                # Add predictions to dataframe
                results_df = df.copy()
                results_df['fraud_score'] = [p['fraud_score'] for p in predictions]
                results_df['prediction'] = [p['prediction'] for p in predictions]

                st.success(f"Batch predictions completed!")

                # Summary
                col1, col2, col3 = st.columns(3)

                with col1:
                    fraud_count = len([p for p in predictions if p['prediction'] == 'Fraud'])
                    st.metric("Flagged as Fraud", fraud_count)

                with col2:
                    fraud_pct = (fraud_count / len(predictions)) * 100
                    st.metric("Fraud Rate", f"{fraud_pct:.2%}")

                with col3:
                    avg_score = np.mean([p['fraud_score'] for p in predictions])
                    st.metric("Avg Fraud Score", f"{avg_score:.2%}")

                # Display results
                st.markdown("#### Prediction Results")
                st.dataframe(results_df, use_container_width=True)

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def show_live_stream(model):
    """Simulate live transaction stream"""
    st.markdown("**Simulate live transaction stream:**")

    col1, col2 = st.columns(2)

    with col1:
        stream_rate = st.slider("Transactions per second:", 1, 100, 10)

    with col2:
        duration = st.number_input("Duration (seconds):", 5, 60, 10)

    if st.button("Start Stream", type="primary"):
        st.markdown("#### Live Stream")

        stream_placeholder = st.empty()
        metrics_placeholder = st.empty()

        import time

        total_processed = 0
        total_fraud = 0

        for i in range(duration):
            # Generate transactions for this second
            transactions = []

            for _ in range(stream_rate):
                fraud_score = np.random.uniform(0.1, 0.95)
                is_fraud = fraud_score > 0.5

                transactions.append({
                    'Time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                    'Amount': f"${np.random.uniform(10, 5000):,.2f}",
                    'Score': f"{fraud_score:.2%}",
                    'Result': 'Fraud' if is_fraud else 'Legit'
                })

                total_processed += 1
                if is_fraud:
                    total_fraud += 1

            # Display recent transactions
            stream_df = pd.DataFrame(transactions[-10:])
            stream_placeholder.dataframe(stream_df, use_container_width=True, hide_index=True)

            # Update metrics
            col1, col2, col3, col4 = metrics_placeholder.columns(4)

            with col1:
                col1.metric("Processed", total_processed)

            with col2:
                col2.metric("Fraud Detected", total_fraud)

            with col3:
                fraud_rate = (total_fraud / total_processed) * 100 if total_processed > 0 else 0
                col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            with col4:
                col4.metric("TPS", stream_rate)

            time.sleep(1)

        st.success(f"Stream completed! Processed {total_processed} transactions")


def show_model_monitoring():
    st.markdown("### Model Monitoring")
    st.markdown("Monitor model performance and health")

    # Select model
    active_models = [m for m in st.session_state.deployed_models if m['status'] == 'Active']

    if len(active_models) == 0:
        st.warning("No active models to monitor")
        return

    selected_model_name = st.selectbox(
        "Select model:",
        [f"{m['name']} (v{m['version']})" for m in active_models]
    )

    selected_model = next(m for m in active_models if f"{m['name']} (v{m['version']})" == selected_model_name)

    # Time range
    col1, col2 = st.columns(2)

    with col1:
        time_range = st.selectbox("Time Range:", ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"])

    with col2:
        refresh = st.button("Refresh")

    st.markdown("---")

    # Performance metrics
    st.markdown("#### Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Requests", f"{selected_model['predictions_today']:,}",
                 delta="+12.5%")

    with col2:
        st.metric("Avg Latency", f"{selected_model['avg_latency_ms']:.1f}ms",
                 delta="-2.3ms")

    with col3:
        st.metric("Success Rate", "99.8%", delta="+0.1%")

    with col4:
        st.metric("Fraud Detection", "7.2%", delta="+0.5%")

    with col5:
        st.metric("Uptime", "99.95%")

    # Charts
    st.markdown("---")
    st.markdown("#### Monitoring Charts")

    # Generate sample data
    hours = pd.date_range(end=datetime.now(), periods=24, freq='H')

    col1, col2 = st.columns(2)

    with col1:
        # Request volume
        requests_data = pd.DataFrame({
            'time': hours,
            'requests': np.random.randint(400, 800, 24)
        })

        fig = px.line(requests_data, x='time', y='requests',
                     title='Request Volume',
                     labels={'time': 'Time', 'requests': 'Requests'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Latency
        latency_data = pd.DataFrame({
            'time': hours,
            'latency': np.random.uniform(30, 60, 24)
        })

        fig = px.line(latency_data, x='time', y='latency',
                     title='Average Latency',
                     labels={'time': 'Time', 'latency': 'Latency (ms)'})
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Prediction distribution
        pred_dist = pd.DataFrame({
            'Prediction': ['Legitimate', 'Fraud'],
            'Count': [selected_model['predictions_today'] * 0.928,
                     selected_model['predictions_today'] * 0.072]
        })

        fig = px.pie(pred_dist, values='Count', names='Prediction',
                    title='Prediction Distribution',
                    color_discrete_map={'Legitimate': '#744ada', 'Fraud': '#000000'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Error rate
        error_data = pd.DataFrame({
            'time': hours,
            'error_rate': np.random.uniform(0, 0.5, 24)
        })

        fig = px.line(error_data, x='time', y='error_rate',
                     title='Error Rate (%)',
                     labels={'time': 'Time', 'error_rate': 'Error Rate'})
        st.plotly_chart(fig, use_container_width=True)

    # Model health
    st.markdown("---")
    st.markdown("#### Model Health")

    health_metrics = pd.DataFrame({
        'Component': ['API Endpoint', 'Model Server', 'Database', 'Cache', 'Load Balancer'],
        'Status': ['Healthy', 'Healthy', 'Healthy', 'Degraded', 'Healthy'],
        'Response Time': ['15ms', '35ms', '5ms', '120ms', '8ms'],
        'Last Check': ['1 min ago', '1 min ago', '1 min ago', '1 min ago', '1 min ago']
    })

    def color_status(val):
        if val == 'Healthy':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Degraded':
            return 'background-color: #fff3cd; color: #856404'
        else:
            return 'background-color: #f8d7da; color: #721c24'

    styled_health = health_metrics.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_health, use_container_width=True, hide_index=True)

    # Alerts
    st.markdown("---")
    st.markdown("#### Recent Alerts")

    alerts = [
        {'Time': '10 min ago', 'Severity': 'Warning', 'Message': 'Cache response time elevated'},
        {'Time': '2 hours ago', 'Severity': 'Info', 'Message': 'Model redeployed successfully'},
        {'Time': '1 day ago', 'Severity': 'Warning', 'Message': 'Unusual spike in fraud predictions'}
    ]

    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)


def show_model_management():
    st.markdown("### Model Management")
    st.markdown("Manage deployed models and configurations")

    deployed_models = st.session_state.deployed_models

    if len(deployed_models) == 0:
        st.warning("No deployed models")
        return

    # Select model
    selected_model_name = st.selectbox(
        "Select model:",
        [f"{m['name']} (v{m['version']})" for m in deployed_models]
    )

    selected_model = next(m for m in deployed_models if f"{m['name']} (v{m['version']})" == selected_model_name)

    # Model actions
    st.markdown("---")
    st.markdown("#### Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Update Model"):
            st.info("Model update wizard")

    with col2:
        if st.button("Pause Model"):
            st.warning("Model paused")

    with col3:
        if st.button("Export Metrics"):
            st.success("Metrics exported")

    with col4:
        if st.button("Delete Model"):
            st.error("Delete confirmation required")

    # Configuration
    st.markdown("---")
    st.markdown("#### Model Configuration")

    with st.expander("Scaling Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            min_instances = st.number_input("Min Instances:", 1, 10, 1, key="min_inst")
            max_instances = st.number_input("Max Instances:", 1, 50, 5, key="max_inst")

        with col2:
            target_latency = st.number_input("Target Latency (ms):", 10, 1000, 50)
            scale_up_threshold = st.slider("Scale Up Threshold:", 50, 100, 80)

    with st.expander("Performance Tuning", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            batch_size = st.number_input("Batch Size:", 1, 128, 32, key="batch")
            timeout = st.number_input("Timeout (s):", 1, 60, 30, key="timeout")

        with col2:
            cache_enabled = st.checkbox("Enable Caching", value=True)
            compression = st.checkbox("Enable Compression", value=True)

    with st.expander("Monitoring & Alerts", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            alert_latency = st.number_input("Alert on Latency > (ms):", 50, 1000, 100)
            alert_error_rate = st.number_input("Alert on Error Rate > (%):", 1, 20, 5)

        with col2:
            notification_email = st.text_input("Notification Email:")
            slack_webhook = st.text_input("Slack Webhook URL:")

    if st.button("Save Configuration", type="primary"):
        st.success("Configuration saved successfully!")

    # Version history
    st.markdown("---")
    st.markdown("#### Version History")

    versions = pd.DataFrame({
        'Version': ['3.0', '2.1', '2.0', '1.5', '1.0'],
        'Deployed': ['2024-12-10', '2024-11-15', '2024-10-20', '2024-09-10', '2024-08-01'],
        'Status': ['Active', 'Archived', 'Archived', 'Archived', 'Archived'],
        'Accuracy': ['94.2%', '93.8%', '92.5%', '91.2%', '89.8%'],
        'Notes': ['Current production', 'Stable', 'Baseline', 'Initial deployment', 'v1.0']
    })

    st.dataframe(versions, use_container_width=True, hide_index=True)

    # Rollback option
    st.markdown("#### Rollback")

    col1, col2 = st.columns([2, 1])

    with col1:
        rollback_version = st.selectbox("Select version to rollback to:", versions['Version'].tolist()[1:])

    with col2:
        if st.button("Rollback", type="secondary"):
            st.warning(f"Rollback to version {rollback_version} requires confirmation")
