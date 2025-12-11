"""
Feature Monitoring Module
Monitor feature drift, quality, and data health
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import sys
sys.path.append('/home/user/ArgusAI')
from src.utils.data_generator import generate_fraud_data


def show():
    st.markdown('<p class="main-header">📊 Feature Monitoring</p>', unsafe_allow_html=True)
    st.markdown("Monitor feature drift, data quality, and feature health")

    # Initialize session state
    if 'monitoring_data' not in st.session_state:
        st.session_state.monitoring_data = None
    if 'baseline_data' not in st.session_state:
        st.session_state.baseline_data = None
    if 'feature_alerts' not in st.session_state:
        st.session_state.feature_alerts = []

    # Create tabs
    tabs = st.tabs([
        "Feature Drift",
        "Data Quality",
        "Feature Statistics",
        "Alerts & Anomalies",
        "Distribution Analysis",
        "Feature Importance Tracking"
    ])

    with tabs[0]:
        show_feature_drift()

    with tabs[1]:
        show_data_quality()

    with tabs[2]:
        show_feature_statistics()

    with tabs[3]:
        show_alerts_anomalies()

    with tabs[4]:
        show_distribution_analysis()

    with tabs[5]:
        show_feature_importance_tracking()


def show_feature_drift():
    st.markdown("### Feature Drift Detection")
    st.markdown("Detect and monitor feature distribution drift over time")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Baseline Data")
        if st.button("Load Baseline Data"):
            with st.spinner("Loading baseline data..."):
                st.session_state.baseline_data = generate_fraud_data(n_samples=5000, fraud_rate=0.05)
                st.success("Baseline data loaded (5,000 records)")

    with col2:
        st.markdown("#### Current Data")
        if st.button("Load Current Data"):
            with st.spinner("Loading current data..."):
                # Simulate drift by modifying distributions
                st.session_state.monitoring_data = generate_fraud_data(n_samples=5000, fraud_rate=0.05)
                # Add drift to some features
                st.session_state.monitoring_data['transaction_amount'] *= np.random.uniform(1.2, 1.5)
                st.success("Current data loaded (5,000 records)")

    if st.session_state.baseline_data is not None and st.session_state.monitoring_data is not None:
        st.markdown("---")
        st.markdown("#### Drift Analysis")

        baseline = st.session_state.baseline_data
        current = st.session_state.monitoring_data

        # Select feature to analyze
        numeric_features = baseline.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraud' in numeric_features:
            numeric_features.remove('is_fraud')

        selected_feature = st.selectbox("Select feature to analyze:", numeric_features)

        # Calculate drift metrics
        drift_metrics = calculate_drift_metrics(
            baseline[selected_feature],
            current[selected_feature]
        )

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("KS Statistic", f"{drift_metrics['ks_statistic']:.4f}")

        with col2:
            psi_color = "" if drift_metrics['psi'] < 0.1 else "" if drift_metrics['psi'] < 0.2 else ""
            st.metric("PSI", f"{psi_color} {drift_metrics['psi']:.4f}")

        with col3:
            st.metric("Mean Shift", f"{drift_metrics['mean_shift']:.2%}")

        with col4:
            st.metric("Std Shift", f"{drift_metrics['std_shift']:.2%}")

        # Drift interpretation
        if drift_metrics['psi'] < 0.1:
            st.success("No significant drift detected")
        elif drift_metrics['psi'] < 0.2:
            st.warning("Moderate drift detected - monitor closely")
        else:
            st.error("Significant drift detected - investigation required")

        # Visualization
        st.markdown("---")
        st.markdown("#### Distribution Comparison")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Baseline Distribution', 'Current Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Baseline histogram
        fig.add_trace(
            go.Histogram(x=baseline[selected_feature], name='Baseline',
                        marker_color='#744ada', opacity=0.7, nbinsx=50),
            row=1, col=1
        )

        # Current histogram
        fig.add_trace(
            go.Histogram(x=current[selected_feature], name='Current',
                        marker_color='#000000', opacity=0.7, nbinsx=50),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=True, title_text=f"{selected_feature} - Distribution Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Overlay comparison
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=baseline[selected_feature], name='Baseline',
                                   marker_color='#744ada', opacity=0.5, nbinsx=50))
        fig2.add_trace(go.Histogram(x=current[selected_feature], name='Current',
                                   marker_color='#000000', opacity=0.5, nbinsx=50))
        fig2.update_layout(
            title=f"{selected_feature} - Overlay Comparison",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Drift over time simulation
        st.markdown("---")
        st.markdown("#### Drift Trends Over Time")

        time_periods = pd.date_range(end=datetime.now(), periods=30, freq='D')
        psi_values = np.random.uniform(0.05, 0.3, 30)
        psi_values = np.sort(psi_values)  # Simulate increasing drift

        drift_df = pd.DataFrame({
            'date': time_periods,
            'PSI': psi_values
        })

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=drift_df['date'], y=drift_df['PSI'],
                                 mode='lines+markers', name='PSI',
                                 line=dict(color='#744ada', width=2)))

        # Add threshold lines
        fig3.add_hline(y=0.1, line_dash="dash", line_color="#744ada",
                      annotation_text="Low Drift Threshold")
        fig3.add_hline(y=0.2, line_dash="dash", line_color="#666666",
                      annotation_text="High Drift Threshold")

        fig3.update_layout(
            title=f"{selected_feature} - PSI Trend (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="PSI",
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

        # All features drift summary
        st.markdown("---")
        st.markdown("#### All Features Drift Summary")

        drift_summary = []
        for feature in numeric_features[:10]:  # Limit to first 10 features
            metrics = calculate_drift_metrics(baseline[feature], current[feature])
            status = "GOOD" if metrics['psi'] < 0.1 else "MONITOR" if metrics['psi'] < 0.2 else "ALERT"

            drift_summary.append({
                'Feature': feature,
                'PSI': f"{metrics['psi']:.4f}",
                'KS Statistic': f"{metrics['ks_statistic']:.4f}",
                'Mean Shift': f"{metrics['mean_shift']:.2%}",
                'Status': status
            })

        drift_summary_df = pd.DataFrame(drift_summary)
        st.dataframe(drift_summary_df, use_container_width=True, hide_index=True)

    else:
        st.info("👆 Load both baseline and current data to begin drift analysis")


def calculate_drift_metrics(baseline, current):
    """Calculate drift metrics between baseline and current data"""
    # Remove NaN values
    baseline_clean = baseline.dropna()
    current_clean = current.dropna()

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(baseline_clean, current_clean)

    # PSI calculation
    psi = calculate_psi(baseline_clean, current_clean)

    # Mean and std shift
    mean_shift = (current_clean.mean() - baseline_clean.mean()) / (baseline_clean.mean() + 1e-10)
    std_shift = (current_clean.std() - baseline_clean.std()) / (baseline_clean.std() + 1e-10)

    return {
        'ks_statistic': ks_stat,
        'ks_pval': ks_pval,
        'psi': psi,
        'mean_shift': mean_shift,
        'std_shift': std_shift
    }


def calculate_psi(baseline, current, bins=10):
    """Calculate Population Stability Index"""
    # Create bins based on baseline
    breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    # Calculate distributions
    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts = np.histogram(current, bins=breakpoints)[0]

    # Normalize
    baseline_perc = baseline_counts / len(baseline)
    current_perc = current_counts / len(current)

    # Avoid division by zero
    baseline_perc = np.where(baseline_perc == 0, 0.0001, baseline_perc)
    current_perc = np.where(current_perc == 0, 0.0001, current_perc)

    # Calculate PSI
    psi = np.sum((current_perc - baseline_perc) * np.log(current_perc / baseline_perc))

    return psi


def show_data_quality():
    st.markdown("### Data Quality Monitoring")
    st.markdown("Monitor data quality metrics and completeness")

    # Generate or load data
    if st.button("Load Data for Quality Check"):
        with st.spinner("Loading data..."):
            st.session_state.monitoring_data = generate_fraud_data(n_samples=10000, fraud_rate=0.05)
            # Introduce some quality issues for demonstration
            df = st.session_state.monitoring_data
            # Add missing values
            df.loc[df.sample(frac=0.05).index, 'transaction_amount'] = np.nan
            df.loc[df.sample(frac=0.03).index, 'merchant_category'] = np.nan
            st.success("Data loaded (10,000 records)")

    if st.session_state.monitoring_data is not None:
        df = st.session_state.monitoring_data

        # Overall quality score
        st.markdown("---")
        st.markdown("#### Overall Data Quality Score")

        quality_metrics = calculate_quality_metrics(df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            completeness = quality_metrics['completeness']
            color = "" if completeness > 95 else "" if completeness > 90 else ""
            st.metric("Completeness", f"{color} {completeness:.1f}%")

        with col2:
            validity = quality_metrics['validity']
            color = "" if validity > 95 else "" if validity > 90 else ""
            st.metric("Validity", f"{color} {validity:.1f}%")

        with col3:
            consistency = quality_metrics['consistency']
            color = "" if consistency > 95 else "" if consistency > 90 else ""
            st.metric("Consistency", f"{color} {consistency:.1f}%")

        with col4:
            overall_score = (completeness + validity + consistency) / 3
            color = "" if overall_score > 95 else "" if overall_score > 90 else ""
            st.metric("Overall Score", f"{color} {overall_score:.1f}%")

        # Missing values analysis
        st.markdown("---")
        st.markdown("#### Missing Values Analysis")

        missing_data = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100

            if missing_count > 0:
                missing_data.append({
                    'Feature': col,
                    'Missing Count': missing_count,
                    'Missing %': f"{missing_pct:.2f}%",
                    'Status': 'GOOD' if missing_pct < 5 else 'MONITOR' if missing_pct < 10 else 'ALERT'
                })

        if missing_data:
            missing_df = pd.DataFrame(missing_data)
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

            # Visualize missing data
            fig = px.bar(missing_df, x='Feature', y='Missing Count',
                        title='Missing Values by Feature',
                        color='Missing Count',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values detected")

        # Outlier detection
        st.markdown("---")
        st.markdown("#### Outlier Detection")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraud' in numeric_cols:
            numeric_cols.remove('is_fraud')

        outlier_summary = []
        for col in numeric_cols[:10]:  # First 10 numeric features
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
            outlier_pct = (outliers / len(df)) * 100

            outlier_summary.append({
                'Feature': col,
                'Outliers': outliers,
                'Outlier %': f"{outlier_pct:.2f}%",
                'Status': 'GOOD' if outlier_pct < 5 else 'MONITOR' if outlier_pct < 10 else 'ALERT'
            })

        outlier_df = pd.DataFrame(outlier_summary)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)

        # Data freshness
        st.markdown("---")
        st.markdown("#### Data Freshness")

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            latest_timestamp = df['timestamp'].max()
            oldest_timestamp = df['timestamp'].min()
            data_age = (datetime.now() - latest_timestamp).total_seconds() / 3600

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Latest Record", latest_timestamp.strftime('%Y-%m-%d %H:%M'))

            with col2:
                st.metric("Data Age", f"{data_age:.1f} hours")

            with col3:
                status = "FRESH" if data_age < 1 else "Moderate" if data_age < 24 else "Stale"
                st.metric("Freshness Status", status)

        # Quality trends
        st.markdown("---")
        st.markdown("#### Quality Trends (Last 30 Days)")

        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        completeness_trend = np.random.uniform(92, 99, 30)
        validity_trend = np.random.uniform(94, 99, 30)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=completeness_trend, mode='lines+markers',
                                name='Completeness', line=dict(color='#744ada')))
        fig.add_trace(go.Scatter(x=dates, y=validity_trend, mode='lines+markers',
                                name='Validity', line=dict(color='#744ada')))

        fig.add_hline(y=95, line_dash="dash", line_color="#000000",
                     annotation_text="Target Threshold (95%)")

        fig.update_layout(
            title='Data Quality Trends',
            xaxis_title='Date',
            yaxis_title='Quality Score (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Load data to begin quality monitoring")


def calculate_quality_metrics(df):
    """Calculate data quality metrics"""
    # Completeness: % of non-null values
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100

    # Validity: % of values within expected ranges (simplified)
    validity_checks = []
    if 'transaction_amount' in df.columns:
        validity_checks.append((df['transaction_amount'] >= 0).sum() / len(df))
    if 'customer_age' in df.columns:
        validity_checks.append(((df['customer_age'] >= 18) & (df['customer_age'] <= 100)).sum() / len(df))

    validity = np.mean(validity_checks) * 100 if validity_checks else 100

    # Consistency: % of records without contradictions (simplified)
    consistency = 98.5  # Placeholder

    return {
        'completeness': completeness,
        'validity': validity,
        'consistency': consistency
    }


def show_feature_statistics():
    st.markdown("### Feature Statistics")
    st.markdown("Detailed statistical analysis of features over time")

    if st.session_state.monitoring_data is None:
        if st.button("Load Data"):
            st.session_state.monitoring_data = generate_fraud_data(n_samples=10000, fraud_rate=0.05)
            st.success("Data loaded")

    if st.session_state.monitoring_data is not None:
        df = st.session_state.monitoring_data

        # Feature selector
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraud' in numeric_features:
            numeric_features.remove('is_fraud')

        selected_feature = st.selectbox("Select feature:", numeric_features)

        # Statistics summary
        st.markdown("---")
        st.markdown(f"#### 📊 {selected_feature} - Summary Statistics")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Mean", f"{df[selected_feature].mean():.2f}")

        with col2:
            st.metric("Median", f"{df[selected_feature].median():.2f}")

        with col3:
            st.metric("Std Dev", f"{df[selected_feature].std():.2f}")

        with col4:
            st.metric("Min", f"{df[selected_feature].min():.2f}")

        with col5:
            st.metric("Max", f"{df[selected_feature].max():.2f}")

        # Distribution plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribution', 'Box Plot')
        )

        fig.add_trace(
            go.Histogram(x=df[selected_feature], nbinsx=50, name='Distribution'),
            row=1, col=1
        )

        fig.add_trace(
            go.Box(y=df[selected_feature], name='Box Plot'),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Percentiles
        st.markdown("---")
        st.markdown("#### Percentile Distribution")

        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = [df[selected_feature].quantile(p/100) for p in percentiles]

        percentile_df = pd.DataFrame({
            'Percentile': [f"{p}th" for p in percentiles],
            'Value': [f"{v:.2f}" for v in percentile_values]
        })

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(percentile_df, use_container_width=True, hide_index=True)

        with col2:
            fig = px.bar(percentile_df, x='Percentile', y=[float(v) for v in percentile_df['Value']],
                        title='Percentile Values')
            st.plotly_chart(fig, use_container_width=True)

        # Time series if timestamp available
        if 'timestamp' in df.columns:
            st.markdown("---")
            st.markdown("#### Time Series Analysis")

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date

            daily_stats = df.groupby('date')[selected_feature].agg(['mean', 'std', 'min', 'max']).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['mean'],
                                    mode='lines+markers', name='Mean'))
            fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['std'],
                                    mode='lines+markers', name='Std Dev'))

            fig.update_layout(title=f'{selected_feature} - Daily Statistics',
                            xaxis_title='Date',
                            yaxis_title='Value',
                            height=400)
            st.plotly_chart(fig, use_container_width=True)


def show_alerts_anomalies():
    st.markdown("### Alerts & Anomalies")
    st.markdown("Monitor and manage feature-related alerts")

    # Alert configuration
    st.markdown("#### Alert Configuration")

    with st.expander("Configure Alert Thresholds", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Drift Thresholds:**")
            psi_threshold = st.slider("PSI Alert Threshold:", 0.0, 0.5, 0.2, 0.05)
            ks_threshold = st.slider("KS Statistic Threshold:", 0.0, 1.0, 0.2, 0.05)

        with col2:
            st.markdown("**Quality Thresholds:**")
            missing_threshold = st.slider("Missing Value % Alert:", 0, 50, 10)
            outlier_threshold = st.slider("Outlier % Alert:", 0, 50, 15)

    # Generate alerts
    if st.button("Scan for Anomalies"):
        with st.spinner("Scanning for anomalies..."):
            alerts = generate_feature_alerts()
            st.session_state.feature_alerts = alerts
            st.success(f"Scan complete - {len(alerts)} alerts found")

    # Display alerts
    if len(st.session_state.feature_alerts) > 0:
        st.markdown("---")
        st.markdown("#### Active Alerts")

        # Alert summary
        col1, col2, col3 = st.columns(3)

        critical_alerts = [a for a in st.session_state.feature_alerts if a['severity'] == 'Critical']
        warning_alerts = [a for a in st.session_state.feature_alerts if a['severity'] == 'Warning']
        info_alerts = [a for a in st.session_state.feature_alerts if a['severity'] == 'Info']

        with col1:
            st.metric("CRITICAL", len(critical_alerts))

        with col2:
            st.metric("Warning", len(warning_alerts))

        with col3:
            st.metric("🔵 Info", len(info_alerts))

        # Alert list
        for alert in st.session_state.feature_alerts:
            severity_icon = "" if alert['severity'] == 'Critical' else "" if alert['severity'] == 'Warning' else "🔵"

            with st.expander(f"{severity_icon} {alert['title']}", expanded=(alert['severity'] == 'Critical')):
                st.markdown(f"**Feature:** {alert['feature']}")
                st.markdown(f"**Severity:** {alert['severity']}")
                st.markdown(f"**Description:** {alert['description']}")
                st.markdown(f"**Detected:** {alert['timestamp']}")
                st.markdown(f"**Metric Value:** {alert['value']}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                        st.info("Alert acknowledged")

                with col2:
                    if st.button("Investigate", key=f"inv_{alert['id']}"):
                        st.info("Opening investigation...")

                with col3:
                    if st.button("Dismiss", key=f"dis_{alert['id']}"):
                        st.success("Alert dismissed")

    else:
        st.info("No alerts detected. Click 'Scan for Anomalies' to check for issues.")


def generate_feature_alerts():
    """Generate sample feature alerts"""
    alerts = [
        {
            'id': 'ALERT_001',
            'title': 'High Feature Drift Detected',
            'feature': 'transaction_amount',
            'severity': 'Critical',
            'description': 'PSI value of 0.28 exceeds threshold of 0.20. Significant distribution shift detected.',
            'value': 'PSI: 0.28',
            'timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')
        },
        {
            'id': 'ALERT_002',
            'title': 'Increased Missing Values',
            'feature': 'merchant_category',
            'severity': 'Warning',
            'description': 'Missing value percentage increased from 2% to 12% in last 24 hours.',
            'value': 'Missing: 12%',
            'timestamp': (datetime.now() - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M')
        },
        {
            'id': 'ALERT_003',
            'title': 'Outlier Spike',
            'feature': 'distance_from_home',
            'severity': 'Warning',
            'description': 'Outlier percentage increased to 18%, above threshold of 15%.',
            'value': 'Outliers: 18%',
            'timestamp': (datetime.now() - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')
        },
        {
            'id': 'ALERT_004',
            'title': 'Mean Shift Detected',
            'feature': 'transactions_24h',
            'severity': 'Info',
            'description': 'Mean value shifted by +15% compared to baseline.',
            'value': 'Shift: +15%',
            'timestamp': (datetime.now() - timedelta(hours=12)).strftime('%Y-%m-%d %H:%M')
        },
        {
            'id': 'ALERT_005',
            'title': 'Data Freshness Issue',
            'feature': 'timestamp',
            'severity': 'Critical',
            'description': 'No new data received in last 3 hours. Data pipeline may be down.',
            'value': 'Last update: 3h ago',
            'timestamp': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M')
        }
    ]

    return alerts


def show_distribution_analysis():
    st.markdown("### Distribution Analysis")
    st.markdown("Analyze feature distributions across different segments")

    if st.session_state.monitoring_data is None:
        if st.button("Load Data"):
            st.session_state.monitoring_data = generate_fraud_data(n_samples=10000, fraud_rate=0.05)
            st.success("Data loaded")

    if st.session_state.monitoring_data is not None:
        df = st.session_state.monitoring_data

        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraud' in numeric_features:
            numeric_features.remove('is_fraud')

        selected_feature = st.selectbox("Select feature to analyze:", numeric_features, key='dist_feature')

        # Distribution by fraud status
        st.markdown("---")
        st.markdown("#### Distribution by Fraud Status")

        fig = px.histogram(df, x=selected_feature, color='is_fraud',
                          title=f'{selected_feature} Distribution by Fraud Status',
                          nbins=50, barmode='overlay',
                          color_discrete_map={0: '#744ada', 1: '#000000'},
                          labels={'is_fraud': 'Fraud Status'})
        st.plotly_chart(fig, use_container_width=True)

        # Statistical comparison
        fraud_stats = df[df['is_fraud'] == 1][selected_feature].describe()
        legit_stats = df[df['is_fraud'] == 0][selected_feature].describe()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Fraudulent Transactions:**")
            st.dataframe(fraud_stats, use_container_width=True)

        with col2:
            st.markdown("**Legitimate Transactions:**")
            st.dataframe(legit_stats, use_container_width=True)

        # Categorical analysis
        if 'merchant_category' in df.columns:
            st.markdown("---")
            st.markdown("#### Distribution by Merchant Category")

            category_stats = df.groupby('merchant_category')[selected_feature].agg(['mean', 'median', 'std']).reset_index()

            fig = px.bar(category_stats, x='merchant_category', y='mean',
                        title=f'Average {selected_feature} by Merchant Category',
                        error_y='std')
            st.plotly_chart(fig, use_container_width=True)


def show_feature_importance_tracking():
    st.markdown("### Feature Importance Tracking")
    st.markdown("Track how feature importance changes over time")

    # Simulated feature importance over time
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

    features = ['transaction_amount', 'transactions_24h', 'distance_from_home',
                'merchant_category', 'transaction_hour']

    st.markdown("#### Feature Importance Trends")

    # Generate importance data
    importance_data = []
    for feature in features:
        # Simulate importance trend
        base_importance = np.random.uniform(0.1, 0.3)
        trend = np.random.uniform(-0.05, 0.05, 30)
        importance = base_importance + np.cumsum(trend)
        importance = np.clip(importance, 0, 1)

        for date, imp in zip(dates, importance):
            importance_data.append({
                'date': date,
                'feature': feature,
                'importance': imp
            })

    importance_df = pd.DataFrame(importance_data)

    # Plot trends
    fig = px.line(importance_df, x='date', y='importance', color='feature',
                 title='Feature Importance Over Time',
                 labels={'importance': 'Importance Score', 'date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    # Current importance ranking
    st.markdown("---")
    st.markdown("#### Current Feature Ranking")

    current_importance = importance_df[importance_df['date'] == importance_df['date'].max()]
    current_importance = current_importance.sort_values('importance', ascending=False)

    fig = px.bar(current_importance, x='feature', y='importance',
                title='Current Feature Importance',
                color='importance',
                color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

    # Importance change
    st.markdown("---")
    st.markdown("#### Importance Change (Last 30 Days)")

    first_day = importance_df[importance_df['date'] == importance_df['date'].min()]
    last_day = importance_df[importance_df['date'] == importance_df['date'].max()]

    change_data = []
    for feature in features:
        first_imp = first_day[first_day['feature'] == feature]['importance'].values[0]
        last_imp = last_day[last_day['feature'] == feature]['importance'].values[0]
        change = ((last_imp - first_imp) / first_imp) * 100

        change_data.append({
            'Feature': feature,
            'Initial': f"{first_imp:.3f}",
            'Current': f"{last_imp:.3f}",
            'Change %': f"{change:+.2f}%",
            'Trend': '📈' if change > 0 else '📉'
        })

    change_df = pd.DataFrame(change_data)
    st.dataframe(change_df, use_container_width=True, hide_index=True)
