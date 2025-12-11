"""
Model Monitoring Module
Monitor model performance, fraud detection KPIs, and model health
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import sys
sys.path.append('/home/user/ArgusAI')


def show():
    st.markdown('<p class="main-header">🤖 Model Performance Monitoring</p>', unsafe_allow_html=True)
    st.markdown("Monitor model performance, fraud detection KPIs, and business metrics")

    # Initialize session state
    if 'model_performance_data' not in st.session_state:
        st.session_state.model_performance_data = generate_performance_data()

    # Create tabs
    tabs = st.tabs([
        "Performance Dashboard",
        "Fraud Detection KPIs",
        "Performance Trends",
        "Prediction Analysis",
        "Business Impact",
        "Model Health & Alerts",
        "Segment Performance"
    ])

    with tabs[0]:
        show_performance_dashboard()

    with tabs[1]:
        show_fraud_kpis()

    with tabs[2]:
        show_performance_trends()

    with tabs[3]:
        show_prediction_analysis()

    with tabs[4]:
        show_business_impact()

    with tabs[5]:
        show_model_health()

    with tabs[6]:
        show_segment_performance()


def generate_performance_data():
    """Generate sample performance data for monitoring"""
    # Generate daily metrics for last 30 days
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

    data = {
        'date': dates,
        'precision': np.random.uniform(0.85, 0.95, 30),
        'recall': np.random.uniform(0.80, 0.92, 30),
        'f1_score': np.random.uniform(0.83, 0.93, 30),
        'accuracy': np.random.uniform(0.92, 0.98, 30),
        'roc_auc': np.random.uniform(0.90, 0.98, 30),
        'fraud_detection_rate': np.random.uniform(0.78, 0.90, 30),
        'false_positive_rate': np.random.uniform(0.02, 0.08, 30),
        'total_predictions': np.random.randint(8000, 15000, 30),
        'fraud_detected': np.random.randint(500, 1200, 30),
        'true_positives': np.random.randint(450, 1000, 30),
        'false_positives': np.random.randint(50, 300, 30),
        'true_negatives': np.random.randint(7000, 13000, 30),
        'false_negatives': np.random.randint(50, 200, 30)
    }

    return pd.DataFrame(data)


def show_performance_dashboard():
    st.markdown("### Real-Time Performance Dashboard")
    st.markdown("Current model performance metrics and status")

    perf_data = st.session_state.model_performance_data
    latest = perf_data.iloc[-1]

    # Time range selector
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("#### Time Period")

    with col2:
        if st.button("Refresh Data"):
            st.session_state.model_performance_data = generate_performance_data()
            st.rerun()

    # Key metrics
    st.markdown("---")
    st.markdown("#### Current Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        prev_precision = perf_data.iloc[-2]['precision']
        delta_precision = ((latest['precision'] - prev_precision) / prev_precision) * 100
        st.metric("Precision", f"{latest['precision']:.2%}",
                 delta=f"{delta_precision:+.2f}%")

    with col2:
        prev_recall = perf_data.iloc[-2]['recall']
        delta_recall = ((latest['recall'] - prev_recall) / prev_recall) * 100
        st.metric("Recall", f"{latest['recall']:.2%}",
                 delta=f"{delta_recall:+.2f}%")

    with col3:
        prev_f1 = perf_data.iloc[-2]['f1_score']
        delta_f1 = ((latest['f1_score'] - prev_f1) / prev_f1) * 100
        st.metric("F1-Score", f"{latest['f1_score']:.2%}",
                 delta=f"{delta_f1:+.2f}%")

    with col4:
        prev_auc = perf_data.iloc[-2]['roc_auc']
        delta_auc = ((latest['roc_auc'] - prev_auc) / prev_auc) * 100
        st.metric("ROC-AUC", f"{latest['roc_auc']:.2%}",
                 delta=f"{delta_auc:+.2f}%")

    with col5:
        prev_acc = perf_data.iloc[-2]['accuracy']
        delta_acc = ((latest['accuracy'] - prev_acc) / prev_acc) * 100
        st.metric("Accuracy", f"{latest['accuracy']:.2%}",
                 delta=f"{delta_acc:+.2f}%")

    # Performance gauge charts
    st.markdown("---")
    st.markdown("#### Performance Gauges")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = create_gauge_chart(latest['precision'], "Precision", 0.85, 0.90)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_gauge_chart(latest['recall'], "Recall", 0.80, 0.85)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = create_gauge_chart(latest['roc_auc'], "ROC-AUC", 0.90, 0.95)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.markdown("---")
    st.markdown("#### Current Confusion Matrix")

    cm = np.array([
        [latest['true_negatives'], latest['false_positives']],
        [latest['false_negatives'], latest['true_positives']]
    ])

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.imshow(cm,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Legitimate', 'Fraud'],
                       y=['Legitimate', 'Fraud'],
                       text_auto=True,
                       color_continuous_scale='Blues',
                       aspect="auto")
        fig.update_layout(title='Confusion Matrix (Today)', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Breakdown:**")
        st.metric("True Positives", f"{int(latest['true_positives']):,}")
        st.metric("True Negatives", f"{int(latest['true_negatives']):,}")
        st.metric("False Positives", f"{int(latest['false_positives']):,}")
        st.metric("False Negatives", f"{int(latest['false_negatives']):,}")

    # Model status
    st.markdown("---")
    st.markdown("#### Model Health Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "HEALTHY" if latest['precision'] > 0.85 else "DEGRADED" if latest['precision'] > 0.75 else "CRITICAL"
        st.metric("Overall Status", status)

    with col2:
        predictions_status = "NORMAL" if latest['total_predictions'] > 10000 else "Low"
        st.metric("Prediction Volume", predictions_status)

    with col3:
        drift_status = "STABLE" if abs(delta_precision) < 2 else "Drifting"
        st.metric("Performance Drift", drift_status)

    with col4:
        last_updated = datetime.now() - timedelta(minutes=5)
        st.metric("Last Updated", last_updated.strftime('%H:%M'))


def create_gauge_chart(value, title, threshold_low, threshold_high):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': threshold_high * 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold_low * 100], 'color': "lightgray"},
                {'range': [threshold_low * 100, threshold_high * 100], 'color': "#9b7fe8"},
                {'range': [threshold_high * 100, 100], 'color': "#9b7fe8"}
            ],
            'threshold': {
                'line': {'color': "#000000", 'width': 4},
                'thickness': 0.75,
                'value': threshold_high * 100
            }
        }
    ))

    fig.update_layout(height=250)
    return fig


def show_fraud_kpis():
    st.markdown("### Fraud Detection KPIs")
    st.markdown("Key performance indicators for fraud detection")

    perf_data = st.session_state.model_performance_data
    latest = perf_data.iloc[-1]

    # Primary KPIs
    st.markdown("#### Primary Detection KPIs")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Fraud Detection Rate",
                 f"{latest['fraud_detection_rate']:.2%}",
                 help="% of actual fraud cases detected")

    with col2:
        st.metric("False Positive Rate",
                 f"{latest['false_positive_rate']:.2%}",
                 delta=f"-{np.random.uniform(0.1, 0.5):.2%}",
                 help="% of legitimate transactions flagged as fraud")

    with col3:
        fdr = latest['false_positives'] / (latest['false_positives'] + latest['true_positives'])
        st.metric("False Discovery Rate",
                 f"{fdr:.2%}",
                 help="% of fraud predictions that are wrong")

    with col4:
        fnr = latest['false_negatives'] / (latest['false_negatives'] + latest['true_positives'])
        st.metric("False Negative Rate",
                 f"{fnr:.2%}",
                 help="% of fraud cases missed")

    # Detection efficiency
    st.markdown("---")
    st.markdown("#### Detection Efficiency")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Predictions Today",
                 f"{int(latest['total_predictions']):,}")

    with col2:
        st.metric("Fraud Cases Detected",
                 f"{int(latest['fraud_detected']):,}")

    with col3:
        detection_rate = (latest['fraud_detected'] / latest['total_predictions']) * 100
        st.metric("Detection Rate",
                 f"{detection_rate:.2f}%")

    # KPI trends
    st.markdown("---")
    st.markdown("#### KPI Trends (Last 30 Days)")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Fraud Detection Rate', 'False Positive Rate',
                       'Precision', 'Recall')
    )

    # Fraud Detection Rate
    fig.add_trace(
        go.Scatter(x=perf_data['date'], y=perf_data['fraud_detection_rate'],
                  mode='lines+markers', name='Detection Rate', line=dict(color='#744ada')),
        row=1, col=1
    )

    # False Positive Rate
    fig.add_trace(
        go.Scatter(x=perf_data['date'], y=perf_data['false_positive_rate'],
                  mode='lines+markers', name='FP Rate', line=dict(color='#000000')),
        row=1, col=2
    )

    # Precision
    fig.add_trace(
        go.Scatter(x=perf_data['date'], y=perf_data['precision'],
                  mode='lines+markers', name='Precision', line=dict(color='#744ada')),
        row=2, col=1
    )

    # Recall
    fig.add_trace(
        go.Scatter(x=perf_data['date'], y=perf_data['recall'],
                  mode='lines+markers', name='Recall', line=dict(color='#5a38ad')),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=False, title_text="Key Performance Indicators Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Detection breakdown
    st.markdown("---")
    st.markdown("#### Detection Breakdown")

    breakdown_data = pd.DataFrame({
        'Category': ['True Positives', 'False Positives', 'True Negatives', 'False Negatives'],
        'Count': [
            int(latest['true_positives']),
            int(latest['false_positives']),
            int(latest['true_negatives']),
            int(latest['false_negatives'])
        ],
        'Color': ['#744ada', '#000000', '#744ada', '#f39c12']
    })

    fig = px.bar(breakdown_data, x='Category', y='Count',
                color='Category',
                color_discrete_map={
                    'True Positives': '#744ada',
                    'False Positives': '#000000',
                    'True Negatives': '#744ada',
                    'False Negatives': '#f39c12'
                },
                title='Detection Results Breakdown')
    st.plotly_chart(fig, use_container_width=True)

    # Performance targets
    st.markdown("---")
    st.markdown("#### Performance vs Targets")

    targets = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Detection Rate'],
        'Current': [
            f"{latest['precision']:.2%}",
            f"{latest['recall']:.2%}",
            f"{latest['f1_score']:.2%}",
            f"{latest['fraud_detection_rate']:.2%}"
        ],
        'Target': ['90%', '85%', '87%', '85%'],
        'Status': [
            '' if latest['precision'] >= 0.90 else '',
            '' if latest['recall'] >= 0.85 else '',
            '' if latest['f1_score'] >= 0.87 else '',
            '' if latest['fraud_detection_rate'] >= 0.85 else ''
        ]
    }

    targets_df = pd.DataFrame(targets)
    st.dataframe(targets_df, use_container_width=True, hide_index=True)


def show_performance_trends():
    st.markdown("### Performance Trends Analysis")
    st.markdown("Analyze model performance trends over time")

    perf_data = st.session_state.model_performance_data

    # Time range selector
    time_range = st.selectbox("Select time range:", ["Last 7 Days", "Last 14 Days", "Last 30 Days"])

    days = 7 if time_range == "Last 7 Days" else 14 if time_range == "Last 14 Days" else 30
    data_filtered = perf_data.tail(days)

    # Main metrics trend
    st.markdown("#### Core Metrics Trend")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data_filtered['date'], y=data_filtered['precision'],
                            mode='lines+markers', name='Precision',
                            line=dict(color='#744ada', width=2)))
    fig.add_trace(go.Scatter(x=data_filtered['date'], y=data_filtered['recall'],
                            mode='lines+markers', name='Recall',
                            line=dict(color='#744ada', width=2)))
    fig.add_trace(go.Scatter(x=data_filtered['date'], y=data_filtered['f1_score'],
                            mode='lines+markers', name='F1-Score',
                            line=dict(color='#5a38ad', width=2)))
    fig.add_trace(go.Scatter(x=data_filtered['date'], y=data_filtered['roc_auc'],
                            mode='lines+markers', name='ROC-AUC',
                            line=dict(color='#666666', width=2)))

    fig.update_layout(
        title='Model Performance Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Score',
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistical summary
    st.markdown("---")
    st.markdown("#### Statistical Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Precision Statistics:**")
        precision_stats = data_filtered['precision'].describe()
        st.dataframe(precision_stats, use_container_width=True)

    with col2:
        st.markdown("**Recall Statistics:**")
        recall_stats = data_filtered['recall'].describe()
        st.dataframe(recall_stats, use_container_width=True)

    # Rolling averages
    st.markdown("---")
    st.markdown("#### Rolling Averages (7-day)")

    data_filtered['precision_ma'] = data_filtered['precision'].rolling(window=7, min_periods=1).mean()
    data_filtered['recall_ma'] = data_filtered['recall'].rolling(window=7, min_periods=1).mean()

    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(x=data_filtered['date'], y=data_filtered['precision'],
                            mode='markers', name='Precision (actual)',
                            marker=dict(color='#9b7fe8')))
    fig.add_trace(go.Scatter(x=data_filtered['date'], y=data_filtered['precision_ma'],
                            mode='lines', name='Precision (7-day MA)',
                            line=dict(color='#744ada', width=3)))

    fig.add_trace(go.Scatter(x=data_filtered['date'], y=data_filtered['recall'],
                            mode='markers', name='Recall (actual)',
                            marker=dict(color='#9b7fe8')))
    fig.add_trace(go.Scatter(x=data_filtered['date'], y=data_filtered['recall_ma'],
                            mode='lines', name='Recall (7-day MA)',
                            line=dict(color='#744ada', width=3)))

    fig.update_layout(
        title='Actual vs Moving Average',
        xaxis_title='Date',
        yaxis_title='Score',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Performance degradation detection
    st.markdown("---")
    st.markdown("#### Performance Degradation Analysis")

    # Calculate trends
    recent_7 = data_filtered.tail(7)
    previous_7 = data_filtered.iloc[-14:-7] if len(data_filtered) >= 14 else data_filtered.head(7)

    degradation_analysis = []

    for metric in ['precision', 'recall', 'f1_score', 'roc_auc']:
        recent_avg = recent_7[metric].mean()
        previous_avg = previous_7[metric].mean()
        change = ((recent_avg - previous_avg) / previous_avg) * 100

        status = "STABLE" if abs(change) < 2 else "Watch" if abs(change) < 5 else "Degrading"

        degradation_analysis.append({
            'Metric': metric.replace('_', ' ').title(),
            'Previous Avg': f"{previous_avg:.2%}",
            'Recent Avg': f"{recent_avg:.2%}",
            'Change': f"{change:+.2f}%",
            'Status': status
        })

    degradation_df = pd.DataFrame(degradation_analysis)
    st.dataframe(degradation_df, use_container_width=True, hide_index=True)


def show_prediction_analysis():
    st.markdown("### Prediction Analysis")
    st.markdown("Analyze model predictions and confidence scores")

    # Generate sample prediction data
    n_samples = 1000
    predictions = {
        'prediction_score': np.random.beta(2, 5, n_samples),  # Skewed towards lower scores
        'true_label': np.random.binomial(1, 0.05, n_samples),
        'transaction_amount': np.random.lognormal(5, 2, n_samples)
    }

    pred_df = pd.DataFrame(predictions)
    pred_df['predicted_label'] = (pred_df['prediction_score'] > 0.5).astype(int)

    # Prediction score distribution
    st.markdown("#### Prediction Score Distribution")

    fig = px.histogram(pred_df, x='prediction_score', color='true_label',
                      nbins=50, barmode='overlay',
                      title='Prediction Score Distribution by True Label',
                      color_discrete_map={0: '#744ada', 1: '#000000'},
                      labels={'true_label': 'True Label'})
    st.plotly_chart(fig, use_container_width=True)

    # Score calibration
    st.markdown("---")
    st.markdown("#### Prediction Calibration")

    # Bin predictions
    bins = np.linspace(0, 1, 11)
    pred_df['score_bin'] = pd.cut(pred_df['prediction_score'], bins=bins)

    calibration_data = pred_df.groupby('score_bin', observed=True).agg({
        'true_label': ['mean', 'count']
    }).reset_index()

    calibration_data.columns = ['score_bin', 'actual_fraud_rate', 'count']
    calibration_data['predicted_score'] = [b.mid for b in calibration_data['score_bin']]

    fig = go.Figure()

    # Calibration curve
    fig.add_trace(go.Scatter(x=calibration_data['predicted_score'],
                            y=calibration_data['actual_fraud_rate'],
                            mode='markers+lines',
                            name='Actual Calibration',
                            marker=dict(size=10)))

    # Perfect calibration line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Perfect Calibration',
                            line=dict(dash='dash', color='#666666')))

    fig.update_layout(
        title='Calibration Curve',
        xaxis_title='Predicted Probability',
        yaxis_title='Actual Fraud Rate',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    st.markdown("---")
    st.markdown("#### ROC Curve")

    col1, col2 = st.columns(2)

    with col1:
        # Generate ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.4)  # Simulated ROC curve
        roc_auc_val = np.trapz(tpr, fpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                name=f'ROC Curve (AUC = {roc_auc_val:.3f})',
                                line=dict(color='#744ada', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                name='Random Classifier',
                                line=dict(dash='dash', color='#666666')))

        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Precision-Recall curve
        recall = np.linspace(0, 1, 100)
        precision = 1 / (1 + recall)  # Simulated PR curve

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                name='PR Curve',
                                line=dict(color='#744ada', width=2)))

        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Threshold analysis
    st.markdown("---")
    st.markdown("#### Threshold Analysis")

    thresholds = np.linspace(0, 1, 21)
    threshold_metrics = []

    for threshold in thresholds:
        preds = (pred_df['prediction_score'] > threshold).astype(int)
        tp = ((preds == 1) & (pred_df['true_label'] == 1)).sum()
        fp = ((preds == 1) & (pred_df['true_label'] == 0)).sum()
        tn = ((preds == 0) & (pred_df['true_label'] == 0)).sum()
        fn = ((preds == 0) & (pred_df['true_label'] == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        threshold_metrics.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    threshold_df = pd.DataFrame(threshold_metrics)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=threshold_df['threshold'], y=threshold_df['precision'],
                            mode='lines', name='Precision'))
    fig.add_trace(go.Scatter(x=threshold_df['threshold'], y=threshold_df['recall'],
                            mode='lines', name='Recall'))
    fig.add_trace(go.Scatter(x=threshold_df['threshold'], y=threshold_df['f1'],
                            mode='lines', name='F1-Score'))

    fig.update_layout(
        title='Metrics vs Classification Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def show_business_impact():
    st.markdown("### Business Impact Metrics")
    st.markdown("Financial and operational impact of fraud detection")

    # Business metrics
    st.markdown("#### Financial Impact")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fraud_prevented = np.random.uniform(150000, 300000)
        st.metric("Fraud Prevented ($)", f"${fraud_prevented:,.0f}",
                 delta=f"+${np.random.uniform(10000, 30000):,.0f}")

    with col2:
        false_decline_cost = np.random.uniform(20000, 50000)
        st.metric("False Decline Cost ($)", f"${false_decline_cost:,.0f}",
                 delta=f"-${np.random.uniform(2000, 5000):,.0f}")

    with col3:
        net_savings = fraud_prevented - false_decline_cost
        st.metric("Net Savings ($)", f"${net_savings:,.0f}")

    with col4:
        roi = (net_savings / 50000) * 100  # Assuming $50k operating cost
        st.metric("ROI", f"{roi:.1f}%")

    # Monthly trends
    st.markdown("---")
    st.markdown("#### Monthly Financial Trends")

    months = pd.date_range(end=datetime.now(), periods=12, freq='M')
    financial_data = pd.DataFrame({
        'month': months,
        'fraud_prevented': np.random.uniform(100000, 300000, 12),
        'false_decline_cost': np.random.uniform(15000, 45000, 12),
        'investigation_cost': np.random.uniform(5000, 15000, 12)
    })

    financial_data['net_savings'] = (financial_data['fraud_prevented'] -
                                     financial_data['false_decline_cost'] -
                                     financial_data['investigation_cost'])

    fig = go.Figure()

    fig.add_trace(go.Bar(x=financial_data['month'], y=financial_data['fraud_prevented'],
                        name='Fraud Prevented', marker_color='#744ada'))
    fig.add_trace(go.Bar(x=financial_data['month'], y=-financial_data['false_decline_cost'],
                        name='False Decline Cost', marker_color='#000000'))
    fig.add_trace(go.Bar(x=financial_data['month'], y=-financial_data['investigation_cost'],
                        name='Investigation Cost', marker_color='#666666'))

    fig.update_layout(
        title='Monthly Financial Impact',
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        barmode='relative',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Operational metrics
    st.markdown("---")
    st.markdown("#### Operational Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cases Investigated", "1,234",
                 delta="-56")

    with col2:
        avg_investigation_time = 45.3
        st.metric("Avg Investigation Time (min)", f"{avg_investigation_time:.1f}",
                 delta="-3.2")

    with col3:
        automation_rate = 78.5
        st.metric("Automation Rate (%)", f"{automation_rate:.1f}%",
                 delta="+2.3%")

    # Customer impact
    st.markdown("---")
    st.markdown("#### Customer Impact")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Customers Protected", "45,678")

    with col2:
        false_declines = 234
        st.metric("False Declines", false_declines,
                 delta=f"-{np.random.randint(10, 30)}")

    with col3:
        satisfaction_score = 4.2
        st.metric("Customer Satisfaction", f"{satisfaction_score:.1f}/5.0")

    with col4:
        churn_prevented = 12
        st.metric("Churn Prevented", churn_prevented)

    # Cost-benefit analysis
    st.markdown("---")
    st.markdown("#### Cost-Benefit Analysis")

    cost_benefit = pd.DataFrame({
        'Category': ['Fraud Losses Prevented', 'False Positive Costs', 'Model Operating Costs',
                    'Investigation Costs', 'Customer Retention Value'],
        'Amount': [250000, -35000, -50000, -30000, 80000],
        'Type': ['Benefit', 'Cost', 'Cost', 'Cost', 'Benefit']
    })

    fig = px.bar(cost_benefit, x='Category', y='Amount', color='Type',
                color_discrete_map={'Benefit': '#744ada', 'Cost': '#000000'},
                title='Cost-Benefit Breakdown')

    st.plotly_chart(fig, use_container_width=True)

    total_benefit = cost_benefit[cost_benefit['Type'] == 'Benefit']['Amount'].sum()
    total_cost = abs(cost_benefit[cost_benefit['Type'] == 'Cost']['Amount'].sum())
    net_benefit = total_benefit - total_cost

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Benefits", f"${total_benefit:,.0f}")

    with col2:
        st.metric("Total Costs", f"${total_cost:,.0f}")

    with col3:
        st.metric("Net Benefit", f"${net_benefit:,.0f}")


def show_model_health():
    st.markdown("### Model Health & Alerts")
    st.markdown("Monitor model health and performance alerts")

    # Overall health score
    st.markdown("#### Overall Model Health")

    health_score = np.random.uniform(85, 98)
    health_status = "HEALTHY" if health_score > 90 else "FAIR" if health_score > 75 else "Poor"

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric("Health Score", f"{health_score:.1f}/100")

    with col2:
        st.metric("Status", health_status)

    with col3:
        st.metric("Last Health Check", datetime.now().strftime('%Y-%m-%d %H:%M'))

    # Health components
    st.markdown("---")
    st.markdown("#### Health Components")

    components = [
        {'Component': 'Performance', 'Score': 92, 'Status': 'GOOD'},
        {'Component': 'Data Quality', 'Score': 88, 'Status': 'GOOD'},
        {'Component': 'Prediction Volume', 'Score': 95, 'Status': 'GOOD'},
        {'Component': 'Latency', 'Score': 90, 'Status': 'GOOD'},
        {'Component': 'Error Rate', 'Score': 85, 'Status': 'FAIR'},
        {'Component': 'Feature Drift', 'Score': 78, 'Status': 'FAIR'}
    ]

    components_df = pd.DataFrame(components)
    st.dataframe(components_df, use_container_width=True, hide_index=True)

    # Active alerts
    st.markdown("---")
    st.markdown("#### Active Alerts")

    alerts = [
        {
            'Severity': 'Warning',
            'Alert': 'Performance Degradation',
            'Description': 'Precision dropped by 3% in last 48 hours',
            'Time': '2 hours ago'
        },
        {
            'Severity': 'Warning',
            'Alert': 'Feature Drift Detected',
            'Description': 'transaction_amount showing moderate drift (PSI: 0.18)',
            'Time': '5 hours ago'
        },
        {
            'Severity': '🔵 Info',
            'Alert': 'High Prediction Volume',
            'Description': 'Prediction volume 15% above baseline',
            'Time': '1 hour ago'
        }
    ]

    for alert in alerts:
        with st.expander(f"{alert['Severity']} - {alert['Alert']}"):
            st.markdown(f"**Description:** {alert['Description']}")
            st.markdown(f"**Time:** {alert['Time']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Acknowledge", key=f"ack_{alert['Alert']}"):
                    st.success("Alert acknowledged")

            with col2:
                if st.button("Investigate", key=f"inv_{alert['Alert']}"):
                    st.info("Opening investigation...")

            with col3:
                if st.button("🔕 Snooze", key=f"snooze_{alert['Alert']}"):
                    st.info("Alert snoozed for 24 hours")

    # Model retraining recommendations
    st.markdown("---")
    st.markdown("#### Retraining Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        days_since_training = 45
        st.metric("Days Since Last Training", days_since_training)

        if days_since_training > 30:
            st.warning("Model retraining recommended")
        else:
            st.success("Model is recent")

    with col2:
        performance_drift = 4.2
        st.metric("Performance Drift (%)", f"{performance_drift:.1f}%")

        if performance_drift > 5:
            st.error("Significant drift - retraining required")
        elif performance_drift > 3:
            st.warning("Moderate drift - schedule retraining")
        else:
            st.success("Minimal drift")


def show_segment_performance():
    st.markdown("### Segment Performance Analysis")
    st.markdown("Analyze model performance across different segments")

    # Performance by merchant category
    st.markdown("#### Performance by Merchant Category")

    categories = ['retail', 'online', 'groceries', 'gas_station', 'restaurant',
                 'gambling', 'crypto', 'electronics']

    category_performance = []
    for cat in categories:
        category_performance.append({
            'Category': cat,
            'Precision': np.random.uniform(0.75, 0.95),
            'Recall': np.random.uniform(0.70, 0.92),
            'F1-Score': np.random.uniform(0.72, 0.93),
            'Sample Size': np.random.randint(500, 5000)
        })

    category_df = pd.DataFrame(category_performance)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=category_df['Category'], y=category_df['Precision'],
                        name='Precision', marker_color='#744ada'))
    fig.add_trace(go.Bar(x=category_df['Category'], y=category_df['Recall'],
                        name='Recall', marker_color='#744ada'))
    fig.add_trace(go.Bar(x=category_df['Category'], y=category_df['F1-Score'],
                        name='F1-Score', marker_color='#5a38ad'))

    fig.update_layout(
        title='Performance by Merchant Category',
        xaxis_title='Category',
        yaxis_title='Score',
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Performance by transaction amount
    st.markdown("---")
    st.markdown("#### Performance by Transaction Amount")

    amount_ranges = ['$0-$50', '$50-$100', '$100-$500', '$500-$1K', '$1K-$5K', '$5K+']
    amount_performance = pd.DataFrame({
        'Range': amount_ranges,
        'Precision': np.random.uniform(0.80, 0.95, 6),
        'Recall': np.random.uniform(0.75, 0.90, 6),
        'Volume': np.random.randint(1000, 10000, 6)
    })

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=amount_performance['Range'], y=amount_performance['Precision'],
                  mode='lines+markers', name='Precision', line=dict(color='#744ada')),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=amount_performance['Range'], y=amount_performance['Recall'],
                  mode='lines+markers', name='Recall', line=dict(color='#744ada')),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(x=amount_performance['Range'], y=amount_performance['Volume'],
              name='Volume', marker_color='lightgray', opacity=0.3),
        secondary_y=True
    )

    fig.update_layout(title='Performance by Transaction Amount', height=400)
    fig.update_xaxes(title_text='Transaction Amount Range')
    fig.update_yaxes(title_text='Performance Score', secondary_y=False)
    fig.update_yaxes(title_text='Transaction Volume', secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Performance by time of day
    st.markdown("---")
    st.markdown("#### Performance by Time of Day")

    hours = list(range(0, 24))
    hourly_performance = pd.DataFrame({
        'Hour': hours,
        'Precision': np.random.uniform(0.82, 0.94, 24),
        'Fraud Rate': np.random.uniform(0.02, 0.12, 24)
    })

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=hourly_performance['Hour'], y=hourly_performance['Precision'],
                  mode='lines+markers', name='Precision', line=dict(color='#744ada')),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=hourly_performance['Hour'], y=hourly_performance['Fraud Rate'],
                  mode='lines+markers', name='Fraud Rate', line=dict(color='#000000')),
        secondary_y=True
    )

    fig.update_layout(title='Performance by Hour of Day', height=400)
    fig.update_xaxes(title_text='Hour of Day')
    fig.update_yaxes(title_text='Precision', secondary_y=False)
    fig.update_yaxes(title_text='Fraud Rate', secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Worst performing segments
    st.markdown("---")
    st.markdown("#### Segments Requiring Attention")

    attention_segments = category_df.nsmallest(3, 'F1-Score')[['Category', 'Precision', 'Recall', 'F1-Score']]
    attention_segments['Recommendation'] = [
        'Increase training data',
        'Review feature engineering',
        'Adjust classification threshold'
    ]

    st.dataframe(attention_segments, use_container_width=True, hide_index=True)
