# ArgusAI - Fraud Detection & Monitoring Platform

A comprehensive Streamlit dashboard for fraud detection, featuring feature monitoring, model performance tracking, rule-based monitoring, case management, and model deployment capabilities.

## Features

### 1. Feature Monitoring
Advanced feature drift and data quality monitoring:
- **Feature Drift Detection**: PSI, KS statistic, distribution comparison
- **Data Quality Monitoring**: Completeness, validity, consistency metrics
- **Feature Statistics**: Detailed statistical analysis over time
- **Alerts & Anomalies**: Configurable thresholds and anomaly detection
- **Distribution Analysis**: Segment-based distribution comparisons
- **Feature Importance Tracking**: Monitor how feature importance evolves

### 2. Model Performance Monitoring
Comprehensive model evaluation and fraud detection KPIs:
- **Performance Dashboard**: Real-time metrics (precision, recall, F1, ROC-AUC)
- **Fraud Detection KPIs**: Detection rate, false positive rate, false discovery rate
- **Performance Trends**: Historical trend analysis with degradation detection
- **Prediction Analysis**: Score distributions, calibration curves, ROC/PR curves
- **Business Impact**: Financial metrics, ROI, customer impact
- **Model Health & Alerts**: Health scoring, component monitoring, retraining recommendations
- **Segment Performance**: Performance breakdown by merchant, amount, time

### 3. Rule Editor & Fraud Monitoring
Sophisticated rule-based fraud detection system:
- **Rule Manager**: Create, edit, and manage fraud detection rules
- **SQL Templates**: Define rules using SQL-like templates with thresholds
- **Rule Testing**: Test rules against sample or real data
- **Real-Time Monitoring**: Live dashboard with transaction metrics
- **Performance Analytics**: Track rule effectiveness over time

### 4. Case Management
Investigation platform for suspicious transactions:
- **Case Dashboard**: Overview of all fraud investigation cases
- **Advanced Filtering**: Filter by status, priority, type, and assigned analyst
- **Case Investigation**: Detailed transaction analysis with customer history
- **Timeline Visualization**: Visual representation of transaction patterns
- **Notes & Actions**: Add investigation notes and take actions
- **Workflow Configuration**: Configure automated case workflows

### 5. Model Deployment
Production-ready model deployment platform:
- **Model Registry**: Browse and manage trained models
- **Deployment Configuration**: Configure scaling, monitoring, and alerts
- **Real-Time Inference**: Test models with manual input, JSON, batch upload, or live stream
- **Model Monitoring**: Track performance metrics and health
- **Model Management**: Update, pause, or rollback deployments
- **Version Control**: Maintain model version history

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ArgusAI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the dashboard:
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
ArgusAI/
├── app.py                               # Main Streamlit application
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── .streamlit/
│   └── config.toml                     # Streamlit configuration
└── src/
    ├── pages/
    │   ├── feature_monitoring.py       # Feature drift & data quality
    │   ├── model_monitoring_detailed.py # Model performance & KPIs
    │   ├── rule_editor.py              # Rule Editor page
    │   ├── case_management.py          # Case Management page
    │   └── model_deployment.py         # Model Deployment page
    └── utils/
        ├── data_generator.py           # Synthetic data generation
        ├── feature_engineering.py      # Feature engineering utilities
        └── rule_engine.py              # Rule engine implementation
```

## Quick Start Guide

### 1. Feature Monitoring

1. Navigate to "Feature Monitoring" from the sidebar
2. **Feature Drift Tab**:
   - Click "Load Baseline Data" to establish baseline distributions
   - Click "Load Current Data" to load recent data
   - Select features to analyze drift (PSI, KS statistic)
   - Review drift trends and all-features summary

3. **Data Quality Tab**:
   - Monitor completeness, validity, and consistency scores
   - Review missing values and outlier analysis
   - Track data freshness and quality trends

4. **Alerts & Anomalies Tab**:
   - Configure alert thresholds for drift and quality
   - Click "Scan for Anomalies" to detect issues
   - Review and manage active alerts

### 2. Model Monitoring

1. Navigate to "Model Monitoring" from the sidebar
2. **Performance Dashboard**:
   - View current performance metrics and gauges
   - Review confusion matrix and model health status
   - Monitor real-time performance

3. **Fraud Detection KPIs**:
   - Track detection rate, false positive rate
   - Monitor precision, recall, F1-score trends
   - Review performance vs targets

4. **Performance Trends**:
   - Analyze metrics over different time ranges
   - Review rolling averages and statistical summaries
   - Detect performance degradation

5. **Business Impact**:
   - Track financial impact (fraud prevented, costs)
   - Monitor operational metrics
   - Review customer impact and satisfaction

6. **Model Health & Alerts**:
   - Check overall model health score
   - Review component health status
   - Manage active alerts and retraining recommendations

### 3. Rule Editor

1. Navigate to "Rule Editor" from the sidebar
2. In "Rule Manager":
   - View existing rules
   - Click "Create New Rule" to add custom rules
   - Configure rule type, thresholds, and actions
   - Save and activate rules

3. In "Rule Testing":
   - Load test data
   - Select rules to apply
   - Run rules and view flagged transactions

4. Monitor rules in real-time in "Monitoring Dashboard"

### 4. Case Management

1. Navigate to "Case Management"
2. In "Case Dashboard":
   - Filter cases by status, priority, type
   - Search for specific cases
   - Select a case for investigation

3. In "Case Investigation":
   - Review transaction details
   - Analyze customer history
   - Add investigation notes
   - Take actions (mark as fraud, escalate, etc.)

### 5. Model Deployment

1. Navigate to "Model Deployment"
2. In "Model Registry":
   - Browse available models
   - View performance metrics
   - Upload new models

3. In "Real-Time Inference":
   - Test deployed models
   - Use manual entry, JSON, batch upload, or live stream
   - View predictions and fraud scores

4. Monitor model performance in "Model Monitoring"

## Key Features in Detail

### Feature Drift Detection
Advanced drift detection capabilities:
- **PSI (Population Stability Index)**: Industry-standard drift metric
- **KS Statistic**: Distribution comparison using Kolmogorov-Smirnov test
- **Mean & Std Shifts**: Track central tendency and spread changes
- **Thresholds**: Configurable thresholds (PSI < 0.1: Good, 0.1-0.2: Monitor, >0.2: Alert)
- **Visual Analysis**: Distribution overlays, time series trends

### Data Quality Monitoring
Comprehensive quality metrics:
- **Completeness**: % of non-null values
- **Validity**: % of values within expected ranges
- **Consistency**: % of records without contradictions
- **Freshness**: Track data age and update frequency
- **Outlier Detection**: IQR-based outlier identification

### Fraud Detection KPIs
Critical performance indicators:
- **Detection Rate**: % of fraud cases successfully identified
- **False Positive Rate**: % of legitimate transactions incorrectly flagged
- **False Discovery Rate**: % of fraud predictions that are incorrect
- **False Negative Rate**: % of fraud cases missed
- **Precision & Recall**: Standard ML metrics
- **ROC-AUC**: Area under the ROC curve

### Business Impact Tracking
Financial and operational metrics:
- **Fraud Prevented**: $ amount of fraud stopped
- **False Decline Cost**: Revenue loss from incorrect flags
- **Net Savings**: Total financial benefit
- **ROI**: Return on investment
- **Customer Impact**: Satisfaction scores, churn prevented
- **Operational Efficiency**: Investigation time, automation rate

### Model Health Monitoring
Comprehensive health checks:
- **Overall Health Score**: Aggregate health metric (0-100)
- **Component Monitoring**: Performance, data quality, latency, error rate
- **Degradation Detection**: Identify performance decline
- **Retraining Triggers**: Automatic recommendations based on drift/degradation
- **Alert Management**: Configurable alerts with severity levels

### Segment Performance Analysis
Detailed performance breakdown:
- **By Merchant Category**: Performance across different merchants
- **By Transaction Amount**: Performance across amount ranges
- **By Time of Day**: Hourly performance patterns
- **By Geography**: Regional performance differences
- **Worst Performers**: Identify segments needing attention

## Monitoring Best Practices

### Feature Monitoring
- **Baseline**: Establish baseline on stable, representative data
- **Frequency**: Monitor daily for critical features, weekly for others
- **Thresholds**: PSI < 0.1 (stable), 0.1-0.2 (monitor), > 0.2 (investigate)
- **Actions**: Retrain models when significant drift detected

### Model Monitoring
- **Real-Time**: Monitor key metrics (precision, recall) in real-time
- **Trends**: Review weekly trends to detect gradual degradation
- **Segments**: Analyze performance by segment monthly
- **Thresholds**: Set alerts for precision < 85%, recall < 80%
- **Retraining**: Schedule retraining every 30-60 days or when drift > threshold

### Alert Configuration
- **Critical**: Performance drop > 5%, data freshness > 24h
- **Warning**: Performance drop 2-5%, drift PSI > 0.2
- **Info**: Minor changes, usage spikes

## Data Schema

The platform works with transaction data containing:

**Required Fields:**
- `transaction_id`: Unique transaction identifier
- `customer_id`: Customer identifier
- `transaction_amount`: Transaction amount
- `timestamp`: Transaction timestamp
- `is_fraud`: Fraud label (0 or 1)

**Optional Fields:**
- `merchant_id`, `merchant_category`: Merchant information
- `country`, `city`: Location information
- `device_type`, `browser`: Device information
- `distance_from_home`: Geographic distance
- `transactions_24h`, `transactions_7d`: Velocity metrics

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Statistics**: SciPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **UI Components**: streamlit-option-menu

## Configuration

Customize the dashboard by editing `.streamlit/config.toml`:
- Theme colors
- Server settings
- Port configuration

## Development

To extend the platform:

1. **Add new pages**: Create files in `src/pages/`
2. **Add utilities**: Add helper functions in `src/utils/`
3. **Add metrics**: Extend monitoring metrics in feature/model monitoring pages
4. **Add rules**: Create rule templates in `rule_engine.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Open an issue on GitHub
- Contact the development team

## Roadmap

Future enhancements:
- Real-time data pipeline integration (Kafka, Kinesis)
- Advanced drift detection (multivariate drift, domain adaptation)
- Automated retraining workflows
- A/B testing framework for models
- Model explainability (SHAP, LIME)
- Custom metric definitions
- Integration with MLOps tools (MLflow, Kubeflow)
- Advanced alerting (PagerDuty, Slack integration)
- Multi-model comparison
- Automated root cause analysis

## Acknowledgments

Built with Streamlit and powered by scikit-learn for machine learning capabilities. Drift detection uses industry-standard PSI and KS statistics.
