# ArgusAI - Fraud Detection & Monitoring Platform

A comprehensive Streamlit dashboard for fraud detection, featuring real-time model inference with KNIME API integration, feature monitoring, model performance tracking, rule-based monitoring, case management, and model deployment capabilities.

## Features

### 1. Real-time Model Inference ⚡ NEW
**Production-ready inference module with KNIME API integration:**
- **API Configuration**: Configure KNIME or external model endpoints with multiple authentication options (API Key, Bearer Token, Basic Auth)
- **Single Transaction Scoring**: Score individual transactions via JSON editor or form input
- **Batch Inference**: Upload CSV files for bulk transaction scoring
- **Inference History**: Track all predictions with downloadable history
- **API Request Logging**: Full request/response logging for debugging and monitoring
- **Performance Metrics**: Real-time latency and success rate tracking
- **Flexible Integration**: Supports external APIs (KNIME) or local simulation mode
- **Automatic Score Interpretation**: Handles various response formats from different APIs

### 2. Feature Monitoring
Advanced feature drift and data quality monitoring:
- **Feature Drift Detection**: PSI, KS statistic, distribution comparison
- **Data Quality Monitoring**: Completeness, validity, consistency metrics
- **Feature Statistics**: Detailed statistical analysis over time
- **Alerts & Anomalies**: Configurable thresholds and anomaly detection
- **Distribution Analysis**: Segment-based distribution comparisons
- **Feature Importance Tracking**: Monitor how feature importance evolves

### 3. Model Performance Monitoring
Comprehensive model evaluation and fraud detection KPIs:
- **Performance Dashboard**: Real-time metrics (precision, recall, F1, ROC-AUC)
- **Fraud Detection KPIs**: Detection rate, false positive rate, false discovery rate
- **Performance Trends**: Historical trend analysis with degradation detection
- **Prediction Analysis**: Score distributions, calibration curves, ROC/PR curves
- **Business Impact**: Financial metrics, ROI, customer impact
- **Model Health & Alerts**: Health scoring, component monitoring, retraining recommendations
- **Segment Performance**: Performance breakdown by merchant, amount, time

### 4. Rule Editor & Fraud Monitoring
Sophisticated rule-based fraud detection system:
- **Rule Manager**: Create, edit, and manage fraud detection rules
- **SQL Templates**: Define rules using SQL-like templates with thresholds
- **Rule Testing**: Test rules against sample or real data
- **Real-Time Monitoring**: Live dashboard with transaction metrics
- **Performance Analytics**: Track rule effectiveness over time

### 5. Case Management
Investigation platform for suspicious transactions:
- **Case Dashboard**: Overview of all fraud investigation cases
- **Advanced Filtering**: Filter by status, priority, type, and assigned analyst
- **Case Investigation**: Detailed transaction analysis with customer history
- **Timeline Visualization**: Visual representation of transaction patterns
- **Notes & Actions**: Add investigation notes and take actions
- **Workflow Configuration**: Configure automated case workflows

### 6. Model Deployment
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
├── docs/
│   └── API_INTEGRATION.md              # Detailed API integration guide
└── src/
    ├── pages/
    │   ├── realtime_inference.py       # Real-time model inference (NEW)
    │   ├── feature_monitoring.py       # Feature drift & data quality
    │   ├── model_monitoring_detailed.py # Model performance & KPIs
    │   ├── rule_editor.py              # Rule Editor page
    │   ├── case_management.py          # Case Management page
    │   └── model_deployment.py         # Model Deployment page
    └── utils/
        ├── api_client.py               # External API client (NEW)
        ├── data_generator.py           # Synthetic data generation
        ├── feature_engineering.py      # Feature engineering utilities
        └── rule_engine.py              # Rule engine implementation
```

## Quick Start Guide

### 0. Real-time Model Inference (KNIME Integration)

**Step 1: Configure Your KNIME API**
1. Navigate to "Real-time Inference" from the sidebar
2. Go to "API Configuration" tab
3. Enter your KNIME endpoint URL:
   ```
   https://your-knime-server.com/api/v1/predict
   ```
4. Select authentication type:
   - **API Key**: Adds `X-API-Key` header
   - **Bearer Token**: Adds `Authorization: Bearer {token}` header
   - **Basic Auth**: Uses username:password authentication
   - **None**: No authentication
5. Enter your credentials
6. Click **Test Connection** to verify setup
7. Click **Save Configuration**

**Step 2: Score a Single Transaction**

**Option A: JSON Editor**
1. Go to "Single Transaction" tab
2. Select "JSON Editor" input method
3. Paste your transaction JSON:
   ```json
   {
     "transaction_amount": 1250.00,
     "merchant_category": "online",
     "transaction_hour": 14,
     "customer_age": 35,
     "account_age_days": 730,
     "previous_transactions": 125,
     "is_foreign": false,
     "distance_from_home": 5.2,
     "device_type": "mobile"
   }
   ```
4. Click "Score Transaction"

**Option B: Form Input**
1. Select "Form Input" method
2. Fill in transaction details using the form fields
3. Click "Score Transaction"

**Step 3: View Results**
The system displays:
- **Status**: API call success/failure
- **Fraud Score**: Model prediction score
- **Latency**: API response time in milliseconds
- **Full Response**: Complete API response JSON
- **Interpreted Results**: Extracted fraud score, prediction, confidence
- **Risk Level**: High/Low based on score

**Step 4: Batch Scoring**
For multiple transactions:
1. Go to "Batch Inference" tab
2. Upload CSV file with transaction features
3. Click "Run Batch Inference"
4. Download results CSV with fraud scores

**Step 5: Monitor Activity**
- **Inference History**: View all predictions with timestamps, scores, and latency
- **API Logs**: Inspect detailed request/response logs for debugging

**Expected API Response Format:**
Your KNIME API should return JSON like:
```json
{
  "fraud_score": 0.75,
  "prediction": "fraud",
  "confidence": 0.85,
  "risk_level": "high"
}
```

The system also supports alternative field names:
- Score fields: `fraud_score`, `score`, `probability`, `fraud_probability`
- Prediction fields: `prediction`, `label`, `class`

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

## API Documentation

For detailed API integration documentation including:
- Complete authentication examples
- KNIME integration guide
- Error handling best practices
- Security considerations
- Code examples and references

See **[docs/API_INTEGRATION.md](docs/API_INTEGRATION.md)**

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Statistics**: SciPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **HTTP Client**: Requests (for external API integration)
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
- GraphQL API support
- WebSocket real-time streaming
- Multi-region API load balancing

## Changelog

### Version 1.2.0 (December 2024) - Real-time Inference Release
**New Features:**
- ✨ **Real-time Model Inference Module**: Dedicated inference module with full KNIME API integration
- ✨ **API Client Utility**: Reusable HTTP client for external model APIs
- ✨ **Multi-Auth Support**: API Key, Bearer Token, Basic Auth, and No Auth
- ✨ **Batch Inference**: CSV upload for bulk transaction scoring
- ✨ **Inference History**: Complete tracking with downloadable exports
- ✨ **API Logging**: Full request/response logging for debugging
- ✨ **Performance Monitoring**: Real-time latency and success rate metrics
- 📚 **Comprehensive Documentation**: API integration guide with examples

**Improvements:**
- Enhanced navigation with lightning-fast inference access
- Professional color scheme maintained across new modules
- Improved error handling and user feedback
- Better session state management

**Files Added:**
- `src/pages/realtime_inference.py` - Real-time inference module
- `src/utils/api_client.py` - External API client
- `docs/API_INTEGRATION.md` - API documentation

**Dependencies Updated:**
- Added `requests==2.31.0` for HTTP API calls

### Version 1.1.0 (November 2024)
- Added Feature Monitoring module
- Added Model Monitoring module
- Replaced Data Science Workflow
- Professional theme with #744ada color palette
- Removed all emoji icons

### Version 1.0.0 (October 2024)
- Initial release
- 5 core modules: Feature Monitoring, Model Monitoring, Rule Editor, Case Management, Model Deployment
- Professional UI design

## Acknowledgments

Built with Streamlit and powered by scikit-learn for machine learning capabilities. Drift detection uses industry-standard PSI and KS statistics. API integration powered by Python Requests library.
