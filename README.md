# ArgusAI - Fraud Detection & Data Science Platform

A comprehensive Streamlit dashboard for fraud detection, featuring complete data science workflows, rule-based monitoring, case management, and model deployment capabilities.

## Features

### 1. Data Science Workflow
Complete end-to-end data science pipeline including:
- **Data Extraction**: Load data from various sources (CSV, database, generate sample data)
- **Data Cleaning**: Handle missing values, remove duplicates, and outliers
- **Exploratory Data Analysis (EDA)**: Interactive visualizations and statistical analysis
- **Feature Engineering**: Create time-based, amount-based, velocity, and aggregate features
- **Feature Selection**: Multiple methods including correlation, mutual information, chi-squared, and RF importance
- **Customer Profiling**: Analyze customer behavior and segment profiles
- **Train/Test Split**: Stratified splitting with feature scaling
- **Model Building**: Train Logistic Regression, Random Forest, and Gradient Boosting models
- **Model Evaluation**: Comprehensive metrics, confusion matrix, ROC curves
- **Final Metrics**: Compare all models and get recommendations

### 2. Rule Editor & Fraud Monitoring
Sophisticated rule-based fraud detection system:
- **Rule Manager**: Create, edit, and manage fraud detection rules
- **SQL Templates**: Define rules using SQL-like templates with thresholds
- **Rule Testing**: Test rules against sample or real data
- **Real-Time Monitoring**: Live dashboard with transaction metrics
- **Performance Analytics**: Track rule effectiveness over time

### 3. Case Management
Investigation platform for suspicious transactions:
- **Case Dashboard**: Overview of all fraud investigation cases
- **Advanced Filtering**: Filter by status, priority, type, and assigned analyst
- **Case Investigation**: Detailed transaction analysis with customer history
- **Timeline Visualization**: Visual representation of transaction patterns
- **Notes & Actions**: Add investigation notes and take actions
- **Workflow Configuration**: Configure automated case workflows

### 4. Model Deployment
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
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── src/
    ├── pages/
    │   ├── ds_workflow.py         # Data Science Workflow page
    │   ├── rule_editor.py         # Rule Editor page
    │   ├── case_management.py     # Case Management page
    │   └── model_deployment.py    # Model Deployment page
    └── utils/
        ├── data_generator.py      # Synthetic data generation
        ├── feature_engineering.py # Feature engineering utilities
        └── rule_engine.py         # Rule engine implementation
```

## Quick Start Guide

### 1. Data Science Workflow

1. Navigate to "Data Science Workflow" from the sidebar
2. Go to "Data Extraction" tab:
   - Select "Generate Sample Data"
   - Set sample size (e.g., 10,000 records)
   - Set fraud rate (e.g., 5%)
   - Click "Extract Data"

3. Move through the workflow tabs sequentially:
   - Clean the data in "Data Cleaning"
   - Explore data in "EDA"
   - Create features in "Feature Engineering"
   - Select important features in "Feature Selection"
   - Split data in "Train/Test Split"
   - Train models in "Model Building"
   - Evaluate in "Model Evaluation"
   - Review final metrics

### 2. Rule Editor

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

### 3. Case Management

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

### 4. Model Deployment

1. First, train models in "Data Science Workflow"
2. Navigate to "Model Deployment"
3. In "Deploy Model":
   - Select a trained model
   - Configure deployment settings
   - Deploy to production/testing

4. In "Real-Time Inference":
   - Test deployed models
   - Use manual entry, JSON, batch upload, or live stream
   - View predictions and fraud scores

5. Monitor model performance in "Model Monitoring"

## Key Features in Detail

### Rule-Based Detection
The rule editor allows you to create sophisticated fraud detection rules:
- **SQL Templates**: Use SQL-like syntax for flexible rule definition
- **Thresholds**: Set dynamic thresholds for each rule
- **Priority Levels**: Assign priority (Low, Medium, High, Critical)
- **Actions**: Configure actions (Flag, Alert, Review, Block, Notify)
- **Testing**: Test rules before deployment

Example rules:
- High amount transactions (>$5,000)
- Multiple transactions at same merchant
- Foreign country transactions
- High-risk merchant categories
- Unusual transaction hours

### Machine Learning Models
Support for multiple ML algorithms:
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Robust ensemble method
- **Gradient Boosting**: High-performance gradient boosting

Features:
- Automated feature engineering
- Feature selection
- Cross-validation
- Hyperparameter tuning (coming soon)
- Model comparison and selection

### Real-Time Inference
Multiple inference modes:
- **Manual Entry**: Single transaction prediction
- **JSON Input**: API-style JSON input
- **Batch Upload**: Process CSV files
- **Live Stream**: Simulate real-time processing

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
3. **Add models**: Extend model types in `ds_workflow.py`
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
- Database integration (PostgreSQL, MongoDB)
- Real-time streaming with Kafka
- Deep learning models (LSTM, Autoencoders)
- Advanced feature engineering (graph features)
- A/B testing framework
- Model explainability (SHAP, LIME)
- API endpoints for external integration
- User authentication and roles
- Audit logging
- Multi-language support

## Acknowledgments

Built with Streamlit and powered by scikit-learn for machine learning capabilities.
