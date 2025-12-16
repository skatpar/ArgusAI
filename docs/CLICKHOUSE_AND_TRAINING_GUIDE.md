# ClickHouse Integration & Model Training Guide

## Overview

ArgusAI now provides end-to-end integration for:
1. **Data Loading** from ClickHouse, CSV, or synthetic generation
2. **Model Training** using your custom scripts (train_all.py) or built-in algorithms
3. **Real-time Inference** using your production API (localhost:5000/predict)
4. **Workflow Orchestration** connecting all components

## Table of Contents
- [Data Loading Module](#data-loading-module)
- [ClickHouse Integration](#clickhouse-integration)
- [Model Training Module](#model-training-module)
- [Production API Integration](#production-api-integration)
- [Complete Workflow](#complete-workflow)
- [Troubleshooting](#troubleshooting)

---

## Data Loading Module

### Overview
The Data Loading module provides multiple ways to load transaction data for analysis and model training.

### Features
- **ClickHouse Connection** with authentication
- **Visual Query Builder** for easy data extraction
- **Custom SQL Queries** for advanced users
- **CSV Upload** for external data
- **Synthetic Data Generator** for testing
- **Data Preview & Export** (CSV/Parquet)

### Getting Started

#### 1. Connect to ClickHouse

**Navigate to:** Data Loading → ClickHouse Connection

**Configuration:**
```
Host: localhost (or your ClickHouse server)
Port: 8123 (HTTP interface port)
Username: default (or your username)
Password: [your password]
Database: default (or your database name)
```

**Steps:**
1. Enter connection details
2. Click "Test Connection" to verify
3. Click "Save & Connect" to save configuration

**Verify Connection:**
- Green success message confirms connection
- Shows ClickHouse version and available tables

#### 2. Load Data with Query Builder

**Navigate to:** Data Loading → Load from ClickHouse → Query Builder

**Steps:**
1. Select table from dropdown
2. Click "Load Schema" to see table structure
3. Set row limit (default: 10,000)
4. Add filters (optional):
   - Click "Number of filters" to add conditions
   - Example: Column: `is_fraud`, Operator: `=`, Value: `1`
5. Click "Load Data"

**Example Filter:**
```
Filter 1: merchant_category = 'online'
Filter 2: transaction_amount > 1000
Filter 3: timestamp >= '2025-01-01'
```

#### 3. Custom SQL Queries

**Navigate to:** Data Loading → Load from ClickHouse → Custom SQL Query

**Example Queries:**

**Load fraud transactions:**
```sql
SELECT * FROM transactions
WHERE is_fraud = 1
AND timestamp >= '2025-01-01'
LIMIT 10000
```

**Load high-value transactions:**
```sql
SELECT * FROM transactions
WHERE transaction_amount > 1000
ORDER BY transaction_amount DESC
LIMIT 5000
```

**Aggregated fraud stats:**
```sql
SELECT
    merchant_category,
    COUNT(*) as total_transactions,
    SUM(is_fraud) as fraud_count,
    AVG(transaction_amount) as avg_amount
FROM transactions
WHERE timestamp >= '2025-03-01'
GROUP BY merchant_category
ORDER BY fraud_count DESC
```

#### 4. Quick Filters

**Navigate to:** Data Loading → Load from ClickHouse → Quick Filters

**Use Cases:**
- Load recent transactions (last 30 days)
- Load only fraud cases
- Quick data exploration

**Steps:**
1. Select table
2. Set date range
3. Check "Fraud transactions only" (optional)
4. Click "Load Data"

#### 5. CSV Upload

**Navigate to:** Data Loading → Load from CSV

**Steps:**
1. Click "Upload CSV file"
2. Select your CSV file
3. Preview loaded data
4. Click "Use This Data"

**CSV Format Requirements:**
- Must include column headers
- Recommended columns: `transaction_id`, `is_fraud`, `transaction_amount`, etc.

#### 6. Generate Synthetic Data

**Navigate to:** Data Loading → Generate Synthetic Data

**Use Cases:**
- Testing without real data
- Development and debugging
- Model prototyping

**Configuration:**
- Number of records: 100 - 100,000
- Fraud ratio: 0% - 50%
- Date range: Start and end dates

#### 7. Data Preview & Export

**Navigate to:** Data Loading → Data Preview & Export

**Features:**
- View loaded data (first 100 rows)
- See data statistics (mean, std, min, max)
- Check missing values
- Download as CSV or Parquet

**Export Options:**
- **CSV**: Compatible with Excel, Pandas
- **Parquet**: Efficient for large datasets, Spark-compatible

---

## ClickHouse Integration

### Connection Configuration

**Connection String Format:**
```python
connector = ClickHouseConnector(
    host='localhost',
    port=8123,
    username='default',
    password='your_password',
    database='fraud_detection'
)
```

### Python API Examples

**Basic Query:**
```python
from src.utils.clickhouse_connector import ClickHouseConnector

# Connect
connector = ClickHouseConnector(
    host='localhost',
    port=8123,
    username='default',
    password='',
    database='default'
)
connector.connect()

# Load transactions
df = connector.get_transactions(
    table='transactions',
    filters={'is_fraud': 1},
    limit=10000
)

print(f"Loaded {len(df)} fraud transactions")
```

**Date Range Query:**
```python
# Load transactions for specific date range
df = connector.get_transactions_by_date_range(
    table='transactions',
    start_date='2025-03-01',
    end_date='2025-06-30',
    additional_filters={'merchant_category': 'online'},
    limit=100000
)
```

**Aggregations:**
```python
# Get aggregated statistics
stats = connector.get_aggregated_stats(
    table='transactions',
    group_by_cols=['merchant_category', 'is_fraud'],
    agg_cols={
        'transaction_amount': 'AVG',
        'transaction_id': 'COUNT'
    }
)
```

**Custom Query:**
```python
# Execute any SQL query
query = """
SELECT
    toYYYYMM(timestamp) as month,
    COUNT(*) as total,
    SUM(is_fraud) as fraud_count
FROM transactions
WHERE timestamp >= '2025-01-01'
GROUP BY month
ORDER BY month
"""

df = connector.execute_custom_query(query)
```

### ClickHouse Optimization Tips

**1. Use LIMIT for Large Tables:**
```sql
SELECT * FROM transactions
LIMIT 100000  -- Prevent memory issues
```

**2. Filter Early:**
```sql
-- Good: Filter first, then process
SELECT * FROM transactions
WHERE timestamp >= '2025-01-01'
  AND is_fraud = 1
LIMIT 10000

-- Avoid: Processing all rows first
```

**3. Use Proper Date Filtering:**
```sql
-- Use date columns for better performance
WHERE toDate(timestamp) >= '2025-01-01'
```

**4. Index Usage:**
```sql
-- Leverage primary key and indexes
WHERE transaction_id = '12345'  -- Fast if indexed
```

---

## Model Training Module

### Overview
Train fraud detection models using your custom scripts (train_all.py) or built-in scikit-learn algorithms.

### Custom Training Script Integration

**Navigate to:** Model Training → Custom Training Script

#### Your train_all.py Script

**Script Location:** `src/train_all.py`

**Detected Features:**
- PySpark-based pipeline
- Multiple models: Decision Tree, Random Forest, Logistic Regression, GBT
- Training period: 2025-03-01 to 2025-06-30
- Evaluation period: 2025-07-01 to 2025-07-31
- Downsampling support (10% non-fraud)

#### Configuration Steps

**1. Set Script Path:**
```
Training Script Path: src/train_all.py
```

**2. Select Data Source:**
- **Use Loaded Data**: Use data from Data Loading module
- **Specify Data Path**: Provide direct file path
- **Use ClickHouse Query**: Execute query directly

**3. Configure Arguments:**

**Common Arguments for train_all.py:**
```bash
--data_path /path/to/data.csv
--model_type random_forest
--test_size 0.2
--random_state 42
--epochs 100
--save_model
--output_dir ./models
```

**Custom Arguments:**
Add any additional arguments your script accepts:
```
--learning_rate 0.01
--max_depth 10
--n_estimators 100
```

**4. Run Training:**
- Click "Start Training"
- Monitor real-time output
- View logs in training output area
- Check training history after completion

#### Example Training Command

```bash
python src/train_all.py \
  --data_path /tmp/training_data_20251216.csv \
  --model_type random_forest \
  --test_size 0.2 \
  --random_state 42 \
  --save_model \
  --output_dir ./trained_models
```

### Built-in Training

**Navigate to:** Model Training → Built-in Training

**Use Cases:**
- Quick prototyping
- Baseline model creation
- No custom script needed

**Steps:**
1. Ensure data is loaded (check "loaded_data" status)
2. Select features to use
3. Choose algorithm:
   - Random Forest
   - Gradient Boosting
   - Logistic Regression
   - XGBoost
4. Configure hyperparameters
5. Click "Train Model"

**Example:**
```
Algorithm: Random Forest
Features: transaction_amount, merchant_category, customer_age, ...
Number of Trees: 100
Max Depth: 10
Test Split: 0.2
```

**Output:**
- Classification report
- ROC-AUC score
- Model saved to session state
- Ready for inference

### Training History

**Navigate to:** Model Training → Training History

**Features:**
- View all training runs
- Check status (Success/Failed)
- Review command used
- Read full training output
- Compare model performance

---

## Production API Integration

### Overview
Score transactions using your production model API at `localhost:5000/predict`.

### API Configuration

**Navigate to:** Real-time Inference → API Configuration

#### Quick Setup (Production API)

**Steps:**
1. Select "Production API (localhost:5000)" from presets
2. Click "Use Production API Preset"
3. API configured automatically

**Configuration:**
```json
{
  "endpoint_url": "http://localhost:5000/predict",
  "auth_type": "None",
  "timeout": 30,
  "request_format": "transaction_id"
}
```

#### Manual Configuration

**If you need custom settings:**
```
Endpoint URL: http://your-server:port/predict
Authentication: None (or configure as needed)
Timeout: 30 seconds
Content Type: application/json
```

### Scoring Transactions

**Navigate to:** Real-time Inference → Single Transaction

#### Transaction ID Format

**When using Production API preset:**

**Input:**
```
Transaction ID: 83881056341
```

**Click "Score Transaction"**

**API Request:**
```json
{
  "transaction_id": "83881056341"
}
```

**Example Response:**
```json
{
  "fraud_score": 0.85,
  "prediction": "fraud",
  "confidence": 0.92,
  "risk_level": "high"
}
```

#### Full Transaction Data

**When using KNIME or custom API:**

**JSON Input:**
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

### Batch Inference

**Navigate to:** Real-time Inference → Batch Inference

**Steps:**
1. Upload CSV with transaction IDs or full features
2. Click "Run Batch Inference"
3. Monitor progress
4. Download results with fraud scores

**Input CSV Example:**
```csv
transaction_id
83881056341
83881056342
83881056343
```

**Output CSV:**
```csv
transaction_id,fraud_score,prediction
83881056341,0.85,fraud
83881056342,0.12,legitimate
83881056343,0.67,fraud
```

### Inference History & Logs

**Navigate to:** Real-time Inference → Inference History / API Logs

**Features:**
- View all predictions
- Check success/failure status
- Review latency metrics
- Export history as CSV
- Debug API issues with detailed logs

---

## Complete Workflow

### End-to-End Fraud Detection Workflow

#### Step 1: Load Training Data

**Option A: ClickHouse**
```
1. Go to Data Loading → ClickHouse Connection
2. Configure and connect to database
3. Go to Load from ClickHouse → Query Builder
4. Select table: transactions
5. Add filters: timestamp >= '2025-03-01' AND timestamp <= '2025-06-30'
6. Set limit: 100,000
7. Click "Load Data"
```

**Option B: CSV Upload**
```
1. Go to Data Loading → Load from CSV
2. Upload your training data CSV
3. Click "Use This Data"
```

#### Step 2: Train Model

**Option A: Custom Script (train_all.py)**
```
1. Go to Model Training → Custom Training Script
2. Script Path: src/train_all.py
3. Data Source: Use Loaded Data
4. Arguments:
   - model_type: random_forest
   - test_size: 0.2
   - save_model: checked
5. Click "Start Training"
6. Monitor output in real-time
```

**Option B: Built-in Training**
```
1. Go to Model Training → Built-in Training
2. Select features to use
3. Algorithm: Random Forest
4. Configure: n_estimators=100, max_depth=10
5. Click "Train Model"
6. Model saved to session
```

#### Step 3: Configure Production API

```
1. Go to Real-time Inference → API Configuration
2. Select "Production API (localhost:5000)"
3. Click "Use Production API Preset"
4. Click "Test Connection" to verify
```

#### Step 4: Score Transactions

**Single Transaction:**
```
1. Go to Real-time Inference → Single Transaction
2. Enter Transaction ID: 83881056341
3. Click "Score Transaction"
4. View fraud score, prediction, risk level
```

**Batch Scoring:**
```
1. Go to Real-time Inference → Batch Inference
2. Upload CSV with transaction_ids
3. Click "Run Batch Inference"
4. Download results
```

#### Step 5: Monitor Performance

**Model Monitoring:**
```
1. Go to Model Monitoring
2. View real-time metrics
3. Track fraud detection KPIs
4. Monitor performance trends
```

**Feature Monitoring:**
```
1. Go to Feature Monitoring
2. Check for feature drift
3. Monitor data quality
4. Set up alerts
```

---

## API Reference

### Production API (localhost:5000/predict)

**Endpoint:** `POST http://localhost:5000/predict`

**Request Format:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "83881056341"}'
```

**Response Format:**
```json
{
  "transaction_id": "83881056341",
  "fraud_score": 0.85,
  "prediction": "fraud",
  "confidence": 0.92,
  "risk_level": "high",
  "model_version": "v2.1",
  "timestamp": "2025-12-16T10:30:45"
}
```

**Supported Response Fields:**
- `fraud_score` / `score` / `probability` / `fraud_probability`
- `prediction` / `label` / `class`
- `confidence`
- `risk_level`

---

## Troubleshooting

### ClickHouse Connection Issues

**Problem:** Connection failed

**Solutions:**
```
1. Check if ClickHouse server is running:
   ps aux | grep clickhouse

2. Verify port is open:
   netstat -tlnp | grep 8123

3. Test with clickhouse-client:
   clickhouse-client --host localhost --port 9000

4. Check credentials and permissions

5. Verify database exists:
   SHOW DATABASES;
```

**Problem:** "Permission denied" error

**Solution:**
```
-- Grant necessary permissions
GRANT SELECT ON database.* TO user;
```

### Model Training Issues

**Problem:** train_all.py not found

**Solution:**
```
1. Check script exists:
   ls -la src/train_all.py

2. Verify path in UI: src/train_all.py

3. Upload script if missing via UI
```

**Problem:** Training fails with module error

**Solution:**
```
1. Check dependencies:
   pip install -r requirements.txt

2. For Spark issues:
   export PYSPARK_PYTHON=/path/to/python
   export PYSPARK_DRIVER_PYTHON=/path/to/python
```

**Problem:** Out of memory during training

**Solution:**
```
1. Reduce data size in query
2. Use sampling:
   LIMIT 50000
3. Downsample non-fraud class
4. Use Spark for large datasets
```

### Production API Issues

**Problem:** Connection refused to localhost:5000

**Solution:**
```
1. Check if API is running:
   curl http://localhost:5000/health

2. Start API server:
   python your_api_server.py

3. Check firewall:
   sudo ufw allow 5000
```

**Problem:** Invalid response format

**Solution:**
```
1. Check API response structure
2. Supported fields: fraud_score, score, probability
3. Update API to return proper JSON
```

### Data Loading Issues

**Problem:** Query too slow

**Solution:**
```
1. Add WHERE clause to filter early
2. Use LIMIT to restrict rows
3. Add indexes on filter columns:
   ALTER TABLE transactions ADD INDEX idx_timestamp (timestamp) TYPE minmax GRANULARITY 4;
```

**Problem:** Data type errors

**Solution:**
```
1. Check column types in schema:
   DESCRIBE TABLE transactions;

2. Cast types in query:
   SELECT CAST(amount AS Float64) FROM transactions;
```

---

## Best Practices

### Data Loading
1. **Start Small**: Test with LIMIT 1000, then increase
2. **Filter Early**: Use WHERE clauses to reduce data
3. **Use Date Ranges**: Don't load all historical data at once
4. **Export Important Datasets**: Save as Parquet for reuse

### Model Training
1. **Version Control**: Save models with timestamps
2. **Document Args**: Record all training arguments
3. **Track Metrics**: Monitor performance over time
4. **Test First**: Use small datasets for testing

### Production API
1. **Test Endpoint**: Always test connection before batch jobs
2. **Handle Errors**: Implement retry logic
3. **Monitor Latency**: Track API response times
4. **Log Requests**: Keep history for debugging

### Workflow
1. **Iterate Quickly**: Start with small datasets
2. **Validate Data**: Check quality before training
3. **Monitor Drift**: Set up feature monitoring
4. **Document Everything**: Record successful configurations

---

## Support

For issues or questions:
- Check logs in training output
- Review API logs for debugging
- Export data for offline analysis
- Refer to ClickHouse documentation: https://clickhouse.com/docs
