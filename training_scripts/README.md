# Training Scripts Directory

This directory contains custom training scripts for the ArgusAI fraud detection platform.

## How to Add Training Scripts

1. **Place your Python training scripts (.py files) in this directory**
   - Example: `train_all.py`, `train_rf.py`, `train_ensemble.py`

2. **Scripts will be automatically discovered** by the Model Training module

3. **Configure script parameters** in the Configuration tab:
   - Go to Model Training → Configuration → Training Scripts
   - Set default arguments for all scripts
   - Add custom configurations for specific scripts

## Script Requirements

Your training scripts should:

1. **Accept command-line arguments**:
   ```python
   import argparse

   parser = argparse.ArgumentParser()
   parser.add_argument('--data_path', type=str, help='Path to training data')
   parser.add_argument('--model_type', type=str, default='random_forest')
   parser.add_argument('--test_size', type=float, default=0.2)
   parser.add_argument('--random_state', type=int, default=42)
   parser.add_argument('--save_model', action='store_true')
   args = parser.parse_args()
   ```

2. **Load data** from the provided path or from ClickHouse

3. **Train and evaluate** your model

4. **Save artifacts** to the configured directory (default: `artifacts/models/`)

## Example Script Structure

```python
#!/usr/bin/env python
"""
Example Training Script for ArgusAI
"""

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--model_type', default='random_forest')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--save_model', action='store_true')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    # Prepare features
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Train model
    print(f"Training {args.model_type} model...")
    model = RandomForestClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")

    # Save model
    if args.save_model:
        model_path = f"artifacts/models/{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
```

## Global Configuration

The system provides global configuration in `configs/training_config.yaml`:

- **Data Source Configuration**: ClickHouse settings, filters, data limits
- **Model Configuration**: Algorithm type, hyperparameters, training parameters
- **Artifacts Configuration**: Save directories, versioning, formats
- **Script Configuration**: Default arguments, custom script settings

## Using Configuration in Scripts

You can load the global configuration in your scripts:

```python
import yaml

# Load config
with open('configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get settings
data_config = config['data_source']
model_config = config['model']
artifacts_config = config['artifacts']

# Use in your script
database = data_config['clickhouse']['database']
table = data_config['clickhouse']['table']
save_dir = artifacts_config['models']['save_dir']
```

## Logging and Monitoring

For best practices:

1. **Print progress** to stdout (captured by the UI)
2. **Save training logs** to `artifacts/logs/`
3. **Save metrics** in JSON format for tracking
4. **Include timestamps** in all outputs

## Need Help?

- Check the Configuration tab in Model Training for settings
- View `configs/training_config.yaml` for all available options
- See existing scripts in this directory for examples
