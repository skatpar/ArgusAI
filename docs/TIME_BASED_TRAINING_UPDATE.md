# Time-Based Training Module Update

## Requirements Implemented

1. **Time-based data split** (not random test_size)
   - Training period: Start date → End date
   - Evaluation period: Start date → End date
   - Only evaluate on evaluation period data

2. **Remove default hyperparameters**
   - User must provide all hyperparameters
   - No default values

3. **Enhanced feature selection UI**
   - Search/filter features
   - Group by prefix (txn_, user_, channel_, etc.)
   - Quick select by group
   - Select all / Clear all buttons

4. **Use loaded ClickHouse data**
   - Feature Monitoring uses global loaded data
   - No separate data loading

## Key Code Changes for `src/pages/model_training.py`

### Time-Based Split Section (Replace lines 788-842)

```python
st.success(f"Using data: {st.session_state.get('data_source')} ({len(df):,} rows)")
st.info(f"Target column: **{target_col}**")

# Check if cutoff_date exists for time-based split
if 'cutoff_date' not in df.columns:
    st.error("Dataset must contain 'cutoff_date' column for time-based splitting")
    st.info("Please load data with cutoff_date column from ClickHouse")
    return

st.markdown("---")

# Time-based split configuration
st.markdown("#### Time-Based Data Split")
st.info("Specify date ranges for training and evaluation periods")

# Get date range from data
df['cutoff_date'] = pd.to_datetime(df['cutoff_date'])
min_date = df['cutoff_date'].min().date()
max_date = df['cutoff_date'].max().date()

st.text(f"Available date range: {min_date} to {max_date}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Training Period:**")
    train_start = st.date_input(
        "Training Start Date:",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    train_end = st.date_input(
        "Training End Date:",
        value=min_date + pd.Timedelta(days=90),
        min_value=min_date,
        max_value=max_date
    )

with col2:
    st.markdown("**Evaluation Period:**")
    eval_start = st.date_input(
        "Evaluation Start Date:",
        value=train_end + pd.Timedelta(days=1),
        min_value=min_date,
        max_value=max_date
    )
    eval_end = st.date_input(
        "Evaluation End Date:",
        value=min(train_end + pd.Timedelta(days=31), max_date),
        min_value=min_date,
        max_value=max_date
    )

# Validate date ranges
if train_start >= train_end:
    st.error("Training end date must be after training start date")
    return

if eval_start >= eval_end:
    st.error("Evaluation end date must be after evaluation start date")
    return

if eval_start <= train_end:
    st.warning("⚠️ Warning: Evaluation period overlaps with training period")

# Split data by time
train_mask = (df['cutoff_date'].dt.date >= train_start) & (df['cutoff_date'].dt.date <= train_end)
eval_mask = (df['cutoff_date'].dt.date >= eval_start) & (df['cutoff_date'].dt.date <= eval_end)

df_train = df[train_mask]
df_eval = df[eval_mask]

st.text(f"Training samples: {len(df_train):,} | Evaluation samples: {len(df_eval):,}")

if len(df_train) == 0:
    st.error("No training data in selected date range")
    return

if len(df_eval) == 0:
    st.error("No evaluation data in selected date range")
    return
```

### Enhanced Feature Selection UI (Replace lines 793-812)

```python
st.markdown("---")
st.markdown("#### Feature Selection")

# Exclude non-feature columns
exclude_cols = ['transaction_id', 'id', 'timestamp', 'date', 'cutoff_date',
               'mbar_account_type_name', 'ac_to', 'ac_from', 'end_balance', target_col]

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_options = [col for col in numeric_cols if col not in exclude_cols]

# Enhanced feature selection UI
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(f"**Available Features:** {len(feature_options)} features")

    # Search/filter features
    search_term = st.text_input(
        "🔍 Search features:",
        placeholder="e.g., txn_, user_, channel_",
        help="Filter features by name"
    )

    if search_term:
        filtered_features = [f for f in feature_options if search_term.lower() in f.lower()]
        st.caption(f"Showing {len(filtered_features)} features matching '{search_term}'")
    else:
        filtered_features = feature_options

    # Group features by prefix
    feature_groups = {}
    for feat in filtered_features:
        prefix = feat.split('_')[0] if '_' in feat else 'other'
        if prefix not in feature_groups:
            feature_groups[prefix] = []
        feature_groups[prefix].append(feat)

    # Show feature groups
    st.markdown("**Feature Groups:**")
    for prefix, features in sorted(feature_groups.items()):
        st.text(f"  {prefix}_*: {len(features)} features")

with col2:
    st.markdown("**Quick Actions:**")
    if st.button("✓ Select All", use_container_width=True):
        st.session_state.selected_features_builtin = filtered_features
        st.rerun()

    if st.button("✗ Clear All", use_container_width=True):
        st.session_state.selected_features_builtin = []
        st.rerun()

    # Group selection
    st.markdown("**Select by Group:**")
    for prefix in sorted(feature_groups.keys()):
        if st.button(f"+ {prefix}_*", use_container_width=True, key=f"group_{prefix}"):
            current = st.session_state.get('selected_features_builtin', [])
            st.session_state.selected_features_builtin = list(set(current + feature_groups[prefix]))
            st.rerun()

# Main feature multiselect
selected_features = st.multiselect(
    "Selected Features:",
    options=filtered_features,
    default=st.session_state.get('selected_features_builtin', []),
    help="Select features to use for training"
)

# Store selection
st.session_state.selected_features_builtin = selected_features

if not selected_features:
    st.warning("Please select at least one feature")
    return

st.info(f"**{len(selected_features)}** features selected for training")
```

### Remove Default Hyperparameters (Replace lines 819-839)

```python
st.markdown("---")
st.markdown("#### Model Configuration")

col1, col2 = st.columns(2)

with col1:
    algorithm = st.selectbox(
        "Algorithm:",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    )

with col2:
    st.markdown("**Hyperparameters:**")
    # NO DEFAULTS - user must provide values
    if algorithm == "Random Forest":
        n_estimators = st.number_input("Number of Trees:", min_value=10, max_value=1000, value=None, placeholder="e.g., 100")
        max_depth = st.number_input("Max Depth:", min_value=1, max_value=50, value=None, placeholder="e.g., 10")

        if n_estimators is None or max_depth is None:
            st.warning("⚠️ Please provide all hyperparameters")
            return

    elif algorithm == "Gradient Boosting":
        n_estimators = st.number_input("Number of Trees:", min_value=10, max_value=1000, value=None, placeholder="e.g., 100")
        learning_rate = st.number_input("Learning Rate:", min_value=0.001, max_value=1.0, value=None, placeholder="e.g., 0.1", format="%.3f")

        if n_estimators is None or learning_rate is None:
            st.warning("⚠️ Please provide all hyperparameters")
            return

    elif algorithm == "Logistic Regression":
        C = st.number_input("Regularization (C):", min_value=0.001, max_value=100.0, value=None, placeholder="e.g., 1.0", format="%.3f")
        max_iter = st.number_input("Max Iterations:", min_value=100, max_value=10000, value=None, placeholder="e.g., 1000")

        if C is None or max_iter is None:
            st.warning("⚠️ Please provide all hyperparameters")
            return
```

### Update Training Logic (Replace lines 843-898)

```python
st.markdown("---")

if st.button("Train Model", type="primary"):
    with st.spinner("Training model..."):
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

            # Prepare training data
            X_train = df_train[selected_features]
            y_train = df_train[target_col]

            # Prepare evaluation data
            X_eval = df_eval[selected_features]
            y_eval = df_eval[target_col]

            # Train model
            if algorithm == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    random_state=42
                )
            elif algorithm == "Gradient Boosting":
                model = GradientBoostingClassifier(
                    n_estimators=int(n_estimators),
                    learning_rate=float(learning_rate),
                    random_state=42
                )
            elif algorithm == "Logistic Regression":
                model = LogisticRegression(
                    C=float(C),
                    max_iter=int(max_iter),
                    random_state=42
                )

            model.fit(X_train, y_train)

            # Evaluate ONLY on evaluation data
            y_pred = model.predict(X_eval)
            y_pred_proba = model.predict_proba(X_eval)[:, 1]

            st.success("Model trained successfully!")

            # Display results
            st.markdown("---")
            st.markdown("#### Training & Evaluation Results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Training Samples", f"{len(X_train):,}")

            with col2:
                st.metric("Evaluation Samples", f"{len(X_eval):,}")

            with col3:
                roc_auc = roc_auc_score(y_eval, y_pred_proba)
                st.metric("ROC-AUC (Eval)", f"{roc_auc:.4f}")

            with col4:
                fraud_rate = (y_eval.sum() / len(y_eval) * 100)
                st.metric("Fraud Rate (Eval)", f"{fraud_rate:.2f}%")

            # Show period information
            st.markdown("**Training Period:** {} to {}".format(train_start, train_end))
            st.markdown("**Evaluation Period:** {} to {}".format(eval_start, eval_end))

            # Classification report (ONLY on evaluation data)
            st.markdown("---")
            st.markdown("**Classification Report (Evaluation Data):**")
            report = classification_report(y_eval, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
```

### Update Metrics Saving (Replace lines 993-1006)

```python
# Save metrics
metrics_path = os.path.join(logs_dir, f"{model_name}_metrics.json")
import json
with open(metrics_path, 'w') as f:
    json.dump({
        'algorithm': algorithm,
        'features': selected_features,
        'roc_auc': roc_auc,
        'training_period': {
            'start': str(train_start),
            'end': str(train_end),
            'samples': len(X_train)
        },
        'evaluation_period': {
            'start': str(eval_start),
            'end': str(eval_end),
            'samples': len(X_eval)
        },
        'fraud_rate_eval': float(fraud_rate),
        'trained_at': timestamp,
        'classification_report': report,
        'hyperparameters': {
            'n_estimators': int(n_estimators) if algorithm in ["Random Forest", "Gradient Boosting"] else None,
            'max_depth': int(max_depth) if algorithm == "Random Forest" else None,
            'learning_rate': float(learning_rate) if algorithm == "Gradient Boosting" else None,
            'C': float(C) if algorithm == "Logistic Regression" else None,
            'max_iter': int(max_iter) if algorithm == "Logistic Regression" else None
        }
    }, f, indent=2)
```

## Implementation Steps

1. Back up current `src/pages/model_training.py`
2. Apply the code changes section by section
3. Test with actual ClickHouse data containing cutoff_date column
4. Verify time-based splits work correctly
5. Confirm no default hyperparameters are used

## Testing Checklist

- [ ] Load data with cutoff_date column
- [ ] Select training date range
- [ ] Select evaluation date range
- [ ] Use enhanced feature selection
- [ ] Provide all hyperparameters (no defaults)
- [ ] Train model
- [ ] Verify evaluation only uses evaluation period data
- [ ] Check saved metrics contain period information
