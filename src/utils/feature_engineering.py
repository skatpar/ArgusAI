"""
Feature Engineering Utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime


def create_features(df, time_features=True, amount_features=True, velocity_features=True,
                   aggregate_features=True, ratio_features=True, interaction_features=False):
    """
    Create engineered features for fraud detection

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    time_features : bool
        Create time-based features
    amount_features : bool
        Create amount-based features
    velocity_features : bool
        Create velocity features
    aggregate_features : bool
        Create aggregated customer features
    ratio_features : bool
        Create ratio features
    interaction_features : bool
        Create interaction features

    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    df_featured = df.copy()

    # Time-based features
    if time_features:
        df_featured = create_time_features(df_featured)

    # Amount-based features
    if amount_features:
        df_featured = create_amount_features(df_featured)

    # Velocity features
    if velocity_features:
        df_featured = create_velocity_features(df_featured)

    # Aggregate features
    if aggregate_features:
        df_featured = create_aggregate_features(df_featured)

    # Ratio features
    if ratio_features:
        df_featured = create_ratio_features(df_featured)

    # Interaction features
    if interaction_features:
        df_featured = create_interaction_features(df_featured)

    return df_featured


def create_time_features(df):
    """Create time-based features"""
    df = df.copy()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Hour features
        if 'transaction_hour' not in df.columns:
            df['transaction_hour'] = df['timestamp'].dt.hour

        df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['transaction_hour'] >= 9) & (df['transaction_hour'] <= 17)).astype(int)

        # Day features
        if 'transaction_day' not in df.columns:
            df['transaction_day'] = df['timestamp'].dt.dayofweek

        df['is_weekend'] = (df['transaction_day'] >= 5).astype(int)

        # Month features
        if 'transaction_month' not in df.columns:
            df['transaction_month'] = df['timestamp'].dt.month

        df['is_month_start'] = (df['timestamp'].dt.day <= 5).astype(int)
        df['is_month_end'] = (df['timestamp'].dt.day >= 25).astype(int)

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['transaction_day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['transaction_day'] / 7)

    return df


def create_amount_features(df):
    """Create amount-based features"""
    df = df.copy()

    if 'transaction_amount' in df.columns:
        # Log transform
        df['amount_log'] = np.log1p(df['transaction_amount'])

        # Square root transform
        df['amount_sqrt'] = np.sqrt(df['transaction_amount'])

        # Binned amounts
        df['amount_bin'] = pd.cut(df['transaction_amount'],
                                  bins=[0, 50, 100, 500, 1000, 5000, np.inf],
                                  labels=[0, 1, 2, 3, 4, 5])
        df['amount_bin'] = df['amount_bin'].astype(float)

        # Z-score normalization
        df['amount_zscore'] = (df['transaction_amount'] - df['transaction_amount'].mean()) / df['transaction_amount'].std()

        # Deviation from customer mean (if customer data available)
        if 'customer_id' in df.columns:
            customer_avg = df.groupby('customer_id')['transaction_amount'].transform('mean')
            df['amount_vs_customer_avg'] = df['transaction_amount'] - customer_avg
            df['amount_vs_customer_avg_ratio'] = df['transaction_amount'] / (customer_avg + 1)

    return df


def create_velocity_features(df):
    """Create transaction velocity features"""
    df = df.copy()

    if 'customer_id' in df.columns and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by customer and timestamp
        df = df.sort_values(['customer_id', 'timestamp'])

        # Time since last transaction
        df['time_since_last_txn'] = df.groupby('customer_id')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(86400)  # Fill first with 24 hours

        # Transaction frequency
        df['txn_count_1h'] = df.groupby('customer_id')['timestamp'].transform(
            lambda x: x.rolling('1H').count()
        )

        # Rolling counts
        if 'transactions_24h' not in df.columns:
            df['transactions_24h'] = df.groupby('customer_id').cumcount() + 1

        if 'transactions_7d' not in df.columns:
            df['transactions_7d'] = df.groupby('customer_id').cumcount() + 1

        # Merchant velocity
        if 'merchant_id' in df.columns:
            df['merchant_velocity'] = df.groupby(['customer_id', 'merchant_id']).cumcount() + 1

    return df


def create_aggregate_features(df):
    """Create aggregated customer-level features"""
    df = df.copy()

    if 'customer_id' in df.columns:
        # Customer transaction statistics
        if 'transaction_amount' in df.columns:
            customer_stats = df.groupby('customer_id')['transaction_amount'].agg([
                ('customer_total_amount', 'sum'),
                ('customer_avg_amount', 'mean'),
                ('customer_std_amount', 'std'),
                ('customer_min_amount', 'min'),
                ('customer_max_amount', 'max'),
                ('customer_txn_count', 'count')
            ]).reset_index()

            df = df.merge(customer_stats, on='customer_id', how='left')

            # Fill NaN std with 0
            df['customer_std_amount'] = df['customer_std_amount'].fillna(0)

        # Merchant diversity
        if 'merchant_id' in df.columns:
            merchant_diversity = df.groupby('customer_id')['merchant_id'].nunique().reset_index()
            merchant_diversity.columns = ['customer_id', 'merchant_diversity']
            df = df.merge(merchant_diversity, on='customer_id', how='left')

        # Category diversity
        if 'merchant_category' in df.columns:
            category_diversity = df.groupby('customer_id')['merchant_category'].nunique().reset_index()
            category_diversity.columns = ['customer_id', 'category_diversity']
            df = df.merge(category_diversity, on='customer_id', how='left')

    return df


def create_ratio_features(df):
    """Create ratio features"""
    df = df.copy()

    # Amount ratios
    if 'transaction_amount' in df.columns and 'customer_avg_amount' in df.columns:
        df['amount_to_avg_ratio'] = df['transaction_amount'] / (df['customer_avg_amount'] + 1)
        df['amount_to_max_ratio'] = df['transaction_amount'] / (df['customer_max_amount'] + 1)

    # Velocity ratios
    if 'transactions_24h' in df.columns and 'customer_txn_count' in df.columns:
        df['recent_txn_ratio'] = df['transactions_24h'] / (df['customer_txn_count'] + 1)

    # Distance ratios
    if 'distance_from_home' in df.columns:
        df['distance_log'] = np.log1p(df['distance_from_home'])
        df['is_far_from_home'] = (df['distance_from_home'] > 100).astype(int)

    return df


def create_interaction_features(df):
    """Create interaction features"""
    df = df.copy()

    # Amount x Hour interaction
    if 'transaction_amount' in df.columns and 'transaction_hour' in df.columns:
        df['amount_hour_interaction'] = df['transaction_amount'] * df['transaction_hour']

    # Amount x Weekend interaction
    if 'transaction_amount' in df.columns and 'is_weekend' in df.columns:
        df['amount_weekend_interaction'] = df['transaction_amount'] * df['is_weekend']

    # Distance x Amount interaction
    if 'distance_from_home' in df.columns and 'transaction_amount' in df.columns:
        df['distance_amount_interaction'] = df['distance_from_home'] * df['transaction_amount']

    return df


def get_feature_importance(model, feature_names):
    """
    Get feature importance from trained model

    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names

    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df
    else:
        return None


def create_risk_score(df, weights=None):
    """
    Create a simple rule-based risk score

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    weights : dict
        Dictionary of feature weights

    Returns:
    --------
    pd.Series
        Risk scores
    """
    if weights is None:
        weights = {
            'high_amount': 0.3,
            'night_time': 0.2,
            'foreign': 0.25,
            'high_velocity': 0.15,
            'far_from_home': 0.1
        }

    risk_score = 0

    # High amount
    if 'transaction_amount' in df.columns:
        risk_score += (df['transaction_amount'] > 1000).astype(int) * weights['high_amount']

    # Night time
    if 'is_night' in df.columns:
        risk_score += df['is_night'] * weights['night_time']

    # Foreign transaction
    if 'country' in df.columns:
        risk_score += (df['country'] != 'USA').astype(int) * weights['foreign']

    # High velocity
    if 'transactions_24h' in df.columns:
        risk_score += (df['transactions_24h'] > 5).astype(int) * weights['high_velocity']

    # Far from home
    if 'distance_from_home' in df.columns:
        risk_score += (df['distance_from_home'] > 100).astype(int) * weights['far_from_home']

    return risk_score


if __name__ == '__main__':
    # Test feature engineering
    from data_generator import generate_fraud_data

    df = generate_fraud_data(n_samples=1000)
    print("Original features:", df.shape[1])

    df_featured = create_features(df)
    print("After feature engineering:", df_featured.shape[1])
    print("\nNew features created:")
    new_features = set(df_featured.columns) - set(df.columns)
    for feature in sorted(new_features):
        print(f"  - {feature}")
