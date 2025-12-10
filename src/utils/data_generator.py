"""
Data Generator - Generate synthetic fraud detection data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_fraud_data(n_samples=10000, fraud_rate=0.05, random_state=42):
    """
    Generate synthetic fraud detection dataset

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    fraud_rate : float
        Proportion of fraudulent transactions (0-1)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Synthetic fraud dataset
    """
    np.random.seed(random_state)

    # Generate base features
    data = {
        'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
        'customer_id': [f'CUST_{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
        'timestamp': generate_timestamps(n_samples),
    }

    # Transaction features
    data['transaction_amount'] = generate_transaction_amounts(n_samples)
    data['merchant_id'] = [f'MERCH_{np.random.randint(100, 999)}' for _ in range(n_samples)]
    data['merchant_category'] = np.random.choice(
        ['retail', 'groceries', 'gas_station', 'restaurant', 'online',
         'electronics', 'gambling', 'crypto', 'wire_transfer', 'travel'],
        size=n_samples,
        p=[0.20, 0.15, 0.10, 0.15, 0.15, 0.10, 0.05, 0.03, 0.02, 0.05]
    )

    # Location features
    data['country'] = np.random.choice(
        ['USA', 'UK', 'Canada', 'Germany', 'France', 'Nigeria', 'Russia', 'China', 'Brazil', 'India'],
        size=n_samples,
        p=[0.50, 0.15, 0.10, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03]
    )

    data['city'] = generate_cities(n_samples, data['country'])
    data['distance_from_home'] = np.abs(np.random.normal(10, 50, n_samples))

    # Device features
    data['device_type'] = np.random.choice(['Desktop', 'Mobile', 'Tablet'], size=n_samples,
                                          p=[0.4, 0.5, 0.1])
    data['browser'] = np.random.choice(['Chrome', 'Safari', 'Firefox', 'Edge', 'Other'],
                                      size=n_samples, p=[0.4, 0.25, 0.15, 0.15, 0.05])

    # Customer features
    data['customer_age'] = np.random.randint(18, 80, n_samples)
    data['account_age_days'] = np.random.randint(1, 3650, n_samples)

    # Velocity features
    data['transactions_24h'] = np.random.poisson(2, n_samples)
    data['transactions_7d'] = np.random.poisson(8, n_samples)

    # Card features
    data['card_type'] = np.random.choice(['Visa', 'Mastercard', 'Amex', 'Discover'],
                                        size=n_samples, p=[0.45, 0.35, 0.15, 0.05])

    # Time features
    df = pd.DataFrame(data)
    df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['transaction_day'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['transaction_month'] = pd.to_datetime(df['timestamp']).dt.month

    # Generate fraud labels
    n_fraud = int(n_samples * fraud_rate)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    df['is_fraud'] = 0
    df.loc[fraud_indices, 'is_fraud'] = 1

    # Adjust features for fraud cases to make them more realistic
    df = add_fraud_patterns(df)

    return df


def generate_timestamps(n_samples):
    """Generate realistic timestamps"""
    start_date = datetime.now() - timedelta(days=30)
    timestamps = []

    for _ in range(n_samples):
        random_days = np.random.randint(0, 30)
        random_hours = np.random.randint(0, 24)
        random_minutes = np.random.randint(0, 60)
        random_seconds = np.random.randint(0, 60)

        timestamp = start_date + timedelta(
            days=random_days,
            hours=random_hours,
            minutes=random_minutes,
            seconds=random_seconds
        )
        timestamps.append(timestamp)

    return sorted(timestamps)


def generate_transaction_amounts(n_samples):
    """Generate realistic transaction amounts with long tail"""
    # Mix of normal transactions and some high-value ones
    normal_amounts = np.abs(np.random.lognormal(3.5, 1.2, int(n_samples * 0.95)))
    high_amounts = np.abs(np.random.lognormal(7, 1.5, int(n_samples * 0.05)))

    amounts = np.concatenate([normal_amounts, high_amounts])
    np.random.shuffle(amounts)

    return np.round(amounts[:n_samples], 2)


def generate_cities(n_samples, countries):
    """Generate cities based on countries"""
    city_map = {
        'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'],
        'UK': ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Liverpool'],
        'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Ottawa'],
        'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne'],
        'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'],
        'Nigeria': ['Lagos', 'Abuja', 'Kano', 'Ibadan', 'Port Harcourt'],
        'Russia': ['Moscow', 'St Petersburg', 'Novosibirsk', 'Yekaterinburg', 'Kazan'],
        'China': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Chengdu'],
        'Brazil': ['Sao Paulo', 'Rio de Janeiro', 'Brasilia', 'Salvador', 'Fortaleza'],
        'India': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai']
    }

    cities = []
    for country in countries:
        if country in city_map:
            cities.append(np.random.choice(city_map[country]))
        else:
            cities.append('Unknown')

    return cities


def add_fraud_patterns(df):
    """Add realistic patterns to fraud transactions"""
    fraud_mask = df['is_fraud'] == 1

    # Fraudulent transactions tend to have:
    # 1. Higher amounts
    df.loc[fraud_mask, 'transaction_amount'] *= np.random.uniform(1.5, 3.0, fraud_mask.sum())

    # 2. High-risk merchant categories
    high_risk_cats = ['gambling', 'crypto', 'wire_transfer', 'online']
    fraud_indices = df[fraud_mask].index
    change_category = np.random.choice(fraud_indices, int(len(fraud_indices) * 0.6), replace=False)
    df.loc[change_category, 'merchant_category'] = np.random.choice(high_risk_cats, len(change_category))

    # 3. Foreign transactions
    foreign_countries = ['Nigeria', 'Russia', 'China', 'Brazil']
    change_country = np.random.choice(fraud_indices, int(len(fraud_indices) * 0.4), replace=False)
    df.loc[change_country, 'country'] = np.random.choice(foreign_countries, len(change_country))

    # 4. Unusual hours
    unusual_hours = list(range(0, 6)) + list(range(22, 24))
    change_hours = np.random.choice(fraud_indices, int(len(fraud_indices) * 0.5), replace=False)
    df.loc[change_hours, 'transaction_hour'] = np.random.choice(unusual_hours, len(change_hours))

    # 5. Larger distance from home
    df.loc[fraud_mask, 'distance_from_home'] *= np.random.uniform(3.0, 10.0, fraud_mask.sum())

    # 6. Higher velocity
    df.loc[fraud_mask, 'transactions_24h'] *= np.random.uniform(2.0, 5.0, fraud_mask.sum())

    return df


def generate_customer_data(n_customers=1000, random_state=42):
    """Generate customer profile data"""
    np.random.seed(random_state)

    customers = {
        'customer_id': [f'CUST_{i:04d}' for i in range(1000, 1000 + n_customers)],
        'age': np.random.randint(18, 80, n_customers),
        'account_age_days': np.random.randint(1, 3650, n_customers),
        'total_transactions': np.random.poisson(50, n_customers),
        'total_amount': np.random.lognormal(8, 2, n_customers),
        'avg_transaction_amount': np.random.lognormal(4, 1, n_customers),
        'fraud_history': np.random.binomial(1, 0.02, n_customers),
        'credit_score': np.random.randint(300, 850, n_customers),
        'income_level': np.random.choice(['Low', 'Medium', 'High', 'Very High'],
                                        size=n_customers, p=[0.2, 0.4, 0.3, 0.1])
    }

    return pd.DataFrame(customers)


if __name__ == '__main__':
    # Test data generation
    df = generate_fraud_data(n_samples=1000)
    print("Generated dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nFraud distribution:")
    print(df['is_fraud'].value_counts())
    print("\nData types:")
    print(df.dtypes)
