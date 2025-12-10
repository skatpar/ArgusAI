"""
Rule Engine for Fraud Detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable


class Rule:
    """
    Represents a fraud detection rule
    """

    def __init__(self, rule_id: str, name: str, condition: Callable,
                 priority: str = 'Medium', description: str = ''):
        """
        Initialize a fraud detection rule

        Parameters:
        -----------
        rule_id : str
            Unique rule identifier
        name : str
            Rule name
        condition : Callable
            Function that takes a dataframe and returns boolean Series
        priority : str
            Rule priority (Low, Medium, High, Critical)
        description : str
            Rule description
        """
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.priority = priority
        self.description = description
        self.triggers = 0
        self.true_positives = 0
        self.false_positives = 0

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate rule on dataframe

        Parameters:
        -----------
        df : pd.DataFrame
            Transaction data

        Returns:
        --------
        pd.Series
            Boolean series indicating which transactions trigger the rule
        """
        try:
            result = self.condition(df)
            self.triggers += result.sum()
            return result
        except Exception as e:
            print(f"Error evaluating rule {self.rule_id}: {e}")
            return pd.Series([False] * len(df), index=df.index)

    def __repr__(self):
        return f"Rule(id={self.rule_id}, name={self.name}, priority={self.priority})"


class RuleEngine:
    """
    Rule engine for fraud detection
    """

    def __init__(self):
        """Initialize rule engine"""
        self.rules: Dict[str, Rule] = {}
        self.execution_history = []

    def add_rule(self, rule: Rule):
        """
        Add a rule to the engine

        Parameters:
        -----------
        rule : Rule
            Rule to add
        """
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str):
        """
        Remove a rule from the engine

        Parameters:
        -----------
        rule_id : str
            ID of rule to remove
        """
        if rule_id in self.rules:
            del self.rules[rule_id]

    def get_rule(self, rule_id: str) -> Rule:
        """
        Get a rule by ID

        Parameters:
        -----------
        rule_id : str
            Rule ID

        Returns:
        --------
        Rule
            The requested rule
        """
        return self.rules.get(rule_id)

    def evaluate_all(self, df: pd.DataFrame, return_details: bool = False) -> pd.DataFrame:
        """
        Evaluate all rules on a dataframe

        Parameters:
        -----------
        df : pd.DataFrame
            Transaction data
        return_details : bool
            Whether to return detailed results for each rule

        Returns:
        --------
        pd.DataFrame
            Dataframe with flagged transactions and triggered rules
        """
        result_df = df.copy()
        result_df['flagged'] = False
        result_df['triggered_rules'] = ''
        result_df['max_priority'] = 'None'

        priority_order = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1, 'None': 0}

        for rule_id, rule in self.rules.items():
            triggered = rule.evaluate(df)

            # Update flagged status
            result_df.loc[triggered, 'flagged'] = True

            # Add rule ID to triggered rules
            result_df.loc[triggered, 'triggered_rules'] += f"{rule_id};"

            # Update max priority
            for idx in result_df[triggered].index:
                current_priority = result_df.loc[idx, 'max_priority']
                if priority_order[rule.priority] > priority_order[current_priority]:
                    result_df.loc[idx, 'max_priority'] = rule.priority

            if return_details:
                result_df[f'rule_{rule_id}'] = triggered

        # Clean up triggered rules string
        result_df['triggered_rules'] = result_df['triggered_rules'].str.rstrip(';')

        # Record execution
        self.execution_history.append({
            'timestamp': pd.Timestamp.now(),
            'transactions_processed': len(df),
            'transactions_flagged': result_df['flagged'].sum()
        })

        return result_df

    def evaluate_single_rule(self, rule_id: str, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate a single rule

        Parameters:
        -----------
        rule_id : str
            Rule ID to evaluate
        df : pd.DataFrame
            Transaction data

        Returns:
        --------
        pd.Series
            Boolean series of triggered transactions
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule {rule_id} not found")

        return self.rules[rule_id].evaluate(df)

    def get_statistics(self) -> pd.DataFrame:
        """
        Get statistics for all rules

        Returns:
        --------
        pd.DataFrame
            Statistics for each rule
        """
        stats = []

        for rule_id, rule in self.rules.items():
            precision = rule.true_positives / rule.triggers if rule.triggers > 0 else 0

            stats.append({
                'rule_id': rule_id,
                'name': rule.name,
                'priority': rule.priority,
                'triggers': rule.triggers,
                'true_positives': rule.true_positives,
                'false_positives': rule.false_positives,
                'precision': precision
            })

        return pd.DataFrame(stats)

    def update_rule_performance(self, rule_id: str, true_positives: int, false_positives: int):
        """
        Update rule performance metrics

        Parameters:
        -----------
        rule_id : str
            Rule ID
        true_positives : int
            Number of true positives
        false_positives : int
            Number of false positives
        """
        if rule_id in self.rules:
            self.rules[rule_id].true_positives += true_positives
            self.rules[rule_id].false_positives += false_positives


# Predefined rule conditions

def high_amount_rule(threshold: float = 5000) -> Callable:
    """Rule: Transaction amount exceeds threshold"""
    def condition(df: pd.DataFrame) -> pd.Series:
        if 'transaction_amount' in df.columns:
            return df['transaction_amount'] > threshold
        return pd.Series([False] * len(df))
    return condition


def velocity_rule(threshold: int = 3) -> Callable:
    """Rule: High transaction velocity"""
    def condition(df: pd.DataFrame) -> pd.Series:
        if 'transactions_24h' in df.columns:
            return df['transactions_24h'] > threshold
        return pd.Series([False] * len(df))
    return condition


def foreign_transaction_rule(home_country: str = 'USA') -> Callable:
    """Rule: Transaction in foreign country"""
    def condition(df: pd.DataFrame) -> pd.Series:
        if 'country' in df.columns:
            return df['country'] != home_country
        return pd.Series([False] * len(df))
    return condition


def high_risk_merchant_rule(categories: List[str] = None) -> Callable:
    """Rule: High-risk merchant category"""
    if categories is None:
        categories = ['gambling', 'crypto', 'wire_transfer']

    def condition(df: pd.DataFrame) -> pd.Series:
        if 'merchant_category' in df.columns:
            return df['merchant_category'].isin(categories)
        return pd.Series([False] * len(df))
    return condition


def unusual_hour_rule(start_hour: int = 6, end_hour: int = 22) -> Callable:
    """Rule: Transaction during unusual hours"""
    def condition(df: pd.DataFrame) -> pd.Series:
        if 'transaction_hour' in df.columns:
            return (df['transaction_hour'] < start_hour) | (df['transaction_hour'] > end_hour)
        return pd.Series([False] * len(df))
    return condition


def distance_rule(threshold: float = 100) -> Callable:
    """Rule: Transaction far from home"""
    def condition(df: pd.DataFrame) -> pd.Series:
        if 'distance_from_home' in df.columns:
            return df['distance_from_home'] > threshold
        return pd.Series([False] * len(df))
    return condition


def new_device_rule() -> Callable:
    """Rule: Transaction from new device"""
    def condition(df: pd.DataFrame) -> pd.Series:
        # Simplified - in practice would check device history
        return pd.Series([False] * len(df))
    return condition


def create_default_rules() -> RuleEngine:
    """
    Create a rule engine with default fraud detection rules

    Returns:
    --------
    RuleEngine
        Rule engine with default rules
    """
    engine = RuleEngine()

    # High amount rule
    engine.add_rule(Rule(
        rule_id='RULE_001',
        name='High Amount Transaction',
        condition=high_amount_rule(5000),
        priority='High',
        description='Transaction amount exceeds $5,000'
    ))

    # Velocity rule
    engine.add_rule(Rule(
        rule_id='RULE_002',
        name='High Velocity',
        condition=velocity_rule(3),
        priority='Medium',
        description='More than 3 transactions in 24 hours'
    ))

    # Foreign transaction rule
    engine.add_rule(Rule(
        rule_id='RULE_003',
        name='Foreign Transaction',
        condition=foreign_transaction_rule('USA'),
        priority='Medium',
        description='Transaction in foreign country'
    ))

    # High-risk merchant rule
    engine.add_rule(Rule(
        rule_id='RULE_004',
        name='High Risk Merchant',
        condition=high_risk_merchant_rule(['gambling', 'crypto', 'wire_transfer']),
        priority='High',
        description='Transaction at high-risk merchant'
    ))

    # Unusual hours rule
    engine.add_rule(Rule(
        rule_id='RULE_005',
        name='Unusual Hours',
        condition=unusual_hour_rule(6, 22),
        priority='Low',
        description='Transaction outside business hours'
    ))

    # Distance rule
    engine.add_rule(Rule(
        rule_id='RULE_006',
        name='Far from Home',
        condition=distance_rule(100),
        priority='Medium',
        description='Transaction more than 100km from home'
    ))

    return engine


if __name__ == '__main__':
    # Test rule engine
    from data_generator import generate_fraud_data

    # Generate test data
    df = generate_fraud_data(n_samples=1000)

    # Create rule engine
    engine = create_default_rules()

    # Evaluate rules
    results = engine.evaluate_all(df, return_details=True)

    print("Rule Engine Test Results")
    print("=" * 50)
    print(f"Total transactions: {len(df)}")
    print(f"Flagged transactions: {results['flagged'].sum()}")
    print(f"Flagged rate: {results['flagged'].mean():.2%}")
    print("\nRule Statistics:")
    print(engine.get_statistics())

    print("\nSample flagged transactions:")
    print(results[results['flagged']][['transaction_id', 'transaction_amount',
                                       'merchant_category', 'triggered_rules']].head())
