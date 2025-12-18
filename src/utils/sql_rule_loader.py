"""
SQL Rule Loader and Executor
Load SQL-based fraud detection rules from files and execute them on DataFrames
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SQLRule:
    """Represents a single SQL-based fraud detection rule"""

    def __init__(self, rule_id: str, name: str, sql_condition: str,
                 priority: str = 'Medium', description: str = '',
                 file_path: str = None):
        """
        Initialize SQL Rule

        Parameters:
        -----------
        rule_id : str
            Unique rule identifier
        name : str
            Rule name
        sql_condition : str
            SQL WHERE clause condition (without 'WHERE' keyword)
        priority : str
            Rule priority (Low, Medium, High, Critical)
        description : str
            Rule description
        file_path : str
            Path to the SQL file
        """
        self.rule_id = rule_id
        self.name = name
        self.sql_condition = sql_condition
        self.priority = priority
        self.description = description
        self.file_path = file_path
        self.triggers = 0
        self.true_positives = 0
        self.false_positives = 0
        self.detection_rate = 0.0
        self.precision = 0.0

    def to_pandas_query(self) -> str:
        """
        Convert SQL WHERE condition to pandas query string

        Returns:
        --------
        str
            Pandas query string
        """
        # Basic SQL to pandas query conversion
        query = self.sql_condition

        # Replace SQL operators with pandas equivalents
        replacements = {
            ' AND ': ' and ',
            ' OR ': ' or ',
            ' NOT ': ' not ',
            ' BETWEEN ': ' between ',
            ' IN ': ' in ',
            ' LIKE ': ' like ',
            '!=': '!=',
            '<>': '!=',
            '=': '==',
        }

        for sql_op, pandas_op in replacements.items():
            query = query.replace(sql_op, pandas_op)

        return query

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate rule on DataFrame

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
            pandas_query = self.to_pandas_query()
            result = df.query(pandas_query, engine='python')

            # Create boolean series
            triggered = pd.Series(False, index=df.index)
            triggered.loc[result.index] = True

            self.triggers = triggered.sum()
            return triggered

        except Exception as e:
            logger.error(f"Error evaluating rule {self.rule_id}: {str(e)}")
            logger.error(f"SQL condition: {self.sql_condition}")
            return pd.Series(False, index=df.index)

    def calculate_performance(self, df: pd.DataFrame, triggered: pd.Series,
                            target_col: str = 'fraud_flag') -> Dict:
        """
        Calculate rule performance metrics

        Parameters:
        -----------
        df : pd.DataFrame
            Transaction data with fraud labels
        triggered : pd.Series
            Boolean series of rule triggers
        target_col : str
            Name of the fraud flag column

        Returns:
        --------
        Dict
            Performance metrics
        """
        if target_col not in df.columns:
            return {}

        # Calculate confusion matrix
        actual_fraud = df[target_col].astype(bool)

        tp = (triggered & actual_fraud).sum()  # True Positives
        fp = (triggered & ~actual_fraud).sum()  # False Positives
        fn = (~triggered & actual_fraud).sum()  # False Negatives
        tn = (~triggered & ~actual_fraud).sum()  # True Negatives

        # Calculate metrics
        total_fraud = actual_fraud.sum()
        total_triggered = triggered.sum()

        self.true_positives = tp
        self.false_positives = fp

        self.detection_rate = (tp / total_fraud * 100) if total_fraud > 0 else 0.0
        self.precision = (tp / total_triggered * 100) if total_triggered > 0 else 0.0
        recall = (tp / total_fraud * 100) if total_fraud > 0 else 0.0
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
        fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
        f1_score = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'total_triggered': int(total_triggered),
            'total_fraud': int(total_fraud),
            'detection_rate': round(self.detection_rate, 2),
            'precision': round(self.precision, 2),
            'recall': round(recall, 2),
            'specificity': round(specificity, 2),
            'false_positive_rate': round(fpr, 2),
            'f1_score': round(f1_score, 4)
        }

    def __repr__(self):
        return f"SQLRule(id={self.rule_id}, name={self.name}, priority={self.priority})"


class SQLRuleLoader:
    """Load and manage SQL-based fraud detection rules"""

    def __init__(self, rules_directory: str = "/root/research-dir/dev/jazzcash-fraud-detection/rules"):
        """
        Initialize SQL Rule Loader

        Parameters:
        -----------
        rules_directory : str
            Directory containing SQL rule files
        """
        self.rules_directory = rules_directory
        self.rules: Dict[str, SQLRule] = {}

    def load_rules_from_directory(self) -> List[SQLRule]:
        """
        Load all SQL rules from the specified directory

        Returns:
        --------
        List[SQLRule]
            List of loaded SQL rules
        """
        if not os.path.exists(self.rules_directory):
            logger.warning(f"Rules directory not found: {self.rules_directory}")
            return []

        loaded_rules = []

        # Find all SQL files
        sql_files = []
        for root, dirs, files in os.walk(self.rules_directory):
            for file in files:
                if file.endswith('.sql') or file.endswith('.txt'):
                    sql_files.append(os.path.join(root, file))

        logger.info(f"Found {len(sql_files)} SQL rule files")

        # Load each file
        for sql_file in sql_files:
            try:
                rules = self.load_rule_from_file(sql_file)
                loaded_rules.extend(rules)
                logger.info(f"Loaded {len(rules)} rule(s) from {os.path.basename(sql_file)}")
            except Exception as e:
                logger.error(f"Error loading {sql_file}: {str(e)}")

        # Store rules
        for rule in loaded_rules:
            self.rules[rule.rule_id] = rule

        logger.info(f"Total rules loaded: {len(loaded_rules)}")
        return loaded_rules

    def load_rule_from_file(self, file_path: str) -> List[SQLRule]:
        """
        Load rule(s) from a single SQL file

        Expected format:
        -- RULE_ID: RULE_001
        -- NAME: High Amount Transaction
        -- PRIORITY: High
        -- DESCRIPTION: Flag transactions above threshold
        WHERE transaction_amount > 5000 AND is_suspicious = 1

        Or multiple rules separated by semicolons

        Parameters:
        -----------
        file_path : str
            Path to SQL file

        Returns:
        --------
        List[SQLRule]
            List of rules from the file
        """
        with open(file_path, 'r') as f:
            content = f.read()

        rules = []

        # Split by semicolons for multiple rules
        rule_blocks = content.split(';')

        for idx, block in enumerate(rule_blocks):
            block = block.strip()
            if not block or len(block) < 10:
                continue

            # Extract metadata from comments
            rule_id = None
            name = None
            priority = 'Medium'
            description = ''

            lines = block.split('\n')
            sql_lines = []

            for line in lines:
                line = line.strip()

                # Parse metadata from comments
                if line.startswith('--'):
                    comment = line[2:].strip()

                    if comment.startswith('RULE_ID:'):
                        rule_id = comment.split(':', 1)[1].strip()
                    elif comment.startswith('NAME:'):
                        name = comment.split(':', 1)[1].strip()
                    elif comment.startswith('PRIORITY:'):
                        priority = comment.split(':', 1)[1].strip()
                    elif comment.startswith('DESCRIPTION:'):
                        description = comment.split(':', 1)[1].strip()
                else:
                    # SQL line
                    sql_lines.append(line)

            # Combine SQL lines
            sql_condition = ' '.join(sql_lines).strip()

            # Remove WHERE keyword if present
            if sql_condition.upper().startswith('WHERE '):
                sql_condition = sql_condition[6:].strip()

            # Generate rule_id if not provided
            if not rule_id:
                rule_id = f"RULE_{os.path.basename(file_path)}_{idx+1:03d}"

            # Generate name if not provided
            if not name:
                name = f"Rule from {os.path.basename(file_path)}"

            # Create rule
            rule = SQLRule(
                rule_id=rule_id,
                name=name,
                sql_condition=sql_condition,
                priority=priority,
                description=description,
                file_path=file_path
            )

            rules.append(rule)

        return rules

    def execute_rules(self, df: pd.DataFrame, target_col: str = 'fraud_flag') -> Dict[str, Dict]:
        """
        Execute all loaded rules on DataFrame and calculate performance

        Parameters:
        -----------
        df : pd.DataFrame
            Transaction data
        target_col : str
            Name of fraud flag column

        Returns:
        --------
        Dict[str, Dict]
            Dictionary mapping rule_id to performance metrics
        """
        results = {}

        for rule_id, rule in self.rules.items():
            # Evaluate rule
            triggered = rule.evaluate(df)

            # Calculate performance if target column exists
            if target_col in df.columns:
                metrics = rule.calculate_performance(df, triggered, target_col)
            else:
                metrics = {
                    'total_triggered': int(triggered.sum()),
                    'detection_rate': 0.0,
                    'precision': 0.0
                }

            # Add rule info to results
            results[rule_id] = {
                'rule': rule,
                'triggered_count': int(triggered.sum()),
                'metrics': metrics,
                'triggered_indices': df[triggered].index.tolist()
            }

        return results

    def get_rule_summary(self) -> pd.DataFrame:
        """
        Get summary of all loaded rules

        Returns:
        --------
        pd.DataFrame
            Summary DataFrame
        """
        summary_data = []

        for rule_id, rule in self.rules.items():
            summary_data.append({
                'Rule ID': rule.rule_id,
                'Name': rule.name,
                'Priority': rule.priority,
                'Description': rule.description,
                'SQL Condition': rule.sql_condition[:100] + '...' if len(rule.sql_condition) > 100 else rule.sql_condition,
                'File': os.path.basename(rule.file_path) if rule.file_path else 'N/A'
            })

        return pd.DataFrame(summary_data)


def create_sample_rules(rules_directory: str):
    """
    Create sample SQL rule files for demonstration

    Parameters:
    -----------
    rules_directory : str
        Directory to create sample rules in
    """
    os.makedirs(rules_directory, exist_ok=True)

    sample_rules = [
        {
            'filename': 'high_amount_rules.sql',
            'content': """-- RULE_ID: RULE_001
-- NAME: High Amount Transaction
-- PRIORITY: High
-- DESCRIPTION: Flag transactions with very high amounts
WHERE trx_amt > 100000;

-- RULE_ID: RULE_002
-- NAME: Suspicious High Amount at Night
-- PRIORITY: Critical
-- DESCRIPTION: High amount transaction during night hours
WHERE trx_amt > 75000 AND hour_of_day >= 22 OR hour_of_day <= 5;
"""
        },
        {
            'filename': 'velocity_rules.sql',
            'content': """-- RULE_ID: RULE_003
-- NAME: High Transaction Velocity
-- PRIORITY: Medium
-- DESCRIPTION: Multiple transactions in short time period
WHERE txn_txns_3d > 10;

-- RULE_ID: RULE_004
-- NAME: Rapid Amount Increase
-- PRIORITY: High
-- DESCRIPTION: Large increase in transaction amounts
WHERE txn_amount_deviation_from_avg > 5.0;
"""
        },
        {
            'filename': 'unusual_pattern_rules.sql',
            'content': """-- RULE_ID: RULE_005
-- NAME: Weekend + Night Combo
-- PRIORITY: Medium
-- DESCRIPTION: Transactions during unusual times
WHERE is_weekend == 1 AND hour_of_day >= 23 OR hour_of_day <= 4;

-- RULE_ID: RULE_006
-- NAME: Low Balance High Amount
-- PRIORITY: Critical
-- DESCRIPTION: High transaction on low balance account
WHERE start_balance < 5000 AND trx_amt > 50000;
"""
        }
    ]

    for sample in sample_rules:
        file_path = os.path.join(rules_directory, sample['filename'])
        with open(file_path, 'w') as f:
            f.write(sample['content'])
        logger.info(f"Created sample rule file: {file_path}")
