"""
Model Artifacts Loader
Load feature importances, SHAP values, and model metadata from saved models
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st


class ModelArtifactsLoader:
    """Load and manage model artifacts for explainability"""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Try to get from settings
            from utils.settings import SettingsManager
            settings_mgr = SettingsManager()
            models_dir = settings_mgr.get('paths.models_dir', '/root/research-dir/dev/jazzcash-fraud-detection/models')

        self.models_dir = models_dir

    def list_available_models(self) -> List[Dict[str, str]]:
        """List all available models with their metadata"""
        models = []

        if not os.path.exists(self.models_dir):
            return models

        # Look for model directories
        for item in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, item)

            if os.path.isdir(model_path):
                # Check if it's a model directory (contains metadata or stages)
                metadata_file = os.path.join(model_path, 'metadata', 'part-00000')
                stages_dir = os.path.join(model_path, 'stages')

                if os.path.exists(metadata_file) or os.path.exists(stages_dir):
                    models.append({
                        'model_id': item,
                        'model_path': model_path,
                        'model_name': item.replace('_', ' ').title(),
                        'type': self._infer_model_type(item)
                    })

        return models

    def _infer_model_type(self, model_id: str) -> str:
        """Infer model type from model ID"""
        model_id_lower = model_id.lower()

        if 'random_forest' in model_id_lower or 'rf' in model_id_lower:
            return 'Random Forest'
        elif 'gbt' in model_id_lower or 'gradient' in model_id_lower:
            return 'Gradient Boosting'
        elif 'logistic' in model_id_lower or 'lr' in model_id_lower:
            return 'Logistic Regression'
        elif 'decision_tree' in model_id_lower or 'dt' in model_id_lower:
            return 'Decision Tree'
        else:
            return 'Unknown'

    def load_feature_importance(self, model_id: str = None, analysis_dir: str = None) -> Optional[pd.DataFrame]:
        """
        Load feature importance from saved CSV files

        Args:
            model_id: Model identifier (e.g., 'random_forest_pipeline_model')
            analysis_dir: Directory containing analysis CSV files

        Returns:
            DataFrame with feature names and importance scores
        """
        if analysis_dir is None:
            from utils.settings import SettingsManager
            settings_mgr = SettingsManager()
            analysis_dir = settings_mgr.get('paths.analysis_dir', '/root/research-dir/dev/jazzcash-fraud-detection/analysis')

        if not os.path.exists(analysis_dir):
            return None

        # Search for feature importance CSV files
        pattern = f"{model_id}_feature_importance" if model_id else "*_feature_importance"

        import glob
        csv_files = glob.glob(os.path.join(analysis_dir, f"{pattern}*.csv"))

        if not csv_files:
            return None

        # Load the most recent file
        csv_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = csv_files[0]

        try:
            df = pd.read_csv(latest_file)
            return df
        except Exception as e:
            st.error(f"Error loading feature importance: {str(e)}")
            return None

    def load_shap_values(self, model_id: str = None, analysis_dir: str = None) -> Optional[Dict]:
        """
        Load SHAP values from saved files

        Args:
            model_id: Model identifier
            analysis_dir: Directory containing SHAP value files

        Returns:
            Dictionary with SHAP values and feature names
        """
        if analysis_dir is None:
            from utils.settings import SettingsManager
            settings_mgr = SettingsManager()
            analysis_dir = settings_mgr.get('paths.analysis_dir', '/root/research-dir/dev/jazzcash-fraud-detection/analysis')

        if not os.path.exists(analysis_dir):
            return None

        # Look for SHAP value files (NPY or CSV format)
        import glob
        pattern = f"{model_id}_shap" if model_id else "*_shap"
        shap_files = glob.glob(os.path.join(analysis_dir, f"{pattern}*.npy")) + \
                     glob.glob(os.path.join(analysis_dir, f"{pattern}*.csv"))

        if not shap_files:
            return None

        # Load the most recent file
        shap_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = shap_files[0]

        try:
            if latest_file.endswith('.npy'):
                shap_values = np.load(latest_file)
                # Try to load feature names
                feature_file = latest_file.replace('.npy', '_features.json')
                if os.path.exists(feature_file):
                    with open(feature_file, 'r') as f:
                        features = json.load(f)
                else:
                    features = [f"feature_{i}" for i in range(shap_values.shape[1])]

                return {
                    'shap_values': shap_values,
                    'feature_names': features
                }
            elif latest_file.endswith('.csv'):
                df = pd.read_csv(latest_file)
                return {
                    'shap_values': df.values,
                    'feature_names': df.columns.tolist()
                }
        except Exception as e:
            st.error(f"Error loading SHAP values: {str(e)}")
            return None

    def get_model_metadata(self, model_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific model

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model metadata
        """
        model_path = os.path.join(self.models_dir, model_id)

        if not os.path.exists(model_path):
            return None

        metadata = {
            'model_id': model_id,
            'model_path': model_path,
            'created_date': None,
            'model_type': self._infer_model_type(model_id)
        }

        # Try to get creation date
        try:
            metadata['created_date'] = pd.Timestamp(os.path.getctime(model_path), unit='s').strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass

        return metadata

    def get_latest_model_of_type(self, model_type: str) -> Optional[str]:
        """
        Get the latest model ID of a specific type

        Args:
            model_type: Type of model (e.g., 'random_forest', 'gbt')

        Returns:
            Model ID string
        """
        models = self.list_available_models()

        # Filter by type
        filtered = [m for m in models if model_type.lower() in m['model_id'].lower()]

        if not filtered:
            return None

        # Sort by modification time
        filtered.sort(key=lambda x: os.path.getmtime(x['model_path']), reverse=True)

        return filtered[0]['model_id']


def calculate_correlation_matrix(df: pd.DataFrame, target_col: str = 'fraud_flag',
                                 method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric features

    Args:
        df: Input DataFrame
        target_col: Target column name
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix DataFrame
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation
    corr_matrix = numeric_df.corr(method=method)

    return corr_matrix


def get_top_correlations(corr_matrix: pd.DataFrame, target_col: str,
                        n: int = 20) -> pd.DataFrame:
    """
    Get top N features correlated with target

    Args:
        corr_matrix: Correlation matrix
        target_col: Target column name
        n: Number of top correlations to return

    Returns:
        DataFrame with top correlations
    """
    if target_col not in corr_matrix.columns:
        return pd.DataFrame()

    target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)

    # Remove self-correlation
    target_corr = target_corr[target_corr.index != target_col]

    return target_corr.head(n).to_frame('correlation')
