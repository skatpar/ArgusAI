"""
MLflow Experiment Tracking Utility
Track model training experiments, hyperparameters, metrics, and artifacts
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.spark
import os
from datetime import datetime
import json
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient


class MLflowTracker:
    """MLflow experiment tracker for fraud detection models"""

    def __init__(self, tracking_uri=None, experiment_name="fraud_detection_pipeline"):
        """
        Initialize MLflow tracker

        Args:
            tracking_uri: MLflow tracking server URI (default: ./mlruns)
            experiment_name: Name of the experiment
        """
        # Set tracking URI
        if tracking_uri is None:
            tracking_uri = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlruns")

        mlflow.set_tracking_uri(f"file://{tracking_uri}")

        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "mlflow")
            )
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id if experiment else None

        self.experiment_name = experiment_name
        self.experiment_id = experiment_id

    def start_run(self, run_name=None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )

        return mlflow.active_run()

    def log_params(self, params):
        """Log parameters"""
        try:
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(params)

            for key, value in flat_params.items():
                # MLflow params must be strings
                mlflow.log_param(key, str(value))
        except Exception as e:
            print(f"Error logging params: {e}")

    def log_metrics(self, metrics, step=None):
        """Log metrics"""
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"Error logging metrics: {e}")

    def log_model(self, model, model_type, artifact_path="model"):
        """
        Log trained model

        Args:
            model: Trained model object
            model_type: Type of model (sklearn, xgboost, lightgbm, etc.)
            artifact_path: Path within run artifacts
        """
        try:
            if model_type in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                mlflow.sklearn.log_model(model, artifact_path)
            elif model_type == 'xgboost':
                mlflow.xgboost.log_model(model, artifact_path)
            elif model_type == 'lightgbm':
                mlflow.lightgbm.log_model(model, artifact_path)
            else:
                # Default to sklearn
                mlflow.sklearn.log_model(model, artifact_path)
        except Exception as e:
            print(f"Error logging model: {e}")

    def log_artifact(self, local_path, artifact_path=None):
        """Log an artifact file"""
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"Error logging artifact: {e}")

    def log_dict(self, dictionary, filename):
        """Log a dictionary as JSON artifact"""
        try:
            mlflow.log_dict(dictionary, filename)
        except Exception as e:
            print(f"Error logging dict: {e}")

    def set_tags(self, tags):
        """Set run tags"""
        try:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        except Exception as e:
            print(f"Error setting tags: {e}")

    def end_run(self):
        """End the current run"""
        mlflow.end_run()

    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def get_experiment_runs(experiment_name="fraud_detection_pipeline"):
        """Get all runs for an experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                return runs
            return None
        except Exception as e:
            print(f"Error getting experiment runs: {e}")
            return None

    @staticmethod
    def load_model(run_id, model_path="model"):
        """Load a model from a specific run"""
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            model = mlflow.sklearn.load_model(model_uri)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    @staticmethod
    def get_runs_by_experiment(tracking_uri, experiment_name):
        """
        Get all runs from a specific experiment at given tracking URI

        Args:
            tracking_uri: MLflow tracking server URI (e.g., 'http://localhost:5001')
            experiment_name: Name of the experiment

        Returns:
            DataFrame with run information or None
        """
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            # Get experiment by name
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                print(f"Experiment '{experiment_name}' not found")
                return None

            # Search for runs in this experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )

            return runs
        except Exception as e:
            print(f"Error getting runs: {e}")
            return None

    @staticmethod
    def download_artifact(tracking_uri, run_id, artifact_path, dest_path=None):
        """
        Download an artifact from a specific run

        Args:
            tracking_uri: MLflow tracking server URI
            run_id: Run ID to download from
            artifact_path: Path to the artifact within the run
            dest_path: Destination path (if None, returns the downloaded path)

        Returns:
            Path to downloaded artifact or None
        """
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            # Download artifact
            downloaded_path = client.download_artifacts(run_id, artifact_path, dst_path=dest_path)
            return downloaded_path
        except Exception as e:
            print(f"Error downloading artifact: {e}")
            return None

    @staticmethod
    def load_feature_importance_from_run(tracking_uri, run_id):
        """
        Load feature importance from MLflow run artifacts

        Args:
            tracking_uri: MLflow tracking server URI
            run_id: Run ID to load from

        Returns:
            DataFrame with feature importance or None
        """
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            # List artifacts in the run
            artifacts = client.list_artifacts(run_id)

            # Look for feature importance CSV in different possible locations
            possible_paths = [
                'analysis',
                'plots',
                ''  # root level
            ]

            feature_importance_file = None
            for path_prefix in possible_paths:
                if path_prefix:
                    artifacts_in_path = client.list_artifacts(run_id, path=path_prefix)
                else:
                    artifacts_in_path = artifacts

                for artifact in artifacts_in_path:
                    if 'feature_importance' in artifact.path and artifact.path.endswith('.csv'):
                        feature_importance_file = artifact.path
                        break

                if feature_importance_file:
                    break

            if not feature_importance_file:
                print(f"Feature importance file not found in run {run_id}")
                return None

            # Download and read the CSV
            local_path = client.download_artifacts(run_id, feature_importance_file)
            df = pd.read_csv(local_path)

            return df

        except Exception as e:
            print(f"Error loading feature importance: {e}")
            return None

    @staticmethod
    def load_shap_values_from_run(tracking_uri, run_id):
        """
        Load SHAP values from MLflow run artifacts

        Args:
            tracking_uri: MLflow tracking server URI
            run_id: Run ID to load from

        Returns:
            numpy array with SHAP values or None
        """
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            # List artifacts in the run
            artifacts = client.list_artifacts(run_id)

            # Look for SHAP files (.npy or .csv)
            possible_paths = ['analysis', 'plots', '']

            shap_file = None
            for path_prefix in possible_paths:
                if path_prefix:
                    artifacts_in_path = client.list_artifacts(run_id, path=path_prefix)
                else:
                    artifacts_in_path = artifacts

                for artifact in artifacts_in_path:
                    if 'shap' in artifact.path.lower() and (artifact.path.endswith('.npy') or artifact.path.endswith('.csv')):
                        shap_file = artifact.path
                        break

                if shap_file:
                    break

            if not shap_file:
                print(f"SHAP values file not found in run {run_id}")
                return None

            # Download the file
            local_path = client.download_artifacts(run_id, shap_file)

            # Load based on file type
            if local_path.endswith('.npy'):
                shap_values = np.load(local_path)
            elif local_path.endswith('.csv'):
                shap_values = pd.read_csv(local_path).values
            else:
                print(f"Unknown SHAP file format: {local_path}")
                return None

            return shap_values

        except Exception as e:
            print(f"Error loading SHAP values: {e}")
            return None

    @staticmethod
    def get_run_metadata(tracking_uri, run_id):
        """
        Get metadata for a specific run

        Args:
            tracking_uri: MLflow tracking server URI
            run_id: Run ID

        Returns:
            Dictionary with run metadata
        """
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            run = client.get_run(run_id)

            metadata = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000.0),
                'end_time': datetime.fromtimestamp(run.info.end_time / 1000.0) if run.info.end_time else None,
                'params': dict(run.data.params),
                'metrics': dict(run.data.metrics),
                'tags': dict(run.data.tags),
                'artifact_uri': run.info.artifact_uri
            }

            return metadata

        except Exception as e:
            print(f"Error getting run metadata: {e}")
            return None

    @staticmethod
    def load_spark_model(tracking_uri, run_id, model_path="spark_model"):
        """
        Load a Spark ML model from MLflow run

        Args:
            tracking_uri: MLflow tracking server URI
            run_id: Run ID
            model_path: Path to model within run artifacts

        Returns:
            Spark PipelineModel or None
        """
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            # Download the spark model artifact
            local_model_path = client.download_artifacts(run_id, model_path)

            # Load using PySpark
            from pyspark.ml import PipelineModel
            model = PipelineModel.load(local_model_path)

            return model

        except Exception as e:
            print(f"Error loading Spark model: {e}")
            return None

    @staticmethod
    def load_model_from_run(tracking_uri, run_id, model_path="model"):
        """
        Load a model from a specific MLflow run

        Args:
            tracking_uri: MLflow tracking server URI
            run_id: Run ID to load model from
            model_path: Path to model within run artifacts

        Returns:
            Loaded model or None
        """
        try:
            mlflow.set_tracking_uri(tracking_uri)
            model_uri = f"runs:/{run_id}/{model_path}"

            # Try to load as sklearn first, then pyfunc as fallback
            try:
                model = mlflow.sklearn.load_model(model_uri)
            except:
                model = mlflow.pyfunc.load_model(model_uri)

            return model
        except Exception as e:
            print(f"Error loading model from run: {e}")
            return None

    @staticmethod
    def extract_tree_rules(model, feature_names=None, max_depth=3):
        """
        Extract decision rules from tree-based models

        Args:
            model: Trained tree-based model (DecisionTree, RandomForest, GradientBoosting)
            feature_names: List of feature names
            max_depth: Maximum depth of rules to extract

        Returns:
            List of rule dictionaries or None
        """
        try:
            from sklearn.tree import _tree

            rules = []

            # Handle different model types
            if hasattr(model, 'estimators_'):  # RandomForest, GradientBoosting
                # Extract from first few trees
                trees = model.estimators_[:3] if hasattr(model.estimators_[0], '__len__') else [model.estimators_[0]]
                tree_models = [t[0] if hasattr(t, '__len__') else t for t in trees]
            elif hasattr(model, 'tree_'):  # DecisionTree
                tree_models = [model]
            else:
                return None

            for idx, tree_model in enumerate(tree_models):
                tree = tree_model.tree_
                feature_name = [
                    feature_names[i] if feature_names and i != _tree.TREE_UNDEFINED else f"feature_{i}"
                    for i in range(len(feature_names) if feature_names else tree.n_features)
                ]

                def recurse(node, depth, path):
                    if depth > max_depth:
                        return

                    if tree.feature[node] != _tree.TREE_UNDEFINED:
                        name = feature_name[tree.feature[node]]
                        threshold = tree.threshold[node]

                        # Left child (<=)
                        left_path = path + [f"{name} <= {threshold:.3f}"]
                        recurse(tree.children_left[node], depth + 1, left_path)

                        # Right child (>)
                        right_path = path + [f"{name} > {threshold:.3f}"]
                        recurse(tree.children_right[node], depth + 1, right_path)
                    else:
                        # Leaf node
                        value = tree.value[node]
                        if len(value[0]) > 1:
                            fraud_prob = value[0][1] / value[0].sum()
                        else:
                            fraud_prob = value[0][0]

                        rules.append({
                            'tree_id': idx,
                            'rule': ' AND '.join(path),
                            'fraud_probability': float(fraud_prob),
                            'samples': int(tree.n_node_samples[node])
                        })

                recurse(0, 0, [])

            # Sort by fraud probability
            rules.sort(key=lambda x: x['fraud_probability'], reverse=True)
            return rules[:20]  # Return top 20 rules

        except Exception as e:
            print(f"Error extracting tree rules: {e}")
            return None

    @staticmethod
    def get_model_params(tracking_uri, run_id):
        """
        Get model parameters/hyperparameters from a run

        Args:
            tracking_uri: MLflow tracking server URI
            run_id: Run ID

        Returns:
            Dictionary of parameters
        """
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            run = client.get_run(run_id)
            params = dict(run.data.params)

            return params
        except Exception as e:
            print(f"Error getting model parameters: {e}")
            return {}

    @staticmethod
    def predict_with_model(model, X, return_proba=True):
        """
        Make predictions with a loaded model

        Args:
            model: Loaded model
            X: Features (DataFrame or array)
            return_proba: Whether to return probabilities

        Returns:
            Predictions (probabilities if return_proba=True, else classes)
        """
        try:
            if return_proba and hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)
                # Return probability of fraud class (usually class 1)
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    return predictions[:, 1]
                return predictions
            else:
                return model.predict(X)
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None

    @staticmethod
    def compute_shap_for_instance(model, X_instance, X_background=None, feature_names=None):
        """
        Compute SHAP values for a specific instance

        Args:
            model: Loaded model
            X_instance: Single instance to explain (1D array or DataFrame row)
            X_background: Background dataset for SHAP (optional, uses X_instance if None)
            feature_names: List of feature names

        Returns:
            Dictionary with SHAP values and explanation
        """
        try:
            import shap

            # Ensure X_instance is 2D
            if len(X_instance.shape) == 1:
                X_instance = X_instance.reshape(1, -1)

            # Create explainer
            if X_background is not None:
                explainer = shap.TreeExplainer(model, X_background)
            else:
                explainer = shap.TreeExplainer(model)

            # Compute SHAP values
            shap_values = explainer.shap_values(X_instance)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification - use positive class
                shap_values = shap_values[1]

            # Flatten if needed
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            # Create feature contributions DataFrame
            if feature_names is not None:
                contributions = pd.DataFrame({
                    'feature': feature_names,
                    'value': X_instance[0] if len(X_instance.shape) > 1 else X_instance,
                    'shap_value': shap_values,
                    'abs_shap': np.abs(shap_values)
                }).sort_values('abs_shap', ascending=False)
            else:
                contributions = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(shap_values))],
                    'value': X_instance[0] if len(X_instance.shape) > 1 else X_instance,
                    'shap_value': shap_values,
                    'abs_shap': np.abs(shap_values)
                }).sort_values('abs_shap', ascending=False)

            return {
                'shap_values': shap_values,
                'contributions': contributions,
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0
            }

        except ImportError:
            print("SHAP library not installed. Install with: pip install shap")
            return None
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return None


def log_training_run(model, model_type, params, metrics, artifacts=None, tags=None):
    """
    Convenience function to log a complete training run

    Args:
        model: Trained model
        model_type: Type of model
        params: Training parameters dict
        metrics: Evaluation metrics dict
        artifacts: List of artifact paths to log
        tags: Dictionary of tags
    """
    tracker = MLflowTracker()

    try:
        tracker.start_run()

        # Log parameters
        tracker.log_params(params)

        # Log metrics
        tracker.log_metrics(metrics)

        # Log model
        tracker.log_model(model, model_type)

        # Log artifacts
        if artifacts:
            for artifact_path in artifacts:
                if os.path.exists(artifact_path):
                    tracker.log_artifact(artifact_path)

        # Set tags
        if tags:
            tracker.set_tags(tags)

        # Add default tags
        tracker.set_tags({
            'model_type': model_type,
            'timestamp': datetime.now().isoformat()
        })

    finally:
        tracker.end_run()
