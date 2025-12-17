"""
MLflow Experiment Tracking Utility
Track model training experiments, hyperparameters, metrics, and artifacts
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import os
from datetime import datetime
import json


class MLflowTracker:
    """MLflow experiment tracker for fraud detection models"""

    def __init__(self, tracking_uri=None, experiment_name="fraud_detection"):
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
    def get_experiment_runs(experiment_name="fraud_detection"):
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
