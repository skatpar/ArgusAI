"""
Global Settings Manager
Handles connection strings, secrets, and global configurations
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import streamlit as st


class SettingsManager:
    """Manage global application settings and secrets"""

    def __init__(self, settings_file: str = None):
        if settings_file is None:
            settings_file = os.path.join(os.path.expanduser("~"), ".argusai", "settings.json")

        self.settings_file = settings_file
        self.settings_dir = os.path.dirname(settings_file)

        # Create directory if it doesn't exist
        Path(self.settings_dir).mkdir(parents=True, exist_ok=True)

        # Load existing settings
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading settings: {str(e)}")
                return self.get_default_settings()
        return self.get_default_settings()

    def save_settings(self, settings: Dict[str, Any] = None):
        """Save settings to file"""
        if settings is not None:
            self.settings = settings

        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")
            return False

    def get_default_settings(self) -> Dict[str, Any]:
        """Return default settings"""
        return {
            "clickhouse": {
                "host": "localhost",
                "port": 9000,
                "http_port": 8123,
                "database": "public",
                "user": "default",
                "password": ""
            },
            "paths": {
                "models_dir": "/root/research-dir/dev/jazzcash-fraud-detection/models",
                "rules_dir": "/root/research-dir/dev/jazzcash-fraud-detection/rules",
                "analysis_dir": "/root/research-dir/dev/jazzcash-fraud-detection/analysis",
                "logs_dir": "/root/research-dir/dev/jazzcash-fraud-detection/logs"
            },
            "api": {
                "endpoint_url": "http://localhost:5000/predict",
                "auth_type": "None",
                "api_key": "",
                "timeout": 30
            },
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "fraud_detection",
                "artifact_location": ""
            },
            "spark": {
                "executor_memory": "150g",
                "driver_memory": "8g",
                "executor_cores": 32,
                "executor_instances": 2,
                "shuffle_partitions": 200
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value by key (supports nested keys like 'clickhouse.host')"""
        keys = key.split('.')
        value = self.settings

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set setting value by key (supports nested keys)"""
        keys = key.split('.')
        current = self.settings

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        self.save_settings()

    def get_all(self) -> Dict[str, Any]:
        """Get all settings"""
        return self.settings

    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.settings = self.get_default_settings()
        self.save_settings()
