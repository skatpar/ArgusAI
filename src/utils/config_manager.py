"""
Configuration Manager for ArgusAI Training
Handles loading, saving, and managing training configurations
"""

import yaml
import json
import os
from typing import Dict, Any, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage training configurations"""

    def __init__(self, config_dir: str = "configs", config_file: str = "training_config.yaml"):
        """
        Initialize configuration manager

        Args:
            config_dir: Directory containing config files
            config_file: Name of the config file
        """
        self.config_dir = config_dir
        self.config_file = config_file
        self.config_path = os.path.join(config_dir, config_file)
        self.config = None

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Returns:
            Dict containing configuration
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}")
                return self._get_default_config()

            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            logger.info(f"Loaded configuration from {self.config_path}")
            return self.config

        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return self._get_default_config()

    def save_config(self, config: Dict[str, Any], backup: bool = True) -> bool:
        """
        Save configuration to YAML file

        Args:
            config: Configuration dictionary to save
            backup: Whether to create backup of existing config

        Returns:
            bool: Success status
        """
        try:
            # Create config directory if it doesn't exist
            os.makedirs(self.config_dir, exist_ok=True)

            # Backup existing config
            if backup and os.path.exists(self.config_path):
                backup_path = f"{self.config_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.config_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Save new config
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            self.config = config
            logger.info(f"Saved configuration to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update specific configuration values

        Args:
            updates: Dictionary of updates (supports nested keys with dot notation)

        Returns:
            bool: Success status
        """
        try:
            if self.config is None:
                self.load_config()

            # Update nested values
            for key, value in updates.items():
                self._set_nested_value(self.config, key, value)

            # Save updated config
            return self.save_config(self.config)

        except Exception as e:
            logger.error(f"Error updating config: {str(e)}")
            return False

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'model.type')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self.config is None:
            self.load_config()

        return self._get_nested_value(self.config, key, default)

    def get_data_source_config(self) -> Dict[str, Any]:
        """Get data source configuration"""
        return self.get_value('data_source', {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.get_value('model', {})

    def get_artifacts_config(self) -> Dict[str, Any]:
        """Get artifacts configuration"""
        return self.get_value('artifacts', {})

    def get_scripts_config(self) -> Dict[str, Any]:
        """Get scripts configuration"""
        return self.get_value('scripts', {})

    def get_training_script_args(self, script_name: str) -> List[str]:
        """
        Get custom arguments for a specific training script

        Args:
            script_name: Name of the training script

        Returns:
            List of argument names
        """
        custom_scripts = self.get_value('scripts.custom_scripts', {})

        if script_name in custom_scripts:
            return custom_scripts[script_name].get('args', [])

        return self.get_value('scripts.default_args', {})

    def list_training_scripts(self) -> List[str]:
        """
        List all available training scripts

        Returns:
            List of script filenames
        """
        scripts_dir = self.get_value('scripts.directory', 'training_scripts')

        if not os.path.exists(scripts_dir):
            logger.warning(f"Scripts directory not found: {scripts_dir}")
            return []

        scripts = [f for f in os.listdir(scripts_dir)
                   if f.endswith('.py') and not f.startswith('__')]

        return sorted(scripts)

    def get_model_save_path(self, model_name: str, create_dir: bool = True) -> str:
        """
        Get the save path for a model

        Args:
            model_name: Name of the model
            create_dir: Whether to create directory if it doesn't exist

        Returns:
            Full path for saving the model
        """
        save_dir = self.get_value('artifacts.models.save_dir', 'artifacts/models')

        if create_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Generate version
        version_format = self.get_value('artifacts.models.version_format', 'v{timestamp}')

        if '{timestamp}' in version_format:
            version = version_format.replace('{timestamp}',
                                            datetime.now().strftime('%Y%m%d_%H%M%S'))
        elif '{date}' in version_format:
            version = version_format.replace('{date}',
                                            datetime.now().strftime('%Y%m%d'))
        else:
            version = version_format

        # Build filename
        save_format = self.get_value('artifacts.models.save_format', 'joblib')
        filename = f"{model_name}_{version}.{save_format}"

        return os.path.join(save_dir, filename)

    def export_config_to_json(self, output_path: str = None) -> str:
        """
        Export configuration to JSON format

        Args:
            output_path: Path to save JSON file

        Returns:
            JSON string of configuration
        """
        if self.config is None:
            self.load_config()

        json_str = json.dumps(self.config, indent=2)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)

        return json_str

    def _get_nested_value(self, d: Dict, key: str, default: Any = None) -> Any:
        """Get nested dictionary value using dot notation"""
        keys = key.split('.')
        value = d

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def _set_nested_value(self, d: Dict, key: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key.split('.')
        target = d

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'data_source': {
                'type': 'clickhouse',
                'clickhouse': {
                    'database': 'public',
                    'table': 'stixor_fraud_features_distributed',
                    'limit': 100000
                }
            },
            'model': {
                'type': 'random_forest',
                'training': {
                    'test_size': 0.2,
                    'random_state': 42
                }
            },
            'scripts': {
                'directory': 'training_scripts'
            },
            'artifacts': {
                'models': {
                    'save_dir': 'artifacts/models',
                    'save_format': 'joblib'
                }
            }
        }


# Global instance
config_manager = ConfigManager()
