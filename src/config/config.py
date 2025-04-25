from typing import Dict, Any
import yaml
import os

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.config: Dict[str, Any] = {
            'data': {
                'train_path': 'Dataset/Training Data.csv',
                'test_path': 'Dataset/Test Data.csv',
                'processed_dir': 'processed'
            },
            'models': {
                'default_model': 'd_lstm',
                'train_percentage': 80,
                'random_state': 42
            },
            'features': {
                'categorical_columns': [
                    'Married/Single',
                    'House_Ownership',
                    'Car_Ownership',
                    'Profession',
                    'CITY',
                    'STATE'
                ],
                'target_column': 'Risk_Flag'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
    
    def load_from_yaml(self, yaml_path: str) -> None:
        """Load configuration from a YAML file."""
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self._update_config(self.config, yaml_config)
    
    def _update_config(self, base: Dict, update: Dict) -> None:
        """Recursively update configuration dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value 