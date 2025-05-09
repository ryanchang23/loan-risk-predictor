import numpy as np
from typing import Dict, Type, List
from .base_model import BaseModel
from .d_lstm_model import DLSTMModel
from .mlp_model import MLPModel
from .cnn_lightgbm_model import CNNLightGBMModel
from .dnn_model import DNNModel
from .random_forest_model import RandomForestModel
from .rnn_model import RNNModel
from .xgboost_model import XGBoostModel

class ModelFactory:
    """Factory class for creating model instances."""
    
    _models: Dict[str, Type[BaseModel]] = {
        'd_lstm': DLSTMModel,
        'mlp': MLPModel,
        'cnn_lightgbm': CNNLightGBMModel,
        'dnn': DNNModel,
        'random_forest': RandomForestModel,
        'rnn': RNNModel,
        'xgboost': XGBoostModel
    }
    
    @classmethod
    def create_model(cls, model_name: str) -> BaseModel:
        """Create a model instance by name."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        if model_name not in cls._models:
            raise ValueError(f"Unknown model type: {model_name}")
        
        return cls._models[model_name]()
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model type."""
        cls._models[name] = model_class
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls._models.keys())

    @staticmethod
    def create_model(model_name: str):
        """Create a model instance based on the model name."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        if model_name == "mlp":
            return MLPModel()
        elif model_name == "cnn_lightgbm":
            return CNNLightGBMModel()
        elif model_name == "random_forest":
            return RandomForestModel()
        elif model_name == "rnn":
            return RNNModel()
        elif model_name == "dnn":
            return DNNModel()
        elif model_name == "d_lstm":
            return DLSTMModel()
        elif model_name == "xgboost":
            return XGBoostModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def get_available_models():
        """Get list of available model names."""
        return ["mlp", "cnn_lightgbm", "random_forest", "rnn", "dnn", "d_lstm", "xgboost"] 