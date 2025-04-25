import numpy as np
from typing import Dict, Type, List
from .base_model import BaseModel
from .d_lstm_model import DLSTMModel
from .mlp_model import MLPModel
from .cnn_lightgbm_model import CNNLightGBMModel
from .dnn_model import DNNModel
from .random_forest_model import RandomForestModel
from .rnn_model import RNNModel

class ModelFactory:
    """Factory class for creating model instances."""
    
    _models: Dict[str, Type[BaseModel]] = {
        'd_lstm': DLSTMModel,
        'mlp': MLPModel,
        'cnn_lightgbm': CNNLightGBMModel,
        'dnn': DNNModel,
        'random_forest': RandomForestModel,
        'rnn': RNNModel
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