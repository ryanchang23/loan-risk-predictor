import numpy as np
import xgboost as xgb
from typing import Tuple, List, Dict, Any
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost model for loan risk prediction."""
    
    def __init__(self):
        """Initialize the XGBoost model with default parameters."""
        self.model = None
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': 42
        }
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
        """
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train the model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params['n_estimators'],
            verbose_eval=False
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.  """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X)
        
        # Get probability predictions
        proba = self.model.predict(dtest)
        
        # Convert probabilities to binary predictions
        return (proba > 0.5).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        """Evaluate the model."""
        predictions = self.predict(X_test)
        predictions = (predictions > 0.5).astype(int)
        return self._calculate_metrics(y_test, predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.  """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Get feature importance scores
        importance = self.model.get_score(importance_type='gain')
        
        # Normalize scores
        total = sum(importance.values())
        return {k: v/total for k, v in importance.items()}
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.save_model(path)
 
    def load(self, path: str) -> None:
        """Load the model from disk.  """
        self.model = xgb.Booster()
        self.model.load_model(path) 