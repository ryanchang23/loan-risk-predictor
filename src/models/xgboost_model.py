import numpy as np
import xgboost as xgb
from typing import Tuple, List, Dict, Any
from .base_model import BaseModel
from tqdm import tqdm

class XGBoostModel(BaseModel):
    """XGBoost model for loan risk prediction."""
    
    def __init__(self):
        """Initialize the XGBoost model with default parameters."""
        super().__init__()
        self.model = None
        self.model_name = "xgboost"
        # Load hyperparameters from config
        self.hyperparams = self.config.get(f"models.hyperparameters.{self.model_name}")
        
        # Set default parameters if not in config
        self.params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': 42,
            'tree_method': 'hist',  # Use histogram-based algorithm for faster training
            'nthread': -1  # Use all available threads
        }
        
        # Update with config parameters if they exist
        if self.hyperparams:
            self.params.update(self.hyperparams)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
        """
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set number of boosting rounds
        num_boost_round = self.params.pop('num_boost_round', 100)
        
        # Create progress bar
        pbar = tqdm(total=num_boost_round, desc=f'Training {self.model_name}')
        
        # Create a proper XGBoost callback
        class ProgressCallback(xgb.callback.TrainingCallback):
            def __init__(self, pbar):
                self.pbar = pbar
                self.losses = []
            
            def after_iteration(self, model, epoch, evals_log):
                self.pbar.update(1)
                if evals_log and 'train' in evals_log:
                    loss = evals_log['train']['logloss'][-1]
                    self.losses.append(loss)
                    self.pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # Create callback instance
        progress_callback = ProgressCallback(pbar)
        
        # Train the model with specified parameters
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            callbacks=[progress_callback],
            verbose_eval=False
        )
        pbar.close()
        
        # Record losses
        if hasattr(progress_callback, 'losses'):
            for loss in progress_callback.losses:
                self.record_loss(loss)
        
        # Plot and save the training loss curve
        self.plot_train_loss()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X)
        
        # Get probability predictions
        proba = self.model.predict(dtest)
        
        return proba
    
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Get feature importance scores
        importance = self.model.get_score(importance_type='gain')
        
        # Normalize scores
        total = sum(importance.values())
        return {k: v/total for k, v in importance.items()}
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.save_model(path)
 
    def load(self, path: str) -> None:
        """Load the model from disk."""
        self.model = xgb.Booster()
        self.model.load_model(path) 