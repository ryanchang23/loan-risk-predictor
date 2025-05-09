import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def __init__(self):
        self.model = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
        """Evaluate the model."""
        predictions = self.predict(X_test)
        predictions = (predictions > 0.5).astype(int)
        return self._calculate_metrics(y_test, predictions)
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        import joblib
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        import joblib
        self.model = joblib.load(path) 