from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np
from sklearn.metrics import confusion_matrix

class BaseModel(ABC):
    """Base class for all models implementing the Strategy pattern."""
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the given data."""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Evaluate the model and return accuracy, sensitivity, specificity, and confusion matrix."""
        predictions = self.predict(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, predictions)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return accuracy, sensitivity, specificity, cm
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate accuracy, sensitivity, and specificity."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        cm = confusion_matrix(y_true, y_pred)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return accuracy, sensitivity, specificity , cm