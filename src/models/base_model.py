from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import os
from ..config.config import ConfigManager

class BaseModel(ABC):
    """Base class for all models implementing the Strategy pattern."""
    
    def __init__(self):
        """Initialize the base model with common attributes."""
        self.config = ConfigManager()
        self.train_losses: List[float] = []
        self.model_name: str = self.__class__.__name__.replace('Model', '').lower()
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the given data."""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Evaluate the model and return accuracy, sensitivity, specificity, f1_score, and confusion matrix."""
        predictions = self.predict(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, predictions)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y, predictions)

        # TN FP
        # FN TP
        return accuracy, sensitivity, specificity, f1, cm
    
    def plot_train_loss(self, save_path: Optional[str] = None) -> None:
        """Plot and optionally save the training loss curve.
        
        Args:
            save_path: Optional path to save the plot. If None, uses the default graph directory from config.
        """
        if not self.train_losses:
            print(f"No training losses recorded for {self.model_name}")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='#2ecc71', linewidth=2)
        plt.title(f'Training Loss Curve - {self.model_name.upper()}', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Add final loss value annotation
        final_loss = self.train_losses[-1]
        plt.annotate(f'Final Loss: {final_loss:.4f}',
                    xy=(len(self.train_losses)-1, final_loss),
                    xytext=(len(self.train_losses)-1, final_loss*1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=10)
        
        # Save the plot if path is provided or use default
        if save_path is None:
            graph_dir = self.config.get("data.train_loss_dir")
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)
            save_path = os.path.join(graph_dir, f"{self.model_name}_train_loss.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def record_loss(self, loss: float) -> None:
        """Record a training loss value.
        
        Args:
            loss: The loss value to record
        """
        self.train_losses.append(loss)
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate accuracy, sensitivity, specificity, and f1_score."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        cm = confusion_matrix(y_true, y_pred)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred)
        
        return accuracy, sensitivity, specificity, f1, cm