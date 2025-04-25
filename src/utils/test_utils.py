import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

class TestDataGenerator:
    """Generate test data for debugging and testing."""
    
    @staticmethod
    def generate_synthetic_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic data for testing."""
        # Generate features
        data = pd.DataFrame({
            'Married/Single': np.random.choice(['Married', 'Single'], n_samples),
            'House_Ownership': np.random.choice(['Own', 'Rent', 'None'], n_samples),
            'Car_Ownership': np.random.choice(['Yes', 'No'], n_samples),
            'Profession': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Other'], n_samples),
            'CITY': np.random.choice(['City1', 'City2', 'City3'], n_samples),
            'STATE': np.random.choice(['State1', 'State2', 'State3'], n_samples),
            'Age': np.random.randint(18, 70, n_samples),
            'Experience': np.random.randint(0, 30, n_samples),
            'Income': np.random.randint(20000, 200000, n_samples),
            'Current_Job_Years': np.random.randint(0, 20, n_samples),
            'Current_House_Years': np.random.randint(0, 30, n_samples)
        })
        
        # Generate labels (with some correlation to features)
        labels = np.zeros(n_samples)
        for i in range(n_samples):
            risk_score = 0
            # Higher risk for single people
            if data['Married/Single'].iloc[i] == 'Single':
                risk_score += 0.2
            # Higher risk for renters
            if data['House_Ownership'].iloc[i] == 'Rent':
                risk_score += 0.15
            # Higher risk for no car
            if data['Car_Ownership'].iloc[i] == 'No':
                risk_score += 0.1
            # Higher risk for lower income
            risk_score += (200000 - data['Income'].iloc[i]) / 200000 * 0.3
            # Higher risk for less experience
            risk_score += (30 - data['Experience'].iloc[i]) / 30 * 0.25
            
            labels[i] = 1 if risk_score > 0.5 else 0
        
        return data, labels

class DebugLogger:
    """Logger for debug information."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.gui_callback = None
    
    def set_gui_callback(self, callback):
        """Set the callback function for GUI updates."""
        self.gui_callback = callback
    
    def _log_to_gui(self, message: str):
        """Log message to GUI if callback is set."""
        if self.gui_callback:
            self.gui_callback(message)
    
    def log_data_info(self, data: pd.DataFrame, title: str = ""):
        """Log information about a DataFrame."""
        message = f"\n\n{title}:\n"
        message += f"Shape: {data.shape}\n"
        message += f"Columns: {data.columns.tolist()}\n"
        message += f"Sample data:\n{data.head()}\n"
        self.logger.debug(message)
        self._log_to_gui(message)
    
    def log_array_info(self, array: np.ndarray, title: str = ""):
        """Log information about a numpy array."""
        message = f"\n\n{title}:\n"
        message += f"Shape: {array.shape}\n"
        message += f"Type: {array.dtype}\n"
        message += f"Sample data:\n{array[:5]}\n"
        self.logger.debug(message)
        self._log_to_gui(message)
    
    def log_metrics(self, metrics: Dict[str, float], title: str = ""):
        """Log model metrics."""
        message = f"\n\n{title}:\n"
        for metric, value in metrics.items():
            message += f"{metric}: {value:.4f}\n"
        self.logger.debug(message)
        self._log_to_gui(message)
    
    def log_message(self, message: str):
        """Log a general message."""
        self.logger.debug(message)
        self._log_to_gui(message)

def test_feature_engineering():
    """Test feature engineering functionality."""
    # Generate test data
    data, labels = TestDataGenerator.generate_synthetic_data(n_samples=100)
    
    # Create debug logger
    logger = DebugLogger("feature_engineering_test")
    
    # Log original data
    logger.log_data_info(data, "Original Data")
    logger.log_array_info(labels, "Labels")
    
    return data, labels

def test_model_training():
    """Test model training functionality."""
    # Generate test data
    data, labels = TestDataGenerator.generate_synthetic_data(n_samples=100)
    
    # Create debug logger
    logger = DebugLogger("model_training_test")
    
    # Log test data
    logger.log_data_info(data, "Test Data")
    logger.log_array_info(labels, "Test Labels")
    
    return data, labels 