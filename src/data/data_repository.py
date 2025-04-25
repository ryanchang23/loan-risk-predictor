import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, List, Generator
import os
from ..config.config import ConfigManager

class DataRepository:
    def __init__(self):
        self.config = ConfigManager()
        self.scaler = StandardScaler()
        self._initialize_directories()
    
    def _initialize_directories(self):
        """Create necessary directories if they don't exist."""
        processed_dir = self.config.get('data.processed_dir')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
    
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and preprocess the training data."""
        data_path = self.config.get('data.train_path')
        target_column = self.config.get('features.target_column')
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Extract labels
        labels = data[target_column].values
        
        # Remove target column
        data = data.drop(target_column, axis=1)
        
        return data, labels
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess the data including categorical encoding and normalization."""
        # Handle categorical columns
        categorical_columns = self.config.get('features.categorical_columns')
        for col in categorical_columns:
            if col in data.columns:
                data[col] = pd.Categorical(data[col]).codes
        
        # Normalize data
        normalized_data = self.scaler.fit_transform(data)
        
        # Save processed data
        processed_path = os.path.join(
            self.config.get('data.processed_dir'),
            'normalized_data.csv'
        )
        np.savetxt(processed_path, normalized_data, delimiter=',', fmt='%s')
        
        return normalized_data
    
    def subsample_data(self, X: np.ndarray, y: np.ndarray, subsample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform stratified subsampling of the data."""
        if subsample_rate >= 1.0:
            return X, y
        
        # Calculate number of samples to keep
        n_samples = int(len(X) * subsample_rate)
        
        # Perform stratified sampling
        X_subsampled, _, y_subsampled, _ = train_test_split(
            X, y,
            train_size=n_samples,
            stratify=y,
            random_state=42
        )
        
        return X_subsampled, y_subsampled
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        train_percentage = self.config.get('models.train_percentage')
        random_state = self.config.get('models.random_state')
        
        test_size = (100 - train_percentage) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_kfold_splits(self, X: np.ndarray, y: np.ndarray, n_splits: int) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Generate K-fold cross-validation splits."""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            yield X_train, X_val, y_train, y_val
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        data, _ = self.load_data()
        return list(data.columns) 