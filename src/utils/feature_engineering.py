import numpy as np
from sklearn.feature_selection import mutual_info_classif
from typing import List, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm 
from ..config.config import ConfigManager

class FeatureEngineer:
    """Feature engineering and selection utilities."""
    
    def __init__(self):
        self.config = ConfigManager()
    
    def calculate_kraskov_entropy(self, X: np.ndarray, k: int = 5) -> np.ndarray:
        """Calculate Kraskov entropy for each feature."""
        n_samples, n_features = X.shape
        entropy = np.zeros(n_features)
        
        for i in range(n_features):
            # Calculate distances between points
            distances = np.zeros((n_samples, n_samples))
            for j in range(n_samples):
                distances[j] = np.abs(X[:, i] - X[j, i])
            
            # Sort distances for each point
            sorted_distances = np.sort(distances, axis=1)
            
            # Calculate entropy using k-nearest neighbors
            entropy[i] = -np.mean(np.log(sorted_distances[:, k] + 1e-10))
        
        return entropy
    
    def select_features(self, X: np.ndarray, y: np.ndarray, n_features: int = 5) -> Tuple[np.ndarray, List[int]]:
        """Select features using mutual information and Kraskov entropy."""
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y)
        
        # Calculate Kraskov entropy
        entropy_scores = self.calculate_kraskov_entropy(X)
        
        # Combine scores (you can adjust the weights)
        combined_scores = 0.7 * mi_scores + 0.3 * entropy_scores
        
        # Select top features
        selected_indices = np.argsort(combined_scores)[-n_features:]
        selected_features = X[:, selected_indices]
        
        return selected_features, selected_indices.tolist()
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using min-max scaling."""
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
    
    def create_feature_combinations(self, X: np.ndarray) -> np.ndarray:
        """Create feature combinations using polynomial features."""
        n_samples, n_features = X.shape
        combinations = []
        
        # Add original features
        combinations.append(X)
        
        # Add squared features
        squared = X ** 2
        combinations.append(squared)
        
        # Add interaction terms
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction = X[:, i:i+1] * X[:, j:j+1]
                combinations.append(interaction)
        
        return np.hstack(combinations)
    
    def process_features(self, X: np.ndarray, y: np.ndarray, n_features: int = 5) -> np.ndarray:
        """Process features through the complete pipeline."""
        # Normalize features
        X_normalized = self.normalize_features(X)
        
        # Create feature combinations
        X_combined = self.create_feature_combinations(X_normalized)
        
        # Select features
        X_selected, _ = self.select_features(X_combined, y, n_features)
        
        return X_selected 