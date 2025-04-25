import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from .base_model import BaseModel

class DNNModel(BaseModel):
    """Deep Neural Network model implementation."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the DNN model architecture."""
        class DNN(nn.Module):
            def __init__(self, input_size):
                super(DNN, self).__init__()
                self.fc1 = nn.Linear(input_size, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, 32)
                self.fc5 = nn.Linear(32, 1)
                self.dropout = nn.Dropout(0.3)
                self.batch_norm1 = nn.BatchNorm1d(256)
                self.batch_norm2 = nn.BatchNorm1d(128)
                self.batch_norm3 = nn.BatchNorm1d(64)
                self.batch_norm4 = nn.BatchNorm1d(32)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.batch_norm1(torch.relu(self.fc1(x)))
                x = self.dropout(x)
                x = self.batch_norm2(torch.relu(self.fc2(x)))
                x = self.dropout(x)
                x = self.batch_norm3(torch.relu(self.fc3(x)))
                x = self.dropout(x)
                x = self.batch_norm4(torch.relu(self.fc4(x)))
                x = self.fc5(x)
                return self.sigmoid(x)
        
        return DNN(input_size)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        # Create model
        self.model = self._create_model(X_train.shape[1]).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(100):  # Number of epochs
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to PyTorch tensor
        X = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        
        return predictions.cpu().numpy()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
        """Evaluate the model."""
        predictions = self.predict(X_test)
        predictions = (predictions > 0.5).astype(int)
        return self._calculate_metrics(y_test, predictions)
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.load_state_dict(torch.load(path)) 