import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from .base_model import BaseModel

class DLSTMModel(BaseModel):
    """Deep LSTM model implementation."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the LSTM model architecture."""
        class LSTM(nn.Module):
            def __init__(self, input_size):
                super(LSTM, self).__init__()
                self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
                self.lstm2 = nn.LSTM(64, 32, batch_first=True)
                self.fc1 = nn.Linear(32, 16)
                self.fc2 = nn.Linear(16, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                lstm1_out, _ = self.lstm1(x)
                lstm2_out, _ = self.lstm2(lstm1_out)
                fc1_out = torch.relu(self.fc1(lstm2_out[:, -1, :]))
                fc2_out = self.fc2(fc1_out)
                return self.sigmoid(fc2_out)
        
        return LSTM(input_size)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        # Reshape input for LSTM (batch_size, sequence_length, features)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        # Create model
        self.model = self._create_model(X_train.shape[2]).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
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
        
        # Reshape input for LSTM
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
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