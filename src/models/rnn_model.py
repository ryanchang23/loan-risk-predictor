import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from .base_model import BaseModel

class RNNModel(BaseModel):
    """Recurrent Neural Network model implementation."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = None  # 新增一個屬性，記錄input_size
        
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the RNN model architecture."""
        class RNN(nn.Module):
            def __init__(self, input_size):
                super(RNN, self).__init__()
                self.hidden_size = 64
                self.num_layers = 2
                self.rnn = nn.RNN(
                    input_size,
                    self.hidden_size,
                    self.num_layers,
                    batch_first=True,
                    dropout=0.3
                )
                self.fc1 = nn.Linear(self.hidden_size, 32)
                self.fc2 = nn.Linear(32, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                out, _ = self.rnn(x, h0)
                out = out[:, -1, :]  # Get the last time step
                out = torch.relu(self.fc1(out))
                out = self.fc2(out)
                return self.sigmoid(out)
        
        return RNN(input_size)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        # Reshape input for RNN (batch_size, sequence_length, features)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        # Create model
        self.input_size = X_train.shape[2]
        self.model = self._create_model(self.input_size).to(self.device)
        
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
            raise ValueError("Model not trained yet.")
        
        # Reshape input for RNN
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
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size
        }
        torch.save(save_data, path)
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_size = checkpoint['input_size']
        self.model = self._create_model(self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
