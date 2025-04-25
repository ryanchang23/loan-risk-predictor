import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from ..config.config import ConfigManager

class Autoencoder(nn.Module):
    """Autoencoder model for feature extraction."""
    
    def __init__(self, input_size: int, encoding_size: int):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_size)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoencoderTrainer:
    """Trainer for the autoencoder model."""
    
    def __init__(self, input_size: int, encoding_size: int):
        self.config = ConfigManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Autoencoder(input_size, encoding_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def train(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        """Train the autoencoder."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        n_samples = X.shape[0]
        
        self.model.train()
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = X_tensor[batch_indices]
                
                # Forward pass
                encoded, decoded = self.model(batch)
                
                # Calculate loss
                loss = self.criterion(decoded, batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode data using the trained autoencoder."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            encoded, _ = self.model(X_tensor)
            return encoded.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        self.model.load_state_dict(torch.load(path)) 