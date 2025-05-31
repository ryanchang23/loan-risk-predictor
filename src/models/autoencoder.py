import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np
from .base_model import BaseModel

class AutoencoderModel(BaseModel):
    """Autoencoder model for feature extraction."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = None
        self.encoding_size = None
        
        # Load hyperparameters from config
        self.hyperparams = self.config.get(f"models.hyperparameters.{self.model_name}")
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create the autoencoder model architecture."""
        class Autoencoder(nn.Module):
            def __init__(self, input_size, encoding_size, hidden_layers, dropout_rate):
                super(Autoencoder, self).__init__()
                
                # Encoder
                encoder_layers = []
                prev_size = input_size
                for hidden_size in hidden_layers:
                    encoder_layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_size = hidden_size
                encoder_layers.append(nn.Linear(prev_size, encoding_size))
                
                # Decoder
                decoder_layers = []
                prev_size = encoding_size
                for hidden_size in reversed(hidden_layers):
                    decoder_layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_size = hidden_size
                decoder_layers.extend([
                    nn.Linear(prev_size, input_size),
                    nn.Sigmoid()
                ])
                
                self.encoder = nn.Sequential(*encoder_layers)
                self.decoder = nn.Sequential(*decoder_layers)
            
            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                """Forward pass through the autoencoder."""
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        return Autoencoder(
            input_size=input_size,
            encoding_size=self.hyperparams['encoding_size'],
            hidden_layers=self.hyperparams['hidden_layers'],
            dropout_rate=self.hyperparams['dropout_rate']
        )
    
    def train(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> None:
        """Train the autoencoder.
        
        Args:
            X_train: Training data
            y_train: Not used in autoencoder training, kept for interface compatibility
        """
        # Store input size for model creation and saving
        self.input_size = X_train.shape[1]
        self.encoding_size = self.hyperparams['encoding_size']
        
        # Convert to PyTorch tensor
        X_train = torch.FloatTensor(X_train).to(self.device)
        
        # Create model
        self.model = self._create_model(self.input_size).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams['learning_rate']
        )
        
        # Training loop
        batch_size = self.hyperparams['batch_size']
        n_samples = X_train.shape[0]
        
        self.model.train()
        for epoch in range(self.hyperparams['epochs']):
            # Shuffle data
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = X_train[batch_indices]
                
                # Forward pass
                encoded, decoded = self.model(batch)
                
                # Calculate loss
                loss = criterion(decoded, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record average loss for the epoch
            self.record_loss(epoch_loss / (n_samples / batch_size))
        
        # Plot and save the training loss curve
        self.plot_train_loss()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Encode the input data using the trained autoencoder.
        
        Args:
            X: Input data to encode
            
        Returns:
            Encoded representation of the input data
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to PyTorch tensor
        X = torch.FloatTensor(X).to(self.device)
        
        # Encode data
        self.model.eval()
        with torch.no_grad():
            encoded, _ = self.model(X)
        
        return encoded.cpu().numpy()
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict method for backward compatibility."""
        return self.predict(X)
    
    def decode(self, X: np.ndarray) -> np.ndarray:
        """Decode the encoded data back to original space.
        
        Args:
            X: Encoded data to decode
            
        Returns:
            Decoded data in original space
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to PyTorch tensor
        X = torch.FloatTensor(X).to(self.device)
        
        # Decode data
        self.model.eval()
        with torch.no_grad():
            _, decoded = self.model(X)
        
        return decoded.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'encoding_size': self.encoding_size,
            'hyperparams': self.hyperparams
        }
        torch.save(save_data, path)
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_size = checkpoint['input_size']
        self.encoding_size = checkpoint['encoding_size']
        self.hyperparams = checkpoint['hyperparams']
        
        self.model = self._create_model(self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict']) 