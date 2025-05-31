import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from typing import Tuple
from .base_model import BaseModel

class CNNLightGBMModel(BaseModel):
    """CNN-LightGBM hybrid model implementation."""
    
    def __init__(self):
        super().__init__()
        self.cnn_model = None
        self.lgb_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = None
        
        # Load hyperparameters from config
        self.hyperparams = self.config.get(f"models.hyperparameters.{self.model_name}")
    
    def _create_cnn_model(self, input_size: int) -> nn.Module:
        """Create the CNN model architecture."""
        class CNN(nn.Module):
            def __init__(self, input_size, conv_channels, kernel_size, dropout_rate):
                super(CNN, self).__init__()
                
                # Create convolutional layers dynamically
                conv_layers = []
                prev_channels = 1  # Input channels
                
                for out_channels in conv_channels:
                    conv_layers.extend([
                        nn.Conv1d(prev_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                        nn.ReLU(),
                        nn.MaxPool1d(2)
                    ])
                    prev_channels = out_channels
                
                # Calculate the size of the flattened features
                with torch.no_grad():
                    dummy_input = torch.zeros(1, 1, input_size)
                    for layer in conv_layers:
                        dummy_input = layer(dummy_input)
                    flatten_size = dummy_input.view(1, -1).size(1)
                
                self.conv_layers = nn.Sequential(*conv_layers)
                self.fc1 = nn.Linear(flatten_size, 32)
                self.fc2 = nn.Linear(32, 16)
                self.dropout = nn.Dropout(dropout_rate)
            
            def forward(self, x):
                x = x.unsqueeze(1)  # Add channel dimension
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)  # Flatten
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return CNN(
            input_size=input_size,
            conv_channels=self.hyperparams['cnn']['conv_channels'],
            kernel_size=self.hyperparams['cnn']['kernel_size'],
            dropout_rate=self.hyperparams['cnn']['dropout_rate']
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        # Train CNN
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        self.cnn_model = self._create_cnn_model(X_train.shape[1]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.cnn_model.parameters(),
            lr=self.hyperparams['cnn']['learning_rate']
        )
        
        # CNN training loop
        batch_size = self.hyperparams['cnn']['batch_size']
        n_samples = X_train.shape[0]
        
        self.cnn_model.train()
        for epoch in range(self.hyperparams['cnn']['epochs']):
            # Shuffle data
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                # Forward pass
                outputs = self.cnn_model(X_batch)
                loss = criterion(outputs, y_batch.long())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record average loss for the epoch
            self.record_loss(epoch_loss / (n_samples / batch_size))
        
        # Plot and save the training loss curve
        self.plot_train_loss()
        
        # Extract features from CNN
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_features = self.cnn_model(X_train_tensor).cpu().numpy()
        
        # Train LightGBM
        lgb_train = lgb.Dataset(cnn_features, y_train)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': self.hyperparams['lightgbm']['num_leaves'],
            'learning_rate': self.hyperparams['lightgbm']['learning_rate'],
            'feature_fraction': self.hyperparams['lightgbm']['feature_fraction']
        }
        
        self.lgb_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=self.hyperparams['lightgbm']['num_boost_round']
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.cnn_model is None or self.lgb_model is None:
            raise ValueError("Model not trained yet")
        
        # Get CNN features
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_features = self.cnn_model(X_tensor).cpu().numpy()
        
        # Make LightGBM predictions
        predictions = self.lgb_model.predict(cnn_features)
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
        """Evaluate the model."""
        predictions = self.predict(X_test)
        predictions = (predictions > 0.5).astype(int)
        return self._calculate_metrics(y_test, predictions)
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        if self.cnn_model is None or self.lgb_model is None:
            raise ValueError("No model to save")
        
        save_data = {
            'cnn_state_dict': self.cnn_model.state_dict(),
            'input_size': self.input_size,
            'hyperparams': self.hyperparams
        }
        torch.save(save_data, f"{path}_cnn.pt")
        self.lgb_model.save_model(f"{path}_lgb.txt")
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        # Load CNN model
        checkpoint = torch.load(f"{path}_cnn.pt", map_location=self.device)
        self.input_size = checkpoint['input_size']
        self.hyperparams = checkpoint['hyperparams']
        
        self.cnn_model = self._create_cnn_model(self.input_size).to(self.device)
        self.cnn_model.load_state_dict(checkpoint['cnn_state_dict'])
        
        # Load LightGBM model
        self.lgb_model = lgb.Booster()
        self.lgb_model.load_model(f"{path}_lgb.txt") 