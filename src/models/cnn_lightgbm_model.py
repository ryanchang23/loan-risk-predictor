import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from typing import Tuple
from .base_model import BaseModel
from tqdm import tqdm

class CNNLightGBMModel(BaseModel):
    """CNN-LightGBM hybrid model implementation."""
    
    def __init__(self):
        super().__init__()
        self.cnn_model = None
        self.lgb_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = None
        
        self.model_name = "cnn_lightgbm"
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
                        nn.BatchNorm1d(out_channels),  # Add batch normalization
                        nn.ReLU(),
                        nn.MaxPool1d(2)
                    ])
                    prev_channels = out_channels
                
                # Calculate the size of the flattened features
                with torch.no_grad():
                    # Create dummy input with correct shape (batch_size, channels, sequence_length)
                    dummy_input = torch.zeros(1, 1, input_size)
                    for layer in conv_layers:
                        dummy_input = layer(dummy_input)
                    flatten_size = dummy_input.view(1, -1).size(1)
                
                self.conv_layers = nn.Sequential(*conv_layers)
                self.fc1 = nn.Linear(flatten_size, 64)  # Increased size
                self.fc2 = nn.Linear(64, 32)  # Increased size
                self.fc3 = nn.Linear(32, 1)  # Final layer for binary classification
                self.dropout = nn.Dropout(dropout_rate)
                self.sigmoid = nn.Sigmoid()
                self.bn1 = nn.BatchNorm1d(64)  # Add batch normalization
                self.bn2 = nn.BatchNorm1d(32)  # Add batch normalization
            
            def forward(self, x, return_features=False):
                # Add channel dimension if not present (batch_size, channels, sequence_length)
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)  # Flatten
                x = self.fc1(x)
                x = self.bn1(x)
                x = torch.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.bn2(x)
                features = torch.relu(x)
                
                if return_features:
                    return features
                
                # For training, return classification output
                x = self.fc3(features)
                return self.sigmoid(x)
        
        return CNN(
            input_size=input_size,
            conv_channels=self.hyperparams['cnn']['conv_channels'],
            kernel_size=self.hyperparams['cnn']['kernel_size'],
            dropout_rate=self.hyperparams['cnn']['dropout_rate']
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        # Convert to PyTorch tensors - no need to reshape here as we'll handle it in the model
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        # Store input size and create CNN model
        self.input_size = X_train.shape[1]  # Number of features
        self.cnn_model = self._create_cnn_model(self.input_size).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.cnn_model.parameters(),
            lr=self.hyperparams['cnn']['learning_rate'],
            weight_decay=1e-5  # Add L2 regularization
        )
        
        # Training loop for CNN
        batch_size = self.hyperparams['cnn']['batch_size']
        n_samples = X_train.shape[0]
        
        self.cnn_model.train()
        # Add progress bar for epochs
        pbar = tqdm(range(self.hyperparams['cnn']['epochs']), desc=f'Training {self.model_name} CNN')
        for epoch in pbar:
            # Shuffle data
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Forward pass - get classification output
                outputs = self.cnn_model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record average loss for the epoch
            avg_loss = epoch_loss / (n_samples / batch_size)
            self.record_loss(avg_loss)
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Extract features using CNN for LightGBM
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_features = self.cnn_model(X_train, return_features=True).cpu().numpy()
        
        # Train LightGBM model with modified parameters
        lgb_params = self.hyperparams['lightgbm'].copy()
        lgb_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data_in_leaf': 20,  # Prevent overfitting
            'min_gain_to_split': 0.1  # Prevent splits with very small gains
        })
        
        lgb_train = lgb.Dataset(cnn_features, y_train.cpu().numpy().ravel())  # Ensure y is 1D
        
        # Create progress bar for LightGBM training
        callbacks = [lgb.log_evaluation(period=0)]  # Disable default logging
        pbar = tqdm(range(self.hyperparams['lightgbm']['num_boost_round']), 
                   desc=f'Training {self.model_name} LightGBM')
        
        def callback(env):
            pbar.update(1)
            if env.evaluation_result_list:  # Check if list is not empty
                pbar.set_postfix({'loss': f'{env.evaluation_result_list[0][2]:.4f}'})
        
        # Train LightGBM with progress bar
        self.lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=self.hyperparams['lightgbm']['num_boost_round'],
            callbacks=[callback]
        )
        pbar.close()
        
        # Plot and save the training loss curve
        self.plot_train_loss()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.cnn_model is None or self.lgb_model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get CNN features for LightGBM
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_features = self.cnn_model(X_tensor, return_features=True).cpu().numpy()
        
        # Make LightGBM predictions
        predictions = self.lgb_model.predict(cnn_features)
        return predictions.reshape(-1)  # Ensure 1D output
    
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