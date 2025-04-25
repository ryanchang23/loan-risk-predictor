import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from typing import Tuple
from .base_model import BaseModel

class CNNLightGBMModel(BaseModel):
    """CNN-LightGBM hybrid model implementation."""
    
    def __init__(self):
        self.cnn_model = None
        self.lgb_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_cnn_model(self, input_size: int) -> nn.Module:
        """Create the CNN model architecture."""
        # class CNN(nn.Module):
        #     def __init__(self, input_size):
        #         super(CNN, self).__init__()
        #         self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        #         self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        #         self.pool = nn.MaxPool1d(2)
        #         self.fc1 = nn.Linear(64 * ((input_size - 4) // 2), 32)
        #         self.fc2 = nn.Linear(32, 16)
        #         self.dropout = nn.Dropout(0.3)
            
        #     def forward(self, x):
        #         x = x.unsqueeze(1)  # Add channel dimension
        #         x = torch.relu(self.conv1(x))
        #         x = self.pool(x)
        #         x = torch.relu(self.conv2(x))
        #         x = self.pool(x)
        #         x = x.view(x.size(0), -1)
        #         x = torch.relu(self.fc1(x))
        #         x = self.dropout(x)
        #         x = self.fc2(x)
        #         return x
        
        # return CNN(input_size)
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        # Train CNN
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        self.cnn_model = self._create_cnn_model(X_train.shape[1]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnn_model.parameters())
        
        # CNN training loop
        self.cnn_model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = self.cnn_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor.long())
            loss.backward()
            optimizer.step()
        
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
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        self.lgb_model = lgb.train(params, lgb_train, num_boost_round=100)
    
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
        
        # Save CNN model
        torch.save(self.cnn_model.state_dict(), f"{path}_cnn.pt")
        
        # Save LightGBM model
        self.lgb_model.save_model(f"{path}_lgb.txt")
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        if self.cnn_model is None:
            raise ValueError("CNN model not initialized")
        
        # Load CNN model
        self.cnn_model.load_state_dict(torch.load(f"{path}_cnn.pt"))
        
        # Load LightGBM model
        self.lgb_model = lgb.Booster(model_file=f"{path}_lgb.txt") 