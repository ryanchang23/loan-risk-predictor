from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from .config.config import ConfigManager
from .data.data_repository import DataRepository
from .models.model_factory import ModelFactory
from .utils.logger import Logger
from .utils.feature_engineering import FeatureEngineer
from .models.autoencoder import AutoencoderTrainer
from .utils.test_utils import TestDataGenerator, DebugLogger

class LoanRiskPredictor:
    """Main application class for loan risk prediction."""
    
    def __init__(self):
        self.config = ConfigManager()
        self.data_repo = DataRepository()
        self.logger = Logger()
        self.feature_engineer = FeatureEngineer()
        self.debug_logger = DebugLogger("loan_risk_predictor")
        # Set random seed for reproducibility
        np.random.seed(42)

    def run(self, model_name: str = None, subsample_rate: float = 1.0, n_folds: int = 5,
            debug_mode: bool = False, use_feature_engineering: bool = True,
            processed_features: np.ndarray = None, processed_labels: np.ndarray = None) -> Dict[str, List[float]]:
        """Run the loan risk prediction pipeline with K-fold cross-validation."""
        try:
            # Load and preprocess data
            self.logger.info("Loading and preprocessing data...")
            if debug_mode:
                # data, labels = TestDataGenerator.generate_synthetic_data(n_samples=1000)
                data, labels = self.data_repo.load_data()
                self.debug_logger.log_data_info(data, "Debug Data")
                self.debug_logger.log_array_info(labels, "Debug Labels")
            else:
                data, labels = self.data_repo.load_data()
            
            normalized_data = self.data_repo.preprocess_data(data)
            if debug_mode:
                self.debug_logger.log_array_info(normalized_data, "Normalized Data")
            self.logger.info("Normalized Data Shape")
            self.logger.info(normalized_data.shape)
            
            # Subsample data if needed
            if subsample_rate < 1.0:
                self.logger.info(f"Subsampling data to {subsample_rate*100}%...")
                normalized_data, labels = self.data_repo.subsample_data(normalized_data, labels, subsample_rate)
                if debug_mode:
                    self.debug_logger.log_array_info(normalized_data, "Subsampled Data")
                    self.debug_logger.log_array_info(labels, "Subsampled Labels")
                self.logger.info("Subsample Data Shape")
                self.logger.info(normalized_data.shape)

                
            
            # Feature engineering
            if use_feature_engineering:
                if processed_features is not None and processed_labels is not None:
                    self.logger.info("Using cached feature engineering results...")
                    encoded_features = processed_features
                    labels = processed_labels
                else:
                    self.logger.info("Performing feature engineering...")
                    n_features = 5  # Number of features to select
                    with tqdm(total=3, desc="Feature Engineering") as pbar:
                        processed_features = self.feature_engineer.process_features(normalized_data, labels, n_features)
                        pbar.update(1)
                        if debug_mode:
                            self.debug_logger.log_array_info(processed_features, "Processed Features")
                        
                        # Autoencoder feature extraction
                        self.logger.info("Performing autoencoder feature extraction...")
                        autoencoder = AutoencoderTrainer(processed_features.shape[1], n_features)
                        autoencoder.train(processed_features)
                        pbar.update(1)
                        encoded_features = autoencoder.encode(processed_features)
                        pbar.update(1)
                        if debug_mode:
                            self.debug_logger.log_array_info(encoded_features, "Encoded Features")
            else:
                encoded_features = normalized_data
            
            # Initialize metrics storage
            metrics = {
                'accuracy': [],
                'sensitivity': [],
                'specificity': [],
                'confusion_matrices': []
            }
            
            # Get model name from config if not provided
            if model_name is None:
                model_name = self.config.get('models.default_model')
            
            # Perform K-fold cross-validation
            self.logger.info(f"Performing {n_folds}-fold cross-validation...")
            for fold, (X_train, X_val, y_train, y_val) in enumerate(tqdm(self.data_repo.get_kfold_splits(encoded_features, labels, n_folds), 
                                                                        total=n_folds, desc="K-fold Cross Validation")):
                self.logger.info(f"Training fold {fold + 1}/{n_folds}...")
                
                if debug_mode:
                    self.debug_logger.log_array_info(X_train, f"Fold {fold + 1} Training Data")
                    self.debug_logger.log_array_info(y_train, f"Fold {fold + 1} Training Labels")
                    self.debug_logger.log_array_info(X_val, f"Fold {fold + 1} Validation Data")
                    self.debug_logger.log_array_info(y_val, f"Fold {fold + 1} Validation Labels")
                
                # Create and train model
                model = ModelFactory.create_model(model_name)
                with tqdm(total=1, desc=f"Training {model_name}") as pbar:
                    model.train(X_train, y_train)
                    pbar.update(1)
                
                # Evaluate model
                acc, sens, spec, cm = model.evaluate(X_val, y_val)
                
                metrics['accuracy'].append(acc)
                metrics['sensitivity'].append(sens)
                metrics['specificity'].append(spec)
                metrics['confusion_matrices'].append(cm)
                
                if debug_mode:
                    self.debug_logger.log_metrics({
                        'accuracy': acc,
                        'sensitivity': sens,
                        'specificity': spec
                    }, f"Fold {fold + 1} Results")
                    self.debug_logger.log_array_info(cm, f"Fold {fold + 1} Confusion Matrix")
                
                self.logger.info(f"Fold {fold + 1} results:")
                self.logger.info(f"Accuracy: {acc:.4f}")
                self.logger.info(f"Sensitivity: {sens:.4f}")
                self.logger.info(f"Specificity: {spec:.4f}")
                # self.logger.log_array_info(cm, f"Fold {fold + 1} Confusion Matrix")

            
            # Calculate and log average metrics
            avg_metrics = {
                'accuracy': np.mean(metrics['accuracy']),
                'sensitivity': np.mean(metrics['sensitivity']),
                'specificity': np.mean(metrics['specificity'])
            }
            
            # Calculate average confusion matrix
            avg_cm = np.mean(metrics['confusion_matrices'], axis=0).astype(int)
            metrics['confusion_matrix'] = avg_cm
            
            if debug_mode:
                self.debug_logger.log_metrics(avg_metrics, "Average Results")
                self.debug_logger.log_array_info(avg_cm, "Average Confusion Matrix")
            
            self.logger.info("\nAverage results across all folds:")
            self.logger.info(f"Accuracy: {avg_metrics['accuracy']:.4f}")
            self.logger.info(f"Sensitivity: {avg_metrics['sensitivity']:.4f}")
            self.logger.info(f"Specificity: {avg_metrics['specificity']:.4f}")
            # self.logger.log_array_info(avg_cm, "Average Confusion Matrix")
            
            # Add processed features to metrics if this is the first run
            if use_feature_engineering and processed_features is None:
                metrics['processed_features'] = encoded_features
                metrics['processed_labels'] = labels
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in loan risk prediction: {str(e)}")
            raise
    
    def run_all_models(self, subsample_rate: float = 1.0, n_folds: int = 5, debug_mode: bool = False) -> Dict[str, Dict[str, List[float]]]:
        """Run all available models with K-fold cross-validation."""
        results = {}
        
        for model_name in ModelFactory.get_available_models():
            try:
                self.logger.info(f"\nRunning {model_name} model...")
                metrics = self.run(model_name, subsample_rate, n_folds, debug_mode)
                results[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Error running {model_name}: {str(e)}")
                continue
        
        return results 