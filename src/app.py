from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from .config.config import ConfigManager
from .data_pipeline.data_repository import DataRepository
from .data_pipeline.feature_engineering import FeatureEngineer
from .models.model_factory import ModelFactory
from .utils.logger import Logger
from .models.autoencoder import AutoencoderModel
from .utils.test_utils import TestDataGenerator, DebugLogger
import pandas as pd
import os

class LoanRiskPredictor:
    """Main application class for loan risk prediction."""
    
    def __init__(self):
        self.config = ConfigManager()
        # Load config from config.yaml (adjust the path if needed)
        self.config.load_from_yaml("config.yaml")
        self.data_repo = DataRepository()
        self.logger = Logger()
        self.feature_engineer = FeatureEngineer()
        self.debug_logger = DebugLogger("loan_risk_predictor")
        # Set random seed for reproducibility
        np.random.seed(42)

    def run(self, model_name: str = None, subsample_rate: float = 1.0, n_folds: int = 5,
            debug_mode: bool = False, use_feature_engineering: bool = True) -> Dict[str, List[float]]:
        """Run the loan risk prediction pipeline with K-fold cross-validation."""
        try:
            # Load and preprocess data
            self.logger.info("Loading and preprocessing data...")
            if debug_mode:
                data, labels = self.data_repo.load_data()
                self.debug_logger.log_data_info(data, "Debug Data")
                self.debug_logger.log_array_info(labels, "Debug Labels")
            else:
                data, labels = self.data_repo.load_data()

            processed_dir = self.config.get('data.processed_dir')

            # self.logger.info(f"{processed_dir, self.config.get('data.normalized_data_path')}")

            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)

            processed_path = os.path.join(processed_dir, self.config.get('data.normalized_data_path'))
            self.logger.info(f"{processed_path}")

            if os.path.exists(processed_path):
                self.logger.info(f"Using {processed_path} data.")
                normalized_data = pd.read_csv(processed_path).values
            else:
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

                self.logger.info(f"{processed_dir, self.config.get('data.fused_features_path')}")
                fused_features_path = self.config.get('data.fused_features_path')
                

                # Check if the paths are valid before joining
                if processed_dir is None or fused_features_path is None:
                    self.logger.error("Configuration for processed_dir or fused_features_path is not set.")
                    raise ValueError("Invalid configuration: processed_dir or fused_features_path is None.")

                fused_features_path = os.path.join(processed_dir, fused_features_path)
                
                # Check if fused features exist and have matching shape
                if os.path.exists(fused_features_path):
                    fused_features = pd.read_csv(fused_features_path).values
                    self.logger.info(f"fused features' shape: {fused_features.shape}")
                    self.logger.info(f"normalized_data's shape: {normalized_data.shape}")
                    if fused_features.shape[0] == normalized_data.shape[0]:
                        self.logger.info(f"Using cached fused features: {fused_features_path}")
                        encoded_features = fused_features
                    else:
                        self.logger.info("Cached fused features shape mismatch, performing feature engineering...")
                        encoded_features = self._perform_feature_engineering(normalized_data, labels, debug_mode)
                        # Save the new fused features
                        pd.DataFrame(encoded_features).to_csv(fused_features_path, index=False)
                else:
                    self.logger.info("No cached fused features found, performing feature engineering...")
                    encoded_features = self._perform_feature_engineering(normalized_data, labels, debug_mode)
                    # Save the fused features
                    pd.DataFrame(encoded_features).to_csv(fused_features_path, index=False)
            else:
                encoded_features = normalized_data
            
            # Initialize metrics storage
            metrics = {
                'accuracy': [],
                'sensitivity': [],
                'specificity': [],
                'f1_score': [],
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
                acc, sens, spec, f1, cm = model.evaluate(X_val, y_val)
                
                metrics['accuracy'].append(acc)
                metrics['sensitivity'].append(sens)
                metrics['specificity'].append(spec)
                metrics['f1_score'].append(f1)
                metrics['confusion_matrices'].append(cm)
                
                if debug_mode:
                    self.debug_logger.log_metrics({
                        'accuracy': acc,
                        'sensitivity': sens,
                        'specificity': spec,
                        'f1_score': f1
                    }, f"Fold {fold + 1} Results")
                    self.debug_logger.log_array_info(cm, f"Fold {fold + 1} Confusion Matrix")
                
                self.logger.info(f"Fold {fold + 1} results:")
                self.logger.info(f"Accuracy: {acc:.4f}")
                self.logger.info(f"Sensitivity: {sens:.4f}")
                self.logger.info(f"Specificity: {spec:.4f}")
                self.logger.info(f"F1-Score: {f1:.4f}")
                self.logger.log_confusion_matrix(cm, "Confusion Matrix: ")
            
            # Calculate and log average metrics
            avg_metrics = {
                'accuracy': np.mean(metrics['accuracy']),
                'sensitivity': np.mean(metrics['sensitivity']),
                'specificity': np.mean(metrics['specificity']),
                'f1_score': np.mean(metrics['f1_score'])
            }
            
            # Calculate total confusion matrix
            total_cm = np.sum(metrics['confusion_matrices'], axis=0).astype(int)
            metrics['confusion_matrix'] = total_cm
            
            if debug_mode:
                self.debug_logger.log_metrics(avg_metrics, "Average Results")
                self.debug_logger.log_array_info(total_cm, "Total Confusion Matrix")
            
            self.logger.info("\nAverage results across all folds:")
            self.logger.info(f"Accuracy: {avg_metrics['accuracy']:.4f}")
            self.logger.info(f"Sensitivity: {avg_metrics['sensitivity']:.4f}")
            self.logger.info(f"Specificity: {avg_metrics['specificity']:.4f}")
            self.logger.info(f"F1-Score: {avg_metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in loan risk prediction: {str(e)}")
            raise

    def _perform_feature_engineering(self, normalized_data: np.ndarray, labels: np.ndarray, debug_mode: bool) -> np.ndarray:
        """Helper method to perform feature engineering and autoencoder encoding."""
        n_features = self.config.get('features.n_features')
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
        
        return encoded_features

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