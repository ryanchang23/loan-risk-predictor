from src.app import LoanRiskPredictor
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Loan Risk Prediction System')
    parser.add_argument('--model', type=str, help='Model to use (default: use all models)')
    parser.add_argument('--subsample', type=float, default=1.0, help='Subsample rate (0.1 to 1.0)')
    parser.add_argument('--folds', type=int, default=5, help='Number of K-fold cross-validation folds')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    if args.gui:
        from src.gui import LoanRiskPredictorGUI
        app = LoanRiskPredictorGUI()
        app.run()
    else:
        predictor = LoanRiskPredictor()
        
        if args.model:
            results = predictor.run(args.model, args.subsample, args.folds, args.debug)
        else:
            results = predictor.run_all_models(args.subsample, args.folds, args.debug)
        
        # Print results
        print("\nResults:")
        print("--------")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"Average Accuracy: {np.mean(metrics['accuracy']):.4f}")
            print(f"Average Sensitivity: {np.mean(metrics['sensitivity']):.4f}")
            print(f"Average Specificity: {np.mean(metrics['specificity']):.4f}")
            print(f"Standard Deviation (Accuracy): {np.std(metrics['accuracy']):.4f}")

if __name__ == "__main__":
    main() 