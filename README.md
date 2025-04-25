# Loan Risk Predictor

A machine learning system for predicting loan risk using multiple models, including D-LSTM, MLP, CNN-LightGBM, DNN, Random Forest, and RNN.

## Features

- Multiple model implementations
- Stratified sampling for training
- K-fold cross-validation
- Subsample rate control
- Interactive GUI with real-time visualization
- Performance metrics visualization (accuracy, sensitivity, specificity)
- Model comparison charts

## Prerequisites

- Python 3.12 or higher
- Poetry (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JiaLong0209/loan-risk-predictor.git
cd loan-risk-predictor
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Usage

### GUI Application

To run the GUI application:

```bash
python -m src.main
```

The GUI provides the following controls:
- Train Split Rate: Adjust the training/test split ratio (50-90%)
- K-Fold Number: Set the number of folds for cross-validation (2-10)
- Subsample Rate: Control the data subsampling rate (10-100%)
- Model Selection: Choose which models to run
- Run Models: Start the training and evaluation process

### Command Line Interface

To run specific models from the command line:

```bash
python -m src.main --model d_lstm  # Run only D-LSTM model
python -m src.main  # Run all models
poetry run python -m src.main --gui # Run GUI
```

## Project Structure

```
src/
├── config/
│   └── config.py
├── data/
│   └── data_repository.py
├── models/
│   ├── base_model.py
│   ├── model_factory.py
│   ├── d_lstm_model.py
│   ├── mlp_model.py
│   ├── cnn_lightgbm_model.py
│   ├── dnn_model.py
│   ├── random_forest_model.py
│   └── rnn_model.py
├── utils/
│   ├── test_utils.py
│   ├── feature_engineering.py
│   └── logger.py
├── app.py
├── gui.py
└── main.py
```

## Model Descriptions

1. **D-LSTM (Deep Long Short-Term Memory)**
   - Deep LSTM architecture for sequence modeling
   - Suitable for capturing temporal dependencies

2. **MLP (Multi-Layer Perceptron)**
   - Classic neural network architecture
   - Good for general classification tasks

3. **CNN-LightGBM**
   - Hybrid model combining CNN for feature extraction
   - LightGBM for final classification

4. **DNN (Deep Neural Network)**
   - Deep neural network with batch normalization
   - Dropout for regularization

5. **Random Forest**
   - Ensemble of decision trees
   - Good for handling non-linear relationships

6. **RNN (Recurrent Neural Network)**
   - Recurrent architecture for sequence modeling
   - Suitable for time-series data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
