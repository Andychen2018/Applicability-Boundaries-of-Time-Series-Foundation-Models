# Applicability Boundaries of Time Series Foundation Models

This repository contains implementations and experiments for analyzing the applicability boundaries of time series foundation models in motor fault diagnosis.

## Project Structure

```
.
├── data_processing/          # Data loading and preprocessing
├── feature_engineering/      # Feature extraction modules
├── traditional_ml/          # Traditional machine learning models
├── deep_learning/           # Deep learning models (CNN, LSTM, etc.)
├── foundation_models/       # Foundation models (Chronos, Moment, etc.)
├── finetune_chronos/        # Fine-tuning scripts for Chronos
├── anomaly_detection/       # Unsupervised anomaly detection methods
├── evaluation/              # Model evaluation utilities
└── *.py                     # Main experiment scripts
```

## Main Scripts

- `01_statistical_ml_models.py` - Statistical and traditional ML models
- `01_fast_statistical_ml.py` - Fast statistical ML implementations
- `02_deep_learning_models.py` - Deep learning model experiments
- `02_enhanced_ml_models.py` - Enhanced ML models
- `03_chronos_models.py` - Chronos foundation model experiments
- `04_transformer_models.py` - Transformer-based models
- `05_comprehensive_analysis.py` - Comprehensive analysis pipeline
- `06_model_ranking.py` - Model performance ranking
- `07_chronos_residual_classification.py` - Residual-based classification

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn

## Installation

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn
```

## Usage

Run individual experiment scripts:

```bash
python 01_statistical_ml_models.py
python 02_deep_learning_models.py
python 03_chronos_models.py
```

## Data

The project uses motor vibration data for fault diagnosis. Data should be organized in the following structure:

```
data/
├── ShengYing/
├── ZhenDong/
└── Fusion/
```

## Models

### Traditional ML
- Random Forest
- SVM
- Gradient Boosting
- Logistic Regression

### Deep Learning
- CNN
- LSTM
- BiLSTM
- Attention-based models

### Foundation Models
- Chronos
- Moment
- TimesFM

## License

MIT License

