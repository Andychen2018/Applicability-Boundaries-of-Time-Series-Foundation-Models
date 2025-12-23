# Applicability Boundaries of Time Series Foundation Models

This repository contains comprehensive implementations and experiments for analyzing the applicability boundaries of time series foundation models in motor fault diagnosis applications. The project systematically compares traditional machine learning, deep learning, and foundation models to understand their strengths and limitations in industrial fault detection scenarios.

## Research Objectives

- Investigate the performance boundaries of time series foundation models (Chronos, Moment, TimesFM) in motor fault diagnosis
- Compare foundation models against traditional ML and deep learning approaches
- Explore different feature extraction and fine-tuning strategies for foundation models
- Analyze model performance across different data modalities (acoustic, vibration, fusion)
- Provide comprehensive benchmarking and analysis tools for time series classification

## Project Structure

```
.
├── data_processing/              # Data loading and preprocessing modules
├── feature_engineering/          # Feature extraction modules
├── traditional_ml/               # Traditional machine learning models
├── deep_learning/                # Deep learning models (CNN, LSTM, BiLSTM, Attention)
├── foundation_models/            # Foundation model implementations (Chronos, Moment, TimesFM)
├── finetune_chronos/             # Chronos fine-tuning experiments
├── anomaly_detection/            # Unsupervised anomaly detection methods
├── evaluation/                   # Model evaluation and visualization utilities
├── 01_statistical_ml_models.py   # Statistical and traditional ML experiments
├── 02_deep_learning_models.py    # Deep learning model experiments
├── 03_chronos_models.py          # Chronos foundation model experiments
├── 04_transformer_models.py      # Transformer-based model experiments
├── 05_comprehensive_analysis.py  # Comprehensive result analysis
├── 06_model_ranking.py           # Model performance ranking
└── 07_chronos_residual_classification.py # Residual-based classification
```

## Main Experiment Scripts

### 1. Statistical and Traditional ML Models
- **01_statistical_ml_models.py** / **01_fast_statistical_ml.py**
  - Extract time-domain and frequency-domain features (mean, std, RMS, FFT, PSD, etc.)
  - Train traditional ML classifiers: Random Forest, SVM, Gradient Boosting, Logistic Regression
  - Support three classification modes: ShengYing (acoustic), ZhenDong (vibration), Fusion (combined)

### 2. Deep Learning Models
- **02_deep_learning_models.py** / **02_enhanced_ml_models.py**
  - CNN (Convolutional Neural Networks) for temporal pattern recognition
  - LSTM/BiLSTM for sequence modeling
  - Attention-based models for feature importance
  - ResNet and Transformer variants

### 3. Foundation Models
- **03_chronos_models.py**
  - Use Chronos as feature extractor and classifier
  - Extract embeddings from pre-trained models
  - Compare with traditional feature extraction

- **04_transformer_models.py**
  - Transformer-based time series classification
  - Self-attention mechanisms for temporal patterns

### 4. Analysis and Evaluation
- **05_comprehensive_analysis.py** - Aggregate and compare all model results
- **06_model_ranking.py** - Rank models by performance metrics
- **07_chronos_residual_classification.py** - Novel residual-based classification approach

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-capable GPU (optional, for faster training)

### Required Packages
```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn joblib tqdm
pip install chronos-forecasting momentfm
```

## Usage

### Run Individual Experiments
```bash
python 01_statistical_ml_models.py
python 02_deep_learning_models.py
python 03_chronos_models.py
python 05_comprehensive_analysis.py
```

### Fine-tune Chronos Models
```bash
cd finetune_chronos
python method_a_residual.py      # Residual-based approach
python method_b_embedding.py     # Embedding extraction
python method_c_all_class.py     # Multi-class fine-tuning
```

## Data Format

Expected data structure:
```
data/
├── ShengYing/          # Acoustic signal data
│   ├── Normal/
│   ├── Fault1/
│   └── Fault2/
├── ZhenDong/           # Vibration signal data
│   ├── Normal/
│   ├── Fault1/
│   └── Fault2/
└── Fusion/             # Combined data
    ├── Normal/
    ├── Fault1/
    └── Fault2/
```

## Models Implemented

### Traditional Machine Learning
- Random Forest, SVM, Gradient Boosting, Logistic Regression, Naive Bayes, Extra Trees

### Deep Learning
- CNN, LSTM, BiLSTM, Attention Models, ResNet, Transformer

### Foundation Models
- Chronos (Amazon), Moment, TimesFM (Google)

## Fine-tuning Strategies

### Method A: Residual-based Classification
- Use Chronos to predict normal patterns
- Classify faults based on prediction residuals

### Method B: Embedding Extraction
- Extract embeddings from Chronos encoder
- Use as features for downstream classifiers

### Method C: Multi-class Fine-tuning
- Fine-tune Chronos on all fault classes
- End-to-end training for classification

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Cross-validation Scores
- Model Ranking and Comparison

## License

MIT License
