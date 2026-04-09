# Applicability Boundaries of Time-Series Foundation Models

**Data-Layer Mismatch in Industrial Motor Fault Diagnosis**

> Z. Chen, K. Song, X. Lv, and G. Guo, "Applicability Boundaries of Time-Series Foundation Models: Data-Layer Mismatch in Industrial Motor Fault Diagnosis," *IEEE Access*, 2025.

This repository provides the complete code, raw data, and configuration tables for reproducing the experiments described in the paper. The study conducts a controlled comparison of 606 experimental configurations across traditional ML, deep learning, and time-series foundation models (Chronos-Bolt, MOMENT) for industrial motor fault diagnosis.

---

## Repository Structure

```
.
├── rawdata/                        # Raw industrial sensor data (1,868 recordings)
│   ├── ZhenDong/                   #   Vibration signals (65,536 Hz)
│   │   ├── normal/                 #     362 healthy motor recordings
│   │   ├── spark/                  #     62 electrical fault recordings
│   │   └── vibrate/                #     510 mechanical fault recordings
│   └── ShengYing/                  #   Acoustic signals (65,536 Hz)
│       ├── normal/                 #     362 healthy motor recordings
│       ├── spark/                  #     62 electrical fault recordings
│       └── vibrate/                #     510 mechanical fault recordings
│
├── experiments/                    # Main experiment scripts (run in order)
│   ├── 01_statistical_ml.py        #   Traditional ML: RF, SVM, LightGBM, etc.
│   ├── 02_enhanced_ml.py           #   Enhanced ML: tuned SVM, MLP, ensembles
│   ├── 03_chronos_models.py        #   Chronos-Bolt embedding + classifiers
│   ├── 04_transformer_models.py    #   Lightweight Transformer (Small/Medium/Large)
│   ├── 05_comprehensive_analysis.py#   Cross-paradigm result aggregation
│   ├── 06_model_ranking.py         #   Performance ranking across 606 configs
│   ├── 07_chronos_residual_classification.py  # Residual-based fault detection
│   └── 08_device_level_split.py    #   Device-level split evaluation (new)
│   └── 09_weak_feature_baseline.py  # Weak-feature baseline (reviewer response)
│
├── models/                         # Model architecture definitions
│   ├── traditional_ml.py           #   sklearn-based ML pipeline
│   ├── deep_learning.py            #   CNN, LSTM, Transformer (PyTorch)
│   ├── advanced_deep_learning.py   #   ResNet, attention variants
│   ├── foundation_models.py        #   TSFM wrappers (Chronos, MOMENT)
│   ├── moment_chronos_pipeline.py  #   MOMENT + Chronos unified pipeline
│   └── unsupervised_anomaly.py     #   Unsupervised anomaly detection baselines
│
├── feature_engineering/            # Feature extraction
│   └── feature_extractor_full.py   #   Complete 71-dim physics-driven features
│
├── data_processing/                # Data loading and preprocessing
│   ├── data_loader.py              #   Multi-modal data loader
│   ├── data_utils.py               #   Utility functions
│   └── preprocessor.py             #   Signal preprocessing pipeline
│
├── finetune_chronos/               # Chronos fine-tuning experiments
│   ├── 00_data_process.py          #   Data preparation for fine-tuning
│   ├── data_split.py               #   Device-aware train/val/test splitting
│   ├── method_a_residual.py        #   Method A: Normal-only + residual features
│   ├── method_b_embedding.py       #   Method B: Normal-only + embedding features
│   ├── method_c_all_class.py       #   Method C: All-class + embedding features
│   └── feature_extraction_classification.py  # Unified feature extraction + classification
│
├── evaluation/                     # Evaluation and visualization
│   ├── experiment_tracker.py       #   Experiment logging
│   └── visualizer.py               #   Result visualization
│
├── tables/                         # Specification and configuration tables
│   ├── feature_list_71_physics_driven.csv  # 71 features with physical interpretations
│   ├── motor_specifications.csv            # Motor, sensor, and fault descriptions
│   └── model_hyperparameters.csv           # Hyperparameters for all models
│
├── .gitignore
└── README.md
```

---

## Raw Data

Each CSV file contains a **one-second recording** sampled at **65,536 Hz**. Data were collected during factory end-of-line inspection of single-phase high-speed brushed motors (~37,000 rpm) under no-load conditions.

| Modality | Class | Samples | Fault Description |
|----------|-------|---------|-------------------|
| Vibration (`ZhenDong`) | normal | 362 | Healthy operation |
| Vibration (`ZhenDong`) | spark | 62 | Electrical brush-commutator fault (4 motor units) |
| Vibration (`ZhenDong`) | vibrate | 510 | Mechanical imbalance (26 motor units) |
| Acoustic (`ShengYing`) | normal | 362 | Healthy operation |
| Acoustic (`ShengYing`) | spark | 62 | Electrical brush-commutator fault |
| Acoustic (`ShengYing`) | vibrate | 510 | Mechanical imbalance |

---

## Quick Start

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn lightgbm joblib tqdm
pip install chronos-forecasting momentfm
```

### Run Experiments

```bash
# Traditional ML (statistical features + classifiers)
python experiments/01_statistical_ml.py

# Enhanced ML (tuned hyperparameters, ensembles)
python experiments/02_enhanced_ml.py

# Foundation models
python experiments/03_chronos_models.py
python experiments/04_transformer_models.py

# Analysis and ranking
python experiments/05_comprehensive_analysis.py
python experiments/06_model_ranking.py

# Device-level split evaluation (reviewer-requested)
python experiments/08_device_level_split.py
```

### Fine-tune Chronos

```bash
cd finetune_chronos
python method_a_residual.py      # Normal-only fine-tuning + residual classification
python method_b_embedding.py     # Normal-only fine-tuning + embedding classification
python method_c_all_class.py     # All-class fine-tuning + embedding classification
```

---

## Key Results

Traditional ML pipelines based on 71 physics-driven features consistently outperform TSFMs by ~5.35 pp in Accuracy, Macro-F1, and AUROC under small-sample, noisy industrial conditions. The study identifies **Data-Layer Mismatch** as the structural root cause and proposes a **Threefold Alignment Principle**: temporal-scale alignment, representation-task alignment, and aggregation-robustness alignment.

### Weak-Feature Baseline (Supplementary Experiment)
- **Weak-10**: 10 basic time-domain statistics (mean, std, rms, skewness, kurtosis, min, max, range, median, iqr)
- **Full**: All ML features used in the device-level evaluation
- **TSFM**: Chronos-t5-base embedding (768-dim) + RandomForest
---

## Citation

```bibtex
@article{chen2025applicability,
  title   = {Applicability Boundaries of Time-Series Foundation Models:
             Data-Layer Mismatch in Industrial Motor Fault Diagnosis},
  author  = {Chen, Zhiqiang and Song, Kangkang and Lv, Xintong and Guo, Guodong},
  journal = {IEEE Access},
  year    = {2025}
}
```

## License

MIT License
