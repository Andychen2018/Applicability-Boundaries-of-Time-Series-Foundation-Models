# Applicability Boundaries of Time-Series Foundation Models

**Data-Layer Mismatch in Industrial Motor Fault Diagnosis**

This repository provides the raw data, supplementary experiment code, and configuration tables for the paper:

> *Applicability Boundaries of Time-Series Foundation Models: Data-Layer Mismatch in Industrial Motor Fault Diagnosis*
> Submitted to IEEE Access

---

## Repository Structure

```
.
├── rawdata/                          # Raw industrial motor sensor data
│   ├── ZhenDong/                     # Vibration signals (65,536 Hz)
│   │   ├── normal/                   # 362 recordings
│   │   ├── spark/                    # 62 recordings
│   │   └── vibrate/                  # 510 recordings
│   └── ShengYing/                    # Acoustic signals (65,536 Hz)
│       ├── normal/                   # 362 recordings
│       ├── spark/                    # 62 recordings
│       └── vibrate/                  # 510 recordings
├── code/                             # Supplementary experiment scripts
│   ├── 08_device_level_split.py      # Device-level split evaluation (new)
│   ├── 08_device_level_evaluation.py # Device-level evaluation utilities (new)
│   └── 09_feature_extractor_full.py  # Full 71-feature physics-driven extractor
├── tables/                           # Configuration and specification tables
│   ├── feature_list_71_physics_driven.csv  # Complete 71-feature list with physical interpretations
│   ├── motor_specifications.csv            # Motor and sensor specifications
│   └── model_hyperparameters.csv           # Hyperparameters for all evaluated models
└── README.md
```

## Data Description

Each CSV file in `rawdata/` contains a one-second recording sampled at **65,536 Hz** (65,536 samples per file). Signals are organized by sensing modality and health state:

| Modality | Class | Samples | Description |
|----------|-------|---------|-------------|
| ZhenDong (Vibration) | normal | 362 | Healthy motor operation |
| ZhenDong (Vibration) | spark | 62 | Electrical brush-commutator fault |
| ZhenDong (Vibration) | vibrate | 510 | Mechanical imbalance fault |
| ShengYing (Acoustic) | normal | 362 | Healthy motor operation |
| ShengYing (Acoustic) | spark | 62 | Electrical brush-commutator fault |
| ShengYing (Acoustic) | vibrate | 510 | Mechanical imbalance fault |

**Total: 934 samples × 2 modalities = 1,868 recordings**

## Supplementary Code

| Script | Description |
|--------|-------------|
| `08_device_level_split.py` | Device-level split experiment ensuring no motor unit appears in both train and test sets. Addresses reviewer concern on cross-device generalization. |
| `08_device_level_evaluation.py` | Evaluation utilities for device-level split experiments. |
| `09_feature_extractor_full.py` | Complete 71-dimensional physics-driven feature extraction pipeline (time-domain, frequency-domain, Hjorth, nonlinear, multi-scale band energy, cepstral, and envelope spectrum features). |

## Configuration Tables

| Table | Description |
|-------|-------------|
| `feature_list_71_physics_driven.csv` | All 71 hand-crafted features with Chinese/English names and physical interpretations |
| `motor_specifications.csv` | Motor type, rated power/voltage/speed, sensor configuration, fault descriptions |
| `model_hyperparameters.csv` | Hyperparameters for all models: LightGBM, SVM, Random Forest, Transformer, MLP, etc. |

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{chen2025applicability,
  title={Applicability Boundaries of Time-Series Foundation Models: Data-Layer Mismatch in Industrial Motor Fault Diagnosis},
  author={Chen, Zhiqiang and Song, Kangkang and Lv, Xintong and Guo, Guodong},
  journal={IEEE Access},
  year={2025}
}
```

## License

This project is released for academic research purposes.
