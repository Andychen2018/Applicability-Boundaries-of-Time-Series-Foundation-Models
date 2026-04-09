#!/usr/bin/env python3
"""
09 - Weak-Feature Baseline Experiment
======================================
Purpose: Address reviewer concern on whether the advantage of traditional
         ML is due to heavy domain-specific feature engineering.

Approach:
  - Construct a 10-dim "weak" (pure time-domain statistical) feature set
    as a strict subset of the full ML features used in 08_device_level_split.py.
  - Evaluate LightGBM and Logistic Regression under:
      (1) Random stratified split
      (2) Device-level split
  - Compare Weak-10 vs Full (all ML features) vs Chronos+RF (TSFM).

All settings (seeds, split ratio, data, downsampling) are identical to
08_device_level_split.py.

Usage:
  python 09_weak_feature_baseline.py \
      --data_dir ../rawdata \
      --chronos_model amazon/chronos-t5-base \
      --output_dir ./output_weak_feature
"""

import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============ Constants ============
SEEDS = [42, 123, 456, 789, 1024]
DOWNSAMPLE_FACTOR = 128
SAMPLING_RATE = 65536
DOWNSAMPLED_SR = SAMPLING_RATE // DOWNSAMPLE_FACTOR  # 512
CLASS_NAMES = ['normal', 'spark', 'vibrate']


# ============ Data Loading (identical to 08_device_level_split) ============

def parse_device_id(filename):
    parts = filename.replace('.csv', '').split('_')
    return parts[-2]


def load_data(data_dir, mode='zhendong'):
    X_list, y_list, dev_list = [], [], []
    for category in CLASS_NAMES:
        if mode == 'zhendong':
            cat_dir = os.path.join(data_dir, 'ZhenDong', category)
        else:
            cat_dir = os.path.join(data_dir, 'ShengYing', category)
        if not os.path.exists(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith('.csv'):
                continue
            fpath = os.path.join(cat_dir, fname)
            try:
                signal = pd.read_csv(fpath, header=None).values.flatten()
            except:
                continue
            if len(signal) > 65536:
                signal = signal[:65536]
            elif len(signal) < 65536:
                pad = np.zeros(65536)
                pad[:len(signal)] = signal
                signal = pad
            device_id = parse_device_id(fname)
            dev_key = f"{category}_{device_id}"
            X_list.append(signal)
            y_list.append(category)
            dev_list.append(dev_key)

    X = np.array(X_list)
    y_str = np.array(y_list)
    devices = np.array(dev_list)
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y = le.transform(y_str)

    print(f"Loaded {len(X)} samples, {len(np.unique(devices))} unique devices")
    for c in CLASS_NAMES:
        m = y_str == c
        print(f"  {c}: {m.sum()} samples, {len(np.unique(devices[m]))} devices")
    return X, y, y_str, devices, le


# ============ Feature Extraction ============

def extract_full_features(signal, sr=512):
    """Full ML feature set, identical to 08_device_level_split.py extract_ml_features.
    Same order, same naming, same computation."""
    f = {}
    f['mean'] = np.mean(signal)
    f['std'] = np.std(signal)
    f['var'] = np.var(signal)
    f['min'] = np.min(signal)
    f['max'] = np.max(signal)
    f['range'] = f['max'] - f['min']
    f['median'] = np.median(signal)
    f['skewness'] = float(stats.skew(signal))
    f['kurtosis'] = float(stats.kurtosis(signal))
    f['rms'] = np.sqrt(np.mean(signal ** 2))
    f['energy'] = np.sum(signal ** 2)
    f['power'] = f['energy'] / len(signal)
    f['peak_to_peak'] = np.ptp(signal)
    f['crest_factor'] = f['max'] / f['rms'] if f['rms'] != 0 else 0
    abs_mean = np.mean(np.abs(signal))
    f['form_factor'] = f['rms'] / abs_mean if abs_mean != 0 else 0
    zc = np.where(np.diff(np.signbit(signal)))[0]
    f['zero_crossing_rate'] = len(zc) / len(signal)
    f['q10'] = np.percentile(signal, 10)
    f['q25'] = np.percentile(signal, 25)
    f['q75'] = np.percentile(signal, 75)
    f['q90'] = np.percentile(signal, 90)
    f['iqr'] = f['q75'] - f['q25']

    # Frequency domain
    try:
        fft_vals = fft(signal)
        fft_mag = np.abs(fft_vals[:len(fft_vals) // 2])
        freqs = fftfreq(len(signal), 1 / sr)[:len(fft_vals) // 2]
        te = np.sum(fft_mag ** 2)
        if te > 0:
            f['spectral_centroid'] = np.sum(freqs * fft_mag) / np.sum(fft_mag)
            f['spectral_bw'] = np.sqrt(np.sum(((freqs - f['spectral_centroid']) ** 2) * fft_mag) / np.sum(fft_mag))
            di = np.argmax(fft_mag)
            f['dominant_freq'] = freqs[di]
            f['dominant_mag'] = fft_mag[di]
            lo = freqs < (sr * 0.03)
            mi = (freqs >= sr * 0.03) & (freqs < sr * 0.3)
            hi = freqs >= sr * 0.3
            f['low_freq_e'] = np.sum(fft_mag[lo] ** 2) / te
            f['mid_freq_e'] = np.sum(fft_mag[mi] ** 2) / te
            f['high_freq_e'] = np.sum(fft_mag[hi] ** 2) / te
            f['spectral_flatness'] = float(stats.gmean(fft_mag + 1e-10)) / (np.mean(fft_mag) + 1e-10)
            f['spectral_kurtosis'] = float(stats.kurtosis(fft_mag))
        else:
            for k in ['spectral_centroid', 'spectral_bw', 'dominant_freq', 'dominant_mag',
                       'low_freq_e', 'mid_freq_e', 'high_freq_e', 'spectral_flatness',
                       'spectral_kurtosis']:
                f[k] = 0
    except:
        for k in ['spectral_centroid', 'spectral_bw', 'dominant_freq', 'dominant_mag',
                   'low_freq_e', 'mid_freq_e', 'high_freq_e', 'spectral_flatness',
                   'spectral_kurtosis']:
            f[k] = 0

    # Hjorth
    try:
        f['hjorth_activity'] = np.var(signal)
        d1 = np.diff(signal)
        mob = np.std(d1) / np.std(signal) if np.std(signal) != 0 else 0
        f['hjorth_mobility'] = mob
        d2 = np.diff(d1)
        mob2 = np.std(d2) / np.std(d1) if np.std(d1) != 0 else 0
        f['hjorth_complexity'] = mob2 / mob if mob != 0 else 0
    except:
        f['hjorth_activity'] = f['hjorth_mobility'] = f['hjorth_complexity'] = 0

    return f


# ---- Weak feature selection ----
# 10 basic time-domain statistical descriptors (strict subset of full features).
# These require NO domain knowledge and are universally applicable to any 1-D signal.
WEAK_FEATURE_NAMES = [
    'mean',                # Mean - central tendency
    'std',                 # Standard deviation - dispersion
    'rms',                 # Root mean square - overall signal level
    'skewness',            # Skewness - asymmetry
    'kurtosis',            # Kurtosis - tail heaviness / impulsiveness
    'min',                 # Minimum value
    'max',                 # Maximum value
    'range',               # Range - max minus min
    'median',              # Median - robust central tendency
    'iqr',                 # Interquartile range - robust dispersion
]


# ============ Split Functions (identical to 08_device_level_split) ============

def random_split(y, seed, test_size=0.2):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return tr, te


def device_split(y, y_str, devices, seed, test_size=0.2):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for cat in CLASS_NAMES:
        cat_mask = y_str == cat
        cat_indices = np.where(cat_mask)[0]
        cat_devs = devices[cat_indices]
        udevs = np.unique(cat_devs)
        rng.shuffle(udevs)
        n_test = max(1, int(len(udevs) * test_size))
        test_devs = set(udevs[:n_test])
        for idx in cat_indices:
            if devices[idx] in test_devs:
                test_idx.append(idx)
            else:
                train_idx.append(idx)
    return np.array(train_idx), np.array(test_idx)


# ============ Evaluation ============

def evaluate(model, X_tr, X_te, y_tr, y_te, le):
    """Train, predict, compute metrics."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)

    # Probabilities for AUROC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_te_s)
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_te_s)
    else:
        y_prob = None

    # Core metrics
    acc = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average='macro')

    # AUROC (one-vs-rest, macro)
    try:
        if y_prob is not None and y_prob.ndim == 2:
            auroc = roc_auc_score(y_te, y_prob, multi_class='ovr', average='macro')
        else:
            auroc = float('nan')
    except:
        auroc = float('nan')

    # Per-class metrics
    spark_i = list(le.classes_).index('spark')
    per_prec = precision_score(y_te, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_te, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_te, y_pred, average=None, zero_division=0)
    spark_rec = per_rec[spark_i]

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'auroc': auroc,
        'spark_recall': spark_rec,
        'per_precision': per_prec,
        'per_recall': per_rec,
        'per_f1': per_f1,
        'cm': confusion_matrix(y_te, y_pred),
    }


def run_repeats(X_feat, y, y_str, devices, le, model_fn, split_type, seeds):
    results = []
    for seed in seeds:
        if split_type == 'random':
            tr, te = random_split(y, seed)
        else:
            tr, te = device_split(y, y_str, devices, seed)
        model = model_fn(seed)
        r = evaluate(model, X_feat[tr], X_feat[te], y[tr], y[te], le)
        results.append(r)
    return results


# ============ Model Factories ============

def lgbm_factory(seed):
    return lgb.LGBMClassifier(
        n_estimators=200, random_state=seed, verbose=-1, n_jobs=-1)

def lr_factory(seed):
    return LogisticRegression(random_state=seed, max_iter=1000)

def rf_factory(seed):
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)


# ============ Main ============

def main(args):
    print("=" * 70)
    print("Weak-Feature Baseline Experiment")
    print("=" * 70)

    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

    # 1. Load data
    print("\n[1/4] Loading data...")
    X_raw, y, y_str, devices, le = load_data(args.data_dir, 'zhendong')
    X_ds = X_raw[:, ::DOWNSAMPLE_FACTOR]
    print(f"  Downsampled: {X_ds.shape}")

    # 2. Extract features
    print("\n[2/4] Extracting features...")
    print(f"  Extracting full feature set...")
    all_records = []
    for i, s in enumerate(X_ds):
        if i % 200 == 0 and i > 0:
            print(f"    Progress: {i}/{len(X_ds)}")
        all_records.append(extract_full_features(s, sr=DOWNSAMPLED_SR))
    full_df = pd.DataFrame(all_records).fillna(0).replace([np.inf, -np.inf], 0)

    X_full = full_df.values  # all features
    X_weak = full_df[WEAK_FEATURE_NAMES].values  # 10 weak features

    print(f"  Full features: {X_full.shape[1]} dims")
    print(f"  Weak features: {X_weak.shape[1]} dims  =>  {WEAK_FEATURE_NAMES}")

    # 3. Also extract Chronos TSFM features for comparison
    print("\n  Extracting Chronos (TSFM) features...")
    try:
        import torch
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            args.chronos_model, device_map="cpu", torch_dtype=torch.float32)
        model_chr = pipeline.model
        tokenizer = pipeline.tokenizer

        feat_list = []
        for i, signal in enumerate(X_ds):
            if i % 50 == 0:
                print(f"    Chronos embedding: {i}/{len(X_ds)}")
            try:
                sig_t = torch.tensor(signal[:512], dtype=torch.float32).unsqueeze(0)
                token_ids, attn_mask, scale = tokenizer._input_transform(sig_t)
                with torch.no_grad():
                    embedding = model_chr.encode(token_ids, attn_mask)
                    if isinstance(embedding, tuple):
                        embedding = embedding[0]
                    pooled = embedding.mean(dim=1).squeeze(0).numpy()
                feat_list.append(pooled)
            except:
                feat_list.append(np.zeros(768))
        X_tsfm = np.nan_to_num(np.array(feat_list), nan=0., posinf=0., neginf=0.)
        print(f"  TSFM features: {X_tsfm.shape}")
        tsfm_available = True
    except Exception as e:
        print(f"  WARNING: Could not load Chronos model: {e}")
        print(f"  TSFM comparison will be skipped.")
        tsfm_available = False

    # 4. Run experiments
    print("\n[3/4] Running experiments (5 seeds)...")

    configs = []

    # Weak features (10 dims)
    for name, factory in [('LR', lr_factory),
                          ('LightGBM', lgbm_factory)]:
        for split in ['random', 'device']:
            configs.append((f'{name} (Weak-10)', X_weak, factory, split))

    # Full features - re-run for direct comparison with same code
    for name, factory in [('LR', lr_factory),
                          ('LightGBM', lgbm_factory)]:
        for split in ['random', 'device']:
            configs.append((f'{name} (Full)', X_full, factory, split))

    # TSFM (Chronos+RF)
    if tsfm_available:
        for split in ['random', 'device']:
            configs.append(('Chronos+RF (TSFM)', X_tsfm, rf_factory, split))

    all_results = {}
    for model_name, X_feat, model_fn, split_type in configs:
        key = f"{model_name}|{split_type}"
        print(f"  {key} ...")
        res = run_repeats(X_feat, y, y_str, devices, le, model_fn, split_type, SEEDS)
        all_results[key] = res

    # 5. Build output tables
    print("\n[4/4] Building output tables...")

    def fmt(vals, pct=False):
        m = np.nanmean(vals)
        s = np.nanstd(vals)
        if pct:
            return f"{m*100:.1f} ± {s*100:.1f}"
        return f"{m:.4f} ± {s:.4f}"

    # ===== TABLE 1: Overall Performance =====
    tbl1_rows = []
    for key, res_list in all_results.items():
        model_name, split_type = key.split('|')
        if 'Weak' in model_name:
            feat_set = 'Weak (10)'
        elif 'TSFM' in model_name:
            feat_set = 'Embedding (768)'
        else:
            feat_set = 'Full'

        accs = [r['accuracy'] for r in res_list]
        f1s = [r['macro_f1'] for r in res_list]
        aurocs = [r['auroc'] for r in res_list]

        tbl1_rows.append({
            'Model': model_name,
            'Feature Set': feat_set,
            'Split': split_type,
            'Accuracy': fmt(accs, True),
            'Macro-F1': fmt(f1s, True),
            'AUROC': fmt(aurocs, True),
        })

    tbl1 = pd.DataFrame(tbl1_rows)
    p1 = os.path.join(output_dir, "tables", "table1_overall_performance.csv")
    tbl1.to_csv(p1, index=False)
    print("\n" + "=" * 100)
    print("TABLE 1: Overall Performance Comparison")
    print("=" * 100)
    print(tbl1.to_string(index=False))
    print(f"\nSaved: {p1}")

    # ===== TABLE 2: Spark Class Performance =====
    tbl2_rows = []
    for key, res_list in all_results.items():
        model_name, split_type = key.split('|')
        if 'Weak' in model_name:
            feat_set = 'Weak (10)'
        elif 'TSFM' in model_name:
            feat_set = 'Embedding (768)'
        else:
            feat_set = 'Full'

        spark_i = list(le.classes_).index('spark')
        spark_recs = [r['spark_recall'] for r in res_list]
        spark_precs = [r['per_precision'][spark_i] for r in res_list]
        spark_f1s = [r['per_f1'][spark_i] for r in res_list]

        tbl2_rows.append({
            'Model': model_name,
            'Feature Set': feat_set,
            'Split': split_type,
            'Spark Precision': fmt(spark_precs, True),
            'Spark Recall': fmt(spark_recs, True),
            'Spark F1': fmt(spark_f1s, True),
        })

    tbl2 = pd.DataFrame(tbl2_rows)
    p2 = os.path.join(output_dir, "tables", "table2_spark_performance.csv")
    tbl2.to_csv(p2, index=False)
    print("\n" + "=" * 100)
    print("TABLE 2: Spark Class Performance")
    print("=" * 100)
    print(tbl2.to_string(index=False))
    print(f"\nSaved: {p2}")

    # ===== JSON Summary =====
    summary = {}
    for key, res_list in all_results.items():
        spark_i = list(le.classes_).index('spark')
        entry = {
            'accuracy_mean': float(np.nanmean([r['accuracy'] for r in res_list])),
            'accuracy_std': float(np.nanstd([r['accuracy'] for r in res_list])),
            'macro_f1_mean': float(np.nanmean([r['macro_f1'] for r in res_list])),
            'macro_f1_std': float(np.nanstd([r['macro_f1'] for r in res_list])),
            'auroc_mean': float(np.nanmean([r['auroc'] for r in res_list])),
            'auroc_std': float(np.nanstd([r['auroc'] for r in res_list])),
            'spark_recall_mean': float(np.nanmean([r['spark_recall'] for r in res_list])),
            'spark_recall_std': float(np.nanstd([r['spark_recall'] for r in res_list])),
            'spark_f1_mean': float(np.nanmean([r['per_f1'][spark_i] for r in res_list])),
            'spark_f1_std': float(np.nanstd([r['per_f1'][spark_i] for r in res_list])),
        }
        for ci, cn in enumerate(CLASS_NAMES):
            entry[f'{cn}_precision_mean'] = float(np.nanmean(
                [r['per_precision'][ci] for r in res_list]))
            entry[f'{cn}_recall_mean'] = float(np.nanmean(
                [r['per_recall'][ci] for r in res_list]))
            entry[f'{cn}_f1_mean'] = float(np.nanmean(
                [r['per_f1'][ci] for r in res_list]))
        summary[key] = entry

    json_path = os.path.join(output_dir, "tables", "full_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary saved: {json_path}")

    # ===== Brief Analysis =====
    print("\n" + "=" * 100)
    print("BRIEF ANALYSIS")
    print("=" * 100)

    device_keys = [(k, v) for k, v in summary.items() if 'device' in k]
    if device_keys:
        print("\n--- Device-Level Split Comparison (sorted by Macro-F1) ---")
        for key, entry in sorted(device_keys, key=lambda x: -x[1]['macro_f1_mean']):
            print(f"  {key:40s}  Acc={entry['accuracy_mean']*100:.1f}±{entry['accuracy_std']*100:.1f}  "
                  f"F1={entry['macro_f1_mean']*100:.1f}±{entry['macro_f1_std']*100:.1f}  "
                  f"AUROC={entry['auroc_mean']*100:.1f}±{entry['auroc_std']*100:.1f}  "
                  f"Spark-R={entry['spark_recall_mean']*100:.1f}±{entry['spark_recall_std']*100:.1f}")

    print("\nAll experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weak-Feature Baseline: Weak-10 vs Full vs TSFM")
    parser.add_argument("--data_dir", type=str, default="../rawdata",
                        help="Path to rawdata directory containing ZhenDong/ShengYing")
    parser.add_argument("--chronos_model", type=str, default="amazon/chronos-t5-base",
                        help="Chronos model name or local path")
    parser.add_argument("--output_dir", type=str, default="./output_weak_feature",
                        help="Directory for output tables and JSON")
    args = parser.parse_args()
    main(args)
