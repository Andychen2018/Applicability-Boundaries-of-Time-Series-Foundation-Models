#!/usr/bin/env python3
"""
08_Device-Level Split Evaluation (精简版)
==========================================
只评估 2 个模型:
  - 最优 ML: GradientBoosting (统计特征)
  - 最优 TSFM: Chronos-t5-base + RandomForest

两种划分:
  - Random stratified split
  - Device-level split (同一电机不跨集)

输出:
  1. 对比表 (Random vs Device): Macro-F1, Spark Recall, Spark F1
  2. 混淆矩阵图: 最优 TSFM, Device-level split

实验规范: 固定种子, 无数据增强, 无类别重平衡, 5次重复
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch

warnings.filterwarnings('ignore')

# ============ 配置 ============
DATA_PATH = "/home/deep/TimeSeries/Zhendong/data3"
CHRONOS_MODEL_PATH = "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-t5-base"
OUTPUT_DIR = "/home/deep/TimeSeries/Zhendong/code/output_device_level"
SEEDS = [42, 123, 456, 789, 1024]
DOWNSAMPLE_FACTOR = 128
SAMPLING_RATE = 65536
CLASS_NAMES = ['normal', 'spark', 'vibrate']

os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tables"), exist_ok=True)


# ============ 数据加载 ============

def parse_device_id(filename):
    """从文件名解析 device_id (倒数第2段)"""
    parts = filename.replace('.csv', '').split('_')
    return parts[-2]


def load_data(mode='zhendong'):
    """加载全部数据，返回信号、标签、设备标识"""
    X_list, y_list, dev_list = [], [], []

    for category in CLASS_NAMES:
        if mode == 'zhendong':
            cat_dir = os.path.join(DATA_PATH, 'ZhenDong', category)
        else:
            cat_dir = os.path.join(DATA_PATH, 'ShengYing', category)
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
            # 标准化长度
            if len(signal) > 65536:
                signal = signal[:65536]
            elif len(signal) < 65536:
                pad = np.zeros(65536)
                pad[:len(signal)] = signal
                signal = pad

            device_id = parse_device_id(fname)
            dev_key = f"{category}_{device_id}"  # 类别+设备号 = 唯一设备

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


# ============ 特征提取 ============

def extract_ml_features(signal, sr=512):
    """统计 + 频域特征 (与 01_fast_statistical_ml 一致)"""
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

    # 频域
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
                       'low_freq_e', 'mid_freq_e', 'high_freq_e', 'spectral_flatness', 'spectral_kurtosis']:
                f[k] = 0
    except:
        for k in ['spectral_centroid', 'spectral_bw', 'dominant_freq', 'dominant_mag',
                   'low_freq_e', 'mid_freq_e', 'high_freq_e', 'spectral_flatness', 'spectral_kurtosis']:
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

    return list(f.values())


def extract_ml_features_batch(signals):
    """批量提取 ML 特征"""
    out = []
    for i, s in enumerate(signals):
        if i % 200 == 0 and i > 0:
            print(f"  ML features: {i}/{len(signals)}")
        out.append(extract_ml_features(s))
    return np.nan_to_num(np.array(out), nan=0., posinf=0., neginf=0.)


def extract_chronos_features_batch(signals, model_path=CHRONOS_MODEL_PATH, ctx_len=512):
    """用 Chronos-t5-base encoder embedding 做特征 (768-dim mean pooling)"""
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        model_path, device_map="cpu", torch_dtype=torch.float32)
    model = pipeline.model
    tokenizer = pipeline.tokenizer

    feat_list = []
    for i, signal in enumerate(signals):
        if i % 50 == 0:
            print(f"  Chronos embedding: {i}/{len(signals)}")
        try:
            sig_t = torch.tensor(signal[:ctx_len], dtype=torch.float32).unsqueeze(0)
            token_ids, attn_mask, scale = tokenizer._input_transform(sig_t)
            with torch.no_grad():
                embedding = model.encode(token_ids, attn_mask)  # (1, seq_len, 768)
                if isinstance(embedding, tuple):
                    embedding = embedding[0]
                pooled = embedding.mean(dim=1).squeeze(0).numpy()  # (768,)
            feat_list.append(pooled)
        except Exception as e:
            feat_list.append(np.zeros(768))

    return np.nan_to_num(np.array(feat_list), nan=0., posinf=0., neginf=0.)


# ============ 数据划分 ============

def random_split(y, seed, test_size=0.2):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return tr, te


def device_split(y, y_str, devices, seed, test_size=0.2):
    """按设备划分: 每个类别内独立抽设备到测试集"""
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


# ============ 评估 ============

def evaluate(model, X_tr, X_te, y_tr, y_te, le):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    spark_i = list(le.classes_).index('spark')
    macro_f1 = f1_score(y_te, y_pred, average='macro')
    spark_rec = recall_score(y_te, y_pred, labels=[spark_i], average=None, zero_division=0)[0]
    spark_f1 = f1_score(y_te, y_pred, labels=[spark_i], average=None, zero_division=0)[0]
    cm = confusion_matrix(y_te, y_pred)

    # Per-class precision, recall, f1
    per_prec = precision_score(y_te, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_te, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_te, y_pred, average=None, zero_division=0)

    return {
        'macro_f1': macro_f1, 'spark_recall': spark_rec, 'spark_f1': spark_f1,
        'cm': cm,
        'per_precision': per_prec,  # shape (3,)
        'per_recall': per_rec,
        'per_f1': per_f1,
    }


def run_repeats(X_feat, y, y_str, devices, le, model_fn, split_fn, seeds):
    results = []
    for seed in seeds:
        if split_fn == 'random':
            tr, te = random_split(y, seed)
        else:
            tr, te = device_split(y, y_str, devices, seed)
        model = model_fn(seed)
        r = evaluate(model, X_feat[tr], X_feat[te], y[tr], y[te], le)
        results.append(r)
    return results


# ============ 主流程 ============

def main():
    print("=" * 70)
    print("Device-Level Split Evaluation (2 models, 5 seeds)")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/5] Loading data...")
    X_raw, y, y_str, devices, le = load_data('zhendong')
    X_ds = X_raw[:, ::DOWNSAMPLE_FACTOR]  # 65536 -> 512
    print(f"  Downsampled: {X_ds.shape}")

    # 2. 提取特征
    print("\n[2/5] Extracting ML features...")
    X_ml = extract_ml_features_batch(X_ds)
    print(f"  ML features: {X_ml.shape}")

    print("\n[3/5] Extracting Chronos (TSFM) features...")
    X_tsfm = extract_chronos_features_batch(X_ds)
    print(f"  TSFM features: {X_tsfm.shape}")

    # 3. 定义模型工厂
    def gb_factory(seed):
        return GradientBoostingClassifier(n_estimators=200, random_state=seed)

    def rf_factory(seed):
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)

    # 4. 跑 4 组实验
    configs = [
        ('GradientBoosting (ML)',  X_ml,   gb_factory, 'random'),
        ('GradientBoosting (ML)',  X_ml,   gb_factory, 'device'),
        ('Chronos+RF (TSFM)',      X_tsfm, rf_factory, 'random'),
        ('Chronos+RF (TSFM)',      X_tsfm, rf_factory, 'device'),
    ]

    rows = {}  # (model, split) -> results
    print("\n[4/5] Running experiments (5 seeds × 4 configs)...")
    for model_name, X_feat, model_fn, split_type in configs:
        print(f"  {model_name} | {split_type} split")
        res = run_repeats(X_feat, y, y_str, devices, le, model_fn, split_type, SEEDS)
        rows[(model_name, split_type)] = res

    # 5. 构建所有输出
    print("\n[5/5] Building tables & figures...")

    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    # ---- 与 redraw_figures_ieee_template.py 完全一致的样式 ----
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "figure.dpi": 600, "savefig.dpi": 600, "text.usetex": False,
    })
    # 与论文 Figure 1 配色一致：白 → 浅蓝 → 蓝紫（对比度足够，PDF可见）
    cm_cmap = LinearSegmentedColormap.from_list(
        'elegant', ['#FFFFFF', '#cde2e8', '#c8d4e9', '#9eaad1'])

    MODEL_NAMES = ['GradientBoosting (ML)', 'Chronos+RF (TSFM)']

    def fmt(vals):
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

    def fmt2(vals):
        return f"{np.mean(vals)*100:.1f} ± {np.std(vals)*100:.1f}"

    # ========== TABLE 1: Comparison (Random vs Device) with Performance Drop ==========
    tbl1_rows = []
    for mn in MODEL_NAMES:
        rr = rows[(mn, 'random')]
        dr = rows[(mn, 'device')]
        r_f1 = np.mean([r['macro_f1'] for r in rr])
        d_f1 = np.mean([r['macro_f1'] for r in dr])
        r_sr = np.mean([r['spark_recall'] for r in rr])
        d_sr = np.mean([r['spark_recall'] for r in dr])
        r_sf1 = np.mean([r['spark_f1'] for r in rr])
        d_sf1 = np.mean([r['spark_f1'] for r in dr])
        tbl1_rows.append({
            'Model': mn,
            'Random Macro-F1':     fmt([r['macro_f1'] for r in rr]),
            'Device Macro-F1':     fmt([r['macro_f1'] for r in dr]),
            'Macro-F1 Drop':       f"{(r_f1 - d_f1)*100:+.1f} pp",
            'Random Spark Recall': fmt([r['spark_recall'] for r in rr]),
            'Device Spark Recall': fmt([r['spark_recall'] for r in dr]),
            'Spark Recall Drop':   f"{(r_sr - d_sr)*100:+.1f} pp",
            'Random Spark F1':     fmt([r['spark_f1'] for r in rr]),
            'Device Spark F1':     fmt([r['spark_f1'] for r in dr]),
            'Spark F1 Drop':       f"{(r_sf1 - d_sf1)*100:+.1f} pp",
        })
    tbl1 = pd.DataFrame(tbl1_rows)
    p1 = os.path.join(OUTPUT_DIR, "tables", "device_vs_random_comparison.csv")
    tbl1.to_csv(p1, index=False)
    print("\n" + "=" * 90)
    print("TABLE 1: Random vs Device-Level Split (with Performance Drop)")
    print("=" * 90)
    print(tbl1.to_string(index=False))
    print(f"Saved: {p1}")

    # ========== TABLE 2: Per-class Precision / Recall / F1 ==========
    tbl2_rows = []
    for mn in MODEL_NAMES:
        for sp in ['random', 'device']:
            res_list = rows[(mn, sp)]
            precs = np.array([r['per_precision'] for r in res_list])   # (5, 3)
            recs  = np.array([r['per_recall'] for r in res_list])
            f1s   = np.array([r['per_f1'] for r in res_list])
            for ci, cn in enumerate(CLASS_NAMES):
                tbl2_rows.append({
                    'Model': mn, 'Split': sp, 'Class': cn,
                    'Precision': fmt2(precs[:, ci]),
                    'Recall':    fmt2(recs[:, ci]),
                    'F1':        fmt2(f1s[:, ci]),
                })
    tbl2 = pd.DataFrame(tbl2_rows)
    p2 = os.path.join(OUTPUT_DIR, "tables", "per_class_metrics.csv")
    tbl2.to_csv(p2, index=False)
    print("\n" + "=" * 90)
    print("TABLE 2: Per-Class Precision / Recall / F1")
    print("=" * 90)
    print(tbl2.to_string(index=False))
    print(f"Saved: {p2}")

    # ========== FIGURE: 2×2 Confusion Matrix Comparison ==========
    cm_labels = ['N', 'S', 'V']
    panel_keys = [
        ('GradientBoosting (ML)', 'random',  'Best ML\n(Random Split)'),
        ('GradientBoosting (ML)', 'device',  'Best ML\n(Device Split)'),
        ('Chronos+RF (TSFM)',     'random',  'Best TSFM\n(Random Split)'),
        ('Chronos+RF (TSFM)',     'device',  'Best TSFM\n(Device Split)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    for idx, (mn, sp, title) in enumerate(panel_keys):
        ax = axes[idx // 2][idx % 2]
        total_cm = sum(r['cm'] for r in rows[(mn, sp)])
        cm_norm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, cmap=cm_cmap, aspect='auto', vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(cm_labels, fontsize=9)
        ax.set_yticklabels(cm_labels, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')

        for i in range(3):
            for j in range(3):
                v = cm_norm[i, j]
                ax.text(j, i, f'{v:.2f}', ha="center", va="center",
                        color='black', fontsize=8, fontweight='bold')

    fig.supxlabel('Predicted Label', fontsize=9, y=0.02)
    fig.supylabel('True Label', fontsize=9, x=0.02)
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])

    for ext in ['pdf', 'png']:
        fpath = os.path.join(OUTPUT_DIR, "figures", f"confusion_matrix_2x2_comparison.{ext}")
        fig.savefig(fpath, dpi=600, bbox_inches='tight', format=ext, transparent=False)
    plt.close()
    print(f"\n2×2 confusion matrix saved: confusion_matrix_2x2_comparison.pdf/.png")

    # ========== FIGURE: Single CM — Best TSFM, Device split (论文内嵌用) ==========
    tsfm_dev = rows[('Chronos+RF (TSFM)', 'device')]
    total_cm = sum(r['cm'] for r in tsfm_dev)
    cm_norm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    im = ax.imshow(cm_norm, cmap=cm_cmap, aspect='auto', vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(np.arange(3)); ax.set_yticks(np.arange(3))
    ax.set_xticklabels(cm_labels, fontsize=10)
    ax.set_yticklabels(cm_labels, fontsize=10)
    ax.set_title('Chronos+RF (Device-Level Split)', fontsize=11, fontweight='bold')
    for i in range(3):
        for j in range(3):
            v = cm_norm[i, j]
            ax.text(j, i, f'{v:.2f}', ha="center", va="center",
                    color='black', fontsize=8, fontweight='bold')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fpath = os.path.join(OUTPUT_DIR, "figures", f"confusion_matrix_tsfm_device.{ext}")
        fig.savefig(fpath, dpi=600, bbox_inches='tight', format=ext, transparent=False)
    plt.close()
    print(f"Single TSFM CM saved: confusion_matrix_tsfm_device.pdf/.png")

    # ========== JSON summary ==========
    summary = {}
    for (mn, sp), res_list in rows.items():
        f1s = [r['macro_f1'] for r in res_list]
        srs = [r['spark_recall'] for r in res_list]
        sf1s = [r['spark_f1'] for r in res_list]
        precs = np.array([r['per_precision'] for r in res_list])
        recs  = np.array([r['per_recall'] for r in res_list])
        pf1s  = np.array([r['per_f1'] for r in res_list])
        entry = {
            'macro_f1_mean': float(np.mean(f1s)),
            'macro_f1_std': float(np.std(f1s)),
            'spark_recall_mean': float(np.mean(srs)),
            'spark_recall_std': float(np.std(srs)),
            'spark_f1_mean': float(np.mean(sf1s)),
            'spark_f1_std': float(np.std(sf1s)),
        }
        for ci, cn in enumerate(CLASS_NAMES):
            entry[f'{cn}_precision_mean'] = float(np.mean(precs[:, ci]))
            entry[f'{cn}_recall_mean'] = float(np.mean(recs[:, ci]))
            entry[f'{cn}_f1_mean'] = float(np.mean(pf1s[:, ci]))
        summary[f"{mn}|{sp}"] = entry
    json_path = os.path.join(OUTPUT_DIR, "tables", "device_vs_random_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved: {json_path}")

    print("\n✅ All outputs generated!")


if __name__ == "__main__":
    main()
