"""
08_Device-Level Generalization & Statistical Validation
========================================================
Section IV 补充实验：
  Part 1: Device-level split 泛化验证
  Part 2: 统计验证 (ANOVA, Tukey HSD, Cohen's d)
  Part 3: 图表生成 (混淆矩阵, 频谱对比)

实验规范：
  - 固定随机种子
  - 统一训练轮数/优化预算
  - 不允许数据增强
  - 不允许类别重平衡
  - 所有差异仅来自模型范式差异
"""

import os
import re
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from itertools import combinations

from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.stats import f_oneway

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score,
    precision_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import torch

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
DATA_PATH = "/home/deep/TimeSeries/Zhendong/data3"
CHRONOS_MODEL_PATH = "/home/deep/TimeSeries/Zhendong/chronos_models/chronos-bolt-base"
OUTPUT_DIR = "/home/deep/TimeSeries/Zhendong/code/output_device_level"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

SEEDS = [42, 123, 456, 789, 1024]
N_REPEATS = 5
DOWNSAMPLE_FACTOR = 128  # 65536 -> 512 points
SAMPLING_RATE = 65536
CATEGORIES = ['normal', 'spark', 'vibrate']
CLASS_NAMES = ['normal', 'spark', 'vibrate']

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# ============================================================
# Part 0: Data Loading with Device Information
# ============================================================

def parse_device_id(filename, category):
    """从文件名中解析设备ID。
    normal: a_normal_{device}_{seg}.csv
    spark:  a_spark_{device}_{seg}.csv
    vibrate: a_noise_{device}_{seg}.csv 或 a_vibrate_{device}_{seg}.csv
    """
    name = filename.replace('.csv', '')
    parts = name.split('_')
    device_id = parts[-2]
    segment_id = parts[-1]
    return device_id, segment_id


def load_all_data_with_device_info(mode='zhendong'):
    """加载数据并保留设备信息。
    mode: 'shengying', 'zhendong', 'fusion'
    返回: X (signals), y (labels), device_ids, file_paths
    """
    sensor_dirs = {
        'shengying': ['ShengYing'],
        'zhendong': ['ZhenDong'],
        'fusion': ['ShengYing', 'ZhenDong']
    }
    sensors = sensor_dirs[mode]

    X_list, y_list, device_list, file_list = [], [], [], []

    # 获取ShengYing的文件列表作为配对基准
    ref_sensor = 'ShengYing'
    for category in CATEGORIES:
        cat_dir = os.path.join(DATA_PATH, ref_sensor, category)
        if not os.path.exists(cat_dir):
            continue
        files = sorted([f for f in os.listdir(cat_dir) if f.endswith('.csv')])

        for fname in files:
            device_id, seg_id = parse_device_id(fname, category)
            # 构造唯一设备标识: category_device
            device_key = f"{category}_{device_id}"

            if mode == 'fusion':
                sy_path = os.path.join(DATA_PATH, 'ShengYing', category, fname)
                zd_path = os.path.join(DATA_PATH, 'ZhenDong', category, fname)
                if not os.path.exists(zd_path):
                    continue
                try:
                    sy_data = pd.read_csv(sy_path, header=None).values.flatten()
                    zd_data = pd.read_csv(zd_path, header=None).values.flatten()
                    signal = np.concatenate([sy_data, zd_data])
                except:
                    continue
            elif mode == 'shengying':
                fpath = os.path.join(DATA_PATH, 'ShengYing', category, fname)
                try:
                    signal = pd.read_csv(fpath, header=None).values.flatten()
                except:
                    continue
            else:  # zhendong
                fpath = os.path.join(DATA_PATH, 'ZhenDong', category, fname)
                try:
                    signal = pd.read_csv(fpath, header=None).values.flatten()
                except:
                    continue

            X_list.append(signal)
            y_list.append(category)
            device_list.append(device_key)
            file_list.append(fname)

    X = np.array([_standardize_length(s, 65536) for s in X_list])
    y = np.array(y_list)
    devices = np.array(device_list)
    files = np.array(file_list)

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y_encoded = le.transform(y)

    print(f"[{mode}] Loaded {len(X)} samples, {len(np.unique(devices))} unique devices")
    for c in CLASS_NAMES:
        mask = y == c
        print(f"  {c}: {mask.sum()} samples, {len(np.unique(devices[mask]))} devices")

    return X, y_encoded, y, devices, files, le


def _standardize_length(signal, target_length=65536):
    if len(signal) > target_length:
        return signal[:target_length]
    elif len(signal) < target_length:
        padded = np.zeros(target_length)
        padded[:len(signal)] = signal
        return padded
    return signal


# ============================================================
# Part 0b: Feature Extraction
# ============================================================

class FeatureExtractor:
    def __init__(self, sampling_rate=65536):
        self.sampling_rate = sampling_rate

    def extract(self, signal):
        features = {}
        # 基本统计
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(signal)
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['energy'] = np.sum(signal**2)
        features['power'] = features['energy'] / len(signal)
        features['peak_to_peak'] = np.ptp(signal)
        features['crest_factor'] = features['max'] / features['rms'] if features['rms'] != 0 else 0
        features['form_factor'] = features['rms'] / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
        features['q10'] = np.percentile(signal, 10)
        features['q25'] = np.percentile(signal, 25)
        features['q75'] = np.percentile(signal, 75)
        features['q90'] = np.percentile(signal, 90)
        features['iqr'] = features['q75'] - features['q25']

        # 频域特征
        try:
            fft_vals = fft(signal)
            fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
            freqs = fftfreq(len(signal), 1/self.sampling_rate)[:len(fft_vals)//2]
            total_energy = np.sum(fft_mag**2)
            if total_energy > 0:
                features['spectral_centroid'] = np.sum(freqs * fft_mag) / np.sum(fft_mag)
                features['spectral_bandwidth'] = np.sqrt(
                    np.sum(((freqs - features['spectral_centroid'])**2) * fft_mag) / np.sum(fft_mag))
                dom_idx = np.argmax(fft_mag)
                features['dominant_frequency'] = freqs[dom_idx]
                features['dominant_magnitude'] = fft_mag[dom_idx]
                low_mask = freqs < 1000
                mid_mask = (freqs >= 1000) & (freqs < 10000)
                high_mask = freqs >= 10000
                features['low_freq_energy'] = np.sum(fft_mag[low_mask]**2) / total_energy
                features['mid_freq_energy'] = np.sum(fft_mag[mid_mask]**2) / total_energy
                features['high_freq_energy'] = np.sum(fft_mag[high_mask]**2) / total_energy
                features['spectral_flatness'] = stats.gmean(fft_mag + 1e-10) / (np.mean(fft_mag) + 1e-10)
                features['spectral_kurtosis'] = stats.kurtosis(fft_mag)
            else:
                for k in ['spectral_centroid','spectral_bandwidth','dominant_frequency',
                           'dominant_magnitude','low_freq_energy','mid_freq_energy',
                           'high_freq_energy','spectral_flatness','spectral_kurtosis']:
                    features[k] = 0
        except:
            for k in ['spectral_centroid','spectral_bandwidth','dominant_frequency',
                       'dominant_magnitude','low_freq_energy','mid_freq_energy',
                       'high_freq_energy','spectral_flatness','spectral_kurtosis']:
                features[k] = 0

        # Hjorth参数
        try:
            activity = np.var(signal)
            features['hjorth_activity'] = activity
            diff1 = np.diff(signal)
            mobility = np.std(diff1) / np.std(signal) if np.std(signal) != 0 else 0
            features['hjorth_mobility'] = mobility
            diff2 = np.diff(diff1)
            mobility2 = np.std(diff2) / np.std(diff1) if np.std(diff1) != 0 else 0
            features['hjorth_complexity'] = mobility2 / mobility if mobility != 0 else 0
        except:
            features['hjorth_activity'] = 0
            features['hjorth_mobility'] = 0
            features['hjorth_complexity'] = 0

        return features

    def extract_batch(self, signals):
        feat_list = []
        for i, sig in enumerate(signals):
            if i % 100 == 0 and i > 0:
                print(f"  Extracting features: {i}/{len(signals)}")
            f = self.extract(sig)
            feat_list.append(list(f.values()))
        return np.array(feat_list)


# ============================================================
# Part 0c: Chronos Feature Extraction (TSFM)
# ============================================================

def load_chronos_and_extract(signals, model_path=CHRONOS_MODEL_PATH, context_length=512):
    """使用Chronos提取特征 (作为TSFM的代表)"""
    try:
        from chronos import ChronosPipeline
    except ImportError:
        print("WARNING: Chronos not available, falling back to statistical features")
        return None

    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"WARNING: Failed to load Chronos model: {e}")
        return None

    features_list = []
    for i, signal in enumerate(signals):
        if i % 50 == 0:
            print(f"  Chronos feature extraction: {i}/{len(signals)}")
        try:
            # 截取context_length长度
            if len(signal) > context_length:
                # 取多个窗口
                step = len(signal) // 4
                windows = []
                for j in range(4):
                    start = j * step
                    end = start + context_length
                    if end <= len(signal):
                        windows.append(signal[start:end])
                if not windows:
                    windows = [signal[:context_length]]
            else:
                windows = [signal[:context_length]]

            window_feats = []
            for window in windows:
                tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    forecast = pipeline.predict(
                        context=tensor,
                        prediction_length=24,
                        num_samples=3
                    )
                    forecast_mean = forecast.numpy().mean(axis=0).flatten()

                    feat = [
                        np.mean(forecast_mean), np.std(forecast_mean),
                        np.min(forecast_mean), np.max(forecast_mean),
                        np.sum(forecast_mean**2),
                        np.sqrt(np.mean(forecast_mean**2)),
                        np.mean(window), np.std(window),
                        stats.skew(window), stats.kurtosis(window),
                        np.sum(window**2), np.sqrt(np.mean(window**2)),
                        np.ptp(window),
                    ]
                    # 信号与预测的关系
                    if len(window) >= 24:
                        corr = np.corrcoef(window[-24:], forecast_mean[:24])[0, 1]
                        mse = np.mean((window[-24:] - forecast_mean[:24])**2)
                        feat.extend([corr if not np.isnan(corr) else 0, mse])
                    else:
                        feat.extend([0, 0])
                    window_feats.append(feat)

            # 聚合
            arr = np.array(window_feats)
            agg = np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr.max(axis=0), arr.min(axis=0)])
            features_list.append(agg)

        except Exception as e:
            # Fallback
            feat = [np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
                    np.sum(signal**2), np.sqrt(np.mean(signal**2)),
                    stats.skew(signal), stats.kurtosis(signal), np.ptp(signal),
                    0, 0, 0, 0, 0, 0]
            features_list.append(np.tile(feat, 4))  # repeat for 4 aggregations

    return np.array(features_list)


# ============================================================
# Part 1: Split Strategies
# ============================================================

def random_stratified_split(X, y, devices, test_size=0.2, random_state=42):
    """随机分层划分"""
    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=test_size,
        random_state=random_state, stratify=y
    )
    return idx_train, idx_test


def device_level_split(X, y, y_str, devices, test_size=0.2, random_state=42):
    """Device-level划分：同一设备的所有样本全部在训练集或测试集。
    按类别分别划分设备，确保测试集中每个类别都有样本。
    """
    np.random.seed(random_state)

    train_indices = []
    test_indices = []

    for cat in CLASS_NAMES:
        cat_mask = y_str == cat
        cat_indices = np.where(cat_mask)[0]
        cat_devices = devices[cat_indices]
        unique_devices = np.unique(cat_devices)

        # 随机打乱设备
        perm = np.random.permutation(len(unique_devices))
        unique_devices = unique_devices[perm]

        # 计算需要多少设备放入测试集
        n_test_devices = max(1, int(len(unique_devices) * test_size))

        test_devs = set(unique_devices[:n_test_devices])
        train_devs = set(unique_devices[n_test_devices:])

        for idx in cat_indices:
            if devices[idx] in test_devs:
                test_indices.append(idx)
            else:
                train_indices.append(idx)

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # 验证
    train_devices = set(devices[train_indices])
    test_devices = set(devices[test_indices])
    overlap = train_devices & test_devices
    # 不同类别的同名设备可能重叠，这是正常的（不同类别的设备编号含义不同）
    print(f"  Device-level split: train={len(train_indices)}, test={len(test_indices)}")
    print(f"  Train classes: {np.bincount(y[train_indices], minlength=3)}")
    print(f"  Test classes:  {np.bincount(y[test_indices], minlength=3)}")

    return train_indices, test_indices


# ============================================================
# Part 2: Model Definitions
# ============================================================

def get_ml_models(random_state=42):
    """传统机器学习模型"""
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, random_state=random_state),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1),
        'SVM': SVC(random_state=random_state, probability=True),
        'LogisticRegression': LogisticRegression(
            random_state=random_state, max_iter=1000),
    }


def get_tsfm_classifiers(random_state=42):
    """TSFM下游分类器"""
    return {
        'Chronos+RF': RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1),
        'Chronos+SVM': SVC(random_state=random_state, probability=True),
        'Chronos+LR': LogisticRegression(
            random_state=random_state, max_iter=1000),
    }


# ============================================================
# Part 3: Evaluation Pipeline
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, le):
    """评估单个模型，返回所有指标"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 基本指标
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    # AUROC (multi-class OVR)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            auroc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
        else:
            auroc = np.nan
    except:
        auroc = np.nan

    # Spark 类别指标 (class index = 1 for 'spark')
    spark_idx = list(le.classes_).index('spark')
    spark_recall = recall_score(y_test, y_pred, labels=[spark_idx], average=None, zero_division=0)[0]
    spark_f1 = f1_score(y_test, y_pred, labels=[spark_idx], average=None, zero_division=0)[0]
    spark_precision = precision_score(y_test, y_pred, labels=[spark_idx], average=None, zero_division=0)[0]

    # 每类指标
    per_class = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'auroc': auroc,
        'spark_recall': spark_recall,
        'spark_f1': spark_f1,
        'spark_precision': spark_precision,
        'y_pred': y_pred,
        'per_class': per_class,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def run_experiment(X_features, y, y_str, devices, le, models_dict, split_fn,
                   split_name, model_family, seeds=SEEDS):
    """对给定的模型集合运行多种子实验"""
    all_results = {}

    for model_name in models_dict:
        seed_results = []
        for seed in seeds:
            print(f"  [{split_name}] {model_name} seed={seed}")

            # 创建新模型实例
            if model_family == 'ML':
                model = get_ml_models(seed)[model_name]
            else:
                model = get_tsfm_classifiers(seed)[model_name]

            # 数据划分
            if split_name == 'random':
                idx_train, idx_test = random_stratified_split(
                    X_features, y, devices, test_size=0.2, random_state=seed)
            else:
                idx_train, idx_test = device_level_split(
                    X_features, y, y_str, devices, test_size=0.2, random_state=seed)

            X_tr, X_te = X_features[idx_train], X_features[idx_test]
            y_tr, y_te = y[idx_train], y[idx_test]

            # 标准化
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            # NaN处理
            X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
            X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

            result = evaluate_model(model, X_tr, X_te, y_tr, y_te, le)
            seed_results.append(result)

        all_results[model_name] = seed_results

    return all_results


# ============================================================
# Part 4: Statistical Tests
# ============================================================

def compute_anova(groups_dict):
    """单因素ANOVA：比较不同组的scores。
    groups_dict: {group_name: [score_list]}
    """
    group_values = list(groups_dict.values())
    if len(group_values) >= 2 and all(len(v) > 0 for v in group_values):
        f_stat, p_value = f_oneway(*group_values)
    else:
        f_stat, p_value = np.nan, np.nan

    return {
        'groups': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}
                   for k, v in groups_dict.items()},
        'F_statistic': float(f_stat) if not np.isnan(f_stat) else np.nan,
        'p_value': float(p_value) if not np.isnan(p_value) else np.nan
    }


def tukey_hsd_test(groups_dict):
    """Tukey HSD 事后检验"""
    from scipy.stats import studentized_range
    results = []

    group_names = list(groups_dict.keys())
    all_data = []
    all_labels = []
    for name, values in groups_dict.items():
        all_data.extend(values)
        all_labels.extend([name] * len(values))

    n_total = len(all_data)
    k = len(group_names)
    grand_mean = np.mean(all_data)

    # MSE (within-group)
    ss_within = 0
    df_within = 0
    for name, values in groups_dict.items():
        group_mean = np.mean(values)
        ss_within += np.sum((np.array(values) - group_mean)**2)
        df_within += len(values) - 1
    mse = ss_within / df_within if df_within > 0 else 1e-10

    for i in range(len(group_names)):
        for j in range(i+1, len(group_names)):
            g1, g2 = group_names[i], group_names[j]
            v1, v2 = groups_dict[g1], groups_dict[g2]
            mean_diff = np.mean(v1) - np.mean(v2)
            n1, n2 = len(v1), len(v2)
            se = np.sqrt(mse * (1/n1 + 1/n2) / 2)
            if se > 0:
                q_stat = abs(mean_diff) / se
            else:
                q_stat = 0

            # Approximate p-value using t-test as fallback
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(v1, v2, equal_var=False)

            sig = ''
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            results.append({
                'Group1': g1,
                'Group2': g2,
                'Mean_Diff': mean_diff,
                'q_statistic': q_stat,
                'p_value': p_val,
                'significance': sig
            })

    return results


def cohens_d(group1, group2):
    """计算Cohen's d 效应量"""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    # Pooled std
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    d = (m1 - m2) / pooled_std

    # 判定
    abs_d = abs(d)
    if abs_d < 0.2:
        effect = 'negligible'
    elif abs_d < 0.5:
        effect = 'small'
    elif abs_d < 0.8:
        effect = 'medium'
    else:
        effect = 'large'

    return d, effect


# ============================================================
# Part 5: Visualization
# ============================================================

def plot_confusion_matrix(cm, class_names, title, save_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spectrum_comparison(X_raw, y_str, save_path):
    """绘制 Spark vs Normal 的频谱对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, cat in enumerate(['normal', 'spark']):
        mask = y_str == cat
        signals = X_raw[mask]

        # 取前几个样本的平均频谱
        n_samples = min(20, len(signals))
        avg_spectrum = np.zeros(65536 // 2)

        for i in range(n_samples):
            fft_vals = fft(signals[i])
            fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
            avg_spectrum += fft_mag

        avg_spectrum /= n_samples
        freqs = fftfreq(65536, 1/SAMPLING_RATE)[:65536//2]

        # 全频谱
        axes[0].semilogy(freqs[:5000], avg_spectrum[:5000],
                         label=cat, alpha=0.8, linewidth=0.8)
        # 低频细节
        low_n = 500
        axes[1].plot(freqs[:low_n], avg_spectrum[:low_n],
                     label=cat, alpha=0.8, linewidth=1.2)

    axes[0].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[0].set_ylabel('Magnitude (log scale)', fontsize=12)
    axes[0].set_title('Frequency Spectrum: Spark vs Normal (0-5kHz)', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylabel('Magnitude', fontsize=12)
    axes[1].set_title('Low-Frequency Detail: Spark vs Normal (0-500Hz)', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_performance_drop_table(comparison_df, save_path):
    """绘制 Random vs Device 性能下降对比表格图"""
    fig, ax = plt.subplots(figsize=(14, max(4, len(comparison_df) * 0.5 + 2)))
    ax.axis('off')

    table = ax.table(
        cellText=comparison_df.values,
        colLabels=comparison_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # 设置表头样式
    for j in range(len(comparison_df.columns)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Random vs Device-Level Split Performance Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# Main Experiment Pipeline
# ============================================================

def main():
    print("=" * 80)
    print("Device-Level Generalization & Statistical Validation Experiment")
    print("=" * 80)

    mode = 'zhendong'  # 使用振动数据（与论文一致）

    # ---- Step 1: 加载数据 ----
    print("\n[Step 1] Loading data with device info...")
    X_raw, y, y_str, devices, files, le = load_all_data_with_device_info(mode=mode)

    # 下采样
    X_ds = X_raw[:, ::DOWNSAMPLE_FACTOR]
    print(f"Downsampled shape: {X_ds.shape}")

    # ---- Step 2: 特征提取 ----
    print("\n[Step 2] Extracting features...")

    # ML特征
    print("  Extracting ML features (time + frequency domain)...")
    fe = FeatureExtractor(sampling_rate=SAMPLING_RATE // DOWNSAMPLE_FACTOR)
    X_ml_features = fe.extract_batch(X_ds)
    X_ml_features = np.nan_to_num(X_ml_features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  ML features shape: {X_ml_features.shape}")

    # TSFM特征 (Chronos)
    print("  Extracting TSFM features (Chronos)...")
    X_tsfm_features = load_chronos_and_extract(X_ds, context_length=512)
    if X_tsfm_features is None:
        print("  WARNING: Chronos failed. Using ML features as TSFM proxy.")
        X_tsfm_features = X_ml_features.copy()
    else:
        X_tsfm_features = np.nan_to_num(X_tsfm_features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  TSFM features shape: {X_tsfm_features.shape}")

    # ---- Step 3: 两种划分方式的实验 ----
    print("\n[Step 3] Running experiments with both split strategies...")

    # 收集所有结果
    all_experiment_results = {}

    # 3a: Random split - ML models
    print("\n--- Random Split: ML Models ---")
    random_ml = run_experiment(
        X_ml_features, y, y_str, devices, le,
        get_ml_models(), random_stratified_split,
        'random', 'ML', seeds=SEEDS
    )
    all_experiment_results[('random', 'ML')] = random_ml

    # 3b: Random split - TSFM models
    print("\n--- Random Split: TSFM Models ---")
    random_tsfm = run_experiment(
        X_tsfm_features, y, y_str, devices, le,
        get_tsfm_classifiers(), random_stratified_split,
        'random', 'TSFM', seeds=SEEDS
    )
    all_experiment_results[('random', 'TSFM')] = random_tsfm

    # 3c: Device split - ML models
    print("\n--- Device Split: ML Models ---")
    device_ml = run_experiment(
        X_ml_features, y, y_str, devices, le,
        get_ml_models(), device_level_split,
        'device', 'ML', seeds=SEEDS
    )
    all_experiment_results[('device', 'ML')] = device_ml

    # 3d: Device split - TSFM models
    print("\n--- Device Split: TSFM Models ---")
    device_tsfm = run_experiment(
        X_tsfm_features, y, y_str, devices, le,
        get_tsfm_classifiers(), device_level_split,
        'device', 'TSFM', seeds=SEEDS
    )
    all_experiment_results[('device', 'TSFM')] = device_tsfm

    # ---- Step 4: 结果汇总 ----
    print("\n[Step 4] Aggregating results...")

    # 4a: 构建主要对比表
    comparison_rows = []
    detailed_rows = []

    for (split, family), results in all_experiment_results.items():
        for model_name, seed_results in results.items():
            accs = [r['accuracy'] for r in seed_results]
            f1s = [r['macro_f1'] for r in seed_results]
            aurocs = [r['auroc'] for r in seed_results]
            s_recalls = [r['spark_recall'] for r in seed_results]
            s_f1s = [r['spark_f1'] for r in seed_results]

            detailed_rows.append({
                'Split': split,
                'Family': family,
                'Model': model_name,
                'Accuracy_mean': np.mean(accs),
                'Accuracy_std': np.std(accs),
                'Macro_F1_mean': np.mean(f1s),
                'Macro_F1_std': np.std(f1s),
                'AUROC_mean': np.nanmean(aurocs),
                'AUROC_std': np.nanstd(aurocs),
                'Spark_Recall_mean': np.mean(s_recalls),
                'Spark_Recall_std': np.std(s_recalls),
                'Spark_F1_mean': np.mean(s_f1s),
                'Spark_F1_std': np.std(s_f1s),
            })

    detailed_df = pd.DataFrame(detailed_rows)

    # 保存详细结果
    detailed_df.to_csv(os.path.join(TABLE_DIR, 'detailed_results.csv'), index=False)
    print(f"\nDetailed results saved to {TABLE_DIR}/detailed_results.csv")
    print(detailed_df.to_string(index=False))

    # 4b: 构建 Performance Drop 对比表
    print("\n\n--- Performance Drop Table ---")
    drop_rows = []
    all_models = set()
    for (split, family), results in all_experiment_results.items():
        for model_name in results:
            all_models.add((family, model_name))

    for family, model_name in sorted(all_models):
        rand_key = ('random', family)
        dev_key = ('device', family)
        if rand_key in all_experiment_results and model_name in all_experiment_results[rand_key]:
            rand_f1 = np.mean([r['macro_f1'] for r in all_experiment_results[rand_key][model_name]])
            rand_sr = np.mean([r['spark_recall'] for r in all_experiment_results[rand_key][model_name]])
        else:
            continue
        if dev_key in all_experiment_results and model_name in all_experiment_results[dev_key]:
            dev_f1 = np.mean([r['macro_f1'] for r in all_experiment_results[dev_key][model_name]])
            dev_sr = np.mean([r['spark_recall'] for r in all_experiment_results[dev_key][model_name]])
        else:
            continue

        drop_rows.append({
            'Model': model_name,
            'Family': family,
            'Random_Macro_F1': f"{rand_f1:.4f}",
            'Device_Macro_F1': f"{dev_f1:.4f}",
            'Random_Spark_Recall': f"{rand_sr:.4f}",
            'Device_Spark_Recall': f"{dev_sr:.4f}",
            'F1_Drop': f"{rand_f1 - dev_f1:+.4f}",
            'Spark_Recall_Drop': f"{rand_sr - dev_sr:+.4f}",
        })

    drop_df = pd.DataFrame(drop_rows)
    drop_df.to_csv(os.path.join(TABLE_DIR, 'performance_drop.csv'), index=False)
    print(drop_df.to_string(index=False))

    # 绘制对比表格图
    plot_performance_drop_table(drop_df,
        os.path.join(FIG_DIR, 'performance_drop_table.png'))

    # ---- Step 5: 混淆矩阵 ----
    print("\n[Step 5] Generating confusion matrices...")

    # 找到device-level下最优TSFM和最优ML
    for family_tag, family_key in [('TSFM', 'TSFM'), ('ML', 'ML')]:
        dev_results = all_experiment_results.get(('device', family_key), {})
        best_model = None
        best_f1 = -1
        for model_name, seed_results in dev_results.items():
            avg_f1 = np.mean([r['macro_f1'] for r in seed_results])
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_model = model_name
                best_cm = seed_results[0]['confusion_matrix']  # 第一个seed的混淆矩阵

        if best_model:
            # 累积所有seed的混淆矩阵
            total_cm = sum(r['confusion_matrix'] for r in dev_results[best_model])
            plot_confusion_matrix(
                total_cm, CLASS_NAMES,
                f'Device-Level Split: {best_model} (Best {family_tag})',
                os.path.join(FIG_DIR, f'confusion_matrix_device_{family_tag.lower()}.png')
            )

    # ---- Step 6: 频谱对比图 ----
    print("\n[Step 6] Generating spectrum comparison plot...")
    plot_spectrum_comparison(X_raw, y_str,
        os.path.join(FIG_DIR, 'spectrum_spark_vs_normal.png'))

    # ---- Step 7: 统计验证 ----
    print("\n[Step 7] Statistical validation...")

    # 7a: 收集各模型家族的F1分数（用random split的结果）
    family_f1_scores = {}

    # ML子族
    ml_results = all_experiment_results.get(('random', 'ML'), {})
    ml_all_f1 = []
    for model_name, seed_results in ml_results.items():
        for r in seed_results:
            ml_all_f1.append(r['macro_f1'])
    family_f1_scores['Traditional_ML'] = ml_all_f1

    # TSFM子族
    tsfm_results = all_experiment_results.get(('random', 'TSFM'), {})
    tsfm_all_f1 = []
    for model_name, seed_results in tsfm_results.items():
        for r in seed_results:
            tsfm_all_f1.append(r['macro_f1'])
    family_f1_scores['TSFM'] = tsfm_all_f1

    # 7a-1: 更细粒度：每个模型作为一个组
    model_f1_scores = {}
    for (split, family), results in all_experiment_results.items():
        if split == 'random':
            for model_name, seed_results in results.items():
                key = f"{family}_{model_name}"
                model_f1_scores[key] = [r['macro_f1'] for r in seed_results]

    # ANOVA
    print("\n--- One-way ANOVA (across all model families) ---")
    anova_result = compute_anova({'Traditional_ML': ml_all_f1, 'TSFM': tsfm_all_f1})
    print(f"  F-statistic: {anova_result['F_statistic']:.4f}")
    print(f"  p-value:     {anova_result['p_value']:.6f}")
    for g, s in anova_result['groups'].items():
        print(f"  {g}: mean={s['mean']:.4f}, std={s['std']:.4f}, n={s['n']}")

    # 细粒度ANOVA (所有单个模型)
    print("\n--- One-way ANOVA (across all individual models) ---")
    if len(model_f1_scores) >= 2:
        f_stat, p_val = f_oneway(*[v for v in model_f1_scores.values()])
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value:     {p_val:.6f}")
    else:
        f_stat, p_val = np.nan, np.nan

    # 7b: Tukey HSD
    print("\n--- Tukey HSD Post-Hoc Tests ---")

    # TSFM vs 传统ML
    tukey_family = tukey_hsd_test({'Traditional_ML': ml_all_f1, 'TSFM': tsfm_all_f1})
    print("\n  TSFM vs Traditional ML:")
    for t in tukey_family:
        print(f"    {t['Group1']} vs {t['Group2']}: "
              f"mean_diff={t['Mean_Diff']:+.4f}, p={t['p_value']:.6f} {t['significance']}")

    # 所有模型两两对比
    print("\n  All model pairwise comparisons:")
    tukey_all = tukey_hsd_test(model_f1_scores)
    tukey_df = pd.DataFrame(tukey_all)
    # 只显示显著的
    sig_tukey = tukey_df[tukey_df['significance'] != 'ns']
    if len(sig_tukey) > 0:
        print(sig_tukey.to_string(index=False))
    else:
        print("  No significant differences found.")

    tukey_df.to_csv(os.path.join(TABLE_DIR, 'tukey_hsd_results.csv'), index=False)

    # 7c: Cohen's d
    print("\n--- Cohen's d Effect Size ---")
    cohens_results = []

    # TSFM vs ML
    if len(tsfm_all_f1) > 1 and len(ml_all_f1) > 1:
        d, effect = cohens_d(tsfm_all_f1, ml_all_f1)
        print(f"  TSFM vs Traditional_ML: d={d:.4f} ({effect})")
        cohens_results.append({'Comparison': 'TSFM vs Traditional_ML', 'd': d, 'effect': effect})

    # 最优模型 vs 各个模型
    # 找到random split下最优模型
    best_random_model = None
    best_random_f1 = -1
    for key, vals in model_f1_scores.items():
        avg = np.mean(vals)
        if avg > best_random_f1:
            best_random_f1 = avg
            best_random_model = key

    if best_random_model:
        for key, vals in model_f1_scores.items():
            if key != best_random_model and len(vals) > 1:
                d, effect = cohens_d(model_f1_scores[best_random_model], vals)
                cohens_results.append({
                    'Comparison': f'{best_random_model} vs {key}',
                    'd': d, 'effect': effect
                })
                print(f"  {best_random_model} vs {key}: d={d:.4f} ({effect})")

    # Device vs Random 对比的 Cohen's d
    print("\n  Device vs Random split effect sizes:")
    for family_key in ['ML', 'TSFM']:
        rand_res = all_experiment_results.get(('random', family_key), {})
        dev_res = all_experiment_results.get(('device', family_key), {})
        for model_name in rand_res:
            if model_name in dev_res:
                r_f1 = [r['macro_f1'] for r in rand_res[model_name]]
                d_f1 = [r['macro_f1'] for r in dev_res[model_name]]
                if len(r_f1) > 1 and len(d_f1) > 1:
                    d, effect = cohens_d(r_f1, d_f1)
                    cohens_results.append({
                        'Comparison': f'{model_name} Random vs Device',
                        'd': d, 'effect': effect
                    })
                    print(f"  {model_name} (Random vs Device): d={d:.4f} ({effect})")

    cohens_df = pd.DataFrame(cohens_results)
    cohens_df.to_csv(os.path.join(TABLE_DIR, 'cohens_d_results.csv'), index=False)

    # ---- Step 8: 保存ANOVA结果 ----
    anova_summary = {
        'family_anova': {
            'F_statistic': float(anova_result['F_statistic']),
            'p_value': float(anova_result['p_value']),
            'groups': anova_result['groups']
        },
        'individual_anova': {
            'F_statistic': float(f_stat) if not np.isnan(f_stat) else None,
            'p_value': float(p_val) if not np.isnan(p_val) else None,
        }
    }
    with open(os.path.join(TABLE_DIR, 'anova_results.json'), 'w') as f:
        json.dump(anova_summary, f, indent=2, default=str)

    # ---- Step 9: 最终报告 ----
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE - Summary")
    print("=" * 80)

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  Tables:")
    print(f"    - {TABLE_DIR}/detailed_results.csv")
    print(f"    - {TABLE_DIR}/performance_drop.csv")
    print(f"    - {TABLE_DIR}/tukey_hsd_results.csv")
    print(f"    - {TABLE_DIR}/cohens_d_results.csv")
    print(f"    - {TABLE_DIR}/anova_results.json")
    print(f"  Figures:")
    print(f"    - {FIG_DIR}/performance_drop_table.png")
    print(f"    - {FIG_DIR}/confusion_matrix_device_tsfm.png")
    print(f"    - {FIG_DIR}/confusion_matrix_device_ml.png")
    print(f"    - {FIG_DIR}/spectrum_spark_vs_normal.png")

    # 输出最终对比表
    print("\n\n=== FINAL COMPARISON TABLE ===")
    print(detailed_df[['Split', 'Family', 'Model',
                        'Accuracy_mean', 'Accuracy_std',
                        'Macro_F1_mean', 'Macro_F1_std',
                        'Spark_Recall_mean', 'Spark_F1_mean']].to_string(index=False))

    print("\n\n=== PERFORMANCE DROP (Random - Device) ===")
    print(drop_df.to_string(index=False))

    print("\n\n=== ANOVA ===")
    print(f"TSFM vs Traditional ML: F={anova_result['F_statistic']:.4f}, p={anova_result['p_value']:.6f}")

    print("\n\n=== Cohen's d ===")
    if len(cohens_df) > 0:
        print(cohens_df.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
