#!/usr/bin/env python3
"""
特征提取模块
提取时域、频域和时频域特征
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis, entropy
import pywt
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path

class FeatureExtractor:
    """特征提取器"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.fs = self.config['data']['sampling_rate']
        self.output_path = Path(self.config['output']['tables'])
        self.image_path = Path(self.config['output']['images'])

    def extract_time_domain_features(self, signal: np.ndarray) -> Dict:
        """提取时域特征"""
        features = {}

        # 基本统计特征
        features['mean'] = float(np.mean(signal))
        features['std'] = float(np.std(signal))
        features['var'] = float(np.var(signal))
        features['skewness'] = float(skew(signal))
        features['kurtosis'] = float(kurtosis(signal))
        features['min'] = float(np.min(signal))
        features['max'] = float(np.max(signal))
        features['median'] = float(np.median(signal))

        # 能量特征
        features['rms'] = float(np.sqrt(np.mean(signal**2)))
        features['energy'] = float(np.sum(signal**2))
        features['power'] = features['energy'] / len(signal)

        # 形状特征
        features['peak_to_peak'] = float(np.ptp(signal))
        features['crest_factor'] = float(np.max(np.abs(signal)) / features['rms']) if features['rms'] != 0 else 0
        features['clearance_factor'] = float(np.max(np.abs(signal)) / np.mean(np.sqrt(np.abs(signal)))**2) if np.mean(np.sqrt(np.abs(signal))) != 0 else 0
        features['shape_factor'] = float(features['rms'] / np.mean(np.abs(signal))) if np.mean(np.abs(signal)) != 0 else 0
        features['impulse_factor'] = float(np.max(np.abs(signal)) / np.mean(np.abs(signal))) if np.mean(np.abs(signal)) != 0 else 0

        # 分位数特征
        features['q25'] = float(np.percentile(signal, 25))
        features['q75'] = float(np.percentile(signal, 75))
        features['iqr'] = features['q75'] - features['q25']

        # 零交叉率
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features['zero_crossing_rate'] = float(len(zero_crossings) / len(signal))

        return features

    def extract_frequency_domain_features(self, signal: np.ndarray) -> Dict:
        """提取频域特征"""
        features = {}

        # FFT分析
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal), 1/self.fs)
        magnitude = np.abs(fft_vals)
        power_spectrum = magnitude**2

        # 只取正频率部分
        positive_idx = freqs > 0
        freqs_pos = freqs[positive_idx]
        magnitude_pos = magnitude[positive_idx]
        power_pos = power_spectrum[positive_idx]

        # 归一化功率谱
        total_power = np.sum(power_pos)
        if total_power > 0:
            normalized_power = power_pos / total_power
        else:
            normalized_power = power_pos

        # 频域统计特征
        features['spectral_centroid'] = float(np.sum(freqs_pos * normalized_power))
        features['spectral_bandwidth'] = float(np.sqrt(np.sum(((freqs_pos - features['spectral_centroid'])**2) * normalized_power)))
        features['spectral_rolloff'] = float(self._calculate_spectral_rolloff(freqs_pos, normalized_power))
        features['spectral_flux'] = float(np.sum(np.diff(magnitude_pos)**2))

        # 主频率
        dominant_idx = np.argmax(magnitude_pos)
        features['dominant_frequency'] = float(freqs_pos[dominant_idx])
        features['dominant_magnitude'] = float(magnitude_pos[dominant_idx])

        # 频带能量分布
        freq_bands = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 10000)]
        for i, (low, high) in enumerate(freq_bands):
            band_mask = (freqs_pos >= low) & (freqs_pos < high)
            features[f'band_{i}_energy'] = float(np.sum(power_pos[band_mask]))
            features[f'band_{i}_ratio'] = features[f'band_{i}_energy'] / total_power if total_power > 0 else 0

        # 频谱熵
        if total_power > 0:
            features['spectral_entropy'] = float(entropy(normalized_power + 1e-12))
        else:
            features['spectral_entropy'] = 0.0

        return features

    def _calculate_spectral_rolloff(self, freqs: np.ndarray, power: np.ndarray, rolloff_percent: float = 0.85) -> float:
        """计算频谱滚降点"""
        cumulative_power = np.cumsum(power)
        total_power = cumulative_power[-1]
        rolloff_threshold = rolloff_percent * total_power

        rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1]

    def extract_time_frequency_features(self, signal: np.ndarray) -> Dict:
        """提取时频域特征"""
        features = {}

        # 小波变换特征
        try:
            # 使用db4小波进行多层分解
            coeffs = pywt.wavedec(signal, 'db4', level=5)

            for i, coeff in enumerate(coeffs):
                features[f'wavelet_energy_level_{i}'] = float(np.sum(coeff**2))
                features[f'wavelet_std_level_{i}'] = float(np.std(coeff))
                features[f'wavelet_mean_level_{i}'] = float(np.mean(np.abs(coeff)))

            # 小波包能量
            total_wavelet_energy = sum(features[f'wavelet_energy_level_{i}'] for i in range(len(coeffs)))
            for i in range(len(coeffs)):
                features[f'wavelet_energy_ratio_level_{i}'] = features[f'wavelet_energy_level_{i}'] / total_wavelet_energy if total_wavelet_energy > 0 else 0

        except Exception as e:
            print(f"小波变换失败: {e}")
            # 如果小波变换失败，设置默认值
            for i in range(6):
                features[f'wavelet_energy_level_{i}'] = 0.0
                features[f'wavelet_std_level_{i}'] = 0.0
                features[f'wavelet_mean_level_{i}'] = 0.0
                features[f'wavelet_energy_ratio_level_{i}'] = 0.0

        # 短时傅里叶变换特征
        try:
            f, t, Zxx = signal.stft(signal, fs=self.fs, nperseg=min(256, len(signal)//4))
            stft_magnitude = np.abs(Zxx)

            # STFT统计特征
            features['stft_mean'] = float(np.mean(stft_magnitude))
            features['stft_std'] = float(np.std(stft_magnitude))
            features['stft_max'] = float(np.max(stft_magnitude))
            features['stft_energy'] = float(np.sum(stft_magnitude**2))

        except Exception as e:
            print(f"STFT计算失败: {e}")
            features['stft_mean'] = 0.0
            features['stft_std'] = 0.0
            features['stft_max'] = 0.0
            features['stft_energy'] = 0.0

        return features

    def extract_all_features(self, signal: np.ndarray) -> Dict:
        """提取所有特征"""
        features = {}

        # 时域特征
        time_features = self.extract_time_domain_features(signal)
        features.update({f'time_{k}': v for k, v in time_features.items()})

        # 频域特征
        freq_features = self.extract_frequency_domain_features(signal)
        features.update({f'freq_{k}': v for k, v in freq_features.items()})

        # 时频域特征
        time_freq_features = self.extract_time_frequency_features(signal)
        features.update({f'tf_{k}': v for k, v in time_freq_features.items()})

        return features

    def extract_features_from_dataset(self, data: Dict) -> Tuple[pd.DataFrame, Dict]:
        """从整个数据集提取特征"""
        print("🔧 开始特征提取...")

        all_features = []
        all_labels = []
        all_sensors = []
        all_files = []

        feature_extraction_log = {}

        for sensor in data.keys():
            feature_extraction_log[sensor] = {}

            for state in data[sensor].keys():
                signals = data[sensor][state]
                print(f"  提取 {sensor}/{state}: {len(signals)} 个信号")

                state_features = []
                failed_count = 0

                for i, signal in enumerate(signals):
                    try:
                        features = self.extract_all_features(signal)
                        all_features.append(features)
                        all_labels.append(state)
                        all_sensors.append(sensor)
                        all_files.append(f"{sensor}_{state}_{i}")
                        state_features.append(features)

                    except Exception as e:
                        print(f"    ❌ 特征提取失败 {sensor}/{state} 信号 {i}: {e}")
                        failed_count += 1
                        continue

                feature_extraction_log[sensor][state] = {
                    'total_signals': len(signals),
                    'successful': len(state_features),
                    'failed': failed_count,
                    'success_rate': len(state_features) / len(signals) if len(signals) > 0 else 0
                }

                print(f"    ✅ 成功提取 {len(state_features)}/{len(signals)} 个信号特征")

        # 创建特征DataFrame
        features_df = pd.DataFrame(all_features)
        features_df['label'] = all_labels
        features_df['sensor'] = all_sensors
        features_df['file_id'] = all_files

        # 保存特征数据
        features_path = self.output_path / 'extracted_features.csv'
        features_df.to_csv(features_path, index=False)
        print(f"💾 特征数据已保存: {features_path}")

        # 保存提取日志
        import json
        from datetime import datetime

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(features_df.columns) - 3,  # 减去label, sensor, file_id
            'total_samples': len(features_df),
            'extraction_log': feature_extraction_log
        }

        log_path = self.output_path / 'feature_extraction_log.json'
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"📋 提取日志已保存: {log_path}")

        print(f"✅ 特征提取完成！共提取 {len(features_df)} 个样本，{len(features_df.columns)-3} 个特征")

        return features_df, feature_extraction_log

    def analyze_feature_importance(self, features_df: pd.DataFrame):
        """分析特征重要性"""
        print("📊 分析特征重要性...")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split

        # 准备数据
        feature_cols = [col for col in features_df.columns if col not in ['label', 'sensor', 'file_id']]
        X = features_df[feature_cols]
        y = features_df['label']

        # 处理缺失值和无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # 编码标签
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # 训练随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y_encoded)

        # 获取特征重要性
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # 保存特征重要性
        importance_path = self.output_path / 'feature_importance.csv'
        feature_importance_df.to_csv(importance_path, index=False)
        print(f"📊 特征重要性已保存: {importance_path}")

        # 可视化特征重要性
        self._plot_feature_importance(feature_importance_df)

        return feature_importance_df

    def _plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20):
        """绘制特征重要性图"""
        plt.figure(figsize=(12, 8))

        top_features = importance_df.head(top_n)

        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        save_path = self.image_path / 'feature_analysis' / 'feature_importance.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 特征重要性图已保存: {save_path}")

    def visualize_feature_distributions(self, features_df: pd.DataFrame):
        """可视化特征分布"""
        print("🎨 生成特征分布可视化...")

        # 选择重要特征进行可视化
        feature_cols = [col for col in features_df.columns if col not in ['label', 'sensor', 'file_id']]

        # 按类别分组的特征
        time_features = [col for col in feature_cols if col.startswith('time_')]
        freq_features = [col for col in feature_cols if col.startswith('freq_')]
        tf_features = [col for col in feature_cols if col.startswith('tf_')]

        # 可视化时域特征
        self._plot_feature_category(features_df, time_features[:12], 'Time Domain Features')

        # 可视化频域特征
        self._plot_feature_category(features_df, freq_features[:12], 'Frequency Domain Features')

        # 可视化时频域特征
        self._plot_feature_category(features_df, tf_features[:12], 'Time-Frequency Features')

        print("✅ 特征分布可视化完成")

    def _plot_feature_category(self, features_df: pd.DataFrame, feature_list: List[str], category_name: str):
        """绘制特定类别的特征分布"""
        if not feature_list:
            return

        n_features = len(feature_list)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        colors = ['green', 'orange', 'red']
        states = features_df['label'].unique()

        for i, feature in enumerate(feature_list):
            row = i // n_cols
            col = i % n_cols

            for j, state in enumerate(states):
                state_data = features_df[features_df['label'] == state][feature]
                # 移除异常值
                state_data = state_data[np.abs(state_data - state_data.mean()) <= 3 * state_data.std()]

                axes[row, col].hist(state_data, bins=20, alpha=0.7,
                                  label=state, color=colors[j % len(colors)], density=True)

            axes[row, col].set_title(feature.replace('_', ' ').title())
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

        plt.suptitle(category_name, fontsize=16)
        plt.tight_layout()

        # 保存图片
        filename = category_name.lower().replace(' ', '_').replace('-', '_') + '.png'
        save_path = self.image_path / 'feature_analysis' / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 {category_name}分布图已保存: {save_path}")

if __name__ == "__main__":
    # 测试特征提取器
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
    from data_loader import MotorDataLoader

    # 配置路径
    config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"

    # 加载数据
    loader = MotorDataLoader(str(config_path))
    data, _ = loader.load_all_data(max_files_per_state=20)

    # 创建特征提取器
    extractor = FeatureExtractor(str(config_path))

    # 提取特征
    features_df, extraction_log = extractor.extract_features_from_dataset(data)

    # 分析特征重要性
    importance_df = extractor.analyze_feature_importance(features_df)

    # 可视化特征分布
    extractor.visualize_feature_distributions(features_df)

    print("\n🎉 特征工程测试完成！")
    print(f"📊 提取特征数量: {len(features_df.columns)-3}")
    print(f"📋 样本数量: {len(features_df)}")
    print(f"📁 结果保存在: {extractor.output_path} 和 {extractor.image_path}")

    # 显示前5个重要特征
    print(f"\n🏆 前5个重要特征:")
    for i, row in importance_df.head().iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
