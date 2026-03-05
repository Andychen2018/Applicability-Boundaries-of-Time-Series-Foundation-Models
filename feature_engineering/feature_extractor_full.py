"""
完整特征提取模块
从时间序列信号中提取丰富的统计特征（55+个特征）
复制自原始实验的01_statistical_ml_models.py
"""

import numpy as np
from typing import Dict
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, hilbert
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractorFull:
    """
    完整特征提取器
    提取时域、频域、时频域和非线性特征（55+个特征）
    """
    
    def __init__(self, sampling_rate=65535):
        self.sampling_rate = sampling_rate
    
    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """提取时域特征"""
        features = {}
        
        # 基本统计特征
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(signal)
        
        # 高阶统计特征
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        
        # 能量特征
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['energy'] = np.sum(signal**2)
        features['power'] = features['energy'] / len(signal)
        
        # 形状特征
        features['peak_to_peak'] = np.ptp(signal)
        features['crest_factor'] = features['max'] / features['rms'] if features['rms'] != 0 else 0
        features['form_factor'] = features['rms'] / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0
        
        # 零交叉率
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)

        # Hjorth参数
        hjorth_features = self._hjorth_parameters(signal)
        features.update(hjorth_features)

        # 样本熵和排列熵
        features['sample_entropy'] = self._sample_entropy(signal)
        features['permutation_entropy'] = self._permutation_entropy(signal)

        # 子窗口特征
        subwindow_features = self._subwindow_features(signal)
        features.update(subwindow_features)

        return features
    
    def extract_frequency_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """提取频域特征 - 充分利用高采样率优势"""
        features = {}

        # 使用完整的高采样率数据，但进行智能分析
        fft_vals = fft(signal)
        fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = fftfreq(len(signal), 1/self.sampling_rate)[:len(fft_vals)//2]

        # 避免除零错误
        if np.sum(fft_magnitude) == 0:
            return self._get_zero_features()

        # 基础频域统计特征
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * fft_magnitude) / np.sum(fft_magnitude))

        # 谱滚降
        cumsum_magnitude = np.cumsum(fft_magnitude)
        rolloff_idx = np.where(cumsum_magnitude >= 0.85 * np.sum(fft_magnitude))[0]
        features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

        # 主频率和幅值
        dominant_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = freqs[dominant_idx]
        features['dominant_magnitude'] = fft_magnitude[dominant_idx]

        # 多尺度频带能量分析 - 充分利用高采样率
        # 低频带（电机基频及其谐波）
        low_freq_bands = [(0, 50), (50, 100), (100, 200), (200, 500)]
        for i, (low, high) in enumerate(low_freq_bands):
            band_mask = (freqs >= low) & (freqs < high)
            features[f'low_band_{i}_energy'] = np.sum(fft_magnitude[band_mask]**2)
            features[f'low_band_{i}_peak'] = np.max(fft_magnitude[band_mask]) if np.any(band_mask) else 0

        # 中频带（机械振动特征）
        mid_freq_bands = [(500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
        for i, (low, high) in enumerate(mid_freq_bands):
            band_mask = (freqs >= low) & (freqs < high)
            features[f'mid_band_{i}_energy'] = np.sum(fft_magnitude[band_mask]**2)
            features[f'mid_band_{i}_peak'] = np.max(fft_magnitude[band_mask]) if np.any(band_mask) else 0

        # 高频带（轴承故障、电气故障特征）
        high_freq_bands = [(10000, 15000), (15000, 20000), (20000, 25000), (25000, 32000)]
        for i, (low, high) in enumerate(high_freq_bands):
            band_mask = (freqs >= low) & (freqs < high)
            features[f'high_band_{i}_energy'] = np.sum(fft_magnitude[band_mask]**2)
            features[f'high_band_{i}_peak'] = np.max(fft_magnitude[band_mask]) if np.any(band_mask) else 0

        # 频谱形状特征
        features['spectral_flatness'] = stats.gmean(fft_magnitude + 1e-10) / np.mean(fft_magnitude + 1e-10)
        features['spectral_slope'] = self._calculate_spectral_slope(freqs, fft_magnitude)

        # 能量分布特征
        total_energy = np.sum(fft_magnitude**2)
        features['low_freq_ratio'] = sum([features[f'low_band_{i}_energy'] for i in range(4)]) / total_energy
        features['mid_freq_ratio'] = sum([features[f'mid_band_{i}_energy'] for i in range(4)]) / total_energy
        features['high_freq_ratio'] = sum([features[f'high_band_{i}_energy'] for i in range(4)]) / total_energy

        # 高级频域特征
        advanced_freq_features = self._extract_advanced_frequency_features(signal, fft_magnitude, freqs)
        features.update(advanced_freq_features)

        return features

    def _get_zero_features(self):
        """返回零特征字典"""
        features = {
            'spectral_centroid': 0, 'spectral_bandwidth': 0, 'spectral_rolloff': 0,
            'dominant_frequency': 0, 'dominant_magnitude': 0, 'spectral_flatness': 0,
            'spectral_slope': 0, 'low_freq_ratio': 0, 'mid_freq_ratio': 0, 'high_freq_ratio': 0
        }
        # 添加频带特征
        for prefix in ['low_band', 'mid_band', 'high_band']:
            for i in range(4):
                features[f'{prefix}_{i}_energy'] = 0
                features[f'{prefix}_{i}_peak'] = 0
        return features

    def _calculate_spectral_slope(self, freqs, magnitude):
        """计算频谱斜率"""
        if len(freqs) < 2:
            return 0
        # 使用线性回归计算斜率
        log_freqs = np.log(freqs[1:] + 1e-10)  # 避免log(0)
        log_magnitude = np.log(magnitude[1:] + 1e-10)
        slope = np.polyfit(log_freqs, log_magnitude, 1)[0]
        return slope
    
    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        """提取所有特征"""
        time_features = self.extract_time_domain_features(signal)
        freq_features = self.extract_frequency_domain_features(signal)

        all_features = {**time_features, **freq_features}
        return all_features

    def _hjorth_parameters(self, signal):
        """计算Hjorth参数：Activity, Mobility, Complexity"""
        features = {}

        # Activity (方差)
        activity = np.var(signal)
        features['hjorth_activity'] = activity

        # Mobility (一阶导数的标准差与信号标准差的比值)
        if len(signal) > 1:
            diff1 = np.diff(signal)
            mobility = np.std(diff1) / np.std(signal) if np.std(signal) != 0 else 0
            features['hjorth_mobility'] = mobility

            # Complexity (二阶导数的mobility与一阶导数的mobility的比值)
            if len(diff1) > 1:
                diff2 = np.diff(diff1)
                mobility2 = np.std(diff2) / np.std(diff1) if np.std(diff1) != 0 else 0
                complexity = mobility2 / mobility if mobility != 0 else 0
                features['hjorth_complexity'] = complexity
            else:
                features['hjorth_complexity'] = 0
        else:
            features['hjorth_mobility'] = 0
            features['hjorth_complexity'] = 0

        return features

    def _sample_entropy(self, signal, m=2, r=None):
        """计算样本熵 - 简化版本"""
        try:
            if r is None:
                r = 0.2 * np.std(signal)

            N = len(signal)
            if N < 100:  # 对于短信号，返回简单的熵估计
                return -np.sum(np.histogram(signal, bins=10)[0] / N * np.log(np.histogram(signal, bins=10)[0] / N + 1e-10))

            # 简化的样本熵计算，只使用前1000个点
            signal_short = signal[:1000] if len(signal) > 1000 else signal
            N = len(signal_short)

            # 简化的模式匹配
            matches_m = 0
            matches_m1 = 0

            for i in range(N - m):
                template = signal_short[i:i + m]
                for j in range(i + 1, N - m):
                    if np.max(np.abs(template - signal_short[j:j + m])) <= r:
                        matches_m += 1
                        if i < N - m - 1 and j < N - m - 1:
                            if abs(signal_short[i + m] - signal_short[j + m]) <= r:
                                matches_m1 += 1

            if matches_m > 0 and matches_m1 > 0:
                return np.log(matches_m / matches_m1)
            else:
                return 0
        except:
            return 0

    def _permutation_entropy(self, signal, order=3, delay=1):
        """计算排列熵 - 简化版本"""
        try:
            # 简化版本，只使用前500个点
            signal_short = signal[:500] if len(signal) > 500 else signal
            N = len(signal_short)

            if N < order * delay:
                return 0

            # 创建嵌入向量
            embedded = []
            for i in range(N - (order-1)*delay):
                embedded.append(signal_short[i:i+order*delay:delay])

            if len(embedded) == 0:
                return 0

            # 简化的排列模式计算
            patterns = []
            for emb in embedded:
                pattern = tuple(np.argsort(emb))
                patterns.append(pattern)

            # 计算模式频率
            pattern_counts = Counter(patterns)

            # 计算概率和熵
            total = len(patterns)
            entropy = 0
            for count in pattern_counts.values():
                prob = count / total
                if prob > 0:
                    entropy -= prob * np.log2(prob)

            return entropy
        except:
            return 0

    def _subwindow_features(self, signal, n_windows=10):
        """计算子窗口特征"""
        features = {}

        window_size = len(signal) // n_windows
        if window_size < 1:
            return {'subwindow_rms_max': 0, 'subwindow_kurtosis_max': 0}

        rms_values = []
        kurtosis_values = []

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(signal))
            window = signal[start_idx:end_idx]

            if len(window) > 0:
                rms_values.append(np.sqrt(np.mean(window**2)))
                kurtosis_values.append(stats.kurtosis(window))

        features['subwindow_rms_max'] = np.max(rms_values) if rms_values else 0
        features['subwindow_kurtosis_max'] = np.max(kurtosis_values) if kurtosis_values else 0

        return features

    def _extract_advanced_frequency_features(self, signal, fft_magnitude, freqs):
        """提取高级频域特征"""
        features = {}

        # 谱峭度 (Spectral Kurtosis)
        features['spectral_kurtosis'] = self._spectral_kurtosis(signal)

        # 倒谱特征
        cepstrum_features = self._cepstrum_features(signal)
        features.update(cepstrum_features)

        # 包络谱特征
        envelope_features = self._envelope_spectrum_features(signal)
        features.update(envelope_features)

        # 谐波能量比
        features['harmonic_energy_ratio'] = self._harmonic_energy_ratio(fft_magnitude, freqs)

        # 带通能量比
        bandpass_features = self._bandpass_energy_ratios(fft_magnitude, freqs)
        features.update(bandpass_features)

        return features

    def _spectral_kurtosis(self, signal):
        """计算谱峭度 - 简化版本"""
        try:
            # 简化的谱峭度计算
            fft_vals = fft(signal)
            fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])

            # 计算频谱的峭度
            if len(fft_magnitude) > 3:
                return stats.kurtosis(fft_magnitude)
            else:
                return 0
        except:
            return 0

    def _cepstrum_features(self, signal):
        """计算倒谱特征"""
        features = {}
        try:
            # 计算倒谱
            fft_signal = fft(signal)
            log_spectrum = np.log(np.abs(fft_signal) + 1e-10)
            cepstrum = np.real(fft(log_spectrum))

            # 倒谱峰值
            cepstrum_magnitude = np.abs(cepstrum[:len(cepstrum)//2])
            features['cepstrum_peak'] = np.max(cepstrum_magnitude)
            features['cepstrum_mean'] = np.mean(cepstrum_magnitude)

            # 找到倒谱中的主要峰值
            peaks, _ = find_peaks(cepstrum_magnitude, height=np.max(cepstrum_magnitude)*0.1)
            features['cepstrum_num_peaks'] = len(peaks)

        except:
            features['cepstrum_peak'] = 0
            features['cepstrum_mean'] = 0
            features['cepstrum_num_peaks'] = 0

        return features

    def _envelope_spectrum_features(self, signal):
        """计算包络谱特征"""
        features = {}
        try:
            # 使用Hilbert变换计算包络
            analytic_signal = hilbert(signal)
            envelope = np.abs(analytic_signal)

            # 计算包络的FFT
            envelope_fft = fft(envelope)
            envelope_magnitude = np.abs(envelope_fft[:len(envelope_fft)//2])

            # 包络谱峰值
            features['envelope_spectrum_peak'] = np.max(envelope_magnitude)
            features['envelope_spectrum_mean'] = np.mean(envelope_magnitude)

            # 找到包络谱中的主要峰值
            peaks, _ = find_peaks(envelope_magnitude, height=np.max(envelope_magnitude)*0.1)
            features['envelope_spectrum_num_peaks'] = len(peaks)

        except:
            features['envelope_spectrum_peak'] = 0
            features['envelope_spectrum_mean'] = 0
            features['envelope_spectrum_num_peaks'] = 0

        return features

    def _harmonic_energy_ratio(self, fft_magnitude, freqs):
        """计算谐波能量比"""
        try:
            # 假设基频在50Hz附近（可以根据实际情况调整）
            fundamental_freq = 50  # Hz

            # 计算前几个谐波的能量
            harmonic_energy = 0
            total_energy = np.sum(fft_magnitude**2)

            for harmonic in range(1, 6):  # 前5个谐波
                target_freq = fundamental_freq * harmonic
                # 在目标频率附近±5Hz范围内寻找能量
                freq_mask = (freqs >= target_freq - 5) & (freqs <= target_freq + 5)
                harmonic_energy += np.sum(fft_magnitude[freq_mask]**2)

            return harmonic_energy / total_energy if total_energy > 0 else 0
        except:
            return 0

    def _bandpass_energy_ratios(self, fft_magnitude, freqs):
        """计算不同频带的能量比"""
        features = {}
        total_energy = np.sum(fft_magnitude**2)

        # 定义关键频带
        bands = {
            'very_low': (0, 10),      # 极低频
            'low': (10, 100),         # 低频
            'motor': (100, 1000),     # 电机频率
            'bearing': (1000, 5000),  # 轴承频率
            'gear': (5000, 15000),    # 齿轮频率
            'high': (15000, 32000)    # 高频
        }

        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs < high_freq)
            band_energy = np.sum(fft_magnitude[band_mask]**2)
            features[f'{band_name}_band_ratio'] = band_energy / total_energy if total_energy > 0 else 0

        return features

    def extract_features_batch(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        批量提取特征

        Args:
            X: 原始信号数据 (n_samples, signal_length)
            verbose: 是否显示进度

        Returns:
            特征矩阵 (n_samples, n_features)
        """
        n_samples = X.shape[0]
        feature_list = []

        for i in range(n_samples):
            if verbose and (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{n_samples} 个样本")

            features_dict = self.extract_all_features(X[i])
            feature_list.append(features_dict)

        # 转换为numpy数组
        if len(feature_list) > 0:
            # 获取特征名称（按字母顺序排序以保证一致性）
            feature_names = sorted(feature_list[0].keys())

            # 构建特征矩阵
            feature_matrix = np.array([[f[name] for name in feature_names] for f in feature_list])

            if verbose:
                print(f"特征提取完成: {feature_matrix.shape}")

            return feature_matrix
        else:
            return np.array([])

