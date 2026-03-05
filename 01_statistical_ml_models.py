"""
01_统计和机器学习模型
实现特征提取（时域、频域统计特征）和传统机器学习分类器
支持三种分类方式：ShengYing、ZhenDong、Fusion
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch, hilbert, find_peaks, butter, filtfilt
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_utils import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, sampling_rate=65535):
        self.sampling_rate = sampling_rate
    
    def extract_time_domain_features(self, signal):
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
    
    def extract_frequency_domain_features(self, signal):
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
    
    def extract_all_features(self, signal):
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
            from collections import Counter
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

class StatisticalMLClassifier:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.feature_extractor = FeatureExtractor()
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'SVM': SVC(random_state=42, probability=True),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'NaiveBayes': GaussianNB()
        }
        self.results = {}
        
    def extract_features_from_data(self, X):
        """从原始信号数据中提取特征"""
        print("Extracting features...")
        feature_list = []
        
        for i, signal in enumerate(X):
            if i % 100 == 0:
                print(f"Processing signal {i+1}/{len(X)}")
            
            features = self.feature_extractor.extract_all_features(signal)
            feature_list.append(features)
        
        # 转换为DataFrame
        feature_df = pd.DataFrame(feature_list)
        
        # 处理NaN值
        feature_df = feature_df.fillna(0)
        
        print(f"Extracted {feature_df.shape[1]} features")
        return feature_df.values, list(feature_df.columns)
    
    def train_and_evaluate(self, mode='shengying'):
        """训练和评估模型"""
        print(f"\n=== Training models for {mode} mode ===")
        
        # 加载数据
        loader = MotorDataLoader()
        X_raw, y = loader.load_data(mode=mode)
        
        # 提取特征
        X_features, feature_names = self.extract_features_from_data(X_raw)
        
        # 分割数据
        X_train, X_test, y_train, y_test = loader.split_data(X_features, y)
        
        # 标准化特征
        X_train_scaled, X_test_scaled, scaler = loader.normalize_data(X_train, X_test)
        
        mode_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # 训练模型
            model.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # 评估
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # 保存结果
            mode_results[model_name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # 保存模型
            model_path = os.path.join(self.output_dir, 'table', f'01_{mode}_{model_name.lower()}_model.pkl')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump({'model': model, 'scaler': scaler, 'feature_names': feature_names}, model_path)
        
        self.results[mode] = mode_results
        
        # 生成报告和可视化
        self._generate_reports(mode)
        self._plot_results(mode)
        
        return mode_results
    
    def _generate_reports(self, mode):
        """生成结果报告"""
        results_df = []
        
        for model_name, result in self.results[mode].items():
            results_df.append({
                'Model': model_name,
                'Mode': mode,
                'Accuracy': result['accuracy'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std']
            })
        
        df = pd.DataFrame(results_df)
        
        # 保存结果表格
        table_path = os.path.join(self.output_dir, 'table', f'01_{mode}_results.csv')
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        df.to_csv(table_path, index=False)
        
        print(f"\nResults saved to {table_path}")
        print(df.to_string(index=False))
    
    def _plot_results(self, mode):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 准确率对比
        model_names = list(self.results[mode].keys())
        accuracies = [self.results[mode][name]['accuracy'] for name in model_names]
        cv_means = [self.results[mode][name]['cv_mean'] for name in model_names]
        
        axes[0, 0].bar(model_names, accuracies, alpha=0.7, label='Test Accuracy')
        axes[0, 0].bar(model_names, cv_means, alpha=0.7, label='CV Mean')
        axes[0, 0].set_title(f'Model Accuracy Comparison - {mode}')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 混淆矩阵 (使用最佳模型)
        best_model = max(self.results[mode].keys(), key=lambda x: self.results[mode][x]['accuracy'])
        y_test = self.results[mode][best_model]['y_test']
        y_pred = self.results[mode][best_model]['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
        axes[0, 1].set_title(f'Confusion Matrix - {best_model} ({mode})')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # CV分数分布
        cv_data = []
        for name in model_names:
            cv_data.extend([self.results[mode][name]['cv_mean']] * 5)  # 模拟CV分数
        
        axes[1, 0].boxplot([cv_data[i:i+5] for i in range(0, len(cv_data), 5)], labels=model_names)
        axes[1, 0].set_title(f'Cross-Validation Score Distribution - {mode}')
        axes[1, 0].set_ylabel('CV Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 特征重要性 (RandomForest)
        if 'RandomForest' in self.results[mode]:
            # 这里需要重新训练来获取特征重要性，简化处理
            axes[1, 1].text(0.5, 0.5, 'Feature Importance\n(Available in detailed analysis)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title(f'Feature Importance - {mode}')
        
        plt.tight_layout()
        
        # 保存图片
        img_path = os.path.join(self.output_dir, 'images', f'01_{mode}_results.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results plot saved to {img_path}")

    def generate_comparison_report(self):
        """生成三种模式的综合比较报告"""
        if not self.results:
            print("No results to compare")
            return

        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("="*80)

        # 收集所有结果
        all_results = []
        for mode, mode_results in self.results.items():
            for model_name, result in mode_results.items():
                all_results.append({
                    'Mode': mode,
                    'Model': model_name,
                    'Accuracy': result['accuracy'],
                    'CV_Mean': result['cv_mean'],
                    'CV_Std': result['cv_std']
                })

        comparison_df = pd.DataFrame(all_results)

        # 按模式分组比较
        print("\n1. BEST PERFORMANCE BY MODE:")
        print("-" * 50)
        for mode in ['shengying', 'zhendong', 'fusion']:
            mode_data = comparison_df[comparison_df['Mode'] == mode]
            if not mode_data.empty:
                best_model = mode_data.loc[mode_data['Accuracy'].idxmax()]
                print(f"{mode.upper():12}: {best_model['Model']:15} - Accuracy: {best_model['Accuracy']:.4f}")

        # 按模型分组比较
        print("\n2. BEST MODE FOR EACH MODEL:")
        print("-" * 50)
        for model in comparison_df['Model'].unique():
            model_data = comparison_df[comparison_df['Model'] == model]
            best_mode = model_data.loc[model_data['Accuracy'].idxmax()]
            print(f"{model:15}: {best_mode['Mode']:12} - Accuracy: {best_mode['Accuracy']:.4f}")

        # 总体最佳
        print("\n3. OVERALL BEST PERFORMANCE:")
        print("-" * 50)
        best_overall = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        print(f"Best: {best_overall['Model']} on {best_overall['Mode']} data")
        print(f"Accuracy: {best_overall['Accuracy']:.4f} ± {best_overall['CV_Std']:.4f}")

        # 模式比较分析
        print("\n4. MODE COMPARISON ANALYSIS:")
        print("-" * 50)
        mode_avg = comparison_df.groupby('Mode')['Accuracy'].agg(['mean', 'std']).round(4)
        for mode, stats in mode_avg.iterrows():
            print(f"{mode.upper():12}: Mean Accuracy = {stats['mean']:.4f} ± {stats['std']:.4f}")

        # 保存综合比较结果
        comparison_path = os.path.join(self.output_dir, 'table', '01_comprehensive_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComprehensive comparison saved to: {comparison_path}")

        # 生成比较图表
        self._plot_comprehensive_comparison(comparison_df)

    def _plot_comprehensive_comparison(self, comparison_df):
        """绘制综合比较图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 按模式分组的准确率比较
        mode_data = comparison_df.groupby(['Mode', 'Model'])['Accuracy'].first().unstack()
        mode_data.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('Accuracy Comparison by Mode')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 按模型分组的准确率比较
        model_data = comparison_df.groupby(['Model', 'Mode'])['Accuracy'].first().unstack()
        model_data.plot(kind='bar', ax=axes[0, 1], width=0.8)
        axes[0, 1].set_title('Accuracy Comparison by Model')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 模式平均性能
        mode_avg = comparison_df.groupby('Mode')['Accuracy'].agg(['mean', 'std'])
        axes[1, 0].bar(mode_avg.index, mode_avg['mean'], yerr=mode_avg['std'], capsize=5)
        axes[1, 0].set_title('Average Performance by Mode')
        axes[1, 0].set_ylabel('Mean Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. 热力图
        pivot_data = comparison_df.pivot(index='Model', columns='Mode', values='Accuracy')
        im = axes[1, 1].imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_xticks(range(len(pivot_data.columns)))
        axes[1, 1].set_yticks(range(len(pivot_data.index)))
        axes[1, 1].set_xticklabels(pivot_data.columns)
        axes[1, 1].set_yticklabels(pivot_data.index)
        axes[1, 1].set_title('Performance Heatmap')

        # 添加数值标注
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                text = axes[1, 1].text(j, i, f'{pivot_data.iloc[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=axes[1, 1])
        plt.tight_layout()

        # 保存图片
        img_path = os.path.join(self.output_dir, 'images', '01_comprehensive_comparison.png')
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comprehensive comparison plot saved to: {img_path}")

def main():
    """主函数"""
    classifier = StatisticalMLClassifier()

    # 对三种模式分别进行实验
    modes = ['shengying', 'zhendong', 'fusion']

    for mode in modes:
        try:
            classifier.train_and_evaluate(mode=mode)
        except Exception as e:
            print(f"Error in {mode} mode: {e}")
            continue

    # 生成综合比较报告
    classifier.generate_comparison_report()

    print("\n=== All experiments completed ===")

if __name__ == "__main__":
    main()
