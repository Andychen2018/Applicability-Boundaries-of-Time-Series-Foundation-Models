"""
01_快速统计机器学习模型
优化版本，专注于最重要的特征，避免计算密集的操作
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
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

class FastFeatureExtractor:
    def __init__(self, sampling_rate=65535):
        self.sampling_rate = sampling_rate
    
    def extract_features(self, signal):
        """提取优化的特征集"""
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
        
        # 百分位数特征
        features['q25'] = np.percentile(signal, 25)
        features['q75'] = np.percentile(signal, 75)
        features['q10'] = np.percentile(signal, 10)
        features['q90'] = np.percentile(signal, 90)
        features['iqr'] = features['q75'] - features['q25']
        
        # 频域特征
        freq_features = self._extract_frequency_features(signal)
        features.update(freq_features)
        
        # 简化的Hjorth参数
        hjorth_features = self._hjorth_parameters(signal)
        features.update(hjorth_features)
        
        # 子窗口特征
        subwindow_features = self._subwindow_features(signal)
        features.update(subwindow_features)
        
        return features
    
    def _extract_frequency_features(self, signal):
        """提取频域特征"""
        features = {}
        
        try:
            # FFT计算
            fft_vals = fft(signal)
            fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])
            freqs = fftfreq(len(signal), 1/self.sampling_rate)[:len(fft_vals)//2]
            
            if np.sum(fft_magnitude) == 0:
                return {f'freq_{i}': 0 for i in range(15)}
            
            # 基础频域统计特征
            features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
            features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * fft_magnitude) / np.sum(fft_magnitude))
            
            # 主频率和幅值
            dominant_idx = np.argmax(fft_magnitude)
            features['dominant_frequency'] = freqs[dominant_idx]
            features['dominant_magnitude'] = fft_magnitude[dominant_idx]
            
            # 频带能量分析
            total_energy = np.sum(fft_magnitude**2)
            
            # 低频带（0-1000Hz）
            low_mask = freqs < 1000
            features['low_freq_energy'] = np.sum(fft_magnitude[low_mask]**2) / total_energy if total_energy > 0 else 0
            
            # 中频带（1000-10000Hz）
            mid_mask = (freqs >= 1000) & (freqs < 10000)
            features['mid_freq_energy'] = np.sum(fft_magnitude[mid_mask]**2) / total_energy if total_energy > 0 else 0
            
            # 高频带（10000Hz以上）
            high_mask = freqs >= 10000
            features['high_freq_energy'] = np.sum(fft_magnitude[high_mask]**2) / total_energy if total_energy > 0 else 0
            
            # 频谱形状特征
            features['spectral_flatness'] = stats.gmean(fft_magnitude + 1e-10) / np.mean(fft_magnitude + 1e-10)
            features['spectral_rolloff'] = self._spectral_rolloff(freqs, fft_magnitude)
            
            # 简化的谱峭度
            features['spectral_kurtosis'] = stats.kurtosis(fft_magnitude) if len(fft_magnitude) > 3 else 0
            
        except Exception as e:
            print(f"Error in frequency feature extraction: {e}")
            # 返回零特征
            features.update({f'freq_{i}': 0 for i in range(10)})
        
        return features
    
    def _spectral_rolloff(self, freqs, magnitude):
        """计算频谱滚降"""
        try:
            cumsum_magnitude = np.cumsum(magnitude)
            rolloff_idx = np.where(cumsum_magnitude >= 0.85 * np.sum(magnitude))[0]
            return freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        except:
            return 0
    
    def _hjorth_parameters(self, signal):
        """计算Hjorth参数"""
        features = {}
        
        try:
            # Activity (方差)
            activity = np.var(signal)
            features['hjorth_activity'] = activity
            
            # Mobility
            if len(signal) > 1:
                diff1 = np.diff(signal)
                mobility = np.std(diff1) / np.std(signal) if np.std(signal) != 0 else 0
                features['hjorth_mobility'] = mobility
                
                # Complexity
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
                
        except:
            features['hjorth_activity'] = 0
            features['hjorth_mobility'] = 0
            features['hjorth_complexity'] = 0
            
        return features
    
    def _subwindow_features(self, signal, n_windows=10):
        """计算子窗口特征"""
        features = {}
        
        try:
            window_size = len(signal) // n_windows
            if window_size < 1:
                return {'subwindow_rms_max': 0, 'subwindow_rms_std': 0, 'subwindow_kurtosis_max': 0}
            
            rms_values = []
            kurtosis_values = []
            
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(signal))
                window = signal[start_idx:end_idx]
                
                if len(window) > 0:
                    rms_values.append(np.sqrt(np.mean(window**2)))
                    if len(window) > 3:
                        kurtosis_values.append(stats.kurtosis(window))
            
            features['subwindow_rms_max'] = np.max(rms_values) if rms_values else 0
            features['subwindow_rms_std'] = np.std(rms_values) if len(rms_values) > 1 else 0
            features['subwindow_kurtosis_max'] = np.max(kurtosis_values) if kurtosis_values else 0
            
        except:
            features['subwindow_rms_max'] = 0
            features['subwindow_rms_std'] = 0
            features['subwindow_kurtosis_max'] = 0
        
        return features

class FastStatisticalMLClassifier:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.feature_extractor = FastFeatureExtractor()
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'SVM': SVC(random_state=42, probability=True),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'NaiveBayes': GaussianNB()
        }
        self.results = {}
        
    def extract_features_batch(self, signals):
        """批量提取特征"""
        features_list = []
        
        for i, signal in enumerate(signals):
            if i % 100 == 0:
                print(f"Processing signal {i+1}/{len(signals)}")
            
            try:
                features = self.feature_extractor.extract_features(signal)
                features_list.append(list(features.values()))
            except Exception as e:
                print(f"Error processing signal {i}: {e}")
                # 使用零特征
                features_list.append([0] * 30)  # 假设有30个特征
        
        return np.array(features_list)
    
    def train_and_evaluate(self, mode='shengying'):
        """训练和评估模型"""
        print(f"\n=== Training fast statistical models for {mode} mode ===")
        
        # 加载数据
        loader = MotorDataLoader()
        X_raw, y = loader.load_data(mode=mode)
        
        # 下采样以加速处理
        downsample_factor = 128  # 从65536降到512
        X_downsampled = X_raw[:, ::downsample_factor]
        
        print(f"Original shape: {X_raw.shape}, Downsampled shape: {X_downsampled.shape}")
        
        # 提取特征
        print("Extracting features...")
        X_features = self.extract_features_batch(X_downsampled)
        
        print(f"Extracted features shape: {X_features.shape}")
        
        # 分割数据
        X_train, X_test, y_train, y_test = loader.split_data(X_features, y)
        
        # 标准化特征
        X_train_scaled, X_test_scaled, scaler = loader.normalize_data(X_train, X_test)
        
        mode_results = {}
        
        # 训练所有模型
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # 训练模型
                model.fit(X_train_scaled, y_train)
                
                # 预测
                y_pred = model.predict(X_test_scaled)
                
                # 评估
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                # 保存结果
                mode_results[model_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"{model_name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                # 保存模型
                model_path = os.path.join(self.output_dir, 'table', f'01_{mode}_{model_name.lower()}_fast_model.pkl')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump({'model': model, 'scaler': scaler}, model_path)
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        self.results[mode] = mode_results
        
        # 生成报告
        self._generate_reports(mode)
        
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
        table_path = os.path.join(self.output_dir, 'table', f'01_{mode}_fast_statistical_results.csv')
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        df.to_csv(table_path, index=False)
        
        print(f"\nResults saved to {table_path}")
        print(df.to_string(index=False))

def main():
    """主函数"""
    classifier = FastStatisticalMLClassifier()
    
    # 对三种模式分别进行实验
    modes = ['shengying', 'zhendong', 'fusion']
    
    for mode in modes:
        try:
            classifier.train_and_evaluate(mode=mode)
        except Exception as e:
            print(f"Error in {mode} mode: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n=== All fast statistical ML experiments completed ===")

if __name__ == "__main__":
    main()
