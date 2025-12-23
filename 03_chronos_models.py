"""
03_Chronos模型
使用Chronos作为特征提取器和直接分类器
支持三种分类方式：ShengYing、ZhenDong、Fusion
"""

import os
import numpy as np
import pandas as pd
import torch
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    print("Chronos not available, using fallback statistical features")
    CHRONOS_AVAILABLE = False
    ChronosPipeline = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_utils import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

class ChronosClassifier:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.chronos_pipeline = None
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.results = {}
        
    def load_chronos_model(self, model_path="chronos_models/chronos-t5-base"):
        """加载本地Chronos模型"""
        if not CHRONOS_AVAILABLE:
            print("Chronos library not available, using statistical features")
            return False

        try:
            print(f"Loading local Chronos model from: {model_path}")
            # 检查本地模型是否存在
            if not os.path.exists(model_path):
                print(f"Local model not found at {model_path}")
                return False

            self.chronos_pipeline = ChronosPipeline.from_pretrained(
                model_path,
                device_map="cpu",  # 使用CPU避免GPU兼容性问题
                torch_dtype=torch.float32,
            )
            print("Chronos model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading Chronos model: {e}")
            print("Falling back to simple statistical features...")
            return False
    
    def extract_chronos_features(self, signals, context_length=1024):
        """使用Chronos提取增强特征"""
        if self.chronos_pipeline is None:
            return self._extract_fallback_features(signals)

        features_list = []

        for i, signal in enumerate(signals):
            if i % 50 == 0:
                print(f"Processing signal {i+1}/{len(signals)} with Chronos")

            try:
                # 使用更长的上下文窗口和更多的预测步长
                if len(signal) > context_length:
                    # 取更多窗口的特征，增加覆盖度
                    windows = []
                    step = len(signal) // 8  # 取8个重叠窗口
                    for j in range(8):
                        start_idx = j * step
                        end_idx = start_idx + context_length
                        if end_idx <= len(signal):
                            windows.append(signal[start_idx:end_idx])

                    if not windows:
                        windows = [signal[:context_length]]
                else:
                    # 零填充
                    padded_signal = np.zeros(context_length)
                    padded_signal[:len(signal)] = signal
                    windows = [padded_signal]

                # 对每个窗口提取更丰富的特征
                window_features = []
                for window in windows:
                    # 转换为torch tensor
                    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

                    # 使用Chronos进行多步预测
                    with torch.no_grad():
                        # 短期预测
                        short_forecast = self.chronos_pipeline.predict(
                            context=window_tensor,
                            prediction_length=24,  # 预测24个时间步
                            num_samples=3  # 多个样本增加鲁棒性
                        )

                        # 长期预测
                        long_forecast = self.chronos_pipeline.predict(
                            context=window_tensor,
                            prediction_length=48,  # 预测48个时间步
                            num_samples=3
                        )

                        # 提取短期预测特征
                        short_forecast_mean = np.mean([f.numpy().flatten() for f in short_forecast], axis=0)
                        long_forecast_mean = np.mean([f.numpy().flatten() for f in long_forecast], axis=0)

                        # 计算更丰富的预测特征
                        features = {}

                        # 短期预测特征
                        features.update({
                            'short_forecast_mean': np.mean(short_forecast_mean),
                            'short_forecast_std': np.std(short_forecast_mean),
                            'short_forecast_min': np.min(short_forecast_mean),
                            'short_forecast_max': np.max(short_forecast_mean),
                            'short_forecast_trend': np.polyfit(range(len(short_forecast_mean)), short_forecast_mean, 1)[0],
                            'short_forecast_energy': np.sum(short_forecast_mean**2),
                            'short_forecast_rms': np.sqrt(np.mean(short_forecast_mean**2)),
                        })

                        # 长期预测特征
                        features.update({
                            'long_forecast_mean': np.mean(long_forecast_mean),
                            'long_forecast_std': np.std(long_forecast_mean),
                            'long_forecast_min': np.min(long_forecast_mean),
                            'long_forecast_max': np.max(long_forecast_mean),
                            'long_forecast_trend': np.polyfit(range(len(long_forecast_mean)), long_forecast_mean, 1)[0],
                            'long_forecast_energy': np.sum(long_forecast_mean**2),
                            'long_forecast_rms': np.sqrt(np.mean(long_forecast_mean**2)),
                        })

                        # 预测一致性特征
                        features.update({
                            'forecast_consistency': np.corrcoef(short_forecast_mean[:24], long_forecast_mean[:24])[0,1] if len(short_forecast_mean) >= 24 else 0,
                            'forecast_volatility': np.std([np.std(f.numpy().flatten()) for f in short_forecast]),
                        })

                        # 原始信号的高级统计特征
                        features.update({
                            'signal_mean': np.mean(window),
                            'signal_std': np.std(window),
                            'signal_skew': self._safe_skew(window),
                            'signal_kurt': self._safe_kurtosis(window),
                            'signal_energy': np.sum(window**2),
                            'signal_rms': np.sqrt(np.mean(window**2)),
                            'signal_peak_to_peak': np.ptp(window),
                            'signal_crest_factor': np.max(np.abs(window)) / np.sqrt(np.mean(window**2)) if np.sqrt(np.mean(window**2)) != 0 else 0,
                        })

                        # 信号与预测的关系特征
                        features.update({
                            'signal_forecast_corr': np.corrcoef(window[-24:], short_forecast_mean[:24])[0,1] if len(window) >= 24 else 0,
                            'signal_forecast_mse': np.mean((window[-24:] - short_forecast_mean[:24])**2) if len(window) >= 24 else 0,
                        })

                        window_features.append(list(features.values()))

                # 聚合多个窗口的特征
                if len(window_features) > 1:
                    # 使用统计聚合而不是简单平均
                    aggregated_features = []
                    window_features_array = np.array(window_features)
                    for j in range(window_features_array.shape[1]):
                        feature_values = window_features_array[:, j]
                        aggregated_features.extend([
                            np.mean(feature_values),
                            np.std(feature_values),
                            np.max(feature_values),
                            np.min(feature_values)
                        ])
                else:
                    aggregated_features = window_features[0]

                features_list.append(aggregated_features)

            except Exception as e:
                print(f"Error processing signal {i}: {e}")
                # 使用备用特征
                fallback_features = self._extract_enhanced_fallback_features(signal)
                features_list.append(fallback_features)

        return np.array(features_list)
    
    def _extract_fallback_features(self, signals):
        """备用特征提取方法"""
        print("Using fallback feature extraction...")
        features_list = []
        
        for signal in signals:
            features = self._extract_simple_features(signal)
            features_list.append(features)
        
        return np.array(features_list)
    
    def _extract_simple_features(self, signal):
        """简单统计特征提取"""
        features = [
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.median(signal),
            self._safe_skew(signal),
            self._safe_kurtosis(signal),
            np.sum(signal**2),
            np.sqrt(np.mean(signal**2)),
            np.ptp(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75)
        ]
        return features

    def _extract_enhanced_fallback_features(self, signal):
        """增强的备用特征提取"""
        features = []

        # 基础统计特征
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.median(signal),
            self._safe_skew(signal),
            self._safe_kurtosis(signal),
            np.sum(signal**2),
            np.sqrt(np.mean(signal**2)),
            np.ptp(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
            np.percentile(signal, 10),
            np.percentile(signal, 90)
        ])

        # 频域特征
        try:
            from scipy.fft import fft
            fft_vals = fft(signal)
            fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])

            features.extend([
                np.mean(fft_magnitude),
                np.std(fft_magnitude),
                np.max(fft_magnitude),
                np.sum(fft_magnitude**2),
                np.argmax(fft_magnitude)  # 主频率索引
            ])
        except:
            features.extend([0, 0, 0, 0, 0])

        # 时域形状特征
        features.extend([
            np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)) if np.sqrt(np.mean(signal**2)) != 0 else 0,  # 峰值因子
            len(np.where(np.diff(np.signbit(signal)))[0]) / len(signal),  # 零交叉率
        ])

        # 子窗口特征
        n_windows = 10
        window_size = len(signal) // n_windows
        if window_size > 0:
            window_rms = []
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(signal))
                window = signal[start_idx:end_idx]
                if len(window) > 0:
                    window_rms.append(np.sqrt(np.mean(window**2)))

            if window_rms:
                features.extend([
                    np.max(window_rms),
                    np.std(window_rms),
                    np.mean(window_rms)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])

        # 确保特征数量一致（填充到与Chronos特征相同的维度）
        target_length = 100  # 目标特征长度
        while len(features) < target_length:
            features.append(0)

        return features[:target_length]
    
    def _safe_skew(self, data):
        """安全的偏度计算"""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            return 0.0
    
    def _safe_kurtosis(self, data):
        """安全的峰度计算"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            return 0.0
    
    def train_and_evaluate(self, mode='shengying'):
        """训练和评估模型"""
        print(f"\n=== Training Chronos models for {mode} mode ===")
        
        # 加载Chronos模型
        chronos_loaded = self.load_chronos_model()
        
        # 加载数据
        loader = MotorDataLoader()
        X_raw, y = loader.load_data(mode=mode)
        
        # 为了减少计算量，对信号进行下采样
        downsample_factor = 128  # 从65536降到512
        X_downsampled = X_raw[:, ::downsample_factor]
        
        print(f"Original shape: {X_raw.shape}, Downsampled shape: {X_downsampled.shape}")
        
        # 使用Chronos提取特征
        X_features = self.extract_chronos_features(X_downsampled)

        print(f"Extracted features shape: {X_features.shape}")

        # 处理NaN值
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_features = imputer.fit_transform(X_features)

        print(f"Features after NaN handling: {X_features.shape}")

        # 分割数据
        X_train, X_test, y_train, y_test = loader.split_data(X_features, y)
        
        # 标准化特征
        X_train_scaled, X_test_scaled, scaler = loader.normalize_data(X_train, X_test)
        
        mode_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name} with Chronos features...")
            
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
            model_path = os.path.join(self.output_dir, 'table', f'03_{mode}_{model_name.lower()}_chronos_model.pkl')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump({'model': model, 'scaler': scaler, 'chronos_loaded': chronos_loaded}, model_path)
        
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
                'Model': f'Chronos+{model_name}',
                'Mode': mode,
                'Accuracy': result['accuracy'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std']
            })
        
        df = pd.DataFrame(results_df)
        
        # 保存结果表格
        table_path = os.path.join(self.output_dir, 'table', f'03_{mode}_chronos_results.csv')
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        df.to_csv(table_path, index=False)
        
        print(f"\nResults saved to {table_path}")
        print(df.to_string(index=False))
    
    def _plot_results(self, mode):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 准确率对比
        model_names = [f'Chronos+{name}' for name in self.results[mode].keys()]
        accuracies = [self.results[mode][name]['accuracy'] for name in self.results[mode].keys()]
        cv_means = [self.results[mode][name]['cv_mean'] for name in self.results[mode].keys()]
        
        axes[0, 0].bar(model_names, accuracies, alpha=0.7, label='Test Accuracy')
        axes[0, 0].bar(model_names, cv_means, alpha=0.7, label='CV Mean')
        axes[0, 0].set_title(f'Chronos Model Accuracy Comparison - {mode}')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 混淆矩阵 (使用最佳模型)
        best_model = max(self.results[mode].keys(), key=lambda x: self.results[mode][x]['accuracy'])
        y_test = self.results[mode][best_model]['y_test']
        y_pred = self.results[mode][best_model]['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
        axes[0, 1].set_title(f'Confusion Matrix - Chronos+{best_model} ({mode})')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # CV分数分布
        axes[1, 0].boxplot([cv_means], labels=['Chronos Models'])
        axes[1, 0].set_title(f'Cross-Validation Score Distribution - {mode}')
        axes[1, 0].set_ylabel('CV Score')
        
        # 性能总结
        axes[1, 1].text(0.1, 0.8, f'Best Model: Chronos+{best_model}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Best Accuracy: {self.results[mode][best_model]["accuracy"]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'CV Score: {self.results[mode][best_model]["cv_mean"]:.4f} ± {self.results[mode][best_model]["cv_std"]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title(f'Performance Summary - {mode}')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        img_path = os.path.join(self.output_dir, 'images', f'03_{mode}_chronos_results.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results plot saved to {img_path}")

def main():
    """主函数"""
    classifier = ChronosClassifier()
    
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
    
    print("\n=== All Chronos experiments completed ===")

if __name__ == "__main__":
    main()
