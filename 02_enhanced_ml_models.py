"""
02_增强机器学习模型
使用sklearn的神经网络和其他高级模型替代深度学习
支持三种分类方式：ShengYing、ZhenDong、Fusion
"""

import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_utils import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

class EnhancedMLClassifier:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.models = {
            # 神经网络模型 - 替代深度学习
            'MLP_Small': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            ),
            'MLP_Large': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30
            ),
            'MLP_Deep': MLPClassifier(
                hidden_layer_sizes=(1024, 512, 256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.0005,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50
            ),
            # 集成学习模型
            'RandomForest_Enhanced': RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees_Enhanced': ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting_Enhanced': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=0.8,
                random_state=42
            ),
            # 支持向量机
            'SVM_RBF': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'SVM_Poly': SVC(
                kernel='poly',
                degree=3,
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            # K近邻
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                algorithm='auto',
                n_jobs=-1
            )
        }
        self.results = {}
        
    def create_ensemble_model(self):
        """创建集成模型"""
        # 选择最佳的基础模型创建集成
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42, max_iter=500)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        
        ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',  # 使用概率投票
            n_jobs=-1
        )
        
        return ensemble
    
    def extract_enhanced_features(self, signals):
        """提取增强特征"""
        features_list = []
        
        for i, signal in enumerate(signals):
            if i % 100 == 0:
                print(f"Extracting features from signal {i+1}/{len(signals)}")
            
            features = []
            
            # 基础统计特征
            features.extend([
                np.mean(signal),
                np.std(signal),
                np.var(signal),
                np.min(signal),
                np.max(signal),
                np.median(signal),
                np.ptp(signal),  # peak-to-peak
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.percentile(signal, 10),
                np.percentile(signal, 90)
            ])
            
            # 高阶统计特征
            try:
                from scipy import stats
                features.extend([
                    stats.skew(signal),
                    stats.kurtosis(signal),
                    stats.moment(signal, moment=3),
                    stats.moment(signal, moment=4)
                ])
            except:
                features.extend([0, 0, 0, 0])
            
            # 能量特征
            features.extend([
                np.sum(signal**2),  # 总能量
                np.sqrt(np.mean(signal**2)),  # RMS
                np.mean(np.abs(signal)),  # 平均绝对值
                np.max(np.abs(signal))  # 最大绝对值
            ])
            
            # 形状特征
            rms = np.sqrt(np.mean(signal**2))
            mean_abs = np.mean(np.abs(signal))
            max_abs = np.max(np.abs(signal))
            
            features.extend([
                max_abs / rms if rms != 0 else 0,  # 峰值因子
                rms / mean_abs if mean_abs != 0 else 0,  # 波形因子
                max_abs / mean_abs if mean_abs != 0 else 0,  # 脉冲因子
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
                    np.argmax(fft_magnitude),  # 主频率索引
                ])
                
                # 频带能量
                n_bands = 10
                band_size = len(fft_magnitude) // n_bands
                for i in range(n_bands):
                    start_idx = i * band_size
                    end_idx = min((i + 1) * band_size, len(fft_magnitude))
                    band_energy = np.sum(fft_magnitude[start_idx:end_idx]**2)
                    features.append(band_energy)
                    
            except:
                features.extend([0] * 15)  # 5个基础频域特征 + 10个频带能量
            
            # 零交叉率
            zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
            features.append(len(zero_crossings) / len(signal))
            
            # 子窗口特征
            n_windows = 20
            window_size = len(signal) // n_windows
            if window_size > 0:
                window_stats = []
                for j in range(n_windows):
                    start_idx = j * window_size
                    end_idx = min((j + 1) * window_size, len(signal))
                    window = signal[start_idx:end_idx]
                    if len(window) > 0:
                        window_stats.extend([
                            np.mean(window),
                            np.std(window),
                            np.max(window),
                            np.min(window)
                        ])
                
                # 统计子窗口特征
                if window_stats:
                    features.extend([
                        np.mean(window_stats),
                        np.std(window_stats),
                        np.max(window_stats),
                        np.min(window_stats)
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0])
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def train_and_evaluate(self, mode='shengying'):
        """训练和评估模型"""
        print(f"\n=== Training enhanced ML models for {mode} mode ===")
        
        # 加载数据
        loader = MotorDataLoader()
        X_raw, y = loader.load_data(mode=mode)
        
        # 适度下采样以保留更多信息
        downsample_factor = 64  # 从65536降到1024
        X_downsampled = X_raw[:, ::downsample_factor]
        
        print(f"Original shape: {X_raw.shape}, Downsampled shape: {X_downsampled.shape}")
        
        # 提取增强特征
        X_features = self.extract_enhanced_features(X_downsampled)
        
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
                model_path = os.path.join(self.output_dir, 'table', f'02_{mode}_{model_name.lower()}_enhanced_model.pkl')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump({'model': model, 'scaler': scaler}, model_path)
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # 训练集成模型
        print(f"\nTraining Ensemble model...")
        try:
            ensemble_model = self.create_ensemble_model()
            ensemble_model.fit(X_train_scaled, y_train)
            
            y_pred_ensemble = ensemble_model.predict(X_test_scaled)
            accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
            cv_scores_ensemble = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5)
            
            mode_results['Ensemble'] = {
                'accuracy': accuracy_ensemble,
                'cv_mean': cv_scores_ensemble.mean(),
                'cv_std': cv_scores_ensemble.std(),
                'y_test': y_test,
                'y_pred': y_pred_ensemble,
                'y_pred_proba': None,
                'classification_report': classification_report(y_test, y_pred_ensemble, output_dict=True)
            }
            
            print(f"Ensemble - Accuracy: {accuracy_ensemble:.4f}, CV: {cv_scores_ensemble.mean():.4f} ± {cv_scores_ensemble.std():.4f}")
            
            # 保存集成模型
            model_path = os.path.join(self.output_dir, 'table', f'02_{mode}_ensemble_enhanced_model.pkl')
            joblib.dump({'model': ensemble_model, 'scaler': scaler}, model_path)
            
        except Exception as e:
            print(f"Error training Ensemble model: {e}")
        
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
        table_path = os.path.join(self.output_dir, 'table', f'02_{mode}_enhanced_ml_results.csv')
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

        x = np.arange(len(model_names))
        width = 0.35

        axes[0, 0].bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.7)
        axes[0, 0].bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.7)
        axes[0, 0].set_title(f'Enhanced ML Model Accuracy Comparison - {mode}')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()

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
        cv_scores = [self.results[mode][name]['cv_mean'] for name in model_names]
        axes[1, 0].boxplot([cv_scores], labels=['Enhanced ML Models'])
        axes[1, 0].set_title(f'Cross-Validation Score Distribution - {mode}')
        axes[1, 0].set_ylabel('CV Score')

        # 性能总结
        axes[1, 1].text(0.1, 0.8, f'Best Model: {best_model}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Best Accuracy: {self.results[mode][best_model]["accuracy"]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'CV Score: {self.results[mode][best_model]["cv_mean"]:.4f} ± {self.results[mode][best_model]["cv_std"]:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title(f'Performance Summary - {mode}')
        axes[1, 1].axis('off')

        plt.tight_layout()

        # 保存图片
        img_path = os.path.join(self.output_dir, 'images', f'02_{mode}_enhanced_ml_results.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Results plot saved to {img_path}")

def main():
    """主函数"""
    classifier = EnhancedMLClassifier()

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

    print("\n=== All Enhanced ML experiments completed ===")

if __name__ == "__main__":
    main()
