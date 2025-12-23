"""
02_深度学习模型
实现CNN和LSTM模型进行时序分类
支持三种分类方式：ShengYing、ZhenDong、Fusion
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

# 强制使用CPU避免CuDNN版本问题
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 设置TensorFlow使用CPU
tf.config.set_visible_devices([], 'GPU')

class DeepLearningClassifier:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.models = {}
        self.histories = {}
        self.results = {}
        
    def create_cnn_model(self, input_shape, num_classes):
        """创建增强的CNN模型 - 更深更宽"""
        model = Sequential([
            # 第一个卷积块
            Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.1),

            # 第二个卷积块
            Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.1),

            # 第三个卷积块
            Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),

            # 第四个卷积块
            Conv1D(filters=1024, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),

            # 全局池化和全连接层
            GlobalAveragePooling1D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # 稍微降低学习率
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def create_lstm_model(self, input_shape, num_classes):
        """创建增强的LSTM模型 - 更深更宽"""
        model = Sequential([
            # 第一层LSTM
            LSTM(256, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),

            # 第二层LSTM
            LSTM(256, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            # 第三层LSTM
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            # 第四层LSTM
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            # 最后一层LSTM
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),

            # 全连接层
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # 稍微降低学习率
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def create_hybrid_model(self, input_shape, num_classes):
        """创建增强的CNN-LSTM混合模型"""
        model = Sequential([
            # CNN特征提取部分
            Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.1),

            Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.1),

            Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),

            # LSTM时序建模部分
            LSTM(256, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(64, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),

            # 全连接分类部分
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def prepare_data_for_dl(self, X, y, sequence_length=1024):
        """为深度学习准备数据"""
        # 重塑数据为序列格式
        if len(X.shape) == 2:
            # 将长序列分割为较短的序列
            n_samples, signal_length = X.shape
            n_sequences = signal_length // sequence_length
            
            X_reshaped = []
            y_reshaped = []
            
            for i in range(n_samples):
                for j in range(n_sequences):
                    start_idx = j * sequence_length
                    end_idx = start_idx + sequence_length
                    X_reshaped.append(X[i, start_idx:end_idx])
                    y_reshaped.append(y[i])
            
            X_reshaped = np.array(X_reshaped)
            y_reshaped = np.array(y_reshaped)
        else:
            X_reshaped = X
            y_reshaped = y
        
        # 重塑为 (samples, timesteps, features)
        X_reshaped = X_reshaped.reshape(X_reshaped.shape[0], X_reshaped.shape[1], 1)
        
        # 转换标签为one-hot编码
        y_categorical = to_categorical(y_reshaped, num_classes=3)
        
        return X_reshaped, y_categorical
    
    def train_and_evaluate(self, mode='shengying'):
        """训练和评估模型"""
        print(f"\n=== Training deep learning models for {mode} mode ===")
        
        # 加载数据
        loader = MotorDataLoader()
        X_raw, y = loader.load_data(mode=mode)
        
        # 使用更长的时间序列，适度下采样以保留更多信息
        downsample_factor = 32  # 从65536降到2048，保留更多时序信息
        X_downsampled = X_raw[:, ::downsample_factor]

        print(f"Original shape: {X_raw.shape}, Downsampled shape: {X_downsampled.shape}")

        # 准备深度学习数据，使用更长的序列
        X_dl, y_dl = self.prepare_data_for_dl(X_downsampled, y, sequence_length=1024)
        
        print(f"Deep learning data shape: {X_dl.shape}, Labels shape: {y_dl.shape}")
        
        # 分割数据
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_dl, y_dl, test_size=0.2, random_state=42, stratify=y_dl.argmax(axis=1)
        )
        
        # 标准化数据
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        num_classes = y_train.shape[1]
        
        # 定义模型
        model_configs = {
            'CNN': self.create_cnn_model,
            'LSTM': self.create_lstm_model,
            'CNN_LSTM': self.create_hybrid_model
        }
        
        mode_results = {}
        
        for model_name, model_func in model_configs.items():
            print(f"\nTraining {model_name}...")
            
            # 创建模型
            model = model_func(input_shape, num_classes)
            
            # 回调函数
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            # 训练模型（减少epochs以加快训练）
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=20,  # 减少epochs
                batch_size=64,  # 增加batch size
                callbacks=callbacks,
                verbose=1
            )
            
            # 预测
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
            
            # 评估
            accuracy = accuracy_score(y_test_labels, y_pred)
            
            # 保存结果
            mode_results[model_name] = {
                'model': model,
                'history': history,
                'accuracy': accuracy,
                'y_test': y_test_labels,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(y_test_labels, y_pred, output_dict=True)
            }
            
            print(f"{model_name} - Final Accuracy: {accuracy:.4f}")
            
            # 保存模型
            model_path = os.path.join(self.output_dir, 'table', f'02_{mode}_{model_name.lower()}_model.h5')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
        
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
                'Final_Train_Loss': result['history'].history['loss'][-1],
                'Final_Val_Loss': result['history'].history['val_loss'][-1],
                'Final_Train_Acc': result['history'].history['accuracy'][-1],
                'Final_Val_Acc': result['history'].history['val_accuracy'][-1]
            })
        
        df = pd.DataFrame(results_df)
        
        # 保存结果表格
        table_path = os.path.join(self.output_dir, 'table', f'02_{mode}_results.csv')
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        df.to_csv(table_path, index=False)
        
        print(f"\nResults saved to {table_path}")
        print(df.to_string(index=False))
    
    def _plot_results(self, mode):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 训练历史
        for model_name, result in self.results[mode].items():
            history = result['history']
            axes[0, 0].plot(history.history['loss'], label=f'{model_name} Train')
            axes[0, 0].plot(history.history['val_loss'], label=f'{model_name} Val', linestyle='--')
        
        axes[0, 0].set_title(f'Training Loss - {mode}')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # 准确率历史
        for model_name, result in self.results[mode].items():
            history = result['history']
            axes[0, 1].plot(history.history['accuracy'], label=f'{model_name} Train')
            axes[0, 1].plot(history.history['val_accuracy'], label=f'{model_name} Val', linestyle='--')
        
        axes[0, 1].set_title(f'Training Accuracy - {mode}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # 最终准确率对比
        model_names = list(self.results[mode].keys())
        accuracies = [self.results[mode][name]['accuracy'] for name in model_names]
        
        axes[1, 0].bar(model_names, accuracies, alpha=0.7)
        axes[1, 0].set_title(f'Final Test Accuracy - {mode}')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 混淆矩阵 (使用最佳模型)
        best_model = max(self.results[mode].keys(), key=lambda x: self.results[mode][x]['accuracy'])
        y_test = self.results[mode][best_model]['y_test']
        y_pred = self.results[mode][best_model]['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
        axes[1, 1].set_title(f'Confusion Matrix - {best_model} ({mode})')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # 保存图片
        img_path = os.path.join(self.output_dir, 'images', f'02_{mode}_results.png')
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
        print("DEEP LEARNING COMPREHENSIVE COMPARISON REPORT")
        print("="*80)

        # 收集所有结果
        all_results = []
        for mode, mode_results in self.results.items():
            for model_name, result in mode_results.items():
                all_results.append({
                    'Mode': mode,
                    'Model': model_name,
                    'Accuracy': result['accuracy']
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
        print(f"Accuracy: {best_overall['Accuracy']:.4f}")

        # 模式比较分析
        print("\n4. MODE COMPARISON ANALYSIS:")
        print("-" * 50)
        mode_avg = comparison_df.groupby('Mode')['Accuracy'].agg(['mean', 'std']).round(4)
        for mode, stats in mode_avg.iterrows():
            print(f"{mode.upper():12}: Mean Accuracy = {stats['mean']:.4f} ± {stats['std']:.4f}")

        # 保存综合比较结果
        comparison_path = os.path.join(self.output_dir, 'table', '02_comprehensive_comparison.csv')
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComprehensive comparison saved to: {comparison_path}")

def main():
    """主函数"""
    classifier = DeepLearningClassifier()

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

    # 生成综合比较报告
    classifier.generate_comparison_report()

    print("\n=== All deep learning experiments completed ===")

if __name__ == "__main__":
    main()
