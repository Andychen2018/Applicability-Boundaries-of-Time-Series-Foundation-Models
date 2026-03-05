#!/usr/bin/env python3
"""
单独深度学习训练器
逐个训练深度学习模型，考虑多测点融合
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import yaml
import json
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MultiSensorDataset(Dataset):
    """多传感器数据集"""
    
    def __init__(self, audio_signals: List[np.ndarray], vibration_signals: List[np.ndarray], 
                 labels: np.ndarray, max_length: int = 2048):
        self.labels = labels
        self.max_length = max_length
        
        # 预处理信号
        self.audio_signals = []
        self.vibration_signals = []
        
        for audio, vibration in zip(audio_signals, vibration_signals):
            # 处理声音信号
            if len(audio) > max_length:
                indices = np.linspace(0, len(audio)-1, max_length, dtype=int)
                audio = audio[indices]
            else:
                audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
            
            # 处理振动信号
            if len(vibration) > max_length:
                indices = np.linspace(0, len(vibration)-1, max_length, dtype=int)
                vibration = vibration[indices]
            else:
                vibration = np.pad(vibration, (0, max_length - len(vibration)), 'constant')
            
            # 标准化
            audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
            vibration = (vibration - np.mean(vibration)) / (np.std(vibration) + 1e-8)
            
            self.audio_signals.append(audio)
            self.vibration_signals.append(vibration)
        
        print(f"📊 多传感器数据预处理完成: {len(self.audio_signals)} 个样本")
    
    def __len__(self):
        return len(self.audio_signals)
    
    def __getitem__(self, idx):
        audio = torch.FloatTensor(self.audio_signals[idx]).unsqueeze(0)
        vibration = torch.FloatTensor(self.vibration_signals[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])
        return audio, vibration, label

class SingleSensorDataset(Dataset):
    """单传感器数据集"""
    
    def __init__(self, signals: List[np.ndarray], labels: np.ndarray, max_length: int = 2048):
        self.labels = labels
        self.max_length = max_length
        
        # 预处理信号
        self.processed_signals = []
        for signal in signals:
            if len(signal) > max_length:
                indices = np.linspace(0, len(signal)-1, max_length, dtype=int)
                signal = signal[indices]
            else:
                signal = np.pad(signal, (0, max_length - len(signal)), 'constant')
            
            # 标准化
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            self.processed_signals.append(signal)
        
        print(f"📊 单传感器数据预处理完成: {len(self.processed_signals)} 个样本")
    
    def __len__(self):
        return len(self.processed_signals)
    
    def __getitem__(self, idx):
        signal = torch.FloatTensor(self.processed_signals[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])
        return signal, label

class DeepCNN1D(nn.Module):
    """深层1D CNN"""
    
    def __init__(self, input_length: int = 2048, num_classes: int = 3):
        super(DeepCNN1D, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DeepLSTM(nn.Module):
    """深层LSTM"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 128, 
                 num_layers: int = 3, num_classes: int = 3):
        super(DeepLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多层LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, 
                                             dropout=0.1, batch_first=True)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)
        
        # 分类
        output = self.classifier(pooled)
        return output

class MultiSensorFusionCNN(nn.Module):
    """多传感器融合CNN"""
    
    def __init__(self, input_length: int = 2048, num_classes: int = 3):
        super(MultiSensorFusionCNN, self).__init__()
        
        # 声音分支
        self.audio_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 振动分支
        self.vibration_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512, 256),  # 256 + 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, audio, vibration):
        # 分别处理两个传感器
        audio_features = self.audio_branch(audio)
        vibration_features = self.vibration_branch(vibration)
        
        # 特征融合
        fused_features = torch.cat([audio_features, vibration_features], dim=1)
        
        # 分类
        output = self.fusion_classifier(fused_features)
        return output

class IndividualDLTrainer:
    """单独深度学习训练器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['output']['tables'])
        self.models_path = self.output_path.parent / 'models'
        self.models_path.mkdir(exist_ok=True)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        self.results = {}
    
    def clear_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_multi_sensor_data(self) -> Tuple:
        """加载多传感器数据"""
        print("📂 加载多传感器数据...")
        
        # 从增强数据加载器加载数据
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
        from enhanced_data_loader import EnhancedMotorDataLoader
        
        config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
        loader = EnhancedMotorDataLoader(str(config_path))
        dataset, _ = loader.load_comprehensive_dataset(enable_augmentation=False)
        
        # 使用多传感器匹配数据
        audio_signals = []
        vibration_signals = []
        labels = []
        
        for state in ['normal', 'spark', 'vibrate']:
            for sample in dataset['multi_sensor'][state]:
                audio_signals.append(sample['ShengYing'])
                vibration_signals.append(sample['ZhenDong'])
                labels.append(state)
        
        print(f"✅ 多传感器数据加载完成: {len(audio_signals)} 个样本")
        return audio_signals, vibration_signals, np.array(labels)
    
    def load_single_sensor_data(self) -> Tuple:
        """加载单传感器数据"""
        print("📂 加载单传感器数据...")
        
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
        from enhanced_data_loader import EnhancedMotorDataLoader
        
        config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
        loader = EnhancedMotorDataLoader(str(config_path))
        dataset, _ = loader.load_comprehensive_dataset(enable_augmentation=False)
        
        # 使用单传感器数据（取部分以节省时间）
        all_signals = []
        all_labels = []
        
        for state in ['normal', 'spark', 'vibrate']:
            signals = dataset['single_sensor'][state][:1000]  # 每类取1000个样本
            all_signals.extend(signals)
            all_labels.extend([state] * len(signals))
        
        print(f"✅ 单传感器数据加载完成: {len(all_signals)} 个样本")
        return all_signals, np.array(all_labels)

    def train_single_model(self, model_name: str, model_class, data_loaders: Tuple,
                          epochs: int = 30, lr: float = 0.001) -> Dict:
        """训练单个模型"""
        print(f"\n🎯 开始训练 {model_name}")
        print("-" * 50)

        # 清理内存
        self.clear_memory()

        train_loader, val_loader, test_loader = data_loaders

        # 创建模型
        model = model_class().to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 总参数数量: {total_params:,}")
        print(f"📊 可训练参数: {trainable_params:,}")

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # 训练历史
        train_losses = []
        val_losses = []
        val_accuracies = []

        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0

            for batch_idx, batch_data in enumerate(train_loader):
                if model_name == "MultiSensorFusionCNN":
                    audio, vibration, labels = batch_data
                    audio = audio.to(self.device)
                    vibration = vibration.to(self.device)
                    labels = labels.squeeze().to(self.device)

                    optimizer.zero_grad()
                    outputs = model(audio, vibration)
                else:
                    signals, labels = batch_data
                    signals = signals.to(self.device)
                    labels = labels.squeeze().to(self.device)

                    optimizer.zero_grad()
                    outputs = model(signals)

                loss = criterion(outputs, labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

                # 显示进度
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_data in val_loader:
                    if model_name == "MultiSensorFusionCNN":
                        audio, vibration, labels = batch_data
                        audio = audio.to(self.device)
                        vibration = vibration.to(self.device)
                        labels = labels.squeeze().to(self.device)
                        outputs = model(audio, vibration)
                    else:
                        signals, labels = batch_data
                        signals = signals.to(self.device)
                        labels = labels.squeeze().to(self.device)
                        outputs = model(signals)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 早停检查
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"  Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # 早停
            if patience_counter >= 8:
                print(f"  早停于 epoch {epoch}")
                break

            # 清理内存
            if epoch % 5 == 0:
                self.clear_memory()

        # 恢复最佳模型
        model.load_state_dict(best_model_state)

        # 测试集评估
        test_metrics = self.evaluate_model(model, test_loader, model_name)

        # 保存模型
        model_path = self.models_path / f'{model_name.lower()}_individual.pth'
        torch.save(model.state_dict(), model_path)
        print(f"💾 模型已保存: {model_path}")

        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'test_metrics': test_metrics
        }

        print(f"✅ {model_name} 训练完成")
        print(f"📊 最佳验证准确率: {best_val_acc:.4f}")
        print(f"📊 测试准确率: {test_metrics['accuracy']:.4f}")
        print(f"📊 测试F1分数: {test_metrics['f1']:.4f}")

        return training_history

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, model_name: str) -> Dict:
        """评估模型"""
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_data in test_loader:
                if model_name == "MultiSensorFusionCNN":
                    audio, vibration, labels = batch_data
                    audio = audio.to(self.device)
                    vibration = vibration.to(self.device)
                    labels = labels.squeeze().to(self.device)
                    outputs = model(audio, vibration)
                else:
                    signals, labels = batch_data
                    signals = signals.to(self.device)
                    labels = labels.squeeze().to(self.device)
                    outputs = model(signals)

                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }

    def save_results(self, all_results: Dict) -> pd.DataFrame:
        """保存结果"""
        print("\n💾 保存深度学习结果...")

        # 整理结果数据
        results_data = []

        for model_name, result in all_results.items():
            test_metrics = result['test_metrics']

            result_row = {
                'model': model_name,
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'best_val_acc': result['best_val_acc']
            }
            results_data.append(result_row)

        # 保存为CSV
        results_df = pd.DataFrame(results_data)
        results_path = self.output_path / 'individual_deep_learning_results.csv'
        results_df.to_csv(results_path, index=False)

        print(f"📊 深度学习结果已保存: {results_path}")

        return results_df

if __name__ == "__main__":
    # 这个文件作为模块使用，不直接运行
    pass
