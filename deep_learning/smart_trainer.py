#!/usr/bin/env python3
"""
智能深度学习训练器
逐个训练模型，优化内存使用
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class OptimizedMotorDataset(Dataset):
    """优化的电机信号数据集"""
    
    def __init__(self, signals: List[np.ndarray], labels: np.ndarray, 
                 max_length: int = 2000, downsample_factor: int = 2):
        self.labels = labels
        self.max_length = max_length
        self.downsample_factor = downsample_factor
        
        # 预处理信号
        self.processed_signals = []
        for signal in signals:
            # 降采样
            if downsample_factor > 1:
                signal = signal[::downsample_factor]
            
            # 截断或填充
            if len(signal) > max_length:
                signal = signal[:max_length]
            else:
                signal = np.pad(signal, (0, max_length - len(signal)), 'constant')
            
            # 标准化
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            self.processed_signals.append(signal)
        
        print(f"📊 数据预处理完成: {len(self.processed_signals)} 个信号，长度 {max_length}")
    
    def __len__(self):
        return len(self.processed_signals)
    
    def __getitem__(self, idx):
        signal = self.processed_signals[idx]
        label = self.labels[idx]
        return torch.FloatTensor(signal).unsqueeze(0), torch.LongTensor([label])

class SimpleCNN(nn.Module):
    """简化的CNN模型"""
    
    def __init__(self, input_length: int = 2000, num_classes: int = 3):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CompactLSTM(nn.Module):
    """紧凑的LSTM模型"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 32, 
                 num_layers: int = 1, num_classes: int = 3):
        super(CompactLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        output = self.classifier(hidden[-1])
        return output

class MiniTransformer(nn.Module):
    """迷你Transformer模型"""
    
    def __init__(self, input_size: int = 1, d_model: int = 64, 
                 nhead: int = 4, num_layers: int = 2, num_classes: int = 3):
        super(MiniTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 简化的位置编码
        self.pos_encoding = nn.Parameter(torch.randn(2000, d_model) * 0.1)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
        x = x.transpose(1, 2)
        seq_len = x.size(1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)
        
        # 分类
        output = self.classifier(x)
        return output

class SmartTrainer:
    """智能训练器"""
    
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
            # 获取GPU信息
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"📊 GPU内存: {gpu_memory:.1f} GB")
            
            # 设置内存分配策略
            torch.cuda.set_per_process_memory_fraction(0.7)  # 使用70%的GPU内存
        
        # 模型定义
        self.model_configs = {
            'SimpleCNN': {
                'class': SimpleCNN,
                'batch_size': 16,
                'epochs': 30,
                'lr': 0.001
            },
            'CompactLSTM': {
                'class': CompactLSTM,
                'batch_size': 8,
                'epochs': 25,
                'lr': 0.001
            },
            'MiniTransformer': {
                'class': MiniTransformer,
                'batch_size': 8,
                'epochs': 20,
                'lr': 0.0005
            }
        }
        
        self.results = {}
    
    def clear_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_and_prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载和准备数据"""
        print("📂 加载数据...")
        
        # 从data_processing模块加载数据
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
        from data_loader import MotorDataLoader
        
        config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
        loader = MotorDataLoader(str(config_path))
        data, _ = loader.load_all_data(max_files_per_state=30)  # 减少数据量
        
        # 整理数据
        all_signals = []
        all_labels = []
        
        label_map = {'normal': 0, 'spark': 1, 'vibrate': 2}
        
        for sensor in data.keys():
            for state in data[sensor].keys():
                signals = data[sensor][state]
                for signal in signals:
                    all_signals.append(signal)
                    all_labels.append(label_map[state])
        
        print(f"✅ 加载完成: {len(all_signals)} 个信号")
        
        # 数据划分
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_signals, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        print(f"📊 数据划分: 训练{len(X_train)}, 验证{len(X_val)}, 测试{len(X_test)}")
        
        return X_train, X_val, X_test, np.array(y_train), np.array(y_val), np.array(y_test)
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                           batch_size: int = 8) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        # 创建优化的数据集
        train_dataset = OptimizedMotorDataset(X_train, y_train, max_length=2000, downsample_factor=2)
        val_dataset = OptimizedMotorDataset(X_val, y_val, max_length=2000, downsample_factor=2)
        test_dataset = OptimizedMotorDataset(X_test, y_test, max_length=2000, downsample_factor=2)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader, test_loader

    def train_single_model(self, model_name: str, train_loader: DataLoader,
                          val_loader: DataLoader) -> Dict:
        """训练单个模型"""
        print(f"\n🎯 开始训练 {model_name}")
        print("-" * 40)

        # 清理内存
        self.clear_memory()

        # 获取模型配置
        config = self.model_configs[model_name]

        # 创建模型
        model = config['class']().to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型参数数量: {total_params:,}")

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        # 训练历史
        train_losses = []
        val_losses = []
        val_accuracies = []

        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0

        epochs = config['epochs']

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0

            for batch_idx, (batch_signals, batch_labels) in enumerate(train_loader):
                batch_signals = batch_signals.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_signals)
                loss = criterion(outputs, batch_labels)
                loss.backward()
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
                for batch_signals, batch_labels in val_loader:
                    batch_signals = batch_signals.to(self.device)
                    batch_labels = batch_labels.squeeze().to(self.device)

                    outputs = model(batch_signals)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

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
            if patience_counter >= 5:
                print(f"  早停于 epoch {epoch}")
                break

            # 清理内存
            if epoch % 5 == 0:
                self.clear_memory()

        # 恢复最佳模型
        model.load_state_dict(best_model_state)

        # 保存模型
        model_path = self.models_path / f'{model_name.lower()}_smart.pth'
        torch.save(model.state_dict(), model_path)
        print(f"💾 模型已保存: {model_path}")

        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'model': model
        }

        print(f"✅ {model_name} 训练完成, 最佳验证准确率: {best_val_acc:.4f}")

        return training_history

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """评估模型"""
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_signals, batch_labels in test_loader:
                batch_signals = batch_signals.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)

                outputs = model(batch_signals)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }

    def train_all_models(self) -> Dict:
        """逐个训练所有模型"""
        print("🚀 开始智能深度学习训练")
        print("=" * 50)

        # 加载数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()

        all_results = {}

        for model_name, config in self.model_configs.items():
            try:
                print(f"\n{'='*20} {model_name} {'='*20}")

                # 创建数据加载器
                train_loader, val_loader, test_loader = self.create_data_loaders(
                    X_train, X_val, X_test, y_train, y_val, y_test,
                    batch_size=config['batch_size']
                )

                # 训练模型
                training_history = self.train_single_model(model_name, train_loader, val_loader)

                # 测试集评估
                test_metrics = self.evaluate_model(training_history['model'], test_loader)

                all_results[model_name] = {
                    'training_history': training_history,
                    'test_metrics': test_metrics
                }

                print(f"🎉 {model_name} - 测试准确率: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

                # 清理内存
                del training_history['model']  # 删除模型引用
                self.clear_memory()

            except Exception as e:
                print(f"❌ {model_name} 训练失败: {e}")
                self.clear_memory()
                continue

        self.results = all_results
        return all_results

    def save_results(self) -> pd.DataFrame:
        """保存结果"""
        print("\n💾 保存训练结果...")

        # 整理结果数据
        results_data = []

        for model_name, result in self.results.items():
            test_metrics = result['test_metrics']

            result_row = {
                'model': model_name,
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'best_val_acc': result['training_history']['best_val_acc']
            }
            results_data.append(result_row)

        # 保存为CSV
        results_df = pd.DataFrame(results_data)
        results_path = self.output_path / 'smart_deep_learning_results.csv'
        results_df.to_csv(results_path, index=False)

        print(f"📊 结果已保存: {results_path}")

        return results_df

if __name__ == "__main__":
    # 测试智能训练器
    config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"

    # 创建训练器
    trainer = SmartTrainer(str(config_path))

    # 训练所有模型
    results = trainer.train_all_models()

    # 保存结果
    results_df = trainer.save_results()

    print("\n🎉 智能深度学习训练完成！")
    if len(results_df) > 0:
        print("📊 模型性能排序 (按测试F1分数):")
        print(results_df.sort_values('test_f1', ascending=False)[['model', 'test_accuracy', 'test_f1']].to_string(index=False))
