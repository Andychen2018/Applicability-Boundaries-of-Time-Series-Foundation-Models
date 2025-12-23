#!/usr/bin/env python3
"""
深度学习模型模块
包含1D-CNN, LSTM, Transformer等深度学习方法
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MotorSignalDataset(Dataset):
    """电机信号数据集"""
    
    def __init__(self, signals: List[np.ndarray], labels: np.ndarray, max_length: int = 5000):
        self.signals = signals
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        
        # 截断或填充到固定长度
        if len(signal) > self.max_length:
            signal = signal[:self.max_length]
        else:
            signal = np.pad(signal, (0, self.max_length - len(signal)), 'constant')
        
        return torch.FloatTensor(signal).unsqueeze(0), torch.LongTensor([label])

class CNN1D(nn.Module):
    """1D卷积神经网络"""
    
    def __init__(self, input_length: int = 5000, num_classes: int = 3):
        super(CNN1D, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 第一层卷积 - 减小卷积核
            nn.Conv1d(1, 32, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            # 第二层卷积
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            # 第三层卷积
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            # 第四层卷积
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

class LSTMModel(nn.Module):
    """LSTM模型 - 优化内存使用"""

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 3):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 减小隐藏层大小以节省内存
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2, bidirectional=False)

        # 简化注意力机制
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)

        # Classification
        output = self.classifier(pooled)
        return output

class TransformerModel(nn.Module):
    """Transformer模型"""
    
    def __init__(self, input_size: int = 1, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 4, num_classes: int = 3):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(5000, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
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

class AutoEncoder(nn.Module):
    """自编码器用于信号重建"""
    
    def __init__(self, input_length: int = 5000, encoding_dim: int = 128):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(encoding_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=4),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=32, stride=4),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=64, stride=4),
        )
        
        # 分类器（基于编码特征）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoding_dim * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
    
    def forward(self, x, return_reconstruction=False):
        # 编码
        encoded = self.encoder(x)
        
        if return_reconstruction:
            # 解码重建
            decoded = self.decoder(encoded)
            # 调整输出长度
            if decoded.size(-1) != x.size(-1):
                decoded = nn.functional.interpolate(decoded, size=x.size(-1), mode='linear', align_corners=False)
            return decoded, encoded
        else:
            # 分类
            classification = self.classifier(encoded)
            return classification

class DeepLearningPipeline:
    """深度学习流水线"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['output']['tables'])
        self.image_path = Path(self.config['output']['images'])
        self.models_path = self.output_path.parent / 'models'
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 初始化模型
        self.models = {
            'CNN1D': CNN1D,
            'LSTM': LSTMModel,
            'Transformer': TransformerModel,
            'AutoEncoder': AutoEncoder
        }
        
        self.results = {}
    
    def load_data(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """加载原始信号数据"""
        print("📂 加载原始信号数据...")
        
        # 从data_processing模块加载数据
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
        from data_loader import MotorDataLoader
        
        config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
        loader = MotorDataLoader(str(config_path))
        data, _ = loader.load_all_data(max_files_per_state=50)
        
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
        return all_signals, np.array(all_labels)
    
    def prepare_data(self, signals: List[np.ndarray], labels: np.ndarray) -> Dict:
        """准备训练数据"""
        print("🔧 准备训练数据...")
        
        # 数据划分
        X_temp, X_test, y_temp, y_test = train_test_split(
            signals, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        # 创建数据集
        train_dataset = MotorSignalDataset(X_train, y_train)
        val_dataset = MotorSignalDataset(X_val, y_val)
        test_dataset = MotorSignalDataset(X_test, y_test)
        
        # 创建数据加载器 - 减小batch size以节省内存
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        print(f"✅ 数据准备完成:")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   验证集: {len(X_val)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        
        return data_loaders

    def train_model(self, model_name: str, data_loaders: Dict, epochs: int = 50) -> Dict:
        """训练单个模型"""
        print(f"🎯 训练 {model_name}...")

        # 创建模型
        model = self.models[model_name]().to(self.device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
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

            for batch_signals, batch_labels in data_loaders['train']:
                batch_signals = batch_signals.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)

                optimizer.zero_grad()

                if model_name == 'AutoEncoder':
                    outputs = model(batch_signals)
                else:
                    outputs = model(batch_signals)

                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_signals, batch_labels in data_loaders['val']:
                    batch_signals = batch_signals.to(self.device)
                    batch_labels = batch_labels.squeeze().to(self.device)

                    if model_name == 'AutoEncoder':
                        outputs = model(batch_signals)
                    else:
                        outputs = model(batch_signals)

                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(data_loaders['train'])
            avg_val_loss = val_loss / len(data_loaders['val'])
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

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # 早停
            if patience_counter >= 10:
                print(f"  早停于 epoch {epoch}")
                break

        # 恢复最佳模型
        model.load_state_dict(best_model_state)

        # 保存模型
        model_path = self.models_path / f'{model_name.lower()}_model.pth'
        torch.save(model.state_dict(), model_path)

        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'model': model
        }

        print(f"  ✅ {model_name} 训练完成, 最佳验证准确率: {best_val_acc:.4f}")

        return training_history

    def evaluate_model(self, model: nn.Module, data_loader: DataLoader, model_name: str) -> Dict:
        """评估模型"""
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch_signals, batch_labels in data_loader:
                batch_signals = batch_signals.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)

                if model_name == 'AutoEncoder':
                    outputs = model(batch_signals)
                else:
                    outputs = model(batch_signals)

                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        # 计算AUC (多类别)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='weighted')
        except:
            auc = 0.0

        return {
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }

    def train_all_models(self, data_loaders: Dict) -> Dict:
        """训练所有深度学习模型"""
        print("🤖 开始训练所有深度学习模型...")

        all_results = {}

        for model_name in self.models.keys():
            try:
                # 训练模型
                training_history = self.train_model(model_name, data_loaders)

                # 测试集评估
                test_metrics = self.evaluate_model(
                    training_history['model'],
                    data_loaders['test'],
                    model_name
                )

                all_results[model_name] = {
                    'training_history': training_history,
                    'test_metrics': test_metrics
                }

                print(f"✅ {model_name} - 测试准确率: {test_metrics['accuracy']:.4f}")

            except Exception as e:
                print(f"❌ {model_name} 训练失败: {e}")
                continue

        self.results = all_results
        return all_results

    def save_results(self):
        """保存深度学习结果"""
        print("💾 保存深度学习结果...")

        # 整理结果数据
        results_data = []

        for model_name, result in self.results.items():
            test_metrics = result['test_metrics']

            result_row = {
                'model': model_name,
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'test_auc': test_metrics['auc'],
                'best_val_acc': result['training_history']['best_val_acc']
            }
            results_data.append(result_row)

        # 保存为CSV
        results_df = pd.DataFrame(results_data)
        results_path = self.output_path / 'deep_learning_results.csv'
        results_df.to_csv(results_path, index=False)

        print(f"📊 深度学习结果已保存: {results_path}")

        # 保存详细结果
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'results': results_data
        }

        json_path = self.output_path / 'deep_learning_detailed_results.json'
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        return results_df

if __name__ == "__main__":
    # 测试深度学习流水线
    config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"

    # 创建流水线
    pipeline = DeepLearningPipeline(str(config_path))

    # 加载数据
    signals, labels = pipeline.load_data()

    # 准备数据
    data_loaders = pipeline.prepare_data(signals, labels)

    # 训练所有模型
    results = pipeline.train_all_models(data_loaders)

    # 保存结果
    results_df = pipeline.save_results()

    print("\n🎉 深度学习实验完成！")
    print("📊 模型性能排序 (按测试F1分数):")
    print(results_df.sort_values('test_f1', ascending=False)[['model', 'test_accuracy', 'test_f1']].to_string(index=False))
