#!/usr/bin/env python3
"""
时序基础模型实验
包含预训练模型和迁移学习方法
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
from pathlib import Path
import yaml
import json
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesFoundationDataset(Dataset):
    """时序基础模型数据集"""
    
    def __init__(self, signals: List[np.ndarray], labels: np.ndarray, 
                 max_length: int = 1024, normalize: bool = True):
        self.labels = labels
        self.max_length = max_length
        
        # 预处理信号
        self.processed_signals = []
        for signal in signals:
            # 重采样到固定长度
            if len(signal) > max_length:
                # 等间隔采样
                indices = np.linspace(0, len(signal)-1, max_length, dtype=int)
                signal = signal[indices]
            else:
                # 填充
                signal = np.pad(signal, (0, max_length - len(signal)), 'constant')
            
            # 标准化
            if normalize:
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            self.processed_signals.append(signal)
        
        print(f"📊 基础模型数据预处理完成: {len(self.processed_signals)} 个信号，长度 {max_length}")
    
    def __len__(self):
        return len(self.processed_signals)
    
    def __getitem__(self, idx):
        signal = self.processed_signals[idx]
        label = self.labels[idx]
        return torch.FloatTensor(signal), torch.LongTensor([label])

class PretrainedTimeSeriesEncoder(nn.Module):
    """预训练时序编码器"""
    
    def __init__(self, input_length: int = 1024, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 6):
        super(PretrainedTimeSeriesEncoder, self).__init__()
        
        self.d_model = d_model
        self.input_length = input_length
        
        # 输入投影
        self.input_projection = nn.Linear(1, d_model)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(input_length, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # 添加特征维度
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer编码
        x = self.transformer(x)
        
        # 层归一化
        x = self.layer_norm(x)
        
        return x

class FoundationModelClassifier(nn.Module):
    """基于基础模型的分类器"""
    
    def __init__(self, input_length: int = 1024, d_model: int = 256, 
                 num_classes: int = 3, freeze_encoder: bool = False):
        super(FoundationModelClassifier, self).__init__()
        
        # 预训练编码器
        self.encoder = PretrainedTimeSeriesEncoder(input_length, d_model)
        
        # 是否冻结编码器
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)  # (batch_size, seq_len, d_model)
        
        # 转置用于池化
        encoded = encoded.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        # 分类
        output = self.classifier(encoded)
        
        return output

class ContrastiveLearningModel(nn.Module):
    """对比学习模型"""
    
    def __init__(self, input_length: int = 1024, d_model: int = 128):
        super(ContrastiveLearningModel, self).__init__()
        
        # 编码器
        self.encoder = PretrainedTimeSeriesEncoder(input_length, d_model, nhead=4, num_layers=3)
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        
        # 转置用于池化
        encoded = encoded.transpose(1, 2)
        
        # 投影
        projected = self.projection_head(encoded)
        
        # L2归一化
        projected = nn.functional.normalize(projected, dim=1)
        
        return projected

class FoundationModelPipeline:
    """基础模型流水线"""
    
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
            torch.cuda.set_per_process_memory_fraction(0.6)  # 使用60%的GPU内存
        
        self.results = {}
    
    def clear_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_and_prepare_data(self) -> Tuple:
        """加载和准备数据"""
        print("📂 加载基础模型数据...")
        
        # 从data_processing模块加载数据
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
        from data_loader import MotorDataLoader
        
        config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
        loader = MotorDataLoader(str(config_path))
        data, _ = loader.load_all_data(max_files_per_state=25)  # 减少数据量
        
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
                           batch_size: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        # 创建数据集
        train_dataset = TimeSeriesFoundationDataset(X_train, y_train, max_length=1024)
        val_dataset = TimeSeriesFoundationDataset(X_val, y_val, max_length=1024)
        test_dataset = TimeSeriesFoundationDataset(X_test, y_test, max_length=1024)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader, test_loader

    def train_foundation_model(self, train_loader: DataLoader, val_loader: DataLoader,
                              model_name: str = "FoundationModel") -> Dict:
        """训练基础模型"""
        print(f"\n🎯 开始训练 {model_name}")
        print("-" * 40)

        # 清理内存
        self.clear_memory()

        # 创建模型
        if model_name == "FoundationModel":
            model = FoundationModelClassifier(input_length=1024, d_model=128, freeze_encoder=False)
        elif model_name == "FineTunedModel":
            model = FoundationModelClassifier(input_length=1024, d_model=128, freeze_encoder=True)
        else:
            model = FoundationModelClassifier(input_length=1024, d_model=64, freeze_encoder=False)

        model = model.to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 总参数数量: {total_params:,}")
        print(f"📊 可训练参数: {trainable_params:,}")

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

        # 训练历史
        train_losses = []
        val_losses = []
        val_accuracies = []

        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0

        epochs = 25

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

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

                # 显示进度
                if batch_idx % 5 == 0:
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
            scheduler.step()

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
            if epoch % 3 == 0:
                self.clear_memory()

        # 恢复最佳模型
        model.load_state_dict(best_model_state)

        # 保存模型
        model_path = self.models_path / f'{model_name.lower()}_foundation.pth'
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

    def run_foundation_experiments(self) -> Dict:
        """运行基础模型实验"""
        print("🚀 开始基础模型实验")
        print("=" * 50)

        # 加载数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()

        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test, batch_size=4
        )

        # 实验配置
        experiments = [
            "FoundationModel",
            "CompactFoundation"
        ]

        all_results = {}

        for model_name in experiments:
            try:
                print(f"\n{'='*20} {model_name} {'='*20}")

                # 训练模型
                training_history = self.train_foundation_model(train_loader, val_loader, model_name)

                # 测试集评估
                test_metrics = self.evaluate_model(training_history['model'], test_loader)

                all_results[model_name] = {
                    'training_history': training_history,
                    'test_metrics': test_metrics
                }

                print(f"🎉 {model_name} - 测试准确率: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

                # 清理内存
                del training_history['model']
                self.clear_memory()

            except Exception as e:
                print(f"❌ {model_name} 实验失败: {e}")
                self.clear_memory()
                continue

        self.results = all_results
        return all_results

    def save_results(self) -> pd.DataFrame:
        """保存结果"""
        print("\n💾 保存基础模型结果...")

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
        results_path = self.output_path / 'foundation_models_results.csv'
        results_df.to_csv(results_path, index=False)

        print(f"📊 结果已保存: {results_path}")

        return results_df

if __name__ == "__main__":
    # 测试基础模型流水线
    config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"

    # 创建流水线
    pipeline = FoundationModelPipeline(str(config_path))

    # 运行实验
    results = pipeline.run_foundation_experiments()

    # 保存结果
    results_df = pipeline.save_results()

    print("\n🎉 基础模型实验完成！")
    if len(results_df) > 0:
        print("📊 模型性能排序 (按测试F1分数):")
        print(results_df.sort_values('test_f1', ascending=False)[['model', 'test_accuracy', 'test_f1']].to_string(index=False))
