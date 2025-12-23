#!/usr/bin/env python3
"""
无监督异常检测方法
包含自编码器、孤立森林、One-Class SVM等方法
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AnomalyDataset(Dataset):
    """异常检测数据集"""
    
    def __init__(self, signals: List[np.ndarray], max_length: int = 2048):
        self.max_length = max_length
        
        # 预处理信号
        self.processed_signals = []
        for signal in signals:
            # 重采样到固定长度
            if len(signal) > max_length:
                indices = np.linspace(0, len(signal)-1, max_length, dtype=int)
                signal = signal[indices]
            else:
                signal = np.pad(signal, (0, max_length - len(signal)), 'constant')
            
            # 标准化
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            self.processed_signals.append(signal)
        
        print(f"📊 异常检测数据预处理完成: {len(self.processed_signals)} 个信号")
    
    def __len__(self):
        return len(self.processed_signals)
    
    def __getitem__(self, idx):
        signal = self.processed_signals[idx]
        return torch.FloatTensor(signal).unsqueeze(0)

class DeepAutoEncoder(nn.Module):
    """深层自编码器"""
    
    def __init__(self, input_length: int = 2048, encoding_dim: int = 64):
        super(DeepAutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            # 第一层
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第二层
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第三层
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第四层
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(encoding_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            # 上采样开始
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        
        # 解码
        decoded = self.decoder(encoded)
        
        # 调整输出长度
        if decoded.size(-1) != x.size(-1):
            decoded = nn.functional.interpolate(decoded, size=x.size(-1), 
                                              mode='linear', align_corners=False)
        
        return decoded, encoded

class VariationalAutoEncoder(nn.Module):
    """变分自编码器"""
    
    def __init__(self, input_length: int = 2048, latent_dim: int = 32):
        super(VariationalAutoEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        
        # 均值和方差
        self.fc_mu = nn.Linear(128 * 64, latent_dim)
        self.fc_logvar = nn.Linear(128 * 64, latent_dim)
        
        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, 128 * 64)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )
    
    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 128, 64)
        return self.decoder_conv(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # 调整输出长度
        if recon.size(-1) != x.size(-1):
            recon = nn.functional.interpolate(recon, size=x.size(-1), 
                                            mode='linear', align_corners=False)
        
        return recon, mu, logvar

class UnsupervisedAnomalyDetector:
    """无监督异常检测器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['output']['tables'])
        self.models_path = self.output_path.parent / 'models'
        self.models_path.mkdir(exist_ok=True)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        self.results = {}
    
    def load_normal_data(self) -> List[np.ndarray]:
        """加载正常数据用于训练"""
        print("📂 加载正常数据...")
        
        # 从增强数据加载器加载数据
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
        from enhanced_data_loader import EnhancedMotorDataLoader
        
        config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
        loader = EnhancedMotorDataLoader(str(config_path))
        dataset, _ = loader.load_comprehensive_dataset(enable_augmentation=False)
        
        # 只使用正常数据训练
        normal_signals = dataset['single_sensor']['normal']
        
        print(f"✅ 加载正常数据: {len(normal_signals)} 个样本")
        return normal_signals
    
    def load_test_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """加载测试数据"""
        print("📂 加载测试数据...")
        
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
        from enhanced_data_loader import EnhancedMotorDataLoader
        
        config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
        loader = EnhancedMotorDataLoader(str(config_path))
        dataset, _ = loader.load_comprehensive_dataset(enable_augmentation=False)
        
        # 收集所有测试数据
        all_signals = []
        all_labels = []
        
        for state in ['normal', 'spark', 'vibrate']:
            signals = dataset['single_sensor'][state]
            all_signals.extend(signals)
            all_labels.extend([state] * len(signals))
        
        print(f"✅ 加载测试数据: {len(all_signals)} 个样本")
        return all_signals, all_labels
    
    def train_autoencoder(self, normal_signals: List[np.ndarray], 
                         model_type: str = "deep") -> nn.Module:
        """训练自编码器"""
        print(f"🎯 训练{model_type}自编码器...")
        
        # 创建数据集
        dataset = AnomalyDataset(normal_signals)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 创建模型
        if model_type == "deep":
            model = DeepAutoEncoder().to(self.device)
        else:
            model = VariationalAutoEncoder().to(self.device)
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练
        model.train()
        for epoch in range(50):
            total_loss = 0
            
            for batch_signals in dataloader:
                batch_signals = batch_signals.to(self.device)
                
                optimizer.zero_grad()
                
                if model_type == "deep":
                    reconstructed, _ = model(batch_signals)
                    loss = nn.MSELoss()(reconstructed, batch_signals)
                else:
                    reconstructed, mu, logvar = model(batch_signals)
                    recon_loss = nn.MSELoss()(reconstructed, batch_signals)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.001 * kl_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        # 保存模型
        model_path = self.models_path / f'{model_type}_autoencoder.pth'
        torch.save(model.state_dict(), model_path)
        
        print(f"✅ {model_type}自编码器训练完成")
        return model
    
    def detect_anomalies_autoencoder(self, model: nn.Module, test_signals: List[np.ndarray], 
                                   model_type: str = "deep") -> np.ndarray:
        """使用自编码器检测异常"""
        print("🔍 使用自编码器检测异常...")
        
        # 创建测试数据集
        test_dataset = AnomalyDataset(test_signals)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch_signals in test_loader:
                batch_signals = batch_signals.to(self.device)
                
                if model_type == "deep":
                    reconstructed, _ = model(batch_signals)
                else:
                    reconstructed, _, _ = model(batch_signals)
                
                # 计算重建误差
                errors = torch.mean((batch_signals - reconstructed) ** 2, dim=(1, 2))
                reconstruction_errors.extend(errors.cpu().numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # 使用阈值检测异常（基于正常数据的重建误差分布）
        threshold = np.percentile(reconstruction_errors[:len(test_signals)//3], 95)  # 假设前1/3是正常数据
        anomaly_scores = reconstruction_errors > threshold
        
        return anomaly_scores.astype(int)
    
    def detect_anomalies_isolation_forest(self, normal_signals: List[np.ndarray], 
                                        test_signals: List[np.ndarray]) -> np.ndarray:
        """使用孤立森林检测异常"""
        print("🌲 使用孤立森林检测异常...")
        
        # 提取统计特征
        def extract_features(signals):
            features = []
            for signal in signals:
                feat = [
                    np.mean(signal), np.std(signal), np.var(signal),
                    np.min(signal), np.max(signal), np.median(signal),
                    np.percentile(signal, 25), np.percentile(signal, 75),
                    np.sum(np.abs(np.diff(signal))),  # 总变化量
                    len(np.where(np.diff(np.sign(signal)))[0]),  # 零交叉数
                ]
                features.append(feat)
            return np.array(features)
        
        # 提取特征
        normal_features = extract_features(normal_signals)
        test_features = extract_features(test_signals)
        
        # 标准化
        scaler = StandardScaler()
        normal_features_scaled = scaler.fit_transform(normal_features)
        test_features_scaled = scaler.transform(test_features)
        
        # 训练孤立森林
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(normal_features_scaled)
        
        # 预测异常
        predictions = iso_forest.predict(test_features_scaled)
        # 转换为0/1标签（-1表示异常，1表示正常）
        anomaly_scores = (predictions == -1).astype(int)
        
        return anomaly_scores
    
    def detect_anomalies_one_class_svm(self, normal_signals: List[np.ndarray], 
                                     test_signals: List[np.ndarray]) -> np.ndarray:
        """使用One-Class SVM检测异常"""
        print("🎯 使用One-Class SVM检测异常...")
        
        # 提取特征（同孤立森林）
        def extract_features(signals):
            features = []
            for signal in signals:
                feat = [
                    np.mean(signal), np.std(signal), np.var(signal),
                    np.min(signal), np.max(signal), np.median(signal),
                    np.percentile(signal, 25), np.percentile(signal, 75),
                    np.sum(np.abs(np.diff(signal))),
                    len(np.where(np.diff(np.sign(signal)))[0]),
                ]
                features.append(feat)
            return np.array(features)
        
        # 提取特征
        normal_features = extract_features(normal_signals)
        test_features = extract_features(test_signals)
        
        # 标准化
        scaler = StandardScaler()
        normal_features_scaled = scaler.fit_transform(normal_features)
        test_features_scaled = scaler.transform(test_features)
        
        # 训练One-Class SVM
        oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        oc_svm.fit(normal_features_scaled)
        
        # 预测异常
        predictions = oc_svm.predict(test_features_scaled)
        anomaly_scores = (predictions == -1).astype(int)
        
        return anomaly_scores
