#!/usr/bin/env python3
"""
高级深度学习模型
更深层、更复杂的网络结构
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
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedMotorDataset(Dataset):
    """高级电机信号数据集"""
    
    def __init__(self, signals: List[np.ndarray], labels: np.ndarray, 
                 max_length: int = 4096, augment: bool = False):
        self.labels = labels
        self.max_length = max_length
        self.augment = augment
        
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
                signal = np.pad(signal, (0, max_length - len(signal)), 'reflect')
            
            # 标准化
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            self.processed_signals.append(signal)
        
        print(f"📊 高级数据预处理完成: {len(self.processed_signals)} 个信号，长度 {max_length}")
    
    def __len__(self):
        return len(self.processed_signals)
    
    def __getitem__(self, idx):
        signal = self.processed_signals[idx].copy()
        label = self.labels[idx]
        
        # 在线数据增强
        if self.augment and np.random.random() > 0.5:
            # 添加噪声
            noise_level = 0.02 * np.std(signal)
            signal += np.random.normal(0, noise_level, len(signal))
            
            # 随机缩放
            scale = np.random.uniform(0.9, 1.1)
            signal *= scale
        
        return torch.FloatTensor(signal).unsqueeze(0), torch.LongTensor([label])

class ResidualBlock1D(nn.Module):
    """1D残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, downsample: bool = False):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class DeepResNet1D(nn.Module):
    """深层ResNet1D模型"""
    
    def __init__(self, input_length: int = 4096, num_classes: int = 3):
        super(DeepResNet1D, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        
        # 第一个块可能需要下采样
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, 
                                     downsample=(stride != 1 or in_channels != out_channels)))
        
        # 其余块
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.classifier(x)
        
        return x

class AttentionLSTM(nn.Module):
    """带注意力机制的深层LSTM"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 256, 
                 num_layers: int = 4, num_classes: int = 3):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多层双向LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, 
                                             dropout=0.1, batch_first=True)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 自注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 残差连接和层归一化
        attn_out = self.layer_norm(attn_out + lstm_out)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)
        
        # 分类
        output = self.classifier(pooled)
        return output

class DeepTransformer(nn.Module):
    """深层Transformer模型"""
    
    def __init__(self, input_size: int = 1, d_model: int = 512, 
                 nhead: int = 8, num_layers: int = 8, num_classes: int = 3):
        super(DeepTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(4096, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
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
        # x shape: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
        x = x.transpose(1, 2)
        seq_len = x.size(1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer编码
        x = self.transformer(x)
        
        # 层归一化
        x = self.layer_norm(x)
        
        # 转置用于池化
        x = x.transpose(1, 2)
        
        # 分类
        output = self.classifier(x)
        return output

class MultiSensorFusionNet(nn.Module):
    """多传感器融合网络"""
    
    def __init__(self, input_length: int = 4096, num_classes: int = 3):
        super(MultiSensorFusionNet, self).__init__()
        
        # 声音传感器分支
        self.audio_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            ResidualBlock1D(64, 128, downsample=True),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 256, downsample=True),
            ResidualBlock1D(256, 256),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 振动传感器分支
        self.vibration_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            ResidualBlock1D(64, 128, downsample=True),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 256, downsample=True),
            ResidualBlock1D(256, 256),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),  # 256 + 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, audio_signal, vibration_signal):
        # 分别处理两个传感器信号
        audio_features = self.audio_branch(audio_signal)
        vibration_features = self.vibration_branch(vibration_signal)
        
        # 特征融合
        fused_features = torch.cat([audio_features, vibration_features], dim=1)
        
        # 分类
        output = self.fusion(fused_features)
        return output
