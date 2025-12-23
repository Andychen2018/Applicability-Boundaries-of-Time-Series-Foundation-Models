"""
数据加载和预处理工具函数
支持三种分类方式：
1. 仅使用ShengYing数据
2. 仅使用ZhenDong数据  
3. 融合两个测点数据
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class MotorDataLoader:
    def __init__(self, data_path: str = "data3"):
        self.data_path = data_path
        self.sampling_rate = 65535  # Hz - 每个文件1秒数据，包含65535个数据点
        self.label_encoder = LabelEncoder()
        
    def get_file_pairs(self) -> Dict[str, List[Tuple[str, str]]]:
        """获取配对的文件列表"""
        file_pairs = {'normal': [], 'spark': [], 'vibrate': []}
        
        for category in ['normal', 'spark', 'vibrate']:
            shengying_dir = os.path.join(self.data_path, 'ShengYing', category)
            zhendong_dir = os.path.join(self.data_path, 'ZhenDong', category)
            
            shengying_files = sorted([f for f in os.listdir(shengying_dir) if f.endswith('.csv')])
            zhendong_files = sorted([f for f in os.listdir(zhendong_dir) if f.endswith('.csv')])
            
            # 根据文件名匹配配对
            for sy_file in shengying_files:
                # 提取文件标识符
                identifier = sy_file.replace('a_', '').replace('.csv', '')
                zd_file = f'a_{identifier}.csv'
                
                if zd_file in zhendong_files:
                    sy_path = os.path.join(shengying_dir, sy_file)
                    zd_path = os.path.join(zhendong_dir, zd_file)
                    file_pairs[category].append((sy_path, zd_path))
        
        return file_pairs
    
    def load_single_file(self, file_path: str) -> np.ndarray:
        """加载单个CSV文件"""
        try:
            data = pd.read_csv(file_path, header=None).values.flatten()
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_data(self, mode: str = 'fusion', max_length: int = 65536) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据
        mode: 'shengying', 'zhendong', 'fusion'
        max_length: 最大信号长度，超过的部分会被截断，不足的部分会被填充
        """
        file_pairs = self.get_file_pairs()

        X_data = []
        y_data = []

        for category, pairs in file_pairs.items():
            print(f"Loading {category} data: {len(pairs)} pairs")

            for sy_path, zd_path in pairs:
                sy_data = self.load_single_file(sy_path)
                zd_data = self.load_single_file(zd_path)

                if sy_data is None or zd_data is None:
                    continue

                # 标准化信号长度
                sy_data = self._standardize_length(sy_data, max_length)
                zd_data = self._standardize_length(zd_data, max_length)

                if mode == 'shengying':
                    X_data.append(sy_data)
                elif mode == 'zhendong':
                    X_data.append(zd_data)
                elif mode == 'fusion':
                    # 简单拼接两个信号
                    fused_data = np.concatenate([sy_data, zd_data])
                    X_data.append(fused_data)

                y_data.append(category)

        X = np.array(X_data)
        y = np.array(y_data)

        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"Data shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y_encoded)}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        return X, y_encoded

    def _standardize_length(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """标准化信号长度"""
        if len(data) > target_length:
            # 截断
            return data[:target_length]
        elif len(data) < target_length:
            # 零填充
            padded = np.zeros(target_length)
            padded[:len(data)] = data
            return padded
        else:
            return data
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """分割训练和测试数据"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, 
                              stratify=y)
    
    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """标准化数据"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    def analyze_data_distribution(self, output_dir: str = "output/images"):
        """分析数据分布"""
        os.makedirs(output_dir, exist_ok=True)
        
        file_pairs = self.get_file_pairs()
        
        # 统计信息
        stats = {}
        for category, pairs in file_pairs.items():
            stats[category] = len(pairs)
        
        # 绘制数据分布
        plt.figure(figsize=(10, 6))
        categories = list(stats.keys())
        counts = list(stats.values())
        
        plt.subplot(1, 2, 1)
        plt.bar(categories, counts)
        plt.title('Data Distribution by Category')
        plt.ylabel('Number of Samples')
        
        # 分析信号长度
        signal_lengths = []
        for category, pairs in file_pairs.items():
            for sy_path, zd_path in pairs[:5]:  # 只分析前5个样本
                sy_data = self.load_single_file(sy_path)
                if sy_data is not None:
                    signal_lengths.append(len(sy_data))
        
        plt.subplot(1, 2, 2)
        plt.hist(signal_lengths, bins=20)
        plt.title('Signal Length Distribution')
        plt.xlabel('Signal Length')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return stats
    
    def visualize_sample_signals(self, output_dir: str = "output/images", n_samples: int = 2):
        """可视化样本信号"""
        os.makedirs(output_dir, exist_ok=True)
        
        file_pairs = self.get_file_pairs()
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        for i, (category, pairs) in enumerate(file_pairs.items()):
            for j in range(min(n_samples, len(pairs))):
                sy_path, zd_path = pairs[j]
                sy_data = self.load_single_file(sy_path)
                zd_data = self.load_single_file(zd_path)
                
                if sy_data is not None and zd_data is not None:
                    # 只显示前1000个点
                    time_axis = np.arange(1000) / self.sampling_rate
                    
                    axes[i, 0].plot(time_axis, sy_data[:1000])
                    axes[i, 0].set_title(f'{category} - ShengYing')
                    axes[i, 0].set_xlabel('Time (s)')
                    axes[i, 0].set_ylabel('Amplitude')
                    
                    axes[i, 1].plot(time_axis, zd_data[:1000])
                    axes[i, 1].set_title(f'{category} - ZhenDong')
                    axes[i, 1].set_xlabel('Time (s)')
                    axes[i, 1].set_ylabel('Amplitude')
                    break
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_signals.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """测试数据加载功能"""
    loader = MotorDataLoader()
    
    # 分析数据分布
    print("Analyzing data distribution...")
    stats = loader.analyze_data_distribution()
    print("Data statistics:", stats)
    
    # 可视化样本信号
    print("Visualizing sample signals...")
    loader.visualize_sample_signals()
    
    # 测试三种加载模式
    for mode in ['shengying', 'zhendong', 'fusion']:
        print(f"\nTesting {mode} mode...")
        X, y = loader.load_data(mode=mode)
        print(f"Data shape: {X.shape}")
        print(f"Unique labels: {np.unique(y)}")

if __name__ == "__main__":
    main()
