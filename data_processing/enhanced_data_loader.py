#!/usr/bin/env python3
"""
增强数据加载器
充分利用多传感器数据，实现数据融合和增强
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedMotorDataLoader:
    """增强的电机数据加载器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = Path(self.config['data']['path'])
        self.output_path = Path(self.config['output']['tables'])
        
        # 传感器配置
        self.sensors = ['ShengYing', 'ZhenDong']  # 声音和振动传感器
        self.states = ['normal', 'spark', 'vibrate']
        
        print(f"📂 数据路径: {self.data_path}")
        print(f"🔧 传感器: {self.sensors}")
        print(f"📊 状态: {self.states}")
    
    def load_all_data_enhanced(self, max_files_per_state: int = None) -> Tuple[Dict, Dict]:
        """增强数据加载 - 充分利用所有数据"""
        print("📂 开始增强数据加载...")
        
        all_data = {}
        file_info = {}
        
        for sensor in self.sensors:
            all_data[sensor] = {}
            file_info[sensor] = {}
            
            sensor_path = self.data_path / sensor
            
            for state in self.states:
                state_path = sensor_path / state
                
                if not state_path.exists():
                    print(f"⚠️ 路径不存在: {state_path}")
                    all_data[sensor][state] = []
                    file_info[sensor][state] = []
                    continue
                
                # 获取所有.csv文件
                csv_files = list(state_path.glob("*.csv"))

                if max_files_per_state:
                    csv_files = csv_files[:max_files_per_state]

                signals = []
                files = []

                for file_path in csv_files:
                    try:
                        # 读取CSV信号数据
                        data = pd.read_csv(file_path, header=None)

                        # 假设信号在第一列，如果有多列取第一列
                        if len(data.columns) > 1:
                            signal = data.iloc[:, 0].values  # 取第一列
                        else:
                            signal = data.values.flatten()

                        # 基本质量检查
                        if len(signal) > 100 and not np.any(np.isnan(signal)):
                            signals.append(signal)
                            files.append(file_path.name)

                    except Exception as e:
                        print(f"❌ 读取失败 {file_path}: {e}")
                        continue
                
                all_data[sensor][state] = signals
                file_info[sensor][state] = files
                
                print(f"✅ 加载 {sensor}/{state}: {len(signals)} 个文件")
        
        return all_data, file_info
    
    def create_multi_sensor_dataset(self, data: Dict, file_info: Dict) -> Tuple[Dict, Dict]:
        """创建多传感器融合数据集"""
        print("🔗 创建多传感器融合数据集...")

        # 按文件名匹配多传感器数据
        matched_data = {}
        single_sensor_data = {}

        for state in self.states:
            matched_data[state] = []
            single_sensor_data[state] = []

            # 获取两个传感器的文件名
            shengying_files = set(f.replace('.csv', '') for f in
                                file_info.get('ShengYing', {}).get(state, []))
            zhendong_files = set(f.replace('.csv', '') for f in
                               file_info.get('ZhenDong', {}).get(state, []))
            
            # 找到匹配的文件
            common_files = shengying_files.intersection(zhendong_files)
            
            print(f"  {state}: 声音{len(shengying_files)}, 振动{len(zhendong_files)}, 匹配{len(common_files)}")
            
            # 创建匹配的多传感器样本
            for file_base in common_files:
                try:
                    # 找到对应的信号
                    sy_idx = next(i for i, f in enumerate(file_info['ShengYing'][state])
                                if f.replace('.csv', '') == file_base)
                    zd_idx = next(i for i, f in enumerate(file_info['ZhenDong'][state])
                                if f.replace('.csv', '') == file_base)
                    
                    sy_signal = data['ShengYing'][state][sy_idx]
                    zd_signal = data['ZhenDong'][state][zd_idx]
                    
                    # 多传感器融合样本
                    matched_data[state].append({
                        'ShengYing': sy_signal,
                        'ZhenDong': zd_signal,
                        'file_id': file_base
                    })
                    
                except:
                    continue
            
            # 添加单传感器样本
            for sensor in self.sensors:
                if sensor in data and state in data[sensor]:
                    for i, signal in enumerate(data[sensor][state]):
                        single_sensor_data[state].append({
                            'sensor': sensor,
                            'signal': signal,
                            'file_id': f"{sensor}_{state}_{i}"
                        })
        
        return matched_data, single_sensor_data
    
    def augment_data(self, signals: List[np.ndarray], augment_factor: int = 3) -> List[np.ndarray]:
        """数据增强"""
        augmented_signals = []
        
        for signal in signals:
            # 原始信号
            augmented_signals.append(signal)
            
            for _ in range(augment_factor):
                # 添加噪声
                noise_level = 0.01 * np.std(signal)
                noisy_signal = signal + np.random.normal(0, noise_level, len(signal))
                augmented_signals.append(noisy_signal)
                
                # 时间拉伸/压缩
                stretch_factor = np.random.uniform(0.9, 1.1)
                stretched_indices = np.linspace(0, len(signal)-1, 
                                              int(len(signal) * stretch_factor))
                stretched_signal = np.interp(stretched_indices, 
                                           np.arange(len(signal)), signal)
                # 重采样到原长度
                if len(stretched_signal) != len(signal):
                    stretched_signal = np.interp(np.linspace(0, len(stretched_signal)-1, len(signal)),
                                                np.arange(len(stretched_signal)), stretched_signal)
                augmented_signals.append(stretched_signal)
                
                # 幅值缩放
                scale_factor = np.random.uniform(0.8, 1.2)
                scaled_signal = signal * scale_factor
                augmented_signals.append(scaled_signal)
        
        print(f"📈 数据增强: {len(signals)} -> {len(augmented_signals)} 个样本")
        return augmented_signals
    
    def create_fusion_features(self, matched_data: Dict) -> Tuple[List, List]:
        """创建融合特征"""
        print("🔗 创建多传感器融合特征...")
        
        fusion_signals = []
        fusion_labels = []
        
        for state in self.states:
            for sample in matched_data[state]:
                sy_signal = sample['ShengYing']
                zd_signal = sample['ZhenDong']
                
                # 确保信号长度一致
                min_len = min(len(sy_signal), len(zd_signal))
                sy_signal = sy_signal[:min_len]
                zd_signal = zd_signal[:min_len]
                
                # 多种融合策略
                fusion_methods = [
                    # 1. 简单拼接
                    np.concatenate([sy_signal, zd_signal]),
                    
                    # 2. 加权平均
                    0.6 * sy_signal + 0.4 * zd_signal,
                    
                    # 3. 差值特征
                    sy_signal - zd_signal,
                    
                    # 4. 乘积特征
                    sy_signal * zd_signal,
                    
                    # 5. 最大值特征
                    np.maximum(sy_signal, zd_signal),
                ]
                
                for fused_signal in fusion_methods:
                    fusion_signals.append(fused_signal)
                    fusion_labels.append(state)
        
        print(f"🔗 融合特征创建完成: {len(fusion_signals)} 个样本")
        return fusion_signals, fusion_labels
    
    def load_comprehensive_dataset(self, max_files_per_state: int = None, 
                                 enable_augmentation: bool = True) -> Tuple[Dict, Dict]:
        """加载综合数据集"""
        print("🚀 开始加载综合数据集...")
        
        # 1. 加载原始数据
        raw_data, file_info = self.load_all_data_enhanced(max_files_per_state)
        
        # 2. 创建多传感器数据集
        matched_data, single_sensor_data = self.create_multi_sensor_dataset(raw_data, file_info)
        
        # 3. 创建最终数据集
        final_dataset = {
            'single_sensor': {},
            'multi_sensor': {},
            'fusion_features': {}
        }
        
        # 单传感器数据
        for state in self.states:
            final_dataset['single_sensor'][state] = []
            
            # 收集所有单传感器信号
            all_signals = []
            for sensor in self.sensors:
                if sensor in raw_data and state in raw_data[sensor]:
                    all_signals.extend(raw_data[sensor][state])
            
            # 数据增强
            if enable_augmentation and len(all_signals) > 0:
                all_signals = self.augment_data(all_signals, augment_factor=2)
            
            final_dataset['single_sensor'][state] = all_signals
        
        # 多传感器匹配数据
        for state in self.states:
            final_dataset['multi_sensor'][state] = []
            for sample in matched_data[state]:
                # 添加原始多传感器样本
                final_dataset['multi_sensor'][state].append({
                    'ShengYing': sample['ShengYing'],
                    'ZhenDong': sample['ZhenDong']
                })
        
        # 融合特征
        fusion_signals, fusion_labels = self.create_fusion_features(matched_data)
        for state in self.states:
            final_dataset['fusion_features'][state] = [
                sig for sig, label in zip(fusion_signals, fusion_labels) if label == state
            ]
        
        # 统计信息
        print("\n📊 综合数据集统计:")
        for dataset_type in final_dataset:
            print(f"  {dataset_type}:")
            for state in self.states:
                count = len(final_dataset[dataset_type][state])
                print(f"    {state}: {count} 个样本")
        
        return final_dataset, file_info
    
    def save_dataset_info(self, dataset: Dict, file_info: Dict):
        """保存数据集信息"""
        # 创建统计信息
        stats = {}
        total_samples = 0
        
        for dataset_type in dataset:
            stats[dataset_type] = {}
            for state in self.states:
                count = len(dataset[dataset_type][state])
                stats[dataset_type][state] = count
                total_samples += count
        
        stats['total_samples'] = total_samples
        
        # 保存统计信息
        import json
        from datetime import datetime
        
        info_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'file_info': file_info
        }
        
        info_path = self.output_path / 'enhanced_dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info_data, f, indent=2, default=str)
        
        print(f"📋 数据集信息已保存: {info_path}")
        print(f"📊 总样本数: {total_samples}")

if __name__ == "__main__":
    # 测试增强数据加载器
    config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
    
    # 创建加载器
    loader = EnhancedMotorDataLoader(str(config_path))
    
    # 加载综合数据集
    dataset, file_info = loader.load_comprehensive_dataset(
        max_files_per_state=None,  # 使用所有数据
        enable_augmentation=True
    )
    
    # 保存数据集信息
    loader.save_dataset_info(dataset, file_info)
    
    print("\n🎉 增强数据加载完成！")
