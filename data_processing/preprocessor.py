#!/usr/bin/env python3
"""
数据预处理模块
包含信号去噪、滤波、标准化等预处理功能
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path

class SignalPreprocessor:
    """信号预处理器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sampling_rate = self.config['data']['sampling_rate']
        self.output_path = Path(self.config['output']['tables'])
    
    def remove_outliers(self, signal: np.ndarray, method: str = 'iqr', 
                       threshold: float = 3.0) -> np.ndarray:
        """移除异常值"""
        if method == 'iqr':
            Q1 = np.percentile(signal, 25)
            Q3 = np.percentile(signal, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 将异常值替换为边界值
            signal_clean = np.clip(signal, lower_bound, upper_bound)
            
        elif method == 'zscore':
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            z_scores = np.abs((signal - mean_val) / std_val)
            
            # 将超过阈值的点替换为均值
            signal_clean = signal.copy()
            outlier_mask = z_scores > threshold
            signal_clean[outlier_mask] = mean_val
            
        else:
            raise ValueError(f"Unknown outlier removal method: {method}")
        
        return signal_clean
    
    def apply_filter(self, signal: np.ndarray, filter_type: str = 'lowpass',
                    cutoff: float = 1000, order: int = 4) -> np.ndarray:
        """应用数字滤波器"""
        nyquist = self.sampling_rate / 2
        
        if filter_type == 'lowpass':
            b, a = butter(order, cutoff / nyquist, btype='low')
        elif filter_type == 'highpass':
            b, a = butter(order, cutoff / nyquist, btype='high')
        elif filter_type == 'bandpass':
            if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
                low, high = cutoff
                b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
            else:
                raise ValueError("Bandpass filter requires two cutoff frequencies")
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # 使用零相位滤波
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    def smooth_signal(self, signal: np.ndarray, method: str = 'savgol',
                     window_length: int = 51, polyorder: int = 3) -> np.ndarray:
        """信号平滑"""
        if method == 'savgol':
            # 确保窗口长度为奇数且小于信号长度
            window_length = min(window_length, len(signal))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length < polyorder + 1:
                window_length = polyorder + 1
                if window_length % 2 == 0:
                    window_length += 1
            
            smoothed = savgol_filter(signal, window_length, polyorder)
            
        elif method == 'moving_average':
            smoothed = np.convolve(signal, np.ones(window_length)/window_length, mode='same')
            
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        return smoothed
    
    def normalize_signal(self, signal: np.ndarray, method: str = 'zscore') -> Tuple[np.ndarray, dict]:
        """信号标准化"""
        if method == 'zscore':
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            normalized = (signal - mean_val) / std_val if std_val > 0 else signal - mean_val
            params = {'mean': mean_val, 'std': std_val}
            
        elif method == 'minmax':
            min_val = np.min(signal)
            max_val = np.max(signal)
            range_val = max_val - min_val
            normalized = (signal - min_val) / range_val if range_val > 0 else signal - min_val
            params = {'min': min_val, 'max': max_val}
            
        elif method == 'robust':
            median_val = np.median(signal)
            mad = np.median(np.abs(signal - median_val))
            normalized = (signal - median_val) / mad if mad > 0 else signal - median_val
            params = {'median': median_val, 'mad': mad}
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, params
    
    def segment_signal(self, signal: np.ndarray, segment_length: int,
                      overlap: float = 0.5) -> List[np.ndarray]:
        """信号分段"""
        step = int(segment_length * (1 - overlap))
        segments = []
        
        for start in range(0, len(signal) - segment_length + 1, step):
            segment = signal[start:start + segment_length]
            segments.append(segment)
        
        return segments
    
    def preprocess_signal(self, signal: np.ndarray, 
                         remove_outliers: bool = True,
                         apply_filter: bool = True,
                         smooth: bool = False,
                         normalize: bool = True,
                         **kwargs) -> Tuple[np.ndarray, dict]:
        """完整的信号预处理流程"""
        processed_signal = signal.copy()
        processing_info = {'original_length': len(signal)}
        
        # 1. 移除异常值
        if remove_outliers:
            outlier_method = kwargs.get('outlier_method', 'iqr')
            processed_signal = self.remove_outliers(processed_signal, method=outlier_method)
            processing_info['outlier_removal'] = outlier_method
        
        # 2. 滤波
        if apply_filter:
            filter_type = kwargs.get('filter_type', 'lowpass')
            cutoff = kwargs.get('cutoff', 1000)
            processed_signal = self.apply_filter(processed_signal, filter_type, cutoff)
            processing_info['filter'] = {'type': filter_type, 'cutoff': cutoff}
        
        # 3. 平滑
        if smooth:
            smooth_method = kwargs.get('smooth_method', 'savgol')
            window_length = kwargs.get('window_length', 51)
            processed_signal = self.smooth_signal(processed_signal, smooth_method, window_length)
            processing_info['smoothing'] = {'method': smooth_method, 'window': window_length}
        
        # 4. 标准化
        if normalize:
            norm_method = kwargs.get('norm_method', 'zscore')
            processed_signal, norm_params = self.normalize_signal(processed_signal, norm_method)
            processing_info['normalization'] = {'method': norm_method, 'params': norm_params}
        
        processing_info['final_length'] = len(processed_signal)
        
        return processed_signal, processing_info
    
    def preprocess_dataset(self, data: Dict, **preprocessing_kwargs) -> Tuple[Dict, Dict]:
        """预处理整个数据集"""
        print("🔧 开始数据预处理...")
        
        processed_data = {}
        processing_logs = {}
        
        for sensor in data.keys():
            processed_data[sensor] = {}
            processing_logs[sensor] = {}
            
            for state in data[sensor].keys():
                signals = data[sensor][state]
                processed_signals = []
                state_logs = []
                
                print(f"  处理 {sensor}/{state}: {len(signals)} 个信号")
                
                for i, signal in enumerate(signals):
                    try:
                        processed_signal, info = self.preprocess_signal(signal, **preprocessing_kwargs)
                        processed_signals.append(processed_signal)
                        state_logs.append(info)
                        
                    except Exception as e:
                        print(f"    ❌ 处理失败 {sensor}/{state} 信号 {i}: {e}")
                        continue
                
                processed_data[sensor][state] = processed_signals
                processing_logs[sensor][state] = state_logs
                
                print(f"    ✅ 完成 {len(processed_signals)}/{len(signals)} 个信号")
        
        # 保存处理日志
        self._save_processing_logs(processing_logs)
        
        print("✅ 数据预处理完成")
        return processed_data, processing_logs
    
    def _save_processing_logs(self, logs: Dict):
        """保存预处理日志"""
        import json
        from datetime import datetime
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'processing_logs': logs
        }
        
        log_path = self.output_path / 'preprocessing_logs.json'
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"📋 预处理日志已保存: {log_path}")
    
    def create_train_test_split(self, data: Dict, test_ratio: float = 0.2,
                               val_ratio: float = 0.1, random_state: int = 42) -> Dict:
        """创建训练/验证/测试集划分"""
        print("📊 创建数据集划分...")
        
        np.random.seed(random_state)
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for sensor in data.keys():
            for split in splits.keys():
                splits[split][sensor] = {}
        
        split_info = []
        
        for sensor in data.keys():
            for state in data[sensor].keys():
                signals = data[sensor][state]
                n_signals = len(signals)
                
                if n_signals == 0:
                    continue
                
                # 随机打乱索引
                indices = np.random.permutation(n_signals)
                
                # 计算划分点
                n_test = max(1, int(n_signals * test_ratio))
                n_val = max(1, int(n_signals * val_ratio))
                n_train = n_signals - n_test - n_val
                
                # 确保至少有一个样本在训练集
                if n_train <= 0:
                    n_train = 1
                    n_val = max(0, n_signals - n_train - n_test)
                    n_test = n_signals - n_train - n_val
                
                # 划分数据
                train_indices = indices[:n_train]
                val_indices = indices[n_train:n_train + n_val]
                test_indices = indices[n_train + n_val:]
                
                splits['train'][sensor][state] = [signals[i] for i in train_indices]
                splits['val'][sensor][state] = [signals[i] for i in val_indices]
                splits['test'][sensor][state] = [signals[i] for i in test_indices]
                
                # 记录划分信息
                split_info.append({
                    'sensor': sensor,
                    'state': state,
                    'total': n_signals,
                    'train': len(train_indices),
                    'val': len(val_indices),
                    'test': len(test_indices)
                })
        
        # 保存划分信息
        split_df = pd.DataFrame(split_info)
        split_path = self.output_path / 'data_splits.csv'
        split_df.to_csv(split_path, index=False)
        
        print(f"📊 数据划分信息已保存: {split_path}")
        print("📋 划分统计:")
        print(split_df.groupby('state')[['train', 'val', 'test']].sum())
        
        return splits

if __name__ == "__main__":
    # 测试预处理器
    from data_loader import MotorDataLoader
    
    # 加载数据
    loader = MotorDataLoader("../../experiments/configs/config.yaml")
    data, _ = loader.load_all_data(max_files_per_state=10)
    
    # 创建预处理器
    preprocessor = SignalPreprocessor("../../experiments/configs/config.yaml")
    
    # 预处理数据
    processed_data, logs = preprocessor.preprocess_dataset(
        data,
        remove_outliers=True,
        apply_filter=True,
        normalize=True,
        filter_type='lowpass',
        cutoff=1000
    )
    
    # 创建数据集划分
    splits = preprocessor.create_train_test_split(processed_data)
    
    print("\n🎉 预处理测试完成！")
    print(f"📁 结果保存在: {preprocessor.output_path}")
