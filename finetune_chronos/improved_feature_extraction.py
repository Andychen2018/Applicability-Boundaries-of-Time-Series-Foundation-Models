#!/usr/bin/env python3
"""
改进的特征提取方法：
方法1: 直接使用48个预测残差作为特征
方法2: 提取更丰富的时频域特征
"""

import pandas as pd
import numpy as np
import os
import json
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from scipy import signal
from scipy.stats import skew, kurtosis

def method_1_direct_residuals():
    """方法1: 直接使用48个预测残差作为特征"""
    print("\n" + "="*60)
    print("方法1: 直接使用48个预测残差作为特征")
    print("="*60)
    
    # 加载Normal-only微调模型
    model_path = "/home/deep/TimeSeries/Zhendong/output/normal_only_finetune/ZhenDong/predictor"
    print(f"加载模型: {model_path}")
    predictor = TimeSeriesPredictor.load(model_path)
    
    # 加载测试数据
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]
    
    print(f"ZhenDong测试数据: {zhendong_test['item_id'].nunique()} 个序列")
    
    # 提取48维残差特征
    features_list = []
    labels_list = []
    item_ids_list = []
    
    prediction_length = 48
    context_length = 2048
    
    unique_items = zhendong_test['item_id'].unique()
    print(f"开始提取48维残差特征...")
    
    for i, item_id in enumerate(unique_items):
        if i % 20 == 0:
            print(f"进度: {i}/{len(unique_items)}")
        
        item_data = zhendong_test[zhendong_test['item_id'] == item_id].sort_values('timestamp')
        
        if len(item_data) >= context_length + prediction_length:
            # 准备数据
            context_data = item_data.iloc[:context_length].copy()
            true_future = item_data.iloc[context_length:context_length + prediction_length]['target'].values
            
            context_data['timestamp'] = range(len(context_data))
            chronos_data = TimeSeriesDataFrame(context_data)
            
            try:
                # 预测
                predictions = predictor.predict(chronos_data)
                
                # 处理预测结果
                if isinstance(predictions, pd.DataFrame):
                    pred_values = predictions.iloc[:, 0].values
                else:
                    pred_values = list(predictions.values())[0].values
                
                # 确保长度匹配
                min_length = min(len(true_future), len(pred_values))
                true_future = true_future[:min_length]
                pred_values = pred_values[:min_length]
                
                # 直接使用残差作为特征
                residual = true_future - pred_values
                
                # 如果残差长度不足48，用0填充；如果超过48，截取前48个
                if len(residual) < 48:
                    residual_features = np.pad(residual, (0, 48 - len(residual)), 'constant')
                else:
                    residual_features = residual[:48]
                
                features_list.append(residual_features)
                labels_list.append(item_data['label'].iloc[0])
                item_ids_list.append(item_id)
                
            except Exception as e:
                print(f"处理 {item_id} 时出错: {e}")
                continue
    
    if len(features_list) == 0:
        print("没有成功提取到特征")
        return None
    
    # 转换为DataFrame
    features_array = np.array(features_list)
    feature_cols = [f'residual_{i}' for i in range(48)]
    features_df = pd.DataFrame(features_array, columns=feature_cols)
    features_df['item_id'] = item_ids_list
    features_df['label'] = labels_list
    
    print(f"成功提取 {len(features_df)} 个序列的48维残差特征")
    print("标签分布:", features_df['label'].value_counts().to_dict())
    
    # 训练分类器
    X = features_df[feature_cols]
    y = features_df['label']
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练分类器
    classifiers = {
        'LightGBM': lgb.LGBMClassifier(random_state=123, verbosity=-1),
        'SVM': SVC(random_state=123, probability=True),
        'RandomForest': RandomForestClassifier(random_state=123, n_estimators=100),
    }
    
    results = {}
    for name, clf in classifiers.items():
        print(f"\n训练 {name}...")
        
        if name == 'LightGBM':
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"{name} 准确率: {accuracy:.4f}")
        print(f"分类报告:\n{classification_report(y_test, y_pred)}")
    
    # 保存结果
    output_dir = "/home/deep/TimeSeries/Zhendong/output/method_1_direct_residuals"
    os.makedirs(output_dir, exist_ok=True)
    
    features_df.to_csv(os.path.join(output_dir, "direct_residual_features.csv"), index=False)
    
    with open(os.path.join(output_dir, "classification_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n方法1结果保存到: {output_dir}")
    
    return results

def extract_rich_features(time_series, sampling_rate=65536):
    """提取丰富的时频域特征"""
    
    features = {}
    
    # 时域特征
    features['mean'] = np.mean(time_series)
    features['std'] = np.std(time_series)
    features['var'] = np.var(time_series)
    features['min'] = np.min(time_series)
    features['max'] = np.max(time_series)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(time_series)
    features['q25'] = np.percentile(time_series, 25)
    features['q75'] = np.percentile(time_series, 75)
    features['iqr'] = features['q75'] - features['q25']
    features['skewness'] = skew(time_series)
    features['kurtosis'] = kurtosis(time_series)
    
    # RMS特征
    features['rms'] = np.sqrt(np.mean(time_series**2))
    
    # 峰值特征
    features['peak_to_peak'] = np.ptp(time_series)
    features['crest_factor'] = features['max'] / features['rms'] if features['rms'] != 0 else 0
    
    # 形状因子
    features['shape_factor'] = features['rms'] / np.mean(np.abs(time_series)) if np.mean(np.abs(time_series)) != 0 else 0
    
    # 脉冲因子
    features['impulse_factor'] = features['max'] / np.mean(np.abs(time_series)) if np.mean(np.abs(time_series)) != 0 else 0
    
    # 余隙因子
    features['clearance_factor'] = features['max'] / (np.mean(np.sqrt(np.abs(time_series)))**2) if np.mean(np.sqrt(np.abs(time_series))) != 0 else 0
    
    # 频域特征
    try:
        # FFT
        fft = np.fft.fft(time_series)
        fft_magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(time_series), 1/sampling_rate)[:len(fft)//2]
        
        # 频域统计特征
        features['fft_mean'] = np.mean(fft_magnitude)
        features['fft_std'] = np.std(fft_magnitude)
        features['fft_max'] = np.max(fft_magnitude)
        
        # 主频率
        dominant_freq_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = freqs[dominant_freq_idx]
        
        # 频率重心
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) != 0 else 0
        
        # 频谱能量分布
        total_energy = np.sum(fft_magnitude**2)
        if total_energy > 0:
            # 低频能量 (0-1000Hz)
            low_freq_mask = freqs <= 1000
            features['low_freq_energy'] = np.sum(fft_magnitude[low_freq_mask]**2) / total_energy
            
            # 中频能量 (1000-10000Hz)
            mid_freq_mask = (freqs > 1000) & (freqs <= 10000)
            features['mid_freq_energy'] = np.sum(fft_magnitude[mid_freq_mask]**2) / total_energy
            
            # 高频能量 (>10000Hz)
            high_freq_mask = freqs > 10000
            features['high_freq_energy'] = np.sum(fft_magnitude[high_freq_mask]**2) / total_energy
        else:
            features['low_freq_energy'] = 0
            features['mid_freq_energy'] = 0
            features['high_freq_energy'] = 0
            
    except Exception as e:
        print(f"频域特征提取错误: {e}")
        # 如果频域特征提取失败，设置默认值
        for key in ['fft_mean', 'fft_std', 'fft_max', 'dominant_frequency', 'spectral_centroid', 
                   'low_freq_energy', 'mid_freq_energy', 'high_freq_energy']:
            features[key] = 0
    
    return features

def method_2_rich_features():
    """方法2: 提取丰富的时频域特征"""
    print("\n" + "="*60)
    print("方法2: 提取丰富的时频域特征")
    print("="*60)
    
    # 加载测试数据
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    zhendong_test = test_df[test_df['item_id'].str.startswith('ZhenDong')]
    
    print(f"ZhenDong测试数据: {zhendong_test['item_id'].nunique()} 个序列")
    
    # 提取丰富特征
    features_list = []
    labels_list = []
    item_ids_list = []
    
    unique_items = zhendong_test['item_id'].unique()
    print(f"开始提取丰富的时频域特征...")
    
    for i, item_id in enumerate(unique_items):
        if i % 20 == 0:
            print(f"进度: {i}/{len(unique_items)}")
        
        item_data = zhendong_test[zhendong_test['item_id'] == item_id].sort_values('timestamp')
        
        if len(item_data) >= 2048:  # 至少需要足够的数据点
            # 使用完整序列或前32768个点
            time_series = item_data['target'].values[:32768]  # 使用前一半数据
            
            try:
                # 提取丰富特征
                features = extract_rich_features(time_series)
                
                features_list.append(features)
                labels_list.append(item_data['label'].iloc[0])
                item_ids_list.append(item_id)
                
            except Exception as e:
                print(f"处理 {item_id} 时出错: {e}")
                continue
    
    if len(features_list) == 0:
        print("没有成功提取到特征")
        return None
    
    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['item_id'] = item_ids_list
    features_df['label'] = labels_list
    
    print(f"成功提取 {len(features_df)} 个序列的丰富特征")
    print(f"特征维度: {len(features_df.columns) - 2}")  # 减去item_id和label
    print("标签分布:", features_df['label'].value_counts().to_dict())
    
    # 训练分类器
    feature_cols = [col for col in features_df.columns if col not in ['item_id', 'label']]
    X = features_df[feature_cols].fillna(0)
    y = features_df['label']
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练分类器
    classifiers = {
        'LightGBM': lgb.LGBMClassifier(random_state=123, verbosity=-1),
        'SVM': SVC(random_state=123, probability=True),
        'RandomForest': RandomForestClassifier(random_state=123, n_estimators=100),
    }
    
    results = {}
    for name, clf in classifiers.items():
        print(f"\n训练 {name}...")
        
        if name == 'LightGBM':
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"{name} 准确率: {accuracy:.4f}")
        print(f"分类报告:\n{classification_report(y_test, y_pred)}")
    
    # 保存结果
    output_dir = "/home/deep/TimeSeries/Zhendong/output/method_2_rich_features"
    os.makedirs(output_dir, exist_ok=True)
    
    features_df.to_csv(os.path.join(output_dir, "rich_features.csv"), index=False)
    
    with open(os.path.join(output_dir, "classification_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n方法2结果保存到: {output_dir}")
    
    return results

def main():
    print("开始改进的特征提取实验...")
    
    # 方法1: 直接使用48个残差
    results_1 = method_1_direct_residuals()
    
    # 方法2: 丰富的时频域特征
    results_2 = method_2_rich_features()
    
    # 对比结果
    print("\n" + "="*60)
    print("改进方法对比总结")
    print("="*60)
    
    if results_1:
        print("方法1 (48维直接残差):")
        for clf_name, result in results_1.items():
            print(f"  {clf_name}: {result['accuracy']:.4f}")
    
    if results_2:
        print("\n方法2 (丰富时频域特征):")
        for clf_name, result in results_2.items():
            print(f"  {clf_name}: {result['accuracy']:.4f}")

if __name__ == "__main__":
    main()
