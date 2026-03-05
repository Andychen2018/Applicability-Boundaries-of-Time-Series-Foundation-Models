#!/usr/bin/env python3
"""
方案A续：使用已训练的Chronos模型进行特征提取和分类
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

def extract_residual_features(y_true, y_pred, prediction_length=1024):
    """
    从预测残差中提取特征
    """
    residual = y_true - y_pred
    
    features = {}
    
    # 整体残差特征
    features['mae'] = np.mean(np.abs(residual))
    features['mse'] = np.mean(residual ** 2)
    features['rmse'] = np.sqrt(features['mse'])
    features['mape'] = np.mean(np.abs(residual / (y_true + 1e-8))) * 100
    features['mdae'] = np.median(np.abs(residual))
    
    # 分段残差特征 (4段)
    segment_size = max(1, prediction_length // 4)
    for i in range(4):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, len(residual))
        if start_idx < len(residual):
            segment_residual = residual[start_idx:end_idx]
            
            features[f'seg{i}_mae'] = np.mean(np.abs(segment_residual))
            features[f'seg{i}_mse'] = np.mean(segment_residual ** 2)
    
    # 分位数残差特征
    for q in [0.1, 0.5, 0.9]:
        features[f'residual_q{int(q*100)}'] = np.percentile(np.abs(residual), q * 100)
    
    # 简单的自相关特征
    for lag in range(1, 6):
        if len(residual) > lag:
            corr = np.corrcoef(residual[:-lag], residual[lag:])[0, 1]
            features[f'acf_lag{lag}'] = corr if not np.isnan(corr) else 0
    
    # 频域特征
    try:
        fft_residual = np.fft.fft(residual)
        freqs = np.fft.fftfreq(len(residual))
        
        # 高频/低频能量比
        low_freq_mask = np.abs(freqs) < 0.2
        high_freq_mask = np.abs(freqs) >= 0.2
        
        low_freq_energy = np.sum(np.abs(fft_residual[low_freq_mask]) ** 2)
        high_freq_energy = np.sum(np.abs(fft_residual[high_freq_mask]) ** 2)
        
        features['freq_ratio'] = high_freq_energy / (low_freq_energy + 1e-8)
    except:
        features['freq_ratio'] = 0
    
    return features

def continue_method_a_single_domain(domain, output_dir, prediction_length=1024, context_length=2048):
    """
    继续方案A：使用已训练的模型进行特征提取和分类
    """
    print(f"\n=== 继续处理 {domain} 域的方案A模型 ===")
    
    # 检查模型是否存在
    domain_output_dir = os.path.join(output_dir, f"methodA_residual_normal_ft/{domain}")
    predictor_path = os.path.join(domain_output_dir, "predictor")
    
    if not os.path.exists(predictor_path):
        print(f"错误: {domain} 域的模型不存在于 {predictor_path}")
        return
    
    # 加载已训练的模型
    print("加载已训练的Chronos模型...")
    predictor = TimeSeriesPredictor.load(predictor_path)
    
    # 读取数据
    train_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/train_data.csv")
    val_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/val_data.csv")
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    
    # 筛选域数据
    train_domain = train_df[train_df['item_id'].str.startswith(domain)]
    val_domain = val_df[val_df['item_id'].str.startswith(domain)]
    test_domain = test_df[test_df['item_id'].str.startswith(domain)]
    
    print(f"{domain} 域数据统计:")
    print(f"  训练集: {len(train_domain)} 条记录, {train_domain['item_id'].nunique()} 个序列")
    print(f"  验证集: {len(val_domain)} 条记录, {val_domain['item_id'].nunique()} 个序列")
    print(f"  测试集: {len(test_domain)} 条记录, {test_domain['item_id'].nunique()} 个序列")
    
    # 合并所有数据用于特征提取
    all_data = pd.concat([train_domain, val_domain, test_domain], ignore_index=True)
    
    # 为每个序列提取特征
    features_list = []
    labels_list = []
    item_ids_list = []
    
    unique_items = all_data['item_id'].unique()
    print(f"开始处理 {len(unique_items)} 个序列...")
    
    for i, item_id in enumerate(unique_items):
        if i % 50 == 0:
            print(f"处理进度: {i}/{len(unique_items)}")
            
        item_data = all_data[all_data['item_id'] == item_id].sort_values('timestamp')
        
        if len(item_data) >= context_length + prediction_length:
            # 准备预测数据
            context_data = item_data.iloc[:context_length].copy()
            true_future = item_data.iloc[context_length:context_length + prediction_length]['target'].values
            
            # 转换为Chronos格式
            chronos_data = TimeSeriesDataFrame(context_data)
            
            try:
                # 进行预测
                predictions = predictor.predict(chronos_data)
                
                # 修复预测结果索引问题
                if isinstance(predictions, pd.DataFrame):
                    # 如果predictions是DataFrame，取第一列的值
                    pred_values = predictions.iloc[:, 0].values
                elif hasattr(predictions, 'values'):
                    pred_values = predictions.values.flatten()
                else:
                    # 尝试直接访问item_id
                    try:
                        pred_values = predictions[item_id].values
                    except:
                        # 如果都失败了，取第一个可用的预测值
                        pred_values = list(predictions.values())[0].values if hasattr(predictions, 'values') else predictions
                
                # 确保pred_values是numpy数组且长度正确
                if not isinstance(pred_values, np.ndarray):
                    pred_values = np.array(pred_values)
                
                # 确保长度匹配
                min_length = min(len(true_future), len(pred_values))
                if min_length < 100:  # 如果预测长度太短，跳过
                    print(f"跳过 {item_id}: 预测长度太短 ({min_length})")
                    continue
                
                true_future_trimmed = true_future[:min_length]
                pred_values_trimmed = pred_values[:min_length]
                
                # 提取残差特征
                features = extract_residual_features(true_future_trimmed, pred_values_trimmed, min_length)
                features_list.append(features)
                labels_list.append(item_data['label'].iloc[0])
                item_ids_list.append(item_id)
                
            except Exception as e:
                print(f"预测 {item_id} 时出错: {str(e)}")
                continue
    
    if len(features_list) == 0:
        print(f"警告: {domain} 域没有成功提取到任何特征")
        return
    
    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['item_id'] = item_ids_list
    features_df['label'] = labels_list
    
    print(f"成功提取 {len(features_df)} 个序列的残差特征")
    print("标签分布:", features_df['label'].value_counts().to_dict())
    
    # 划分特征数据
    train_features = features_df[features_df['item_id'].isin(train_domain['item_id'].unique())]
    val_features = features_df[features_df['item_id'].isin(val_domain['item_id'].unique())]
    test_features = features_df[features_df['item_id'].isin(test_domain['item_id'].unique())]
    
    # 保存特征数据
    train_features.to_csv(os.path.join(domain_output_dir, "residual_features_train.csv"), index=False)
    val_features.to_csv(os.path.join(domain_output_dir, "residual_features_val.csv"), index=False)
    test_features.to_csv(os.path.join(domain_output_dir, "residual_features_test.csv"), index=False)
    
    print(f"特征数据保存完成:")
    print(f"  训练特征: {len(train_features)} 个")
    print(f"  验证特征: {len(val_features)} 个")
    print(f"  测试特征: {len(test_features)} 个")
    
    # 训练分类器
    print("开始训练分类器...")
    
    # 准备特征和标签
    feature_cols = [col for col in features_df.columns if col not in ['item_id', 'label']]
    
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features['label']
    X_val = val_features[feature_cols].fillna(0)
    y_val = val_features['label']
    X_test = test_features[feature_cols].fillna(0)
    y_test = test_features['label']
    
    print(f"训练集特征形状: {X_train.shape}")
    print(f"验证集特征形状: {X_val.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练多个分类器
    classifiers = {
        'LightGBM': lgb.LGBMClassifier(random_state=123, verbosity=-1),
        'RandomForest': RandomForestClassifier(random_state=123, n_estimators=100),
    }
    
    best_classifier = None
    best_score = 0
    best_clf_name = None
    results = {}
    
    for clf_name, clf in classifiers.items():
        print(f"训练 {clf_name}...")
        
        try:
            if clf_name == 'LightGBM':
                clf.fit(X_train, y_train)
                val_pred = clf.predict(X_val)
            else:
                clf.fit(X_train_scaled, y_train)
                val_pred = clf.predict(X_val_scaled)
            
            val_accuracy = accuracy_score(y_val, val_pred)
            results[clf_name] = {
                'val_accuracy': val_accuracy,
                'classifier': clf
            }
            
            print(f"{clf_name} 验证集准确率: {val_accuracy:.4f}")
            
            if val_accuracy > best_score:
                best_score = val_accuracy
                best_classifier = clf
                best_clf_name = clf_name
                
        except Exception as e:
            print(f"训练 {clf_name} 时出错: {str(e)}")
            continue
    
    # 使用最佳分类器进行最终评估
    if best_classifier is not None:
        print(f"\n最佳分类器: {best_clf_name} (验证集准确率: {best_score:.4f})")
        
        # 在测试集上评估
        if best_clf_name == 'LightGBM':
            test_pred = best_classifier.predict(X_test)
        else:
            test_pred = best_classifier.predict(X_test_scaled)
        
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # 保存分类器
        with open(os.path.join(domain_output_dir, "clf_model.pkl"), "wb") as f:
            pickle.dump({
                'classifier': best_classifier,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'classifier_name': best_clf_name
            }, f)
        
        # 保存评估结果
        metrics = {
            'domain': domain,
            'method': 'A_residual_normal_ft',
            'best_classifier': best_clf_name,
            'val_accuracy': best_score,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
        }
        
        with open(os.path.join(domain_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"分类报告:\n{classification_report(y_test, test_pred)}")
        
        return True
    else:
        print("没有成功训练任何分类器")
        return False

def main():
    output_dir = "/home/deep/TimeSeries/Zhendong/output"
    
    # 处理各个域的模型
    domains = ["ShengYing"]  # 先处理已经训练好的ShengYing域
    
    for domain in domains:
        try:
            success = continue_method_a_single_domain(domain, output_dir)
            if success:
                print(f"{domain} 域方案A处理完成")
            else:
                print(f"{domain} 域方案A处理失败")
        except Exception as e:
            print(f"处理 {domain} 域时出错: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
