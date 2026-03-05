#!/usr/bin/env python3
"""
完整的异常检测流程：
1. 使用Normal-only微调的Chronos模型
2. 对所有数据进行预测
3. 提取残差特征
4. 训练分类器进行异常检测
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

def extract_residual_features(y_true, y_pred):
    """提取残差特征"""
    residual = y_true - y_pred
    
    features = {
        # 基本统计特征
        'mae': np.mean(np.abs(residual)),
        'mse': np.mean(residual ** 2),
        'rmse': np.sqrt(np.mean(residual ** 2)),
        'std': np.std(residual),
        'mean': np.mean(residual),
        
        # 分位数特征
        'q10': np.percentile(np.abs(residual), 10),
        'q25': np.percentile(np.abs(residual), 25),
        'q50': np.percentile(np.abs(residual), 50),
        'q75': np.percentile(np.abs(residual), 75),
        'q90': np.percentile(np.abs(residual), 90),
        
        # 分段特征（4段）
        'seg0_mae': np.mean(np.abs(residual[:len(residual)//4])),
        'seg1_mae': np.mean(np.abs(residual[len(residual)//4:len(residual)//2])),
        'seg2_mae': np.mean(np.abs(residual[len(residual)//2:3*len(residual)//4])),
        'seg3_mae': np.mean(np.abs(residual[3*len(residual)//4:])),
    }
    
    # 自相关特征
    for lag in range(1, 6):
        if len(residual) > lag:
            corr = np.corrcoef(residual[:-lag], residual[lag:])[0, 1]
            features[f'acf_lag{lag}'] = corr if not np.isnan(corr) else 0
    
    return features

def anomaly_detection_pipeline(domain='ZhenDong', model_path=None):
    """完整的异常检测流程"""
    
    print(f"开始{domain}域的异常检测流程...")
    
    # ================== 1. 加载数据 ==================
    train_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/train_data.csv")
    val_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/val_data.csv")
    test_df = pd.read_csv("/home/deep/TimeSeries/Zhendong/output/test_data.csv")
    
    # 筛选域数据
    train_domain = train_df[train_df['item_id'].str.startswith(domain)]
    val_domain = val_df[val_df['item_id'].str.startswith(domain)]
    test_domain = test_df[test_df['item_id'].str.startswith(domain)]
    
    print(f"{domain}域数据统计:")
    print(f"  训练集: {train_domain['item_id'].nunique()} 个序列")
    print(f"  验证集: {val_domain['item_id'].nunique()} 个序列")
    print(f"  测试集: {test_domain['item_id'].nunique()} 个序列")
    
    # ================== 2. 加载微调后的模型 ==================
    if model_path is None:
        model_path = f"/home/deep/TimeSeries/Zhendong/output/normal_only_finetune/{domain}/predictor"
    
    if not os.path.exists(model_path):
        print(f"错误：模型不存在 {model_path}")
        print("请先运行 normal_only_finetune.py 训练模型")
        return
    
    print(f"加载模型: {model_path}")
    predictor = TimeSeriesPredictor.load(model_path)
    
    # ================== 3. 特征提取 ==================
    print("开始提取残差特征...")
    
    all_data = pd.concat([train_domain, val_domain, test_domain], ignore_index=True)
    
    features_list = []
    labels_list = []
    item_ids_list = []
    
    prediction_length = 1024
    context_length = 2048
    
    unique_items = all_data['item_id'].unique()
    print(f"处理 {len(unique_items)} 个序列...")
    
    for i, item_id in enumerate(unique_items):
        if i % 50 == 0:
            print(f"进度: {i}/{len(unique_items)}")
        
        item_data = all_data[all_data['item_id'] == item_id].sort_values('timestamp')
        
        if len(item_data) >= context_length + prediction_length:
            # 准备数据
            context_data = item_data.iloc[:context_length].copy()
            true_future = item_data.iloc[context_length:context_length + prediction_length]['target'].values
            
            # 添加timestamp列
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
                
                # 提取特征
                features = extract_residual_features(true_future, pred_values)
                features_list.append(features)
                labels_list.append(item_data['label'].iloc[0])
                item_ids_list.append(item_id)
                
            except Exception as e:
                print(f"处理 {item_id} 时出错: {e}")
                continue
    
    if len(features_list) == 0:
        print("没有成功提取到特征")
        return
    
    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['item_id'] = item_ids_list
    features_df['label'] = labels_list
    
    print(f"成功提取 {len(features_df)} 个序列的特征")
    print("标签分布:", features_df['label'].value_counts().to_dict())
    
    # ================== 4. 数据划分 ==================
    train_features = features_df[features_df['item_id'].isin(train_domain['item_id'].unique())]
    val_features = features_df[features_df['item_id'].isin(val_domain['item_id'].unique())]
    test_features = features_df[features_df['item_id'].isin(test_domain['item_id'].unique())]
    
    # ================== 5. 训练分类器 ==================
    print("训练分类器...")
    
    feature_cols = [col for col in features_df.columns if col not in ['item_id', 'label']]
    
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features['label']
    X_val = val_features[feature_cols].fillna(0)
    y_val = val_features['label']
    X_test = test_features[feature_cols].fillna(0)
    y_test = test_features['label']
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练分类器
    classifiers = {
        'LightGBM': lgb.LGBMClassifier(random_state=123, verbosity=-1),
        'RandomForest': RandomForestClassifier(random_state=123, n_estimators=100),
    }
    
    best_classifier = None
    best_score = 0
    best_name = None
    
    for name, clf in classifiers.items():
        if name == 'LightGBM':
            clf.fit(X_train, y_train)
            val_pred = clf.predict(X_val)
        else:
            clf.fit(X_train_scaled, y_train)
            val_pred = clf.predict(X_val_scaled)
        
        val_acc = accuracy_score(y_val, val_pred)
        print(f"{name} 验证集准确率: {val_acc:.4f}")
        
        if val_acc > best_score:
            best_score = val_acc
            best_classifier = clf
            best_name = name
    
    # ================== 6. 最终评估 ==================
    if best_classifier is not None:
        print(f"\n最佳分类器: {best_name}")
        
        if best_name == 'LightGBM':
            test_pred = best_classifier.predict(X_test)
        else:
            test_pred = best_classifier.predict(X_test_scaled)
        
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"测试集准确率: {test_acc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, test_pred))
        
        # ================== 7. 保存结果 ==================
        output_dir = f"/home/deep/TimeSeries/Zhendong/output/anomaly_detection/{domain}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        with open(os.path.join(output_dir, "classifier.pkl"), "wb") as f:
            pickle.dump({
                'classifier': best_classifier,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'classifier_name': best_name
            }, f)
        
        # 保存结果
        results = {
            'domain': domain,
            'method': 'normal_only_finetune_residual',
            'val_accuracy': best_score,
            'test_accuracy': test_acc,
            'best_classifier': best_name,
            'classification_report': classification_report(y_test, test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
        }
        
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # 保存特征
        features_df.to_csv(os.path.join(output_dir, "features.csv"), index=False)
        
        print(f"结果保存到: {output_dir}")
        
        return test_acc
    else:
        print("分类器训练失败")
        return None

if __name__ == "__main__":
    # 运行异常检测流程
    domains = ['ZhenDong', 'ShengYing']
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"处理 {domain} 域")
        print('='*60)
        
        try:
            accuracy = anomaly_detection_pipeline(domain)
            if accuracy:
                print(f"{domain} 域异常检测准确率: {accuracy:.4f}")
        except Exception as e:
            print(f"{domain} 域处理失败: {e}")
            import traceback
            traceback.print_exc()

# ================== 使用说明 ==================
"""
完整的异常检测流程：

1. 前置条件：
   - 需要先运行 normal_only_finetune.py 训练Normal-only模型

2. 流程步骤：
   - 加载Normal-only微调的Chronos模型
   - 对所有数据（normal, spark, vibrate）进行预测
   - 计算预测残差并提取特征
   - 训练分类器进行三分类

3. 预期效果：
   - Normal数据的残差应该较小
   - Spark和Vibrate数据的残差应该较大
   - 基于残差特征可以有效区分三种类型

运行方法：
1. python normal_only_finetune.py  # 先训练模型
2. python anomaly_detection_pipeline.py  # 再运行异常检测
"""
