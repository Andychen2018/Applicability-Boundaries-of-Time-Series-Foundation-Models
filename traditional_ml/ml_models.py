#!/usr/bin/env python3
"""
传统机器学习模型模块
包含多种传统机器学习算法的训练和评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import joblib
import json
from datetime import datetime

class TraditionalMLPipeline:
    """传统机器学习流水线"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['output']['tables'])
        self.image_path = Path(self.config['output']['images'])
        self.models_path = self.output_path.parent / 'models'
        self.models_path.mkdir(exist_ok=True)
        
        # 初始化模型
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        self.results = {}
        self.trained_models = {}
        
    def load_features(self, features_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """加载特征数据"""
        print("📊 加载特征数据...")
        
        features_df = pd.read_csv(features_path)
        
        # 分离特征和标签
        feature_cols = [col for col in features_df.columns 
                       if col not in ['label', 'sensor', 'file_id']]
        
        X = features_df[feature_cols]
        y = features_df['label']
        
        # 处理缺失值和无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"✅ 加载完成: {len(X)} 个样本, {len(feature_cols)} 个特征")
        print(f"📋 类别分布: {dict(y.value_counts())}")
        
        return features_df, X.values, y.values
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                       test_size: float = 0.2, val_size: float = 0.1) -> Dict:
        """数据预处理和划分"""
        print("🔧 数据预处理...")
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 划分训练集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # 从训练集中划分验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        data_splits = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_raw': X_train,
            'X_val_raw': X_val,
            'X_test_raw': X_test
        }
        
        print(f"✅ 数据划分完成:")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   验证集: {len(X_val)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        
        return data_splits
    
    def train_models(self, data_splits: Dict) -> Dict:
        """训练所有模型"""
        print("🤖 开始训练模型...")
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        
        for name, model in self.models.items():
            print(f"  训练 {name}...")
            
            try:
                # 训练模型
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # 验证集预测
                y_val_pred = model.predict(X_val)
                y_val_prob = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # 计算指标
                metrics = self._calculate_metrics(y_val, y_val_pred, y_val_prob)
                self.results[name] = {
                    'model': model,
                    'val_metrics': metrics,
                    'val_predictions': y_val_pred,
                    'val_probabilities': y_val_prob
                }
                
                print(f"    ✅ {name} - 验证准确率: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"    ❌ {name} 训练失败: {e}")
                continue
        
        print("✅ 模型训练完成")
        return self.results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: Optional[np.ndarray] = None) -> Dict:
        """计算评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # 多类别AUC
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
        
        return metrics
    
    def evaluate_on_test_set(self, data_splits: Dict) -> Dict:
        """在测试集上评估模型"""
        print("📊 测试集评估...")
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        test_results = {}
        
        for name, result in self.results.items():
            model = result['model']
            
            # 测试集预测
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # 计算指标
            test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_prob)
            
            test_results[name] = {
                'test_metrics': test_metrics,
                'test_predictions': y_test_pred,
                'test_probabilities': y_test_prob
            }
            
            print(f"  {name} - 测试准确率: {test_metrics['accuracy']:.4f}")
        
        return test_results
    
    def hyperparameter_tuning(self, data_splits: Dict, model_names: List[str] = None) -> Dict:
        """超参数调优"""
        print("🔧 超参数调优...")
        
        if model_names is None:
            model_names = ['Random Forest', 'XGBoost', 'SVM']
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        
        # 定义参数网格
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
        
        tuned_models = {}
        
        for name in model_names:
            if name in param_grids:
                print(f"  调优 {name}...")
                
                base_model = self.models[name]
                param_grid = param_grids[name]
                
                # 网格搜索
                grid_search = GridSearchCV(
                    base_model, param_grid, cv=3, scoring='accuracy',
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                tuned_models[name] = {
                    'best_model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }
                
                print(f"    ✅ {name} 最佳分数: {grid_search.best_score_:.4f}")
                print(f"    📋 最佳参数: {grid_search.best_params_}")
        
        return tuned_models
    
    def save_models(self):
        """保存训练好的模型"""
        print("💾 保存模型...")
        
        for name, model in self.trained_models.items():
            model_path = self.models_path / f"{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_path)
            print(f"  ✅ {name} 已保存: {model_path}")
        
        # 保存预处理器
        scaler_path = self.models_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        encoder_path = self.models_path / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"✅ 预处理器已保存")
    
    def save_results(self, test_results: Dict):
        """保存实验结果"""
        print("📋 保存实验结果...")
        
        # 整理结果数据
        results_data = []
        
        for name in self.results.keys():
            val_metrics = self.results[name]['val_metrics']
            test_metrics = test_results[name]['test_metrics']
            
            result_row = {
                'model': name,
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_auc': val_metrics['auc'],
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1'],
                'test_auc': test_metrics['auc']
            }
            results_data.append(result_row)
        
        # 保存为CSV
        results_df = pd.DataFrame(results_data)
        results_path = self.output_path / 'traditional_ml_results.csv'
        results_df.to_csv(results_path, index=False)
        
        print(f"📊 结果已保存: {results_path}")
        
        # 保存详细结果为JSON
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'results': results_data,
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        json_path = self.output_path / 'traditional_ml_detailed_results.json'
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        return results_df

if __name__ == "__main__":
    # 测试传统机器学习流水线
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
    features_path = Path(__file__).parent.parent.parent / "output/table/extracted_features.csv"
    
    # 创建流水线
    pipeline = TraditionalMLPipeline(str(config_path))
    
    # 加载特征
    features_df, X, y = pipeline.load_features(str(features_path))
    
    # 数据预处理
    data_splits = pipeline.preprocess_data(X, y)
    
    # 训练模型
    results = pipeline.train_models(data_splits)
    
    # 测试集评估
    test_results = pipeline.evaluate_on_test_set(data_splits)
    
    # 保存模型和结果
    pipeline.save_models()
    results_df = pipeline.save_results(test_results)
    
    print("\n🎉 传统机器学习实验完成！")
    print("📊 模型性能排序 (按测试F1分数):")
    print(results_df.sort_values('test_f1', ascending=False)[['model', 'test_accuracy', 'test_f1']].to_string(index=False))
