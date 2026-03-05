#!/usr/bin/env python3
"""
MomentFM + Chronos 组合异常检测系统
实现预测性异常检测方案
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Chronos imports
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
    print("✅ Chronos 可用")
except ImportError:
    CHRONOS_AVAILABLE = False
    print("❌ Chronos 不可用")

# MomentFM imports
try:
    from momentfm import MOMENTPipeline
    MOMENT_AVAILABLE = True
    print("✅ MomentFM 可用")
except ImportError:
    MOMENT_AVAILABLE = False
    print("❌ MomentFM 不可用，将使用替代方案")

class MomentChronosAnomalyDetector:
    """MomentFM + Chronos 异常检测器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['output']['tables'])
        self.models_path = self.output_path.parent / 'models'
        self.models_path.mkdir(exist_ok=True)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 初始化模型
        self.chronos_pipeline = None
        self.moment_pipeline = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        print("🚀 初始化基础模型...")
        
        # 初始化Chronos
        if CHRONOS_AVAILABLE:
            try:
                print("  加载Chronos模型...")
                self.chronos_pipeline = ChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-small",
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                )
                print("  ✅ Chronos模型加载成功")
            except Exception as e:
                print(f"  ❌ Chronos模型加载失败: {e}")
                self.chronos_pipeline = None
        
        # 初始化MomentFM (如果可用)
        if MOMENT_AVAILABLE:
            try:
                print("  加载MomentFM模型...")
                self.moment_pipeline = MOMENTPipeline.from_pretrained(
                    "AutonLab/MOMENT-1-large", 
                    model_kwargs={'task_name': 'embedding'}
                )
                print("  ✅ MomentFM模型加载成功")
            except Exception as e:
                print(f"  ❌ MomentFM模型加载失败: {e}")
                self.moment_pipeline = None
    
    def load_motor_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """加载电机数据"""
        print("📂 加载电机数据...")
        
        # 从增强数据加载器加载数据
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
        from enhanced_data_loader import EnhancedMotorDataLoader
        
        config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
        loader = EnhancedMotorDataLoader(str(config_path))
        dataset, _ = loader.load_comprehensive_dataset(enable_augmentation=False)
        
        # 收集所有信号
        all_signals = []
        all_labels = []
        
        for state in ['normal', 'spark', 'vibrate']:
            signals = dataset['single_sensor'][state][:100]  # 每类取100个样本进行测试
            all_signals.extend(signals)
            all_labels.extend([state] * len(signals))
        
        print(f"✅ 加载完成: {len(all_signals)} 个信号")
        return all_signals, all_labels
    
    def prepare_time_series_data(self, signals: List[np.ndarray], 
                               context_length: int = 512, 
                               prediction_length: int = 64) -> List[Dict]:
        """准备时序数据"""
        print(f"🔧 准备时序数据 (上下文长度: {context_length}, 预测长度: {prediction_length})...")
        
        prepared_data = []
        
        for i, signal in enumerate(signals):
            # 确保信号足够长
            min_length = context_length + prediction_length
            if len(signal) < min_length:
                # 填充信号
                signal = np.pad(signal, (0, min_length - len(signal)), 'reflect')
            
            # 标准化
            signal_mean = np.mean(signal)
            signal_std = np.std(signal) + 1e-8
            signal_normalized = (signal - signal_mean) / signal_std
            
            # 创建滑动窗口
            for start_idx in range(0, len(signal_normalized) - min_length + 1, prediction_length):
                end_context = start_idx + context_length
                end_prediction = end_context + prediction_length
                
                if end_prediction <= len(signal_normalized):
                    context = signal_normalized[start_idx:end_context]
                    target = signal_normalized[end_context:end_prediction]
                    
                    prepared_data.append({
                        'signal_id': i,
                        'context': context,
                        'target': target,
                        'mean': signal_mean,
                        'std': signal_std
                    })
        
        print(f"✅ 准备完成: {len(prepared_data)} 个时序片段")
        return prepared_data
    
    def extract_moment_embeddings(self, contexts: List[np.ndarray]) -> Optional[np.ndarray]:
        """使用MomentFM提取嵌入特征"""
        if not self.moment_pipeline:
            print("⚠️ MomentFM不可用，跳过嵌入提取")
            return None
        
        print("🔧 提取MomentFM嵌入...")
        
        try:
            # 准备输入数据
            input_data = []
            for context in contexts:
                # MomentFM期望的输入格式
                input_data.append(torch.tensor(context, dtype=torch.float32).unsqueeze(0))
            
            # 批量处理
            embeddings = []
            batch_size = 32
            
            for i in range(0, len(input_data), batch_size):
                batch = input_data[i:i+batch_size]
                batch_tensor = torch.stack(batch).squeeze(1)  # [batch_size, seq_len]
                
                with torch.no_grad():
                    batch_embeddings = self.moment_pipeline(batch_tensor)
                    embeddings.append(batch_embeddings.cpu().numpy())
            
            embeddings = np.concatenate(embeddings, axis=0)
            print(f"✅ 嵌入提取完成: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ MomentFM嵌入提取失败: {e}")
            return None
    
    def chronos_predict(self, contexts: List[np.ndarray], 
                       prediction_length: int = 64) -> Optional[np.ndarray]:
        """使用Chronos进行预测"""
        if not self.chronos_pipeline:
            print("⚠️ Chronos不可用，跳过预测")
            return None
        
        print("🔮 Chronos预测...")
        
        try:
            predictions = []
            batch_size = 16
            
            for i in range(0, len(contexts), batch_size):
                batch_contexts = contexts[i:i+batch_size]
                
                # 转换为tensor
                batch_tensor = torch.tensor(np.array(batch_contexts), dtype=torch.float32)
                
                # Chronos预测
                with torch.no_grad():
                    forecast = self.chronos_pipeline.predict(
                        context=batch_tensor,
                        prediction_length=prediction_length,
                        num_samples=20  # 生成多个样本以获得不确定性
                    )
                
                predictions.append(forecast.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            print(f"✅ 预测完成: {predictions.shape}")
            return predictions
            
        except Exception as e:
            print(f"❌ Chronos预测失败: {e}")
            return None
    
    def detect_anomalies(self, targets: List[np.ndarray], 
                        predictions: np.ndarray, 
                        confidence_level: float = 0.95) -> np.ndarray:
        """检测异常"""
        print("🔍 检测异常...")
        
        anomaly_scores = []
        
        for i, (target, pred_samples) in enumerate(zip(targets, predictions)):
            # 计算预测分布的统计量
            pred_mean = np.mean(pred_samples, axis=0)
            pred_std = np.std(pred_samples, axis=0)
            
            # 计算置信区间
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
            lower_bound = pred_mean - z_score * pred_std
            upper_bound = pred_mean + z_score * pred_std
            
            # 检查目标值是否在置信区间内
            in_bounds = (target >= lower_bound) & (target <= upper_bound)
            anomaly_ratio = 1 - np.mean(in_bounds)
            
            # 计算残差
            residual = np.mean(np.abs(target - pred_mean))
            
            # 综合异常分数
            anomaly_score = anomaly_ratio * 0.7 + min(residual, 1.0) * 0.3
            anomaly_scores.append(anomaly_score)
        
        anomaly_scores = np.array(anomaly_scores)
        
        # 确定异常阈值
        threshold = np.percentile(anomaly_scores, 90)  # 前10%为异常
        anomalies = anomaly_scores > threshold
        
        print(f"✅ 异常检测完成: 发现 {np.sum(anomalies)} 个异常 (总共 {len(anomalies)} 个样本)")
        
        return anomalies, anomaly_scores
    
    def run_anomaly_detection_experiment(self) -> Dict:
        """运行异常检测实验"""
        print("🚀 开始MomentFM + Chronos异常检测实验")
        print("="*60)
        
        # 加载数据
        signals, labels = self.load_motor_data()
        
        # 准备时序数据
        prepared_data = self.prepare_time_series_data(signals)
        
        if len(prepared_data) == 0:
            print("❌ 没有准备好的数据")
            return {}
        
        # 提取上下文和目标
        contexts = [item['context'] for item in prepared_data]
        targets = [item['target'] for item in prepared_data]
        signal_ids = [item['signal_id'] for item in prepared_data]
        
        # 提取MomentFM嵌入（如果可用）
        embeddings = self.extract_moment_embeddings(contexts)
        
        # Chronos预测
        predictions = self.chronos_predict(contexts)
        
        if predictions is None:
            print("❌ 预测失败，使用简单基线方法")
            return self.baseline_anomaly_detection(signals, labels)
        
        # 异常检测
        anomalies, anomaly_scores = self.detect_anomalies(targets, predictions)
        
        # 评估结果
        results = self._evaluate_anomaly_detection(signal_ids, anomalies, anomaly_scores, labels)
        
        # 保存结果
        self._save_results(results, prepared_data, anomalies, anomaly_scores)
        
        return results

    def _evaluate_anomaly_detection(self, signal_ids: List[int], anomalies: np.ndarray,
                                   anomaly_scores: np.ndarray, labels: List[str]) -> Dict:
        """评估异常检测结果"""
        print("📊 评估异常检测结果...")

        # 将片段级别的结果聚合到信号级别
        signal_anomaly_scores = {}
        signal_labels = {}

        for i, (signal_id, anomaly, score) in enumerate(zip(signal_ids, anomalies, anomaly_scores)):
            if signal_id not in signal_anomaly_scores:
                signal_anomaly_scores[signal_id] = []
                signal_labels[signal_id] = labels[signal_id]

            signal_anomaly_scores[signal_id].append(score)

        # 计算每个信号的平均异常分数
        signal_final_scores = {}
        signal_final_predictions = {}

        for signal_id, scores in signal_anomaly_scores.items():
            avg_score = np.mean(scores)
            signal_final_scores[signal_id] = avg_score
            # 如果平均分数超过阈值，则认为是异常
            signal_final_predictions[signal_id] = avg_score > np.percentile(list(signal_anomaly_scores.values()), 70)

        # 转换为评估格式
        true_labels = []
        pred_labels = []

        for signal_id in sorted(signal_final_scores.keys()):
            true_label = 0 if signal_labels[signal_id] == 'normal' else 1
            pred_label = 1 if signal_final_predictions[signal_id] else 0

            true_labels.append(true_label)
            pred_labels.append(pred_label)

        # 计算指标
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')

        results = {
            'method': 'MomentFM_Chronos',
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'anomaly_count': sum(pred_labels),
            'total_samples': len(pred_labels),
            'signal_scores': signal_final_scores,
            'signal_predictions': signal_final_predictions,
            'signal_labels': signal_labels
        }

        print(f"✅ 评估完成:")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   F1分数: {f1:.4f}")
        print(f"   精确率: {precision:.4f}")
        print(f"   召回率: {recall:.4f}")
        print(f"   异常数量: {sum(pred_labels)}/{len(pred_labels)}")

        return results

    def _save_results(self, results: Dict, prepared_data: List[Dict],
                     anomalies: np.ndarray, anomaly_scores: np.ndarray):
        """保存结果"""
        print("💾 保存实验结果...")

        # 保存主要结果
        results_df = pd.DataFrame([results])
        results_path = self.output_path / 'moment_chronos_results.csv'
        results_df.to_csv(results_path, index=False)

        # 保存详细的异常分数
        detailed_results = []
        for i, (data, anomaly, score) in enumerate(zip(prepared_data, anomalies, anomaly_scores)):
            detailed_results.append({
                'segment_id': i,
                'signal_id': data['signal_id'],
                'anomaly_score': score,
                'is_anomaly': anomaly,
                'context_mean': np.mean(data['context']),
                'context_std': np.std(data['context']),
                'target_mean': np.mean(data['target']),
                'target_std': np.std(data['target'])
            })

        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = self.output_path / 'moment_chronos_detailed_results.csv'
        detailed_df.to_csv(detailed_path, index=False)

        # 保存JSON格式的完整结果
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'summary': results,
            'model_info': {
                'chronos_available': CHRONOS_AVAILABLE,
                'moment_available': MOMENT_AVAILABLE,
                'device': str(self.device)
            }
        }

        json_path = self.output_path / 'moment_chronos_experiment.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        print(f"📊 结果已保存:")
        print(f"   主要结果: {results_path}")
        print(f"   详细结果: {detailed_path}")
        print(f"   完整实验: {json_path}")

def run_moment_chronos_experiment():
    """运行MomentFM + Chronos实验"""
    print("🚀 启动MomentFM + Chronos异常检测实验")

    config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"

    # 创建检测器
    detector = MomentChronosAnomalyDetector(str(config_path))

    # 运行实验
    results = detector.run_anomaly_detection_experiment()

    return results

if __name__ == "__main__":
    # 运行实验
    results = run_moment_chronos_experiment()

    if results:
        print("\n🎉 MomentFM + Chronos实验完成！")
        print(f"📊 最终结果:")
        print(f"   方法: {results['method']}")
        print(f"   准确率: {results['accuracy']:.4f}")
        print(f"   F1分数: {results['f1']:.4f}")
    else:
        print("❌ 实验失败")
    
    def baseline_anomaly_detection(self, signals: List[np.ndarray], labels: List[str]) -> Dict:
        """基线异常检测方法"""
        print("🔧 使用基线异常检测方法...")
        
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score
        
        # 提取简单统计特征
        features = []
        for signal in signals:
            feat = [
                np.mean(signal), np.std(signal), np.var(signal),
                np.min(signal), np.max(signal), np.median(signal),
                np.percentile(signal, 25), np.percentile(signal, 75)
            ]
            features.append(feat)
        
        features = np.array(features)
        
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 孤立森林
        iso_forest = IsolationForest(contamination=0.2, random_state=42)
        anomaly_pred = iso_forest.fit_predict(features_scaled)
        
        # 转换标签
        true_labels = [0 if label == 'normal' else 1 for label in labels]
        pred_labels = [1 if pred == -1 else 0 for pred in anomaly_pred]
        
        # 计算指标
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        
        results = {
            'method': 'Baseline_IsolationForest',
            'accuracy': accuracy,
            'f1': f1,
            'anomaly_count': sum(pred_labels),
            'total_samples': len(pred_labels)
        }
        
        print(f"✅ 基线方法完成: 准确率 {accuracy:.4f}, F1 {f1:.4f}")
        
        return results
