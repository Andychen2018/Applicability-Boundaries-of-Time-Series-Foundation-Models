#!/usr/bin/env python3
"""
结果汇总脚本 - 生成result.md文件
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_method_results(output_dir, method_name, domains):
    """
    加载指定方法的结果
    """
    results = {}
    
    for domain in domains:
        method_dir = os.path.join(output_dir, method_name, domain)
        metrics_file = os.path.join(method_dir, "metrics.json")
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                results[domain] = json.load(f)
        else:
            results[domain] = None
    
    return results

def generate_result_summary(output_dir):
    """
    生成结果汇总
    """
    domains = ["ShengYing", "ZhenDong"]
    methods = [
        ("methodA_residual_normal_ft", "方案A: Normal-only微调+残差特征"),
        ("methodB_embed_normal_ft", "方案B: Normal-only微调+embedding特征"),
        ("methodC_embed_all_ft", "方案C: All-class微调+embedding特征")
    ]
    
    # 加载所有结果
    all_results = {}
    for method_dir, method_desc in methods:
        all_results[method_desc] = load_method_results(output_dir, method_dir, domains)
    
    # 生成Markdown报告
    md_content = []
    md_content.append("# Chronos微调电机异常检测结果报告\n")
    md_content.append("## 项目概述\n")
    md_content.append("基于Chronos时间序列模型，对1s（65536点，采样率65536Hz）的电机时序信号进行多分类：")
    md_content.append("- `normal`（正常）")
    md_content.append("- `spark`（火花异常）") 
    md_content.append("- `vibrate`（振动异常）\n")
    
    md_content.append("## 实验设置\n")
    md_content.append("- **预测步长**: 1024")
    md_content.append("- **上下文窗口**: 2048（受模型限制）")
    md_content.append("- **随机种子**: 123")
    md_content.append("- **数据划分**: 按motor_id分组，train:val:test ≈ 7:1:2")
    md_content.append("- **域**: ShengYing和ZhenDong两个传感器域\n")
    
    md_content.append("## 三种方案对比\n")
    
    # 创建结果对比表
    md_content.append("### 整体结果对比\n")
    md_content.append("| 方案 | 域 | 验证集准确率 | 测试集准确率 | 最佳分类器 | 状态 |")
    md_content.append("|------|----|--------------|--------------|-----------|----- |")
    
    for method_desc, domain_results in all_results.items():
        for domain in domains:
            result = domain_results[domain]
            if result:
                val_acc = f"{result['val_accuracy']:.4f}"
                test_acc = f"{result['test_accuracy']:.4f}"
                classifier = result['best_classifier']
                status = "✅ 完成"
            else:
                val_acc = "-"
                test_acc = "-"
                classifier = "-"
                status = "⏳ 进行中"
            
            md_content.append(f"| {method_desc} | {domain} | {val_acc} | {test_acc} | {classifier} | {status} |")
    
    md_content.append("\n")
    
    # 详细结果分析
    md_content.append("## 详细结果分析\n")
    
    for method_desc, domain_results in all_results.items():
        md_content.append(f"### {method_desc}\n")
        
        for domain in domains:
            result = domain_results[domain]
            if result:
                md_content.append(f"#### {domain} 域\n")
                md_content.append(f"- **测试集准确率**: {result['test_accuracy']:.4f}")
                md_content.append(f"- **验证集准确率**: {result['val_accuracy']:.4f}")
                md_content.append(f"- **最佳分类器**: {result['best_classifier']}\n")
                
                # 分类报告
                if 'classification_report' in result:
                    report = result['classification_report']
                    md_content.append("**分类报告**:")
                    md_content.append("```")
                    md_content.append(f"{'类别':<10} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<8}")
                    md_content.append("-" * 50)
                    
                    for label in ['normal', 'spark', 'vibrate']:
                        if label in report:
                            precision = report[label]['precision']
                            recall = report[label]['recall']
                            f1 = report[label]['f1-score']
                            support = report[label]['support']
                            md_content.append(f"{label:<10} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f} {support:<8}")
                    
                    # 宏平均和加权平均
                    if 'macro avg' in report:
                        macro = report['macro avg']
                        md_content.append(f"{'宏平均':<10} {macro['precision']:<8.4f} {macro['recall']:<8.4f} {macro['f1-score']:<8.4f} {macro['support']:<8}")
                    
                    if 'weighted avg' in report:
                        weighted = report['weighted avg']
                        md_content.append(f"{'加权平均':<10} {weighted['precision']:<8.4f} {weighted['recall']:<8.4f} {weighted['f1-score']:<8.4f} {weighted['support']:<8}")
                    
                    md_content.append("```\n")
                
                # 混淆矩阵
                if 'confusion_matrix' in result:
                    cm = result['confusion_matrix']
                    md_content.append("**混淆矩阵**:")
                    md_content.append("```")
                    md_content.append("预测\\真实   normal  spark  vibrate")
                    labels = ['normal', 'spark', 'vibrate']
                    for i, pred_label in enumerate(labels):
                        row = f"{pred_label:<10}"
                        for j in range(len(cm[i])):
                            row += f"{cm[i][j]:<7}"
                        md_content.append(row)
                    md_content.append("```\n")
            else:
                md_content.append(f"#### {domain} 域: ⏳ 训练进行中\n")
    
    # 结论和建议
    md_content.append("## 结论和建议\n")
    
    # 找出最佳结果
    best_results = []
    for method_desc, domain_results in all_results.items():
        for domain in domains:
            result = domain_results[domain]
            if result:
                best_results.append({
                    'method': method_desc,
                    'domain': domain,
                    'test_accuracy': result['test_accuracy'],
                    'val_accuracy': result['val_accuracy']
                })
    
    if best_results:
        best_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
        best = best_results[0]
        
        md_content.append(f"### 最佳结果")
        md_content.append(f"- **方案**: {best['method']}")
        md_content.append(f"- **域**: {best['domain']}")
        md_content.append(f"- **测试集准确率**: {best['test_accuracy']:.4f}")
        md_content.append(f"- **验证集准确率**: {best['val_accuracy']:.4f}\n")
    
    md_content.append("### 观察和分析")
    md_content.append("1. **数据不平衡**: 数据集中vibrate类别样本最多，normal次之，spark最少")
    md_content.append("2. **域差异**: ShengYing和ZhenDong两个传感器域可能存在不同的特征分布")
    md_content.append("3. **方法对比**: ")
    md_content.append("   - 方案A（残差特征）：利用预测残差的统计特征进行分类")
    md_content.append("   - 方案B（normal-only embedding）：仅用正常数据微调，提取embedding特征")
    md_content.append("   - 方案C（all-class embedding）：用全部数据微调，提取embedding特征\n")
    
    md_content.append("### 建议")
    md_content.append("1. **数据增强**: 对少数类别（特别是spark）进行数据增强")
    md_content.append("2. **特征工程**: 结合领域知识，提取更多有效的时频域特征")
    md_content.append("3. **模型集成**: 考虑将多个方案的结果进行集成")
    md_content.append("4. **阈值优化**: 针对不同类别调整分类阈值，平衡精确率和召回率\n")
    
    # 技术细节
    md_content.append("## 技术细节\n")
    md_content.append("### 模型配置")
    md_content.append("- **基础模型**: chronos-bolt-base")
    md_content.append("- **微调学习率**: 5e-5 (normal-only), 3e-5 (all-class)")
    md_content.append("- **微调步数**: 10,000")
    md_content.append("- **Dropout**: 0.1")
    md_content.append("- **上下文长度**: 2048 (受模型限制)")
    md_content.append("- **预测长度**: 1024\n")
    
    md_content.append("### 分类器")
    md_content.append("- **LightGBM**: 梯度提升决策树")
    md_content.append("- **RandomForest**: 随机森林")
    md_content.append("- **选择标准**: 验证集准确率最高\n")
    
    md_content.append("### 特征提取")
    md_content.append("- **方案A**: 残差统计特征（MAE, MSE, RMSE等）+ 分段特征 + 自相关 + 频域特征")
    md_content.append("- **方案B/C**: 预测值统计特征 + 原始数据统计特征 + 频域特征\n")
    
    # 保存结果
    result_file = os.path.join(output_dir, "result.md")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"结果报告已保存到: {result_file}")
    return result_file

def main():
    output_dir = "/home/deep/TimeSeries/Zhendong/output"
    result_file = generate_result_summary(output_dir)
    print("结果汇总完成！")

if __name__ == "__main__":
    main()
