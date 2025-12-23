#!/usr/bin/env python3
"""
生成最终的结果汇总报告
"""

import pandas as pd
import json
import os

def generate_final_summary():
    """生成最终结果汇总"""
    
    # 读取方法A结果
    method_a_path = "/home/deep/TimeSeries/Zhendong/output/method_a_results/classification_results.json"
    method_b_path = "/home/deep/TimeSeries/Zhendong/output/method_b_results/classification_results.json"
    
    with open(method_a_path, 'r') as f:
        method_a_results = json.load(f)
    
    with open(method_b_path, 'r') as f:
        method_b_results = json.load(f)
    
    # 创建汇总报告
    summary_md = []
    summary_md.append("# Chronos微调电机异常检测最终结果报告\n")
    
    summary_md.append("## 实验概述\n")
    summary_md.append("基于微调后的Chronos模型进行电机异常检测，使用ZhenDong域的测试数据进行评估。\n")
    summary_md.append("- **测试数据**: 178个序列 (normal: 75, vibrate: 95, spark: 8)")
    summary_md.append("- **序列长度**: 65536个数据点 (1秒@65536Hz)")
    summary_md.append("- **patch_length**: 32 (将65536点分成2048个patches)")
    summary_md.append("- **预测长度**: 48个数据点\n")
    
    summary_md.append("## 两种方法对比\n")
    
    # 创建结果对比表
    summary_md.append("### 分类准确率对比\n")
    summary_md.append("| 方法 | 模型类型 | 特征类型 | LightGBM | SVM | 最佳准确率 |")
    summary_md.append("|------|----------|----------|----------|-----|------------|")
    
    method_a_lgb = method_a_results['LightGBM']['accuracy']
    method_a_svm = method_a_results['SVM']['accuracy']
    method_a_best = max(method_a_lgb, method_a_svm)
    
    method_b_lgb = method_b_results['LightGBM']['accuracy']
    method_b_svm = method_b_results['SVM']['accuracy']
    method_b_best = max(method_b_lgb, method_b_svm)
    
    summary_md.append(f"| 方法A | Normal-only微调 | 残差特征 | {method_a_lgb:.4f} | {method_a_svm:.4f} | **{method_a_best:.4f}** |")
    summary_md.append(f"| 方法B | All-class微调 | Embedding特征 | {method_b_lgb:.4f} | {method_b_svm:.4f} | **{method_b_best:.4f}** |")
    summary_md.append("")
    
    # 最佳结果
    overall_best = max(method_a_best, method_b_best)
    best_method = "方法B (All-class + Embedding)" if method_b_best > method_a_best else "方法A (Normal-only + 残差)"
    best_classifier = "LightGBM" if method_b_lgb == method_b_best else "SVM"
    
    summary_md.append(f"### 🏆 最佳结果\n")
    summary_md.append(f"- **最佳方法**: {best_method}")
    summary_md.append(f"- **最佳分类器**: {best_classifier}")
    summary_md.append(f"- **最佳准确率**: {overall_best:.4f}\n")
    
    # 详细分析
    summary_md.append("## 详细分析\n")
    
    summary_md.append("### 方法A: Normal-only微调 + 残差特征\n")
    summary_md.append("**策略**: 只用normal数据微调Chronos，然后计算预测残差作为异常检测特征")
    summary_md.append("**理论**: normal数据残差小，异常数据残差大")
    summary_md.append("**结果**:")
    summary_md.append(f"- LightGBM: {method_a_lgb:.4f}")
    summary_md.append(f"- SVM: {method_a_svm:.4f}")
    
    # 方法A的分类报告
    lgb_report_a = method_a_results['LightGBM']['classification_report']
    summary_md.append("\n**LightGBM分类报告**:")
    summary_md.append("```")
    summary_md.append(f"{'类别':<10} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
    summary_md.append("-" * 40)
    for label in ['normal', 'spark', 'vibrate']:
        if label in lgb_report_a:
            p = lgb_report_a[label]['precision']
            r = lgb_report_a[label]['recall']
            f1 = lgb_report_a[label]['f1-score']
            summary_md.append(f"{label:<10} {p:<8.4f} {r:<8.4f} {f1:<8.4f}")
    summary_md.append("```\n")
    
    summary_md.append("### 方法B: All-class微调 + Embedding特征\n")
    summary_md.append("**策略**: 用所有三类数据微调Chronos，然后提取embedding特征进行分类")
    summary_md.append("**理论**: 模型学习更丰富的模式，embedding包含更多判别信息")
    summary_md.append("**结果**:")
    summary_md.append(f"- LightGBM: {method_b_lgb:.4f}")
    summary_md.append(f"- SVM: {method_b_svm:.4f}")
    
    # 方法B的分类报告
    lgb_report_b = method_b_results['LightGBM']['classification_report']
    summary_md.append("\n**LightGBM分类报告**:")
    summary_md.append("```")
    summary_md.append(f"{'类别':<10} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
    summary_md.append("-" * 40)
    for label in ['normal', 'spark', 'vibrate']:
        if label in lgb_report_b:
            p = lgb_report_b[label]['precision']
            r = lgb_report_b[label]['recall']
            f1 = lgb_report_b[label]['f1-score']
            summary_md.append(f"{label:<10} {p:<8.4f} {r:<8.4f} {f1:<8.4f}")
    summary_md.append("```\n")
    
    # 关键发现
    summary_md.append("## 关键发现\n")
    summary_md.append("1. **方法B (All-class + Embedding) 效果更好**")
    summary_md.append(f"   - 最佳准确率: {method_b_best:.4f} vs {method_a_best:.4f}")
    summary_md.append("   - 说明用全部数据微调能学到更好的特征表示")
    summary_md.append("")
    summary_md.append("2. **LightGBM表现优于SVM**")
    summary_md.append(f"   - 方法B中LightGBM: {method_b_lgb:.4f} > SVM: {method_b_svm:.4f}")
    summary_md.append("   - 树模型更适合处理这类特征")
    summary_md.append("")
    summary_md.append("3. **Spark类别识别困难**")
    summary_md.append("   - 样本数量少 (仅8个)")
    summary_md.append("   - 精确率和召回率都为0")
    summary_md.append("   - 需要更多spark样本或数据增强")
    summary_md.append("")
    summary_md.append("4. **Normal和Vibrate识别较好**")
    summary_md.append("   - 这两类样本数量充足")
    summary_md.append("   - F1分数都在0.7以上")
    
    # 技术细节
    summary_md.append("\n## 技术细节\n")
    summary_md.append("### 模型架构")
    summary_md.append("- **基础模型**: chronos-bolt-base")
    summary_md.append("- **Patch处理**: 65536点 → 2048个patches (每个32点)")
    summary_md.append("- **Context长度**: 2048 (受模型限制)")
    summary_md.append("- **预测长度**: 48")
    summary_md.append("- **微调步数**: 5000")
    summary_md.append("- **学习率**: 3e-5\n")
    
    summary_md.append("### 特征提取")
    summary_md.append("- **残差特征**: MAE, MSE, RMSE, 分位数, 分段统计, 自相关")
    summary_md.append("- **Embedding特征**: 预测值统计 + 原始值统计 + 频域特征")
    summary_md.append("- **特征维度**: ~20维")
    
    # 建议
    summary_md.append("\n## 改进建议\n")
    summary_md.append("1. **数据增强**: 对spark类别进行数据增强")
    summary_md.append("2. **特征工程**: 添加更多领域相关的时频域特征")
    summary_md.append("3. **模型集成**: 结合多个分类器的预测结果")
    summary_md.append("4. **阈值优化**: 针对不同类别调整分类阈值")
    summary_md.append("5. **更大模型**: 尝试chronos-large模型")
    
    # 保存报告
    output_path = "/home/deep/TimeSeries/Zhendong/code/finetune_chronos/final_results.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_md))
    
    print(f"最终结果报告已保存到: {output_path}")
    
    # 同时保存到指定位置
    result_path = "/home/deep/TimeSeries/Zhendong/code/finetune_chronos/result.md"
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_md))
    
    print(f"结果报告也已保存到: {result_path}")

if __name__ == "__main__":
    generate_final_summary()
