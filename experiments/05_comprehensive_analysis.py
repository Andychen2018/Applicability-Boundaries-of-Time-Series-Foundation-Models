"""
05_综合分析
汇总所有模型的结果并进行对比分析
包括统计机器学习、深度学习、Chronos和Transformer模型
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAnalyzer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.results_summary = {}
        
    def load_all_results(self):
        """加载所有模型的结果"""
        print("Loading all model results...")

        # 查找所有结果文件
        result_files = glob(os.path.join(self.output_dir, 'table', '*_results.csv'))

        all_results = []

        for file_path in result_files:
            try:
                df = pd.read_csv(file_path)
                filename = os.path.basename(file_path)

                # 解析文件名获取模型类型和模式
                parts = filename.replace('_results.csv', '').split('_')
                if len(parts) >= 3:
                    model_type = parts[0]  # 01, 02, 03, 04
                    mode = parts[1]        # shengying, zhendong, fusion

                    # 添加模型类型信息
                    df['Model_Type'] = self._get_model_type_name(model_type)
                    df['File_Source'] = filename

                    # 统一准确率列名
                    if 'Test_Accuracy' in df.columns and 'Accuracy' not in df.columns:
                        df['Accuracy'] = df['Test_Accuracy']
                    elif 'Accuracy' not in df.columns:
                        # 如果都没有，尝试其他可能的列名
                        for col in df.columns:
                            if 'accuracy' in col.lower():
                                df['Accuracy'] = df[col]
                                break

                    all_results.append(df)
                    print(f"Loaded: {filename}")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if all_results:
            self.combined_results = pd.concat(all_results, ignore_index=True)
            print(f"Total results loaded: {len(self.combined_results)}")
            return True
        else:
            print("No results found!")
            return False
    
    def _get_model_type_name(self, model_type):
        """获取模型类型名称"""
        type_mapping = {
            '01': 'Statistical ML',
            '02': 'Deep Learning',
            '03': 'Chronos',
            '04': 'Transformer'
        }
        return type_mapping.get(model_type, 'Unknown')
    
    def analyze_by_mode(self):
        """按模式分析结果"""
        print("\n=== Analysis by Mode ===")
        
        if not hasattr(self, 'combined_results'):
            print("No results loaded!")
            return
        
        # 确定准确率列名
        accuracy_cols = [col for col in self.combined_results.columns if 'accuracy' in col.lower() or 'Accuracy' in col]
        if not accuracy_cols:
            print("No accuracy column found!")
            return
        
        accuracy_col = accuracy_cols[0]  # 使用第一个找到的准确率列
        print(f"Using accuracy column: {accuracy_col}")
        
        modes = self.combined_results['Mode'].unique()
        
        for mode in modes:
            mode_data = self.combined_results[self.combined_results['Mode'] == mode]
            
            print(f"\n--- {mode.upper()} Mode ---")
            print(f"Number of models: {len(mode_data)}")
            
            # 按模型类型分组
            by_type = mode_data.groupby('Model_Type')[accuracy_col].agg(['mean', 'max', 'min', 'count'])
            print("\nBy Model Type:")
            print(by_type)
            
            # 最佳模型
            best_model = mode_data.loc[mode_data[accuracy_col].idxmax()]
            print(f"\nBest Model: {best_model['Model']} ({best_model['Model_Type']})")
            print(f"Best Accuracy: {best_model[accuracy_col]:.4f}")
            
            self.results_summary[mode] = {
                'best_model': best_model['Model'],
                'best_accuracy': best_model[accuracy_col],
                'best_type': best_model['Model_Type'],
                'by_type': by_type.to_dict()
            }
    
    def create_comprehensive_plots(self):
        """创建综合对比图表"""
        if not hasattr(self, 'combined_results'):
            print("No results loaded!")
            return
        
        # 确定准确率列名
        accuracy_cols = [col for col in self.combined_results.columns if 'accuracy' in col.lower() or 'Accuracy' in col]
        if not accuracy_cols:
            print("No accuracy column found!")
            return
        
        accuracy_col = accuracy_cols[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 所有模型准确率对比
        model_acc = self.combined_results.groupby(['Model_Type', 'Mode'])[accuracy_col].max().reset_index()
        
        pivot_data = model_acc.pivot(index='Model_Type', columns='Mode', values=accuracy_col)
        
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0, 0])
        axes[0, 0].set_title('Best Accuracy by Model Type and Mode', fontsize=14)
        axes[0, 0].set_xlabel('Mode')
        axes[0, 0].set_ylabel('Model Type')
        
        # 2. 按模式的模型性能分布
        modes = self.combined_results['Mode'].unique()
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        for i, mode in enumerate(modes):
            mode_data = self.combined_results[self.combined_results['Mode'] == mode]
            axes[0, 1].bar([f"{row['Model_Type']}\n{row['Model']}" for _, row in mode_data.iterrows()], 
                          mode_data[accuracy_col], 
                          alpha=0.7, 
                          label=mode,
                          color=colors[i % len(colors)])
        
        axes[0, 1].set_title('Model Performance by Mode', fontsize=14)
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        
        # 3. 模型类型性能箱线图
        sns.boxplot(data=self.combined_results, x='Model_Type', y=accuracy_col, ax=axes[1, 0])
        axes[1, 0].set_title('Performance Distribution by Model Type', fontsize=14)
        axes[1, 0].set_xlabel('Model Type')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 最佳模型总结
        best_models = []
        for mode in modes:
            mode_data = self.combined_results[self.combined_results['Mode'] == mode]
            best_idx = mode_data[accuracy_col].idxmax()
            best_model = mode_data.loc[best_idx]
            best_models.append({
                'Mode': mode,
                'Model': best_model['Model'],
                'Type': best_model['Model_Type'],
                'Accuracy': best_model[accuracy_col]
            })
        
        best_df = pd.DataFrame(best_models)
        
        # 创建最佳模型对比
        bars = axes[1, 1].bar(best_df['Mode'], best_df['Accuracy'], 
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_title('Best Model Performance by Mode', fontsize=14)
        axes[1, 1].set_xlabel('Mode')
        axes[1, 1].set_ylabel('Best Accuracy')
        
        # 添加数值标签
        for bar, model, acc in zip(bars, best_df['Model'], best_df['Accuracy']):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{model}\n{acc:.4f}',
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图片
        img_path = os.path.join(self.output_dir, 'images', '05_comprehensive_analysis.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive analysis plot saved to {img_path}")
    
    def generate_summary_report(self):
        """生成总结报告"""
        if not hasattr(self, 'combined_results'):
            print("No results loaded!")
            return
        
        # 确定准确率列名
        accuracy_cols = [col for col in self.combined_results.columns if 'accuracy' in col.lower() or 'Accuracy' in col]
        if not accuracy_cols:
            print("No accuracy column found!")
            return
        
        accuracy_col = accuracy_cols[0]
        
        # 创建总结表格
        summary_data = []
        
        modes = self.combined_results['Mode'].unique()
        model_types = self.combined_results['Model_Type'].unique()
        
        for mode in modes:
            mode_data = self.combined_results[self.combined_results['Mode'] == mode]

            for model_type in model_types:
                type_data = mode_data[mode_data['Model_Type'] == model_type]

                if len(type_data) > 0 and not type_data[accuracy_col].isna().all():
                    # 过滤掉NaN值
                    valid_data = type_data.dropna(subset=[accuracy_col])
                    if len(valid_data) > 0:
                        best_idx = valid_data[accuracy_col].idxmax()
                        summary_data.append({
                            'Mode': mode,
                            'Model_Type': model_type,
                            'Best_Accuracy': valid_data[accuracy_col].max(),
                            'Mean_Accuracy': valid_data[accuracy_col].mean(),
                            'Std_Accuracy': valid_data[accuracy_col].std(),
                            'Model_Count': len(valid_data),
                            'Best_Model': valid_data.loc[best_idx, 'Model']
                        })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存总结表格
        summary_path = os.path.join(self.output_dir, 'table', '05_comprehensive_summary.csv')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nComprehensive summary saved to {summary_path}")
        print("\n=== COMPREHENSIVE SUMMARY ===")
        print(summary_df.to_string(index=False))
        
        # 找出总体最佳模型
        overall_best = self.combined_results.loc[self.combined_results[accuracy_col].idxmax()]
        print(f"\n=== OVERALL BEST MODEL ===")
        print(f"Model: {overall_best['Model']}")
        print(f"Type: {overall_best['Model_Type']}")
        print(f"Mode: {overall_best['Mode']}")
        print(f"Accuracy: {overall_best[accuracy_col]:.4f}")
        
        return summary_df
    
    def run_comprehensive_analysis(self):
        """运行完整的综合分析"""
        print("=== COMPREHENSIVE MODEL ANALYSIS ===")
        
        # 加载所有结果
        if not self.load_all_results():
            print("Failed to load results!")
            return
        
        # 显示数据概览
        print(f"\nLoaded data shape: {self.combined_results.shape}")
        print(f"Columns: {list(self.combined_results.columns)}")
        print(f"Modes: {self.combined_results['Mode'].unique()}")
        print(f"Model Types: {self.combined_results['Model_Type'].unique()}")
        
        # 按模式分析
        self.analyze_by_mode()
        
        # 创建综合图表
        self.create_comprehensive_plots()
        
        # 生成总结报告
        summary_df = self.generate_summary_report()
        
        print("\n=== ANALYSIS COMPLETE ===")
        return summary_df

def main():
    """主函数"""
    analyzer = ComprehensiveAnalyzer()
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
