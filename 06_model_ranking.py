"""
06_模型排名分析
按准确率从高到低列出所有模型，找到最佳分类器
"""

import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

class ModelRanking:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.combined_results = None
        
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
                if len(parts) >= 2:
                    model_prefix = parts[0]  # 01, 02, 03, 04
                    mode = parts[1]          # shengying, zhendong, fusion
                    
                    # 添加模型类型信息
                    df['Model_Type'] = self._get_model_type_name(model_prefix)
                    df['Mode'] = mode
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
                    print(f"Loaded: {filename} ({len(df)} models)")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if all_results:
            self.combined_results = pd.concat(all_results, ignore_index=True)
            print(f"Total results loaded: {len(self.combined_results)}")
            return True
        else:
            print("No results found!")
            return False
    
    def _get_model_type_name(self, prefix):
        """根据前缀获取模型类型名称"""
        type_mapping = {
            '01': 'Statistical ML',
            '02': 'Enhanced ML', 
            '03': 'Chronos',
            '04': 'Transformer'
        }
        return type_mapping.get(prefix, 'Unknown')
    
    def generate_complete_ranking(self):
        """生成完整的模型排名"""
        if self.combined_results is None:
            print("No results loaded!")
            return None
        
        # 确定准确率列名
        accuracy_col = 'Accuracy'
        if accuracy_col not in self.combined_results.columns:
            print("No Accuracy column found!")
            return None
        
        # 过滤掉NaN值
        valid_results = self.combined_results.dropna(subset=[accuracy_col])
        
        # 按准确率从高到低排序
        ranked_results = valid_results.sort_values(by=accuracy_col, ascending=False).reset_index(drop=True)
        
        print(f"\n{'='*100}")
        print(f"🏆 COMPLETE MODEL RANKING - ALL {len(ranked_results)} MODELS BY ACCURACY")
        print(f"{'='*100}")
        
        print(f"{'Rank':<4} | {'Model':<30} | {'Mode':<10} | {'Type':<15} | {'Accuracy':<8} | {'CV Score':<15}")
        print("-" * 100)
        
        detailed_ranking = []
        
        for i, (_, row) in enumerate(ranked_results.iterrows(), 1):
            accuracy = row[accuracy_col]
            cv_mean = row.get('CV_Mean', np.nan)
            cv_std = row.get('CV_Std', np.nan)
            
            # 格式化CV信息
            if pd.notna(cv_mean) and pd.notna(cv_std):
                cv_info = f"{cv_mean:.3f}±{cv_std:.3f}"
            else:
                cv_info = "N/A"
            
            print(f"{i:<4} | {row['Model']:<30} | {row['Mode']:<10} | {row['Model_Type']:<15} | {accuracy:<8.4f} | {cv_info:<15}")
            
            detailed_ranking.append({
                'Rank': i,
                'Model': row['Model'],
                'Mode': row['Mode'],
                'Model_Type': row['Model_Type'],
                'Accuracy': accuracy,
                'CV_Mean': cv_mean if pd.notna(cv_mean) else None,
                'CV_Std': cv_std if pd.notna(cv_std) else None,
                'File_Source': row.get('File_Source', 'N/A')
            })
        
        # 保存完整排名
        ranking_df = pd.DataFrame(detailed_ranking)
        ranking_path = os.path.join(self.output_dir, 'table', '06_complete_model_ranking.csv')
        os.makedirs(os.path.dirname(ranking_path), exist_ok=True)
        ranking_df.to_csv(ranking_path, index=False)
        
        print(f"\n📊 Complete ranking saved to: {ranking_path}")
        
        return ranking_df, ranked_results
    
    def analyze_top_performers(self, ranked_results, top_n=10):
        """分析顶级表现者"""
        print(f"\n{'='*80}")
        print(f"🥇 TOP {top_n} BEST PERFORMING MODELS")
        print(f"{'='*80}")
        
        top_models = ranked_results.head(top_n)
        
        for i, (_, row) in enumerate(top_models.iterrows(), 1):
            accuracy = row['Accuracy']
            cv_mean = row.get('CV_Mean', np.nan)
            cv_std = row.get('CV_Std', np.nan)
            
            print(f"\n🏆 #{i}: {row['Model']}")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   🎯 Mode: {row['Mode']}")
            print(f"   🔧 Type: {row['Model_Type']}")
            if pd.notna(cv_mean) and pd.notna(cv_std):
                print(f"   ✅ CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return top_models
    
    def analyze_by_mode(self, ranked_results):
        """按模式分析最佳模型"""
        print(f"\n{'='*80}")
        print(f"🎯 BEST MODEL FOR EACH MODE")
        print(f"{'='*80}")
        
        mode_best = {}
        for mode in ranked_results['Mode'].unique():
            mode_data = ranked_results[ranked_results['Mode'] == mode]
            best_model = mode_data.iloc[0]  # 第一个就是最佳的
            mode_best[mode] = best_model
            
            accuracy = best_model['Accuracy']
            cv_mean = best_model.get('CV_Mean', np.nan)
            cv_std = best_model.get('CV_Std', np.nan)
            
            print(f"\n🔊 {mode.upper()} Mode:")
            print(f"   🏆 Best: {best_model['Model']}")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   🔧 Type: {best_model['Model_Type']}")
            if pd.notna(cv_mean) and pd.notna(cv_std):
                print(f"   ✅ CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return mode_best
    
    def analyze_by_type(self, ranked_results):
        """按模型类型分析最佳模型"""
        print(f"\n{'='*80}")
        print(f"🔧 BEST MODEL FOR EACH TYPE")
        print(f"{'='*80}")
        
        type_best = {}
        for model_type in ranked_results['Model_Type'].unique():
            type_data = ranked_results[ranked_results['Model_Type'] == model_type]
            best_model = type_data.iloc[0]  # 第一个就是最佳的
            type_best[model_type] = best_model
            
            accuracy = best_model['Accuracy']
            cv_mean = best_model.get('CV_Mean', np.nan)
            cv_std = best_model.get('CV_Std', np.nan)
            
            print(f"\n🤖 {model_type}:")
            print(f"   🏆 Best: {best_model['Model']}")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   🎯 Mode: {best_model['Mode']}")
            if pd.notna(cv_mean) and pd.notna(cv_std):
                print(f"   ✅ CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return type_best
    
    def generate_summary_statistics(self, ranked_results):
        """生成统计摘要"""
        print(f"\n{'='*80}")
        print(f"📈 PERFORMANCE STATISTICS")
        print(f"{'='*80}")
        
        accuracy_col = 'Accuracy'
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Models: {len(ranked_results)}")
        print(f"   Best Accuracy: {ranked_results[accuracy_col].max():.4f}")
        print(f"   Worst Accuracy: {ranked_results[accuracy_col].min():.4f}")
        print(f"   Mean Accuracy: {ranked_results[accuracy_col].mean():.4f}")
        print(f"   Std Accuracy: {ranked_results[accuracy_col].std():.4f}")
        
        # 按模式统计
        print(f"\n🎯 By Mode:")
        for mode in ranked_results['Mode'].unique():
            mode_data = ranked_results[ranked_results['Mode'] == mode]
            print(f"   {mode.upper():<10}: {len(mode_data):2d} models, Best: {mode_data[accuracy_col].max():.4f}, Mean: {mode_data[accuracy_col].mean():.4f}")
        
        # 按类型统计
        print(f"\n🔧 By Type:")
        for model_type in ranked_results['Model_Type'].unique():
            type_data = ranked_results[ranked_results['Model_Type'] == model_type]
            print(f"   {model_type:<15}: {len(type_data):2d} models, Best: {type_data[accuracy_col].max():.4f}, Mean: {type_data[accuracy_col].mean():.4f}")
    
    def create_ranking_visualization(self, ranked_results, top_n=20):
        """创建排名可视化"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Top N 模型条形图
        top_models = ranked_results.head(top_n)
        ax1 = axes[0, 0]
        bars = ax1.barh(range(len(top_models)), top_models['Accuracy'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_models))))
        ax1.set_yticks(range(len(top_models)))
        ax1.set_yticklabels([f"{i+1}. {model}" for i, model in enumerate(top_models['Model'])], fontsize=8)
        ax1.set_xlabel('Accuracy')
        ax1.set_title(f'Top {top_n} Models by Accuracy')
        ax1.invert_yaxis()
        
        # 按模式分组的箱线图
        ax2 = axes[0, 1]
        mode_data = [ranked_results[ranked_results['Mode'] == mode]['Accuracy'].values 
                    for mode in ranked_results['Mode'].unique()]
        ax2.boxplot(mode_data, labels=ranked_results['Mode'].unique())
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Distribution by Mode')
        ax2.tick_params(axis='x', rotation=45)
        
        # 按类型分组的箱线图
        ax3 = axes[1, 0]
        type_data = [ranked_results[ranked_results['Model_Type'] == mtype]['Accuracy'].values 
                    for mtype in ranked_results['Model_Type'].unique()]
        ax3.boxplot(type_data, labels=ranked_results['Model_Type'].unique())
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Distribution by Model Type')
        ax3.tick_params(axis='x', rotation=45)
        
        # 散点图：准确率 vs 排名
        ax4 = axes[1, 1]
        colors = {'shengying': 'red', 'zhendong': 'blue', 'fusion': 'green'}
        for mode in ranked_results['Mode'].unique():
            mode_data = ranked_results[ranked_results['Mode'] == mode]
            ax4.scatter(range(len(mode_data)), mode_data['Accuracy'], 
                       label=mode, alpha=0.7, color=colors.get(mode, 'gray'))
        ax4.set_xlabel('Rank')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Rank by Mode')
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存图片
        img_path = os.path.join(self.output_dir, 'images', '06_model_ranking_analysis.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Ranking visualization saved to: {img_path}")
    
    def run_complete_analysis(self):
        """运行完整的排名分析"""
        print("🚀 STARTING COMPLETE MODEL RANKING ANALYSIS")
        
        # 加载所有结果
        if not self.load_all_results():
            print("❌ Failed to load results!")
            return
        
        # 生成完整排名
        ranking_df, ranked_results = self.generate_complete_ranking()
        
        # 分析顶级表现者
        top_models = self.analyze_top_performers(ranked_results, top_n=10)
        
        # 按模式分析
        mode_best = self.analyze_by_mode(ranked_results)
        
        # 按类型分析
        type_best = self.analyze_by_type(ranked_results)
        
        # 生成统计摘要
        self.generate_summary_statistics(ranked_results)
        
        # 创建可视化
        self.create_ranking_visualization(ranked_results, top_n=20)
        
        # 最终总结
        overall_best = ranked_results.iloc[0]
        print(f"\n{'='*100}")
        print(f"🎉 FINAL RECOMMENDATION FOR PRACTICAL USE")
        print(f"{'='*100}")
        print(f"🏆 BEST CLASSIFIER: {overall_best['Model']}")
        print(f"📊 ACCURACY: {overall_best['Accuracy']:.4f}")
        print(f"🎯 MODE: {overall_best['Mode']}")
        print(f"🔧 TYPE: {overall_best['Model_Type']}")
        if pd.notna(overall_best.get('CV_Mean')) and pd.notna(overall_best.get('CV_Std')):
            print(f"✅ CROSS-VALIDATION: {overall_best['CV_Mean']:.4f} ± {overall_best['CV_Std']:.4f}")
        print(f"📁 SOURCE: {overall_best.get('File_Source', 'N/A')}")
        
        print(f"\n🎯 This is the single best model for motor fault classification!")
        print(f"{'='*100}")
        
        return ranking_df

def main():
    """主函数"""
    analyzer = ModelRanking()
    ranking_df = analyzer.run_complete_analysis()
    
    if ranking_df is not None:
        print(f"\n✅ Analysis complete! Check the output folder for detailed results.")
    else:
        print(f"\n❌ Analysis failed!")

if __name__ == "__main__":
    main()
