#!/usr/bin/env python3
"""
实验跟踪器
记录实验配置、结果和性能指标
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    """实验跟踪器"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.experiments = []
    
    def log_experiment(self, experiment_name: str, config: dict, results: dict):
        """记录实验"""
        experiment = {
            'name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results
        }
        
        self.experiments.append(experiment)
        
        # 保存到JSON文件
        json_path = self.output_path / 'experiments_log.json'
        with open(json_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        # 保存到CSV文件
        self._save_to_csv()
        
        print(f"✅ 实验已记录: {experiment_name}")
    
    def _save_to_csv(self):
        """保存结果到CSV"""
        if not self.experiments:
            return
        
        # 展平结果数据
        rows = []
        for exp in self.experiments:
            row = {
                'experiment_name': exp['name'],
                'timestamp': exp['timestamp']
            }
            row.update(exp['results'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.output_path / 'experiments_results.csv'
        df.to_csv(csv_path, index=False)
    
    def get_best_experiment(self, metric: str = 'accuracy'):
        """获取最佳实验"""
        if not self.experiments:
            return None
        
        best_exp = max(self.experiments, 
                      key=lambda x: x['results'].get(metric, 0))
        return best_exp

if __name__ == "__main__":
    tracker = ExperimentTracker("../../output/table/")
    
    # 示例实验记录
    config = {'model': 'random_forest', 'n_estimators': 100}
    results = {'accuracy': 0.85, 'f1_score': 0.82}
    tracker.log_experiment('RF_baseline', config, results)
