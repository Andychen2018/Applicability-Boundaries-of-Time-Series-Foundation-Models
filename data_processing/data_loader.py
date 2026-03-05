#!/usr/bin/env python3
"""
数据加载器模块
负责从data3目录加载电机时序数据，包含数据统计和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import json
from datetime import datetime

class MotorDataLoader:
    """电机数据加载器"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_path = Path(self.config['data']['path'])
        self.sensors = self.config['data']['sensors']
        self.states = self.config['data']['states']
        self.sampling_rate = self.config['data']['sampling_rate']
        self.output_path = Path(self.config['output']['tables'])
        self.image_path = Path(self.config['output']['images'])

        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (12, 8)

    def load_all_data(self, max_files_per_state: Optional[int] = None) -> Dict:
        """加载所有数据"""
        print("📂 开始加载电机数据...")
        data = {}
        file_info = {}

        for sensor in self.sensors:
            data[sensor] = {}
            file_info[sensor] = {}

            for state in self.states:
                state_path = self.data_path / sensor / state
                if state_path.exists():
                    signals, files = self._load_state_data(state_path, max_files_per_state)
                    data[sensor][state] = signals
                    file_info[sensor][state] = files
                    print(f"✅ 加载 {sensor}/{state}: {len(signals)} 个文件")
                else:
                    print(f"⚠️ 路径不存在: {state_path}")
                    data[sensor][state] = []
                    file_info[sensor][state] = []

        # 保存数据统计信息
        self._save_data_statistics(data, file_info)

        return data, file_info

    def _load_state_data(self, state_path: Path, max_files: Optional[int] = None) -> Tuple[List[np.ndarray], List[str]]:
        """加载特定状态的数据"""
        signals = []
        file_names = []
        csv_files = list(state_path.glob("*.csv"))

        if max_files:
            csv_files = csv_files[:max_files]

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, header=None)
                signal = df.iloc[:, 0].values if len(df.columns) == 1 else df.values.flatten()

                # 基本质量检查
                if len(signal) > 100 and not np.all(np.isnan(signal)):
                    signals.append(signal)
                    file_names.append(file_path.name)

            except Exception as e:
                print(f"❌ 加载失败 {file_path}: {e}")

        return signals, file_names

    def _save_data_statistics(self, data: Dict, file_info: Dict):
        """保存数据统计信息"""
        stats = []

        for sensor in self.sensors:
            for state in self.states:
                signals = data[sensor][state]
                if signals:
                    lengths = [len(signal) for signal in signals]

                    stat_row = {
                        'sensor': sensor,
                        'state': state,
                        'file_count': len(signals),
                        'min_length': min(lengths),
                        'max_length': max(lengths),
                        'mean_length': np.mean(lengths),
                        'std_length': np.std(lengths),
                        'total_samples': sum(lengths)
                    }
                    stats.append(stat_row)

        # 保存为CSV
        stats_df = pd.DataFrame(stats)
        stats_path = self.output_path / 'data_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        print(f"📊 数据统计已保存: {stats_path}")

        # 保存为JSON
        json_path = self.output_path / 'data_info.json'
        data_info = {
            'timestamp': datetime.now().isoformat(),
            'sampling_rate': self.sampling_rate,
            'statistics': stats,
            'file_info': file_info
        }

        with open(json_path, 'w') as f:
            json.dump(data_info, f, indent=2, default=str)
        print(f"📋 数据信息已保存: {json_path}")

    def analyze_data_distribution(self, data: Dict) -> Dict:
        """分析数据分布"""
        print("📈 分析数据分布...")

        analysis = {}

        for sensor in self.sensors:
            analysis[sensor] = {}

            for state in self.states:
                signals = data[sensor][state]
                if not signals:
                    continue

                # 计算统计特征
                all_values = np.concatenate(signals)
                lengths = [len(signal) for signal in signals]

                state_analysis = {
                    'signal_count': len(signals),
                    'total_samples': len(all_values),
                    'length_stats': {
                        'min': min(lengths),
                        'max': max(lengths),
                        'mean': np.mean(lengths),
                        'std': np.std(lengths)
                    },
                    'amplitude_stats': {
                        'min': float(np.min(all_values)),
                        'max': float(np.max(all_values)),
                        'mean': float(np.mean(all_values)),
                        'std': float(np.std(all_values)),
                        'skewness': float(pd.Series(all_values).skew()),
                        'kurtosis': float(pd.Series(all_values).kurtosis())
                    }
                }

                analysis[sensor][state] = state_analysis

        return analysis

    def visualize_data_overview(self, data: Dict, analysis: Dict):
        """可视化数据概览"""
        print("🎨 生成数据可视化...")

        # 1. 数据量统计图
        self._plot_data_counts(data)

        # 2. 信号长度分布
        self._plot_length_distribution(data)

        # 3. 典型信号波形对比
        self._plot_signal_comparison(data)

        # 4. 幅值分布对比
        self._plot_amplitude_distribution(data)

        print("✅ 数据可视化完成")

    def _plot_data_counts(self, data: Dict):
        """绘制数据量统计图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 准备数据
        sensors = []
        states = []
        counts = []

        for sensor in self.sensors:
            for state in self.states:
                sensors.append(sensor)
                states.append(state)
                counts.append(len(data[sensor][state]))

        # 创建DataFrame
        df = pd.DataFrame({
            'Sensor': sensors,
            'State': states,
            'Count': counts
        })

        # 按传感器分组的柱状图
        sensor_counts = df.groupby('Sensor')['Count'].sum()
        axes[0].bar(sensor_counts.index, sensor_counts.values, color=['skyblue', 'lightcoral'])
        axes[0].set_title('Files Count by Sensor')
        axes[0].set_ylabel('Number of Files')

        # 按状态分组的柱状图
        state_counts = df.groupby('State')['Count'].sum()
        colors = ['green', 'orange', 'red']
        axes[1].bar(state_counts.index, state_counts.values, color=colors)
        axes[1].set_title('Files Count by State')
        axes[1].set_ylabel('Number of Files')

        plt.tight_layout()
        save_path = self.image_path / 'data_exploration' / 'data_counts.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 数据统计图已保存: {save_path}")

    def _plot_length_distribution(self, data: Dict):
        """绘制信号长度分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        plot_idx = 0
        colors = ['green', 'orange', 'red']

        for sensor in self.sensors:
            lengths_by_state = {}

            for i, state in enumerate(self.states):
                signals = data[sensor][state]
                if signals:
                    lengths = [len(signal) for signal in signals]
                    lengths_by_state[state] = lengths

                    # 直方图
                    axes[plot_idx].hist(lengths, bins=20, alpha=0.7,
                                      label=state, color=colors[i])

            axes[plot_idx].set_title(f'{sensor} - Signal Length Distribution')
            axes[plot_idx].set_xlabel('Signal Length')
            axes[plot_idx].set_ylabel('Frequency')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # 整体长度分布对比
        all_lengths = {}
        for state in self.states:
            all_lengths[state] = []
            for sensor in self.sensors:
                signals = data[sensor][state]
                if signals:
                    all_lengths[state].extend([len(signal) for signal in signals])

        for i, (state, lengths) in enumerate(all_lengths.items()):
            if lengths:
                axes[plot_idx].hist(lengths, bins=30, alpha=0.7,
                                  label=state, color=colors[i])

        axes[plot_idx].set_title('Overall Signal Length Distribution')
        axes[plot_idx].set_xlabel('Signal Length')
        axes[plot_idx].set_ylabel('Frequency')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(plot_idx + 1, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = self.image_path / 'data_exploration' / 'length_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📏 长度分布图已保存: {save_path}")

    def _plot_signal_comparison(self, data: Dict):
        """绘制典型信号波形对比"""
        fig, axes = plt.subplots(len(self.sensors), len(self.states),
                                figsize=(15, 10))

        if len(self.sensors) == 1:
            axes = axes.reshape(1, -1)

        colors = ['green', 'orange', 'red']

        for i, sensor in enumerate(self.sensors):
            for j, state in enumerate(self.states):
                signals = data[sensor][state]

                if signals:
                    # 选择第一个信号作为代表
                    signal = signals[0]

                    # 只显示前5000个点以提高可视化效果
                    display_length = min(5000, len(signal))
                    time_axis = np.arange(display_length) / self.sampling_rate

                    axes[i, j].plot(time_axis, signal[:display_length],
                                  color=colors[j], linewidth=0.8)
                    axes[i, j].set_title(f'{sensor} - {state}')
                    axes[i, j].set_xlabel('Time (s)')
                    axes[i, j].set_ylabel('Amplitude')
                    axes[i, j].grid(True, alpha=0.3)

                    # 添加统计信息
                    mean_val = np.mean(signal)
                    std_val = np.std(signal)
                    axes[i, j].text(0.02, 0.98,
                                   f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}',
                                   transform=axes[i, j].transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    axes[i, j].text(0.5, 0.5, 'No Data',
                                   transform=axes[i, j].transAxes,
                                   ha='center', va='center')
                    axes[i, j].set_title(f'{sensor} - {state}')

        plt.tight_layout()
        save_path = self.image_path / 'data_exploration' / 'signal_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📈 信号对比图已保存: {save_path}")

    def _plot_amplitude_distribution(self, data: Dict):
        """绘制幅值分布对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        colors = ['green', 'orange', 'red']
        plot_idx = 0

        # 按传感器分别绘制
        for sensor in self.sensors:
            for i, state in enumerate(self.states):
                signals = data[sensor][state]
                if signals:
                    # 合并所有信号的幅值
                    all_amplitudes = np.concatenate(signals)

                    # 移除异常值（超过3个标准差）
                    mean_amp = np.mean(all_amplitudes)
                    std_amp = np.std(all_amplitudes)
                    filtered_amp = all_amplitudes[
                        np.abs(all_amplitudes - mean_amp) <= 3 * std_amp
                    ]

                    axes[plot_idx].hist(filtered_amp, bins=50, alpha=0.7,
                                      label=state, color=colors[i], density=True)

            axes[plot_idx].set_title(f'{sensor} - Amplitude Distribution')
            axes[plot_idx].set_xlabel('Amplitude')
            axes[plot_idx].set_ylabel('Density')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # 整体分布对比
        for i, state in enumerate(self.states):
            all_amplitudes = []
            for sensor in self.sensors:
                signals = data[sensor][state]
                if signals:
                    all_amplitudes.extend(np.concatenate(signals))

            if all_amplitudes:
                all_amplitudes = np.array(all_amplitudes)
                # 移除异常值
                mean_amp = np.mean(all_amplitudes)
                std_amp = np.std(all_amplitudes)
                filtered_amp = all_amplitudes[
                    np.abs(all_amplitudes - mean_amp) <= 3 * std_amp
                ]

                axes[plot_idx].hist(filtered_amp, bins=50, alpha=0.7,
                                  label=state, color=colors[i], density=True)

        axes[plot_idx].set_title('Overall Amplitude Distribution')
        axes[plot_idx].set_xlabel('Amplitude')
        axes[plot_idx].set_ylabel('Density')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(plot_idx + 1, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = self.image_path / 'data_exploration' / 'amplitude_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 幅值分布图已保存: {save_path}")

if __name__ == "__main__":
    # 数据加载和分析
    config_path = Path(__file__).parent.parent.parent / "experiments/configs/config.yaml"
    loader = MotorDataLoader(str(config_path))

    # 加载数据（限制每个状态最多50个文件以加快处理）
    data, file_info = loader.load_all_data(max_files_per_state=50)

    # 分析数据分布
    analysis = loader.analyze_data_distribution(data)

    # 生成可视化
    loader.visualize_data_overview(data, analysis)

    # 打印总结
    total_files = sum(len(data[sensor][state])
                     for sensor in loader.sensors
                     for state in loader.states)
    print(f"\n🎉 数据探索完成！")
    print(f"📊 总计加载 {total_files} 个信号文件")
    print(f"📁 结果保存在: {loader.output_path} 和 {loader.image_path}")

    # 打印简要统计
    print(f"\n📋 数据概览:")
    for sensor in loader.sensors:
        print(f"  {sensor}:")
        for state in loader.states:
            count = len(data[sensor][state])
            print(f"    {state}: {count} 个文件")
