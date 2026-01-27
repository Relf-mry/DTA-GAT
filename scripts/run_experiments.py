#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DTA-GAT模型核心实验脚本

实验内容:
5.2.1: 话题结构特征表示有效性 (3组实验 × 4话题)
5.2.2: 话题吸引力有效性分析 (3组实验 × 4话题)
"""


import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 从新的src目录导入
from src.models.dta_gat import TopicPropagationModel as OptimizedTopicPropagationModel
from src.data.loader import TopicPropagationLoader
from src.utils.trainer import TopicPropagationRunner


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, output_dir='experimental_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储所有实验结果
        self.all_results = {
            '5.2.1': {},  # 结构特征表示
            '5.2.2': {}   # 话题吸引力
        }
        
        # 使用pheme和twitter_covid19数据集
        self.topics = {
            'PHEME': os.path.join(project_root, 'data', 'pheme'),
            'Twitter_COVID19': os.path.join(project_root, 'data', 'twitter_covid19')
        }
    
    def run_experiment_521(self):
        """
        5.2.1: 话题结构特征表示有效性
        
        实验组:
        1. NoW-Node2vec: 无权重 + Node2Vec
        2. NormW-Node2vec: 归一化权重 + Node2Vec
        3. NormW-DeepWalk: 归一化权重 + DeepWalk
        """
        logging.info("\n" + "="*60)
        logging.info("实验 5.2.1: 话题结构特征表示有效性")
        logging.info("="*60)
        
        experiments = [
            'NoW-Node2vec',
            'NormW-Node2vec',
            'NormW-DeepWalk'
        ]
        
        results = {}
        
        for topic_name, topic_file in self.topics.items():
            logging.info(f"\n处理 {topic_name}...")
            topic_results = {}
            
            for exp_name in experiments:
                logging.info(f"  运行 {exp_name}...")
                
                # 运行实验
                mae, rmse = self._run_single_experiment(
                    topic_file=topic_file,
                    experiment_type='structure',
                    variant=exp_name
                )
                
                topic_results[exp_name] = {
                    'MAE': mae,
                    'RMSE': rmse
                }
                
                logging.info(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            results[topic_name] = topic_results
        
        self.all_results['5.2.1'] = results
        
        # 生成图表
        self._generate_figure_521(results)
        
        return results
    
    def run_experiment_522(self):
        """
        5.2.2: 话题吸引力有效性分析
        
        实验组:
        1. Baseline: 无吸引力
        2. Single-Dimension: 单维度吸引力
        3. Multi-Dimension: 多维度吸引力
        """
        logging.info("\n" + "="*60)
        logging.info("实验 5.2.2: 话题吸引力有效性分析")
        logging.info("="*60)
        
        experiments = [
            'Baseline',
            'Single-Dimension',
            'Multi-Dimension'
        ]
        
        results = {}
        
        for topic_name, topic_file in self.topics.items():
            logging.info(f"\n处理 {topic_name}...")
            topic_results = {}
            
            for exp_name in experiments:
                logging.info(f"  运行 {exp_name}...")
                
                # 运行实验
                mae, rmse = self._run_single_experiment(
                    topic_file=topic_file,
                    experiment_type='attractiveness',
                    variant=exp_name
                )
                
                topic_results[exp_name] = {
                    'MAE': mae,
                    'RMSE': rmse
                }
                
                logging.info(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            results[topic_name] = topic_results
        
        self.all_results['5.2.2'] = results
        
        # 生成图表
        self._generate_figure_522(results)
        
        return results
    
    # 移除消融实验 - 用户只需要核心模型实验
    # def run_experiment_523(self):
    #     """5.2.3: DTA-GAT消融实验"""
    #     pass

    
    
    def _run_single_experiment(self, topic_file, experiment_type, variant):
        """
        运行单个实验
        
        Args:
            topic_file: 话题数据文件
            experiment_type: 实验类型 (structure/attractiveness)
            variant: 实验变体
        
        Returns:
            mae, rmse: 评估指标
        """
        try:
            # 创建参数配置
            args = self._create_args(topic_file, experiment_type, variant)
            
            # 加载数据
            data_loader = TopicPropagationLoader(args)
            train_data, valid_data, test_data = data_loader.split_data()
            
            # 创建模型
            model = OptimizedTopicPropagationModel(args, data_loader)
            model = model.to(args.device)
            
            # 训练模型
            runner = TopicPropagationRunner(args)
            test_mae, test_rmse = runner.run(
                model, train_data, valid_data, test_data,
                data_loader, args
            )
            
            return test_mae, test_rmse
            
        except Exception as e:
            logging.error(f"实验运行失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认值以继续运行
            return 999.99, 999.99
    
    def _create_args(self, topic_file, experiment_type, variant):
        """
        根据实验类型和变体创建参数配置
        
        Args:
            topic_file: 数据文件路径
            experiment_type: 实验类型
            variant: 实验变体
        
        Returns:
            args: 参数对象
        """
        class Args:
            def __init__(self):
                # 数据相关 - 从目录路径提取数据集名称
                # topic_file现在是目录路径，如 'data/pheme'
                self.data_path = topic_file  # 数据目录路径
                self.data_name = os.path.basename(topic_file)  # 'pheme' 或 'twitter_covid19'
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # 训练参数
                self.num_epochs = 50
                self.batch_size = 32
                self.learning_rate = 0.001
                self.patience = 10
                
                # 模型基础参数
                self.topic_dim = 128
                self.hidden_dim = 256
                self.num_heads = 4
                self.dropout = 0.3
                
                # Node2vec参数（默认值）
                self.p = 1.0  # 控制返回参数
                self.q = 1.0  # 控制DFS vs BFS
                self.walk_length = 80
                self.num_walks = 10
                self.embedding_dim = 128
                self.window_size = 10
                self.negative = 5
                self.workers = 4
                
                # 其他参数
                self.use_edge_weight = True  # 默认使用边权重
                self.use_deepwalk = False  # 默认使用Node2vec
                self.use_attractiveness = True  # 默认使用吸引力
                self.attractiveness_mode = 'multi'  # multi/single/none
                
                # 时间衰减
                self.time_decay = 0.05
                
                # 吸引力系数（论文中的最优值）
                self.lambda0 = 0.15
                self.lambda1 = 0.6 
                self.lambda2 = 0.25
        
        args = Args()
        
        # 根据实验类型调整参数
        if experiment_type == 'structure':
            # 5.2.1 结构特征表示实验
            if variant == 'NoW-Node2vec':
                # 无权重 + Node2vec
                args.use_edge_weight = False
                args.use_deepwalk = False
                
            elif variant == 'NormW-Node2vec':
                # 归一化权重 + Node2vec（最优配置）
                args.use_edge_weight = True
                args.use_deepwalk = False
                
            elif variant == 'NormW-DeepWalk':
                # 归一化权重 + DeepWalk
                args.use_edge_weight = True
                args.use_deepwalk = True
                # DeepWalk相当于p=1, q=1
                args.p = 1.0
                args.q = 1.0
        
        elif experiment_type == 'attractiveness':
            # 5.2.2 吸引力有效性实验
            if variant == 'Baseline':
                # 无吸引力
                args.use_attractiveness = False
                args.attractiveness_mode = 'none'
                
            elif variant == 'Single-Dimension':
                # 单维度吸引力（仅使用影响力）
                args.use_attractiveness = True
                args.attractiveness_mode = 'single'
                args.lambda1 = 1.0  # 只用影响力
                args.lambda2 = 0.0
                
            elif variant == 'Multi-Dimension':
                # 多维度吸引力（最优配置）
                args.use_attractiveness = True
                args.attractiveness_mode = 'multi'
                args.lambda1 = 0.6
                args.lambda2 = 0.25
        
        return args

    
    def _generate_figure_521(self, results):
        """生成Figure 5-2-1 (4个子图)"""
        logging.info("\n生成 Figure 5-2-1...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Experimental Results of Topic Structure Feature Representation', fontsize=14)
        
        topics = list(results.keys())
        experiments = ['NoW-Node2vec', 'NormW-Node2vec', 'NormW-DeepWalk']
        
        for idx, (topic_name, ax) in enumerate(zip(topics, axes.flat)):
            topic_data = results[topic_name]
            
            mae_values = [topic_data[exp]['MAE'] for exp in experiments]
            rmse_values = [topic_data[exp]['RMSE'] for exp in experiments]
            
            x = np.arange(len(experiments))
            width = 0.35
            
            ax.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
            ax.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
            
            ax.set_xlabel('Experiment')
            ax.set_ylabel('Error')
            ax.set_title(f'({chr(97+idx)}) {topic_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(experiments, rotation=15, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'figure_5_2_1.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"  保存到: {output_file}")
    
    def _generate_figure_522(self, results):
        """生成Figure 5-2-2 (4个子图)"""
        logging.info("\n生成 Figure 5-2-2...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Experimental Results on Quantifying Topic Attractiveness', fontsize=14)
        
        topics = list(results.keys())
        experiments = ['Baseline', 'Single-Dimension', 'Multi-Dimension']
        
        for idx, (topic_name, ax) in enumerate(zip(topics, axes.flat)):
            topic_data = results[topic_name]
            
            mae_values = [topic_data[exp]['MAE'] for exp in experiments]
            rmse_values = [topic_data[exp]['RMSE'] for exp in experiments]
            
            x = np.arange(len(experiments))
            width = 0.35
            
            ax.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
            ax.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
            
            ax.set_xlabel('Experiment')
            ax.set_ylabel('Error')
            ax.set_title(f'({chr(97+idx)}) {topic_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(experiments, rotation=15, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'figure_5_2_2.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"  保存到: {output_file}")
    
    def _generate_table_523(self, results):
        """生成Table 5-2-3 (LaTeX格式)"""
        logging.info("\n生成 Table 5-2-3...")
        
        # 创建LaTeX表格
        latex_content = r"""\begin{table}[htbp]
    \centering
    \caption{Ablation Experiments of the DTA-GAT Model}
    \begin{tabular}{cccc}
        \toprule
        Topic & Experimental Setup & MAE & RMSE \\
        \hline
"""
        
        for topic_name in ['Topic A', 'Topic B', 'Topic C', 'Topic D']:
            topic_data = results[topic_name]
            
            # Experiment 1: With Time-Aware
            mae1 = topic_data['With Time-Aware']['MAE']
            rmse1 = topic_data['With Time-Aware']['RMSE']
            
            # Experiment 2: Without Time-Aware
            mae2 = topic_data['Without Time-Aware']['MAE']
            rmse2 = topic_data['Without Time-Aware']['RMSE']
            
            latex_content += f"        {topic_name[-1]} & Experiment 1 (Including Time-Sensitive Attention Mechanism) & \\textbf{{{mae1:.4f}}} & \\textbf{{{rmse1:.4f}}} \\\\\n"
            latex_content += f"        \\hline\n"
            latex_content += f"        {topic_name[-1]} & Experiment 2 (Excluding Time-Sensitive Attention Mechanism) & {mae2:.4f} & {rmse2:.4f} \\\\\n"
            latex_content += f"        \\hline\n"
        
        latex_content += r"""    \end{tabular}
    \label{Table 5-2-3}
\end{table}
"""
        
        # 保存
        output_file = os.path.join(self.output_dir, 'table_5_2_3.tex')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        logging.info(f"  保存到: {output_file}")
    
    def save_all_results(self):
        """保存所有实验结果"""
        output_file = os.path.join(self.output_dir, 'all_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"\n所有结果保存到: {output_file}")


def main():
    """主函数"""
    logging.info("\n" + "="*60)
    logging.info("DTA-GAT模型核心实验 5.2.1-5.2.2")
    logging.info("="*60)
    
    # 创建实验运行器
    runner = ExperimentRunner()
    
    # 运行实验
    logging.info("\n开始运行模型核心实验...")
    results_521 = runner.run_experiment_521()  # 结构特征表示
    results_522 = runner.run_experiment_522()  # 吸引力有效性
    # 不运行消融实验
    
    # 保存结果
    runner.save_all_results()
    
    logging.info("\n" + "="*60)
    logging.info("所有实验完成!")
    logging.info("="*60)
    logging.info(f"\n结果保存在: {runner.output_dir}/")
    logging.info("  - figure_5_2_1.png (话题结构特征表示)")
    logging.info("  - figure_5_2_2.png (话题吸引力有效性)")
    logging.info("  - all_results.json (所有实验结果)")



if __name__ == '__main__':
    main()
