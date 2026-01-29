#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DTA-GAT模型核心实验脚本
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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.dta_gat import TopicPropagationModel as OptimizedTopicPropagationModel
from src.data.loader import TopicPropagationLoader
from src.utils.trainer import TopicPropagationRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExperimentRunner:
    def __init__(self, output_dir='experimental_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.all_results = {
            '5.2.1': {},
            '5.2.2': {}
        }
        
        self.topics = {
            'PHEME': os.path.join(project_root, 'data', 'pheme'),
            'Twitter_COVID19': os.path.join(project_root, 'data', 'twitter_covid19'),
            'Weibo': os.path.join(project_root, 'data', 'weibo')
        }
    
    def run_experiment_521(self):
        """实验 5.2.1: 话题结构特征表示有效性"""
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
        self._generate_figure_521(results)
        
        return results
    
    def run_experiment_522(self):
        """实验 5.2.2: 话题吸引力有效性分析"""
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
        self._generate_figure_522(results)
        
        return results
    
    def _run_single_experiment(self, topic_file, experiment_type, variant):
        try:
            args = self._create_args(topic_file, experiment_type, variant)
            
            data_loader = TopicPropagationLoader(args)
            train_data, valid_data, test_data = data_loader.split_data()
            
            model = OptimizedTopicPropagationModel(args, data_loader)
            model = model.to(args.device)
            
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
            return 999.99, 999.99
    
    def _create_args(self, topic_file, experiment_type, variant):
        class Args:
            def __init__(self):
                self.data_path = topic_file
                self.data_name = os.path.basename(topic_file)
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                self.num_epochs = 50
                self.batch_size = 32
                self.learning_rate = 0.001
                self.patience = 10
                
                self.topic_dim = 128
                self.hidden_dim = 256
                self.num_heads = 4
                self.dropout = 0.3
                
                self.p = 1.0
                self.q = 1.0
                self.walk_length = 80
                self.num_walks = 10
                self.embedding_dim = 128
                self.window_size = 10
                self.negative = 5
                self.workers = 4
                
                self.use_edge_weight = True
                self.use_deepwalk = False
                self.use_attractiveness = True
                self.attractiveness_mode = 'multi'
                
                self.time_decay = 0.05
                
                self.lambda0 = 0.15
                self.lambda1 = 0.6 
                self.lambda2 = 0.25
        
        args = Args()
        
        if experiment_type == 'structure':
            if variant == 'NoW-Node2vec':
                args.use_edge_weight = False
                args.use_deepwalk = False
                
            elif variant == 'NormW-Node2vec':
                args.use_edge_weight = True
                args.use_deepwalk = False
                
            elif variant == 'NormW-DeepWalk':
                args.use_edge_weight = True
                args.use_deepwalk = True
                args.p = 1.0
                args.q = 1.0
        
        elif experiment_type == 'attractiveness':
            if variant == 'Baseline':
                args.use_attractiveness = False
                args.attractiveness_mode = 'none'
                
            elif variant == 'Single-Dimension':
                args.use_attractiveness = True
                args.attractiveness_mode = 'single'
                args.lambda1 = 1.0
                args.lambda2 = 0.0
                
            elif variant == 'Multi-Dimension':
                args.use_attractiveness = True
                args.attractiveness_mode = 'multi'
                args.lambda1 = 0.6
                args.lambda2 = 0.25
        
        return args

    def _generate_figure_521(self, results):
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
    
    def save_all_results(self):
        output_file = os.path.join(self.output_dir, 'all_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        logging.info(f"\n所有结果保存到: {output_file}")

def main():
    logging.info("\n" + "="*60)
    logging.info("DTA-GAT模型核心实验 5.2.1-5.2.2")
    logging.info("="*60)
    
    runner = ExperimentRunner()
    
    logging.info("\n开始运行模型核心实验...")
    results_521 = runner.run_experiment_521()
    results_522 = runner.run_experiment_522()
    
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
