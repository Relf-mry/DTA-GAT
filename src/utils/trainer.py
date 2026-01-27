# -*- coding: utf-8 -*-
"""
话题传播模型训练器
使用MAE和RMSE作为评估指标
"""

import logging
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class TopicPropagationRunner:
    """话题传播模型训练器"""
    
    def __init__(self, args):
        """
        初始化训练器
        
        Args:
            args: 命令行参数
        """
        self.patience = args.patience
    
    def run(self, model, train_data, valid_data, test_data, data_loader, args):
        """
        执行训练和评估
        
        Args:
            model: 模型
            train_data: 训练数据
            valid_data: 验证数据
            test_data: 测试数据
            data_loader: 数据加载器
            args: 命令行参数
        """
        self.data_loader = data_loader
        
        # 损失函数: MSE用于回归任务
        loss_func = nn.MSELoss()
        
        # 优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate if hasattr(args, 'learning_rate') else 0.001,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-5
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        model.to(args.device)
        
        best_valid_mae = float('inf')
        best_valid_rmse = float('inf')
        epochs_without_improvement = 0
        
        # 获取邻接矩阵和时间矩阵 (论文公式18-21需要)
        adj_matrix = getattr(data_loader, 'adj_matrix', None)
        time_matrix = getattr(data_loader, 'time_matrix', None)
        edge_index = getattr(data_loader, 'edge_index', None)
        edge_time = getattr(data_loader, 'edge_time', None)
        
        # 将edge_index和edge_time存储到模型中,供OptimizedModel使用
        if edge_index is not None:
            model._edge_index = edge_index.to(args.device)
            model._edge_time = edge_time.to(args.device) if edge_time is not None else None
        
        if adj_matrix is not None:
            adj_matrix = adj_matrix.to(args.device)
            time_matrix = time_matrix.to(args.device) if time_matrix is not None else torch.zeros_like(adj_matrix)
            logging.info(f"使用图结构: {data_loader.topic_num} 个节点")
        else:
            # 创建全连接图
            adj_matrix = torch.ones(data_loader.topic_num, data_loader.topic_num).to(args.device)
            time_matrix = torch.zeros(data_loader.topic_num, data_loader.topic_num).to(args.device)
            logging.info("使用全连接图")
        
        for epoch_i in range(args.epoch):
            logging.info(f'\n[ Epoch {epoch_i} ]')
            start = time.time()
            
            # 训练
            train_mae, train_rmse = self.train_epoch(
                model, train_data, loss_func, optimizer, args.device,
                adj_matrix, time_matrix
            )
            logging.info(
                f'  - (Training)   MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, '
                f'Elapsed: {(time.time() - start):.2f}s'
            )
            
            # 验证
            start = time.time()
            valid_mae, valid_rmse = self.test_epoch(
                model, valid_data, args.device, adj_matrix, time_matrix
            )
            logging.info(
                f'  - (Validation) MAE: {valid_mae:.4f}, RMSE: {valid_rmse:.4f}, '
                f'Elapsed: {(time.time() - start):.2f}s'
            )
            
            # 学习率调度
            scheduler.step(valid_mae)
            
            # 检查是否改进
            if valid_mae < best_valid_mae:
                logging.info('  Best validation MAE improved. Saving model...')
                best_valid_mae = valid_mae
                best_valid_rmse = valid_rmse
                torch.save(model.state_dict(), args.model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logging.info(f'  No improvement for {epochs_without_improvement} epoch(s).')
            
            # 早停
            if epochs_without_improvement >= self.patience:
                logging.info(f'  Early stopping triggered after {epoch_i + 1} epochs.')
                break
        
        # 最终测试
        logging.info('\n  - (Final Test)')
        test_mae, test_rmse = self.test_epoch(
            model, test_data, args.device, adj_matrix, time_matrix
        )
        logging.info(f'  Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}')
        
        logging.info("\n- (Finished!) \nBest validation scores:")
        logging.info(f"  MAE: {best_valid_mae:.4f}")
        logging.info(f"  RMSE: {best_valid_rmse:.4f}")
        
        return test_mae, test_rmse
    
    def train_epoch(self, model, training_data, loss_func, optimizer, device,
                   adj_matrix, time_matrix):
        """
        训练一个epoch
        
        Args:
            model: 模型
            training_data: 训练数据
            loss_func: 损失函数
            optimizer: 优化器
            device: 设备
            adj_matrix: 邻接矩阵
            time_matrix: 时间矩阵
        
        Returns:
            avg_mae: 平均MAE (论文公式23)
            avg_rmse: 平均RMSE (论文公式24)
        """
        model.train()
        
        total_mae = 0.0
        total_rmse = 0.0
        n_batches = 0
        
        # 预先获取吸引力特征引用
        attract_dim_data = getattr(self.data_loader, 'attractiveness_features', None)
        
        for batch in tqdm(training_data, desc="  Training", ncols=100, leave=False):
            topic_ids = batch['topic_id'].to(device)
            gold = batch['propagation_scale'].to(device)
            
            # 获取当前batch的吸引力特征
            attractiveness_batch = None
            if attract_dim_data is not None:
                attractiveness_batch = attract_dim_data[topic_ids.cpu()].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            # 直接调用forward获取tensor用于计算梯度
            prediction = model.forward(
                topic_ids, adj_matrix, time_matrix, 
                attractiveness=attractiveness_batch
            )
            prediction = prediction.squeeze()
            
            # 损失计算
            loss = torch.nn.functional.l1_loss(prediction, gold)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_mae += loss.item()
            total_rmse += torch.sqrt(torch.nn.functional.mse_loss(prediction, gold)).item()
            n_batches += 1
        
        avg_mae = total_mae / n_batches
        avg_rmse = total_rmse / n_batches
        
        return avg_mae, avg_rmse
    
    def test_epoch(self, model, validation_data, device, adj_matrix, time_matrix):
        """
        测试一个epoch
        
        Args:
            model: 模型
            validation_data: 验证/测试数据
            device: 设备
            adj_matrix: 邻接矩阵
            time_matrix: 时间矩阵
        
        Returns:
            avg_mae: 平均MAE (论文公式23)
            avg_rmse: 平均RMSE (论文公式24)
        """
        model.eval()
        
        total_mae = 0.0
        total_rmse = 0.0
        n_batches = 0
        
        all_predictions = []
        all_golds = []
        
        # 预先获取吸引力特征引用
        attract_dim_data = getattr(self.data_loader, 'attractiveness_features', None)
        
        with torch.no_grad():
            for batch in tqdm(validation_data, desc="  Evaluating", ncols=100, leave=False):
                topic_ids = batch['topic_id'].to(device)
                gold = batch['propagation_scale'].to(device)
                
                attractiveness_batch = None
                if attract_dim_data is not None:
                    attractiveness_batch = attract_dim_data[topic_ids.cpu()].to(device)
                
                # 前向传播
                prediction = model.forward(
                    topic_ids, adj_matrix, time_matrix,
                    attractiveness=attractiveness_batch
                )
                prediction = prediction.squeeze()
                
                mae = torch.nn.functional.l1_loss(prediction, gold).item()
                rmse = torch.sqrt(torch.nn.functional.mse_loss(prediction, gold)).item()
                
                total_mae += mae
                total_rmse += rmse
                n_batches += 1
                
                # 收集预测值和真实值
                all_predictions.extend(prediction.cpu().numpy().flatten())
                all_golds.extend(gold.cpu().numpy().flatten())
        
        avg_mae = total_mae / n_batches
        avg_rmse = total_rmse / n_batches
        
        return avg_mae, avg_rmse
