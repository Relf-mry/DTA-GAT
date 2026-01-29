# -*- coding: utf-8 -*-
"""
话题传播模型训练器
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
        self.patience = args.patience
    
    def run(self, model, train_data, valid_data, test_data, data_loader, args):
        self.data_loader = data_loader
        
        loss_func = nn.MSELoss()
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate if hasattr(args, 'learning_rate') else 0.001,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=2,
            eta_min=1e-6
        )
        
        model.to(args.device)
        
        best_valid_mae = float('inf')
        best_valid_rmse = float('inf')
        epochs_without_improvement = 0
        
        adj_matrix = getattr(data_loader, 'adj_matrix', None)
        time_matrix = getattr(data_loader, 'time_matrix', None)
        edge_index = getattr(data_loader, 'edge_index', None)
        edge_time = getattr(data_loader, 'edge_time', None)
        
        if edge_index is not None:
            model._edge_index = edge_index.to(args.device)
            model._edge_time = edge_time.to(args.device) if edge_time is not None else None
        
        if adj_matrix is not None:
            adj_matrix = adj_matrix.to(args.device)
            time_matrix = time_matrix.to(args.device) if time_matrix is not None else torch.zeros_like(adj_matrix)
            logging.info(f"使用图结构: {data_loader.topic_num} 个节点")
        else:
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
            scheduler.step()
            
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
        model.train()
        
        total_mae = 0.0
        total_rmse = 0.0
        n_batches = 0
        
        attract_dim_data = getattr(self.data_loader, 'attractiveness_features', None)
        
        for batch in tqdm(training_data, desc="  Training", ncols=100, leave=False):
            topic_ids = batch['topic_id'].to(device)
            if topic_ids.dim() == 0:
                topic_ids = topic_ids.unsqueeze(0)
            gold = batch['propagation_scale'].to(device)
            if gold.dim() == 0:
                gold = gold.unsqueeze(0)
            
            attractiveness_batch = None
            if attract_dim_data is not None:
                attractiveness_batch = attract_dim_data[topic_ids.cpu()].to(device)
            
            optimizer.zero_grad()
            
            prediction = model.forward(
                topic_ids, adj_matrix, time_matrix, 
                attractiveness=attractiveness_batch
            )
            prediction = prediction.squeeze(-1)
            
            loss = torch.nn.functional.l1_loss(prediction, gold)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_mae += loss.item()
            total_rmse += torch.sqrt(torch.nn.functional.mse_loss(prediction, gold)).item()
            n_batches += 1
        
        avg_mae = total_mae / n_batches
        avg_rmse = total_rmse / n_batches
        
        return avg_mae, avg_rmse
    
    def test_epoch(self, model, validation_data, device, adj_matrix, time_matrix):
        model.eval()
        
        all_predictions = []
        all_golds = []
        
        attract_dim_data = getattr(self.data_loader, 'attractiveness_features', None)
        
        with torch.no_grad():
            for batch in tqdm(validation_data, desc="  Evaluating", ncols=100, leave=False):
                topic_ids = batch['topic_id'].to(device)
                if topic_ids.dim() == 0:
                    topic_ids = topic_ids.unsqueeze(0)
                gold = batch['propagation_scale'].to(device)
                if gold.dim() == 0:
                    gold = gold.unsqueeze(0)
                
                attractiveness_batch = None
                if attract_dim_data is not None:
                    attractiveness_batch = attract_dim_data[topic_ids.cpu()].to(device)
                
                prediction = model.forward(
                    topic_ids, adj_matrix, time_matrix,
                    attractiveness=attractiveness_batch
                )
                prediction = prediction.squeeze(-1)
                
                all_predictions.append(prediction.cpu())
                all_golds.append(gold.cpu())
        
        all_predictions = torch.cat(all_predictions)
        all_golds = torch.cat(all_golds)
        
        avg_mae = torch.nn.functional.l1_loss(all_predictions, all_golds).item()
        avg_rmse = torch.sqrt(torch.nn.functional.mse_loss(all_predictions, all_golds)).item()
        
        # 额外指标 (MAPE, R2)
        with torch.no_grad():
            mask = torch.abs(all_golds) > 1e-6
            if mask.sum() > 0:
                mape = torch.mean(torch.abs((all_predictions[mask] - all_golds[mask]) / all_golds[mask])).item() * 100
            else:
                mape = 0.0
            
            target_mean = torch.mean(all_golds)
            ss_tot = torch.sum((all_golds - target_mean) ** 2)
            ss_res = torch.sum((all_golds - all_predictions) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            r2 = r2.item()
            
            logging.info(f"    - Additional Metrics: MAPE={mape:.2f}%, R2={r2:.4f}")
        
        return avg_mae, avg_rmse
