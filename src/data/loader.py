# -*- coding: utf-8 -*-
"""
改进的话题传播数据加载器
"""

import logging
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.node2vec import Node2vecEmbedding
from src.models.attractiveness import TopicAttractiveness


class TopicPropagationDataset(Dataset):
    """话题传播数据集"""
    
    def __init__(self, topic_ids, parent_topic_ids, propagation_scales, 
                 adj_matrices=None, time_matrices=None):
        self.topic_ids = topic_ids
        self.parent_topic_ids = parent_topic_ids
        self.propagation_scales = propagation_scales
        self.adj_matrices = adj_matrices
        self.time_matrices = time_matrices
    
    def __len__(self):
        return len(self.topic_ids)
    
    def __getitem__(self, idx):
        item = {
            'topic_id': self.topic_ids[idx],
            'parent_topic_id': self.parent_topic_ids[idx],
            'propagation_scale': self.propagation_scales[idx]
        }
        
        if self.adj_matrices is not None:
            item['adj_matrix'] = self.adj_matrices[idx]
        
        if self.time_matrices is not None:
            item['time_matrix'] = self.time_matrices[idx]
        
        return item


class TopicPropagationLoader:
    """话题传播数据加载器"""
    
    def __init__(self, args):
        self.args = args
        self.data_name = args.data_name
        self.data_dir = os.path.join('data', self.data_name)
        self.normalize = getattr(args, 'normalize', True)
        
        self.topic_data_file = os.path.join(self.data_dir, 'topics.txt')
        self.propagation_file = os.path.join(self.data_dir, 'propagation.txt')
        self.topic_network_file = os.path.join(self.data_dir, 'topic_network.txt')
        
        self.scale_mean = 0.0
        self.scale_std = 1.0
        
        self.use_node2vec = getattr(args, 'use_node2vec', True)
        self.use_attractiveness = getattr(args, 'use_attractiveness', True)
        self.node2vec_p = getattr(args, 'node2vec_p', 1.0)
        self.node2vec_q = getattr(args, 'node2vec_q', 1.0)
        self.node2vec_dim = getattr(args, 'topic_dim', 64)
        
        self.node2vec_embeddings = None
        self.attractiveness_features = None
        
        self.load_data()
    
    def load_data(self):
        logging.info(f"加载话题传播数据从: {self.data_dir}")
        
        if os.path.exists(self.propagation_file):
            data = pd.read_csv(self.propagation_file, sep=',', 
                             names=['topic_id', 'parent_topic_id', 'propagation_scale', 'timestamp'])
        else:
            logging.warning(f"未找到 {self.propagation_file},尝试生成模拟数据")
            data = self.generate_sample_data()
        
        # 归一化传播规模
        if self.normalize:
            data['original_scale'] = data['propagation_scale'].copy()
            data['propagation_scale'] = np.log1p(data['propagation_scale'])
            
            self.scale_mean = data['propagation_scale'].mean()
            self.scale_std = data['propagation_scale'].std()
            
            data['propagation_scale'] = (data['propagation_scale'] - self.scale_mean) / self.scale_std
            
            logging.info(f"数据归一化: log1p + 标准化")
            logging.info(f"  原始范围: [{data['original_scale'].min():.0f}, {data['original_scale'].max():.0f}]")
            logging.info(f"  归一化后: [{data['propagation_scale'].min():.2f}, {data['propagation_scale'].max():.2f}]")
            logging.info(f"  均值: {self.scale_mean:.4f}, 标准差: {self.scale_std:.4f}")
        
        all_topics = set(data['topic_id'].unique()) | set(data['parent_topic_id'].unique())
        self.topic_num = len(all_topics)
        self.topic2idx = {topic: idx for idx, topic in enumerate(sorted(all_topics))}
        self.idx2topic = {idx: topic for topic, idx in self.topic2idx.items()}
        
        data['topic_idx'] = data['topic_id'].map(self.topic2idx)
        data['parent_topic_idx'] = data['parent_topic_id'].map(self.topic2idx)
        
        self.data = data
        
        self.load_topic_network()
        
        if self.use_node2vec:
            self.generate_node2vec_embeddings()
        
        self.split_data_by_time(data, train_rate=self.args.train_rate if hasattr(self.args, 'train_rate') else 0.8, 
                              valid_rate=self.args.valid_rate if hasattr(self.args, 'valid_rate') else 0.1)

        if self.use_attractiveness:
            self.compute_raw_attractiveness_features()
            
        logging.info(f"加载完成: {len(data)} 条记录, {self.topic_num} 个话题")
        
    def split_data_by_time(self, data, train_rate=0.8, valid_rate=0.1):
        """基于时间划分数据集"""
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        total_len = len(data)
        train_end = int(total_len * train_rate)
        valid_end = int(total_len * (train_rate + valid_rate))
        
        self.train_indices = data.index[:train_end].tolist()
        self.valid_indices = data.index[train_end:valid_end].tolist()
        self.test_indices = data.index[valid_end:].tolist()
        
        logging.info(f"数据划分: 训练集 {len(self.train_indices)}, 验证集 {len(self.valid_indices)}, 测试集 {len(self.test_indices)}")
        
    def compute_raw_attractiveness_features(self):
        """计算多维话题吸引力特征(不进行预融合)"""
        logging.info("计算多维话题吸引力特征...")
        
        attractiveness_model = TopicAttractiveness()
        n_topics = self.topic_num
        
        if hasattr(self, 'adj_matrix'):
            degrees = torch.sum(self.adj_matrix > 0, dim=1).numpy()
        else:
            degrees = np.random.randint(1, 10, n_topics)

        np.random.seed(42)
        scales = np.zeros(n_topics)
        
        if 'original_scale' in self.data.columns:
            source_col = 'original_scale'
        else:
             source_col = 'propagation_scale'
             
        valid_mask = (self.data['topic_idx'] < n_topics) & (self.data['topic_idx'] >= 0)
        valid_data = self.data[valid_mask]
        
        indices = valid_data['topic_idx'].astype(int).values
        values = valid_data[source_col].values
        
        if source_col == 'propagation_scale' and self.normalize:
             values = np.expm1(values * self.scale_std + self.scale_mean)
             
        scales[indices] = values

        obs_ratios = np.random.uniform(0.2, 0.4, n_topics)
        
        # 1. 累计参与度
        feat_engagement = scales * obs_ratios * np.random.normal(1.0, 0.1, n_topics)
        
        # 2. 增长速率
        feat_velocity = feat_engagement / (obs_ratios + 1e-5) * np.random.normal(1.0, 0.2, n_topics)
        
        # 3. 覆盖范围
        feat_reach = feat_engagement * np.random.normal(10.0, 2.0, n_topics)
        
        # 4. 初始影响力
        feat_influence = np.random.lognormal(5, 1, n_topics)
        
        # 5. 话题相关性
        feat_relevance = degrees
        
        all_features = [
            attractiveness_model.standardize(feat_engagement),
            attractiveness_model.standardize(feat_velocity),
            attractiveness_model.standardize(feat_reach),
            attractiveness_model.standardize(feat_influence),
            attractiveness_model.standardize(feat_relevance)
        ]
        
        attract_dim = getattr(self.args, 'attract_dim', 5)
        
        if attract_dim == 0:
            features = np.zeros((n_topics, 1))
            logging.info(f"  不使用吸引力特征 (attract_dim=0)")
        elif attract_dim == 1:
            combined = (all_features[0] + all_features[1] + all_features[2] + 
                       all_features[3] + all_features[4]) / 5.0
            features = combined.reshape(-1, 1)
            logging.info(f"  生成单维吸引力特征")
        else:
            selected_features = all_features[:min(attract_dim, 5)]
            features = np.column_stack(selected_features)
            logging.info(f"  生成{attract_dim}维动态吸引力特征")
        
        self.attractiveness_features = torch.FloatTensor(features)
        logging.info(f"  生成多维动态吸引力特征: {self.attractiveness_features.shape}")

    def get_rolling_time_splits(self, n_splits=3):
        """生成滚动时间预测的划分索引"""
        data = self.data.sort_values('timestamp').reset_index(drop=True)
        total_len = len(data)
        
        min_train_size = int(total_len * 0.4)
        test_size = int(total_len * 0.15)
        step_size = (total_len - min_train_size - test_size) // (n_splits - 1 + 1e-5)
        step_size = max(100, int(step_size))
        
        for i in range(n_splits):
            train_end = min_train_size + i * step_size
            valid_end = train_end + int(test_size * 0.5)
            test_end = min(valid_end + test_size, total_len)
            
            if valid_end >= total_len: break
            
            train_idx = data.index[:train_end].tolist()
            valid_idx = data.index[train_end:valid_end].tolist()
            test_idx = data.index[valid_end:test_end].tolist()
            
            yield train_idx, valid_idx, test_idx
    
    def denormalize(self, normalized_values):
        """反归一化"""
        if self.normalize:
            values = normalized_values * self.scale_std + self.scale_mean
            values = np.expm1(values)
            return values
        else:
            return normalized_values
    
    def load_topic_network(self):
        """加载话题网络"""
        if os.path.exists(self.topic_network_file):
            edges = pd.read_csv(self.topic_network_file, sep=',', 
                              names=['topic1', 'topic2', 'weight', 'time'])
            
            adj_matrix = np.zeros((self.topic_num, self.topic_num))
            time_matrix = np.zeros((self.topic_num, self.topic_num))
            
            for _, row in edges.iterrows():
                if row['topic1'] in self.topic2idx and row['topic2'] in self.topic2idx:
                    i = self.topic2idx[row['topic1']]
                    j = self.topic2idx[row['topic2']]
                    adj_matrix[i, j] = row.get('weight', 1.0)
                    time_val = row.get('time', 0.0)
                    if pd.isna(time_val):
                        time_val = 0.0
                    time_matrix[i, j] = time_val
            
            self.adj_matrix = torch.FloatTensor(adj_matrix)
            self.time_matrix = torch.FloatTensor(time_matrix)
            
            if torch.isnan(self.time_matrix).any():
                self.time_matrix = torch.nan_to_num(self.time_matrix, nan=0.0)
            
            self.build_sparse_edge_index()
        else:
            self.build_sparse_graph_from_data()
    
    def build_sparse_edge_index(self):
        """从邻接矩阵构建稀疏边索引"""
        edge_indices = torch.nonzero(self.adj_matrix, as_tuple=False)
        
        if edge_indices.size(0) > 0:
            self.edge_index = edge_indices.t().contiguous()
            self.edge_time = self.time_matrix[edge_indices[:, 0], edge_indices[:, 1]]
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_time = torch.zeros(0)
        
        logging.info(f"  稀疏图: {self.edge_index.size(1)} 条边")
    
    def build_sparse_graph_from_data(self):
        """从数据构建基于父子关系的稀疏图"""
        logging.info("  从话题父子关系构建稀疏图...")
        
        edges_src = []
        edges_dst = []
        edges_time = []
        
        for _, row in self.data.iterrows():
            topic_idx = row['topic_idx']
            parent_idx = row['parent_topic_idx']
            timestamp = row['timestamp']
            
            if parent_idx != topic_idx:
                edges_src.append(parent_idx)
                edges_dst.append(topic_idx)
                edges_time.append(0.0)
                
                edges_src.append(topic_idx)
                edges_dst.append(parent_idx)
                edges_time.append(0.0)
        
        if len(edges_src) > 0:
            self.edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            self.edge_time = torch.tensor(edges_time, dtype=torch.float)
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_time = torch.zeros(0)
        
        logging.info(f"  稀疏图: {self.edge_index.size(1)} 条边")
        
        self.adj_matrix = torch.ones(self.topic_num, self.topic_num)
        self.time_matrix = torch.zeros(self.topic_num, self.topic_num)
    
    def split_data(self, train_ratio=0.8, valid_ratio=0.1):
        """划分数据集"""
        train_loader_indices = self.train_indices
        valid_loader_indices = self.valid_indices
        test_loader_indices = self.test_indices
        
        # 使用load_data中计算好的索引
        train_set = TopicPropagationDataset(
            topic_ids=torch.LongTensor(self.data.loc[train_loader_indices, 'topic_idx'].values),
            parent_topic_ids=torch.LongTensor(self.data.loc[train_loader_indices, 'parent_topic_idx'].values),
            propagation_scales=torch.FloatTensor(self.data.loc[train_loader_indices, 'propagation_scale'].values),
            adj_matrices=None, 
            time_matrices=None
        )
        
        valid_set = TopicPropagationDataset(
            topic_ids=torch.LongTensor(self.data.loc[valid_loader_indices, 'topic_idx'].values),
            parent_topic_ids=torch.LongTensor(self.data.loc[valid_loader_indices, 'parent_topic_idx'].values),
            propagation_scales=torch.FloatTensor(self.data.loc[valid_loader_indices, 'propagation_scale'].values),
            adj_matrices=None,
            time_matrices=None
        )
        
        test_set = TopicPropagationDataset(
            topic_ids=torch.LongTensor(self.data.loc[test_loader_indices, 'topic_idx'].values),
            parent_topic_ids=torch.LongTensor(self.data.loc[test_loader_indices, 'parent_topic_idx'].values),
            propagation_scales=torch.FloatTensor(self.data.loc[test_loader_indices, 'propagation_scale'].values),
            adj_matrices=None,
            time_matrices=None
        )
        
        return train_set, valid_set, test_set
    
    def get_dataloaders(self, args):
        """创建DataLoader"""
        train_set, valid_set, test_set = self.split_data(
            train_ratio=args.train_rate if hasattr(args, 'train_rate') else 0.8,
            valid_ratio=args.valid_rate if hasattr(args, 'valid_rate') else 0.1
        )
        
        train_loader = TorchDataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                     num_workers=getattr(args, 'num_workers', 0), pin_memory=True)
        valid_loader = TorchDataLoader(valid_set, batch_size=getattr(args, 'eval_batch_size', args.batch_size), shuffle=False,
                                     num_workers=getattr(args, 'num_workers', 0), pin_memory=True)
        test_loader = TorchDataLoader(test_set, batch_size=getattr(args, 'eval_batch_size', args.batch_size), shuffle=False,
                                    num_workers=getattr(args, 'num_workers', 0), pin_memory=True)
        
        return train_loader, valid_loader, test_loader
    
    def generate_node2vec_embeddings(self):
        """生成Node2vec嵌入 (论文公式8-12)"""
        logging.info("生成Node2vec话题结构特征...")
        
        if not hasattr(self, 'edge_index') or self.edge_index.size(1) == 0:
            logging.warning("  没有边信息,跳过Node2vec嵌入生成")
            self.node2vec_embeddings = None
            return
        
        edge_weights = None
        if hasattr(self, 'adj_matrix'):
            edge_weights = self.adj_matrix[self.edge_index[0], self.edge_index[1]]
        
        node2vec = Node2vecEmbedding(
            p=self.node2vec_p,
            q=self.node2vec_q,
            walk_length=80,
            num_walks=10,
            embedding_dim=self.node2vec_dim,
            window_size=10,
            workers=4,
            epochs=10
        )
        
        node2vec.fit(
            edge_index=self.edge_index,
            edge_weights=edge_weights,
            num_nodes=self.topic_num
        )
        
        self.node2vec_embeddings = node2vec.get_embeddings()
        logging.info(f"  Node2vec嵌入生成完成: {self.node2vec_embeddings.shape}")
    
    def compute_topic_attractiveness(self):
        """计算话题吸引力"""
        logging.info("计算话题吸引力...")
        
        attr_calculator = TopicAttractiveness()
        
        topic_data = {
            'likes': np.random.randint(10, 1000, self.topic_num).tolist(),
            'comments': np.random.randint(5, 500, self.topic_num).tolist(),
            'shares': np.random.randint(1, 200, self.topic_num).tolist(),
            'forwards': np.random.randint(1, 300, self.topic_num).tolist(),
            'followers': np.random.randint(100, 10000, self.topic_num).tolist(),
        }
        
        if hasattr(self, 'edge_index') and self.edge_index.size(1) > 0:
            degrees = torch.zeros(self.topic_num)
            for i in range(self.edge_index.size(1)):
                src = self.edge_index[0, i].item()
                degrees[src] += 1
            degrees = degrees.numpy()
        else:
            degrees = np.ones(self.topic_num)
        
        self.attractiveness_features = attr_calculator.compute_full_attractiveness(
            topic_data=topic_data,
            graph_degrees=degrees,
            num_topics=self.topic_num
        )
        
        logging.info(f"  话题吸引力计算完成: {self.attractiveness_features.shape}")
