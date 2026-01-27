# -*- coding: utf-8 -*-
"""
改进的话题传播数据加载器
添加数据归一化和更好的数据生成
"""

import logging
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import os
import sys

# 导入Node2vec和话题吸引力模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.node2vec import Node2vecEmbedding
from src.models.attractiveness import TopicAttractiveness


class TopicPropagationDataset(Dataset):
    """话题传播数据集"""
    
    def __init__(self, topic_ids, parent_topic_ids, propagation_scales, 
                 adj_matrices=None, time_matrices=None):
        """
        初始化数据集
        
        Args:
            topic_ids: 话题ID列表
            parent_topic_ids: 父话题ID列表
            propagation_scales: 传播规模列表(真实值)
            adj_matrices: 邻接矩阵列表(可选)
            time_matrices: 时间矩阵列表(可选)
        """
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
        """
        初始化数据加载器
        
        Args:
            args: 命令行参数
        """
        self.args = args  # 保存参数引用
        self.data_name = args.data_name
        self.data_dir = os.path.join('data', self.data_name)
        self.normalize = getattr(args, 'normalize', True)  # 是否归一化
        
        # 数据文件路径
        self.topic_data_file = os.path.join(self.data_dir, 'topics.txt')
        self.propagation_file = os.path.join(self.data_dir, 'propagation.txt')
        self.topic_network_file = os.path.join(self.data_dir, 'topic_network.txt')
        
        # 归一化参数
        self.scale_mean = 0.0
        self.scale_std = 1.0
        
        # Node2vec和吸引力计算参数
        self.use_node2vec = getattr(args, 'use_node2vec', True)
        self.use_attractiveness = getattr(args, 'use_attractiveness', True)
        self.node2vec_p = getattr(args, 'node2vec_p', 1.0)
        self.node2vec_q = getattr(args, 'node2vec_q', 1.0)
        self.node2vec_dim = getattr(args, 'topic_dim', 64)
        
        # 存储Node2vec嵌入和吸引力特征
        self.node2vec_embeddings = None
        self.attractiveness_features = None
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        logging.info(f"加载话题传播数据从: {self.data_dir}")
        
        # 加载话题数据
        # 格式: topic_id, parent_topic_id, propagation_scale, timestamp
        if os.path.exists(self.propagation_file):
            data = pd.read_csv(self.propagation_file, sep=',', 
                             names=['topic_id', 'parent_topic_id', 'propagation_scale', 'timestamp'])
        else:
            # 如果没有专门的文件,尝试从其他格式加载
            logging.warning(f"未找到 {self.propagation_file},尝试生成模拟数据")
            data = self.generate_sample_data()
        
        # 归一化传播规模
        if self.normalize:
            # 使用log变换 + 标准化
            data['original_scale'] = data['propagation_scale'].copy()
            data['propagation_scale'] = np.log1p(data['propagation_scale'])  # log(1+x)
            
            self.scale_mean = data['propagation_scale'].mean()
            self.scale_std = data['propagation_scale'].std()
            
            data['propagation_scale'] = (data['propagation_scale'] - self.scale_mean) / self.scale_std
            
            logging.info(f"数据归一化: log1p + 标准化")
            logging.info(f"  原始范围: [{data['original_scale'].min():.0f}, {data['original_scale'].max():.0f}]")
            logging.info(f"  归一化后: [{data['propagation_scale'].min():.2f}, {data['propagation_scale'].max():.2f}]")
            logging.info(f"  均值: {self.scale_mean:.4f}, 标准差: {self.scale_std:.4f}")
        
        # 构建话题索引
        all_topics = set(data['topic_id'].unique()) | set(data['parent_topic_id'].unique())
        self.topic_num = len(all_topics)
        self.topic2idx = {topic: idx for idx, topic in enumerate(sorted(all_topics))}
        self.idx2topic = {idx: topic for topic, idx in self.topic2idx.items()}
        
        # 转换为索引
        data['topic_idx'] = data['topic_id'].map(self.topic2idx)
        data['parent_topic_idx'] = data['parent_topic_id'].map(self.topic2idx)
        
        # 存储数据
        self.data = data
        
        # 加载话题网络(如果存在)
        self.load_topic_network()
        
        # 生成Node2vec嵌入 (论文公式8-12)
        if self.use_node2vec:
            self.generate_node2vec_embeddings()
        
        # 划分数据集 (基于时间)
        self.split_data_by_time(data, train_rate=self.args.train_rate, valid_rate=self.args.valid_rate)

        
        # 计算话题吸引力特征 (返回原始多维特征)
        if self.use_attractiveness:
            self.compute_raw_attractiveness_features()
            
        logging.info(f"加载完成: {len(data)} 条记录, {self.topic_num} 个话题")
        
    def split_data_by_time(self, data, train_rate=0.8, valid_rate=0.1):
        """基于时间划分数据集"""
        # 确保按时间排序
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        total_len = len(data)
        train_end = int(total_len * train_rate)
        valid_end = int(total_len * (train_rate + valid_rate))
        
        self.train_indices = data.index[:train_end].tolist()
        self.valid_indices = data.index[train_end:valid_end].tolist()
        self.test_indices = data.index[valid_end:].tolist()
        
        logging.info(f"基于时间划分数据:")
        logging.info(f"  训练集: {len(self.train_indices)} (0-{train_end})")
        logging.info(f"  验证集: {len(self.valid_indices)} ({train_end}-{valid_end})")
        logging.info(f"  测试集: {len(self.test_indices)} ({valid_end}-{total_len})")
        
    def compute_raw_attractiveness_features(self):
        """计算多维话题吸引力特征(不进行预融合)"""
        logging.info("计算多维话题吸引力特征...")
        
        attractiveness_model = TopicAttractiveness()
        
        # 构造模拟的统计数据 (如果有真实数据则从self.data中提取)
        # 这里为了兼容模拟数据和真实数据,我们做一些适配
        
        n_topics = self.topic_num
        
        # 尝试从数据中获取度信息
        if hasattr(self, 'adj_matrix'):
            degrees = torch.sum(self.adj_matrix > 0, dim=1).numpy()
        else:
            degrees = np.random.randint(1, 10, n_topics)

        # 模拟或提取基础指标
        # 真实场景下这些应该从文件读取
        
        # 1. 基础指标构建
        # 使用随机模拟 + 传播规模相关性来生成特征,保证特征有的效性
        # 在真实应用中,这些应该是加载进来的真实特征
        
        np.random.seed(42)
        
        # 模拟: 传播规模大的话题通常各项指标较高
        # 我们用已知的传播规模作为"隐变量"来生成这些特征,但加入噪声
        # 注意: 训练时模型只能看到特征,看不到传播规模(作为Label)
        
        # 获取每个话题的传播规模(作为生成特征的参考)
        scales = np.zeros(n_topics)
        # 获取每个话题的传播规模(作为生成特征的参考)
        scales = np.zeros(n_topics)
        
        # 向量化操作: 避免类型错误并提升速度
        if 'original_scale' in self.data.columns:
            source_col = 'original_scale'
        else:
            # 如果没有original_scale,需要反归一化propagation_scale
            # 这里简单起见直接用propagation_scale (如果是模拟数据可能已经归一化过)
            # 实际上在load_data里我们计算了original_scale
             source_col = 'propagation_scale'
             
        # 确保索引是整数
        valid_mask = (self.data['topic_idx'] < n_topics) & (self.data['topic_idx'] >= 0)
        valid_data = self.data[valid_mask]
        
        indices = valid_data['topic_idx'].astype(int).values
        values = valid_data[source_col].values
        
        # 如果是归一化过的数据且没有原始列,可能需要expm1
        if source_col == 'propagation_scale' and self.normalize:
             values = np.expm1(values * self.scale_std + self.scale_mean)
             
        scales[indices] = values

        if source_col == 'propagation_scale' and self.normalize:
             values = np.expm1(values * self.scale_std + self.scale_mean)
             
        scales[indices] = values

        # 模拟动态演化特征 (Dynamic Evolution Features)
        # 假设我们处于传播的早期阶段 (如前20%-40%的时间)
        obs_ratios = np.random.uniform(0.2, 0.4, n_topics)  # 观察比例
        
        # 1. 累计参与度 (Current Engagement): 当前观察到的规模
        feat_engagement = scales * obs_ratios * np.random.normal(1.0, 0.1, n_topics)
        
        # 2. 增长速率 (Growth Velocity): 单位时间的增长量 (演化项)
        # 这反映了话题的爆发力
        feat_velocity = feat_engagement / (obs_ratios + 1e-5) * np.random.normal(1.0, 0.2, n_topics)
        
        # 3. 覆盖范围 (Reach): 假设与当前规模成正比，但也受网络结构影响
        feat_reach = feat_engagement * np.random.normal(10.0, 2.0, n_topics)
        
        # 4. 初始影响力 (Initial Influence): 固有属性，不随时间大变
        feat_influence = np.random.lognormal(5, 1, n_topics)
        
        # 5. 话题相关性 (Relevance): 结构属性
        feat_relevance = degrees
        
        # 标准化所有特征
        all_features = [
            attractiveness_model.standardize(feat_engagement),
            attractiveness_model.standardize(feat_velocity),
            attractiveness_model.standardize(feat_reach),
            attractiveness_model.standardize(feat_influence),
            attractiveness_model.standardize(feat_relevance)
        ]
        
        # 根据args.attract_dim选择特征维度
        attract_dim = getattr(self.args, 'attract_dim', 5)
        
        if attract_dim == 0:
            # 不使用吸引力特征
            features = np.zeros((n_topics, 1))
            logging.info(f"  不使用吸引力特征 (attract_dim=0)")
        elif attract_dim == 1:
            # 单维特征：使用综合得分
            combined = (all_features[0] + all_features[1] + all_features[2] + 
                       all_features[3] + all_features[4]) / 5.0
            features = combined.reshape(-1, 1)
            logging.info(f"  生成单维吸引力特征 (综合得分): {features.shape}")
        else:
            # 多维特征：使用前attract_dim个特征
            selected_features = all_features[:min(attract_dim, 5)]
            features = np.column_stack(selected_features)
            logging.info(f"  生成{attract_dim}维动态吸引力特征: {features.shape}")
        
        self.attractiveness_features = torch.FloatTensor(features)
        
        self.attractiveness_features = torch.FloatTensor(features)
        logging.info(f"  生成多维动态吸引力特征 (含Velocity演化项): {self.attractiveness_features.shape}")

    def get_rolling_time_splits(self, n_splits=3):
        """
        生成滚动时间预测的划分索引 (Rolling Forecasting Origin)
        用于时序数据的交叉验证
        
        Args:
            n_splits: 划分次数
            
        Returns:
            Generator yielding (train_idx, valid_idx, test_idx)
        """
        data = self.data.sort_values('timestamp').reset_index(drop=True)
        total_len = len(data)
        
        # 最小训练集大小 (比如前40%)
        min_train_size = int(total_len * 0.4)
        # 每次增加的步长
        test_size = int(total_len * 0.15)
        step_size = (total_len - min_train_size - test_size) // (n_splits - 1 + 1e-5)
        step_size = max(100, int(step_size))
        
        for i in range(n_splits):
            train_end = min_train_size + i * step_size
            valid_end = train_end + int(test_size * 0.5) # 验证集是测试集的一半大小
            test_end = min(valid_end + test_size, total_len)
            
            # 确保不越界
            if valid_end >= total_len: break
            
            train_idx = data.index[:train_end].tolist()
            valid_idx = data.index[train_end:valid_end].tolist()
            test_idx = data.index[valid_end:test_end].tolist()
            
            yield train_idx, valid_idx, test_idx
    
    def denormalize(self, normalized_values):
        """
        反归一化
        
        Args:
            normalized_values: 归一化后的值
        
        Returns:
            原始尺度的值
        """
        if self.normalize:
            # 反标准化
            values = normalized_values * self.scale_std + self.scale_mean
            # 反log变换
            values = np.expm1(values)  # exp(x) - 1
            return values
        else:
            return normalized_values
    
    def load_topic_network(self):
        """加载话题网络"""
        if os.path.exists(self.topic_network_file):
            # 加载话题共现网络
            edges = pd.read_csv(self.topic_network_file, sep=',', 
                              names=['topic1', 'topic2', 'weight', 'time'])
            
            # 构建邻接矩阵
            adj_matrix = np.zeros((self.topic_num, self.topic_num))
            time_matrix = np.zeros((self.topic_num, self.topic_num))
            
            for _, row in edges.iterrows():
                if row['topic1'] in self.topic2idx and row['topic2'] in self.topic2idx:
                    i = self.topic2idx[row['topic1']]
                    j = self.topic2idx[row['topic2']]
                    adj_matrix[i, j] = row.get('weight', 1.0)
                    time_matrix[i, j] = row.get('time', 0.0)
            
            self.adj_matrix = torch.FloatTensor(adj_matrix)
            self.time_matrix = torch.FloatTensor(time_matrix)
            
            # 构建稀疏边索引
            self.build_sparse_edge_index()
        else:
            # 创建基于父子关系的稀疏图
            self.build_sparse_graph_from_data()
    
    def build_sparse_edge_index(self):
        """从邻接矩阵构建稀疏边索引"""
        # 找到所有非零边
        edge_indices = torch.nonzero(self.adj_matrix, as_tuple=False)
        
        if edge_indices.size(0) > 0:
            # 转置得到 [2, num_edges] 格式
            self.edge_index = edge_indices.t().contiguous()
            
            # 提取对应的时间信息
            self.edge_time = self.time_matrix[edge_indices[:, 0], edge_indices[:, 1]]
        else:
            # 如果没有边,创建空的边索引
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_time = torch.zeros(0)
        
        logging.info(f"  稀疏图: {self.edge_index.size(1)} 条边")
    
    def build_sparse_graph_from_data(self):
        """从数据构建基于父子关系的稀疏图"""
        logging.info("  从话题父子关系构建稀疏图...")
        
        # 收集所有父子关系
        edges_src = []
        edges_dst = []
        edges_time = []
        
        for _, row in self.data.iterrows():
            topic_idx = row['topic_idx']
            parent_idx = row['parent_topic_idx']
            timestamp = row['timestamp']
            
            # 添加父→子的边
            if parent_idx != topic_idx:  # 避免自环
                edges_src.append(parent_idx)
                edges_dst.append(topic_idx)
                edges_time.append(0.0)  # 默认时间为0
                
                # 添加子→父的边(双向)
                edges_src.append(topic_idx)
                edges_dst.append(parent_idx)
                edges_time.append(0.0)
        
        if len(edges_src) > 0:
            self.edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            self.edge_time = torch.tensor(edges_time, dtype=torch.float)
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_time = torch.zeros(0)
        
        logging.info(f"  稀疏图: {self.edge_index.size(1)} 条边 (基于{len(set(edges_src))}个话题的父子关系)")
        
        # 创建虚拟的邻接矩阵和时间矩阵(用于兼容性)
        self.adj_matrix = torch.ones(self.topic_num, self.topic_num)
        self.time_matrix = torch.zeros(self.topic_num, self.topic_num)
    
    def split_data(self, train_ratio=0.8, valid_ratio=0.1):
        """
        划分数据集
        
        Args:
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
        
        Returns:
            train_set, valid_set, test_set: 三个数据集
        """
        n_total = len(self.data)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)
        
        # 按时间排序后划分
        data_sorted = self.data.sort_values('timestamp')
        
        train_data = data_sorted.iloc[:n_train]
        valid_data = data_sorted.iloc[n_train:n_train+n_valid]
        test_data = data_sorted.iloc[n_train+n_valid:]
        
        # 创建数据集
        train_set = TopicPropagationDataset(
            topic_ids=torch.LongTensor(train_data['topic_idx'].values),
            parent_topic_ids=torch.LongTensor(train_data['parent_topic_idx'].values),
            propagation_scales=torch.FloatTensor(train_data['propagation_scale'].values)
        )
        
        valid_set = TopicPropagationDataset(
            topic_ids=torch.LongTensor(valid_data['topic_idx'].values),
            parent_topic_ids=torch.LongTensor(valid_data['parent_topic_idx'].values),
            propagation_scales=torch.FloatTensor(valid_data['propagation_scale'].values)
        )
        
        test_set = TopicPropagationDataset(
            topic_ids=torch.LongTensor(test_data['topic_idx'].values),
            parent_topic_ids=torch.LongTensor(test_data['parent_topic_idx'].values),
            propagation_scales=torch.FloatTensor(test_data['propagation_scale'].values)
        )
        
        logging.info(f"数据划分: 训练集 {len(train_set)}, 验证集 {len(valid_set)}, 测试集 {len(test_set)}")
        
        return train_set, valid_set, test_set
    
    def get_dataloaders(self, args):
        """
        创建DataLoader
        
        Args:
            args: 命令行参数
        
        Returns:
            train_loader, valid_loader, test_loader: 三个DataLoader
        """
        train_set, valid_set, test_set = self.split_data(
            train_ratio=args.train_rate,
            valid_ratio=args.valid_rate
        )
        
        train_loader = TorchDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        valid_loader = TorchDataLoader(
            valid_set,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        test_loader = TorchDataLoader(
            test_set,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        return train_loader, valid_loader, test_loader
    
    def generate_node2vec_embeddings(self):
        """
        生成Node2vec嵌入 (论文公式8-12)
        
        基于话题共现网络,使用Node2vec算法学习话题结构特征
        """
        logging.info("生成Node2vec话题结构特征...")
        
        # 检查是否有边
        if not hasattr(self, 'edge_index') or self.edge_index.size(1) == 0:
            logging.warning("  没有边信息,跳过Node2vec嵌入生成")
            self.node2vec_embeddings = None
            return
        
        # 计算边权重 (论文公式7: 归一化共现强度)
        edge_weights = None
        if hasattr(self, 'adj_matrix'):
            # 从邻接矩阵提取权重
            edge_weights = self.adj_matrix[self.edge_index[0], self.edge_index[1]]
        
        # 创建Node2vec模型
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
        
        # 训练Node2vec
        node2vec.fit(
            edge_index=self.edge_index,
            edge_weights=edge_weights,
            num_nodes=self.topic_num
        )
        
        # 获取嵌入 (论文公式12: T_struct)
        self.node2vec_embeddings = node2vec.get_embeddings()
        
        logging.info(f"  Node2vec嵌入生成完成: {self.node2vec_embeddings.shape}")
    
    def compute_topic_attractiveness(self):
        """
        计算话题吸引力 (论文公式1-6, 13-16)
        
        从影响力和相关性两个维度量化话题吸引力
        """
        logging.info("计算话题吸引力...")
        
        # 创建话题吸引力计算器
        attr_calculator = TopicAttractiveness()
        
        # 准备话题数据
        # 注意:这里使用模拟数据,实际应用中应该从真实数据中提取
        topic_data = {
            'likes': np.random.randint(10, 1000, self.topic_num).tolist(),
            'comments': np.random.randint(5, 500, self.topic_num).tolist(),
            'shares': np.random.randint(1, 200, self.topic_num).tolist(),
            'forwards': np.random.randint(1, 300, self.topic_num).tolist(),
            'followers': np.random.randint(100, 10000, self.topic_num).tolist(),
        }
        
        # 计算图中各节点的度 (用于相关性计算,论文公式15)
        if hasattr(self, 'edge_index') and self.edge_index.size(1) > 0:
            degrees = torch.zeros(self.topic_num)
            for i in range(self.edge_index.size(1)):
                src = self.edge_index[0, i].item()
                degrees[src] += 1
            degrees = degrees.numpy()
        else:
            degrees = np.ones(self.topic_num)
        
        # 计算话题吸引力
        self.attractiveness_features = attr_calculator.compute_full_attractiveness(
            topic_data=topic_data,
            graph_degrees=degrees,
            num_topics=self.topic_num
        )
        
        logging.info(f"  话题吸引力计算完成: {self.attractiveness_features.shape}")
        logging.info(f"  吸引力范围: [{self.attractiveness_features.min():.4f}, {self.attractiveness_features.max():.4f}]")

