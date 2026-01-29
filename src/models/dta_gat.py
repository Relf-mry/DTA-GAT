# -*- coding: utf-8 -*-
"""
DTA-GAT: 动态时间感知图注意力网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math


class TopicPropagationModel(nn.Module):
    """话题传播预测模型"""
    
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--topic_dim', type=int, default=64)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--n_gat_layers', type=int, default=2)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--time_decay', type=float, default=0.1)
        return parser
    
    def __init__(self, args, data_loader):
        super(TopicPropagationModel, self).__init__()
        
        self.topic_num = data_loader.topic_num
        self.topic_dim = args.topic_dim
        self.hidden_dim = args.hidden_dim
        self.n_heads = args.num_heads
        self.alpha = args.alpha
        self.time_decay = args.time_decay
        self.dropout = args.dropout
        
        # 1. 话题结构特征
        if hasattr(data_loader, 'node2vec_embeddings') and data_loader.node2vec_embeddings is not None:
            self.use_pretrained_struct = True
            self.register_buffer('structural_features', data_loader.node2vec_embeddings)
            struct_dim = data_loader.node2vec_embeddings.size(1)
        else:
            self.use_pretrained_struct = False
            self.topic_embedding = nn.Embedding(self.topic_num, self.topic_dim)
            struct_dim = self.topic_dim
        
        # 2. 话题吸引力特征
        if hasattr(data_loader, 'attractiveness_features') and data_loader.attractiveness_features is not None:
            self.use_pretrained_attract = True
            self.register_buffer('attractiveness_features', data_loader.attractiveness_features)
            if data_loader.attractiveness_features.dim() == 1:
                attract_dim = 1
            else:
                attract_dim = data_loader.attractiveness_features.shape[1]
        else:
            self.use_pretrained_attract = False
            attract_dim = max(1, getattr(args, 'attract_dim', 1))
        
        attract_dim = max(1, attract_dim)
        
        # 3. 特征融合层
        self.struct_transform = nn.Linear(struct_dim, self.hidden_dim)
        self.attract_transform = nn.Linear(attract_dim, self.hidden_dim)
        
        # 4. 动态时间感知图注意力网络 (DTA-GAT)
        self.gat_layers = nn.ModuleList([
            TimeAwareGATLayer(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
                n_heads=self.n_heads,
                alpha=self.alpha,
                dropout=self.dropout,
                time_decay=self.time_decay
            )
            for i in range(args.n_gat_layers)
        ])
        
        # 5. 传播规模预测层
        self.prediction_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.prediction_fc2 = nn.Linear(self.hidden_dim, 1)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.init_weights()
    
    def init_weights(self):
        if not self.use_pretrained_struct:
            init.xavier_normal_(self.topic_embedding.weight)
        init.xavier_normal_(self.struct_transform.weight)
        init.xavier_normal_(self.attract_transform.weight)
        init.xavier_normal_(self.prediction_fc1.weight)
        init.xavier_normal_(self.prediction_fc2.weight)
    
    def forward(self, topic_ids, adj_matrix, time_matrix, attractiveness=None):
        batch_size = topic_ids.size(0)
        device = topic_ids.device
        
        # 1. 获取话题结构特征 (T_struct)
        if self.use_pretrained_struct:
            all_struct_features = self.structural_features
        else:
            all_struct_features = self.topic_embedding.weight
        
        all_struct_features = self.struct_transform(all_struct_features)
        
        # 2. 获取话题吸引力特征 (T_attraction)
        if attractiveness is not None:
            if attractiveness.dim() == 1:
                attract_features = attractiveness.unsqueeze(-1)
            else:
                attract_features = attractiveness
        elif self.use_pretrained_attract:
            attract_features = self.attractiveness_features[topic_ids]
            if attract_features.dim() == 1:
                attract_features = attract_features.unsqueeze(-1)
        else:
            attract_features = torch.ones(batch_size, 1, device=device)
        
        attract_features = self.attract_transform(attract_features)
        
        # 3. 通过DTA-GAT层 (batch子图优化)
        batch_nodes = topic_ids.cpu().numpy()
        
        adj_np = adj_matrix.cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix
        neighbor_nodes = set(batch_nodes)
        for node in batch_nodes:
            neighbors = np.where(adj_np[node] > 0)[0]
            neighbor_nodes.update(neighbors)
        
        subgraph_nodes = sorted(list(neighbor_nodes))
        node_to_subidx = {node: idx for idx, node in enumerate(subgraph_nodes)}
        
        subgraph_features = all_struct_features[subgraph_nodes]
        subgraph_adj = adj_matrix[subgraph_nodes][:, subgraph_nodes]
        subgraph_time = time_matrix[subgraph_nodes][:, subgraph_nodes]
        
        subgraph_features = subgraph_features.unsqueeze(0)
        subgraph_adj = subgraph_adj.unsqueeze(0)
        subgraph_time = subgraph_time.unsqueeze(0)
        
        for gat_layer in self.gat_layers:
            subgraph_features = gat_layer(subgraph_features, subgraph_adj, subgraph_time)
            subgraph_features = self.dropout_layer(subgraph_features)
        
        subgraph_features = subgraph_features.squeeze(0)
        
        batch_indices = [node_to_subidx[node.item()] for node in topic_ids.cpu()]
        topic_features = subgraph_features[batch_indices]
        
        # 4. 融合特征
        combined_features = topic_features + attract_features
        
        # 5. 规模预测
        hidden = F.relu(self.prediction_fc1(combined_features))
        hidden = self.dropout_layer(hidden)
        prediction = self.prediction_fc2(hidden)
        
        return prediction
    
    def get_performance(self, topic_ids, adj_matrix, time_matrix, gold, attractiveness=None):
        prediction = self.forward(topic_ids, adj_matrix, time_matrix, attractiveness)
        
        mae = F.l1_loss(prediction.squeeze(), gold)
        
        mse = F.mse_loss(prediction.squeeze(), gold)
        rmse = torch.sqrt(mse)
        
        return mae, rmse


class TimeAwareGATLayer(nn.Module):
    """时间感知图注意力层"""
    
    def __init__(self, in_features, out_features, n_heads, alpha=0.2, dropout=0.3, time_decay=0.1):
        super(TimeAwareGATLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.alpha = alpha
        self.time_decay = time_decay
        
        self.head_dim = out_features // n_heads
        assert self.head_dim * n_heads == out_features, "out_features必须能被n_heads整除"
        
        self.W = nn.Parameter(torch.zeros(size=(n_heads, in_features, self.head_dim)))
        self.a = nn.Parameter(torch.zeros(size=(n_heads, 2 * self.head_dim, 1)))
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # 优化：LayerNorm提升稳定性
        self.layer_norm = nn.LayerNorm(out_features)
        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def compute_weighted_cooccurrence_time(self, time_matrix, current_time=None):
        """计算加权平均共现时间"""
        if current_time is None:
            current_time = time_matrix.max()
        
        time_diff = current_time - time_matrix
        time_weights = torch.exp(-self.time_decay * time_diff)
        
        return time_weights
    
    def forward(self, node_features, adj_matrix, time_matrix):
        batch_size, num_nodes, _ = node_features.size()
        
        # 1. 计算时间权重
        time_weights = self.compute_weighted_cooccurrence_time(time_matrix)
        
        # 2. 线性变换
        h = torch.einsum('bni,hio->bhno', node_features, self.W)
        
        # 3. 计算注意力分数
        a_src = self.a[:, :self.head_dim, :]
        a_dst = self.a[:, self.head_dim:, :]
        
        e_src = torch.einsum('bhni,hio->bhno', h, a_src).squeeze(-1)
        e_dst = torch.einsum('bhni,hio->bhno', h, a_dst).squeeze(-1)
        
        e = e_src.unsqueeze(3) + e_dst.unsqueeze(2)
        e = self.leakyrelu(e)
        
        time_weights_expanded = time_weights.unsqueeze(1)
        e = e * time_weights_expanded
        
        # 4. 注意力归一化
        mask_raw = adj_matrix
        
        # 优化：DropEdge
        if self.training:
            drop_prob = 0.1
            drop_mask = torch.rand_like(mask_raw) > drop_prob
            mask_raw = mask_raw * drop_mask.float()
            
        mask = mask_raw.unsqueeze(1)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        has_neighbors = (mask.sum(dim=-1) > 0)
        e = torch.where(
            has_neighbors.unsqueeze(-1),
            e,
            torch.zeros_like(e)
        )
        
        attention = F.softmax(e, dim=-1)
        
        attention = torch.where(
            torch.isnan(attention),
            torch.zeros_like(attention),
            attention
        )
        
        attention = self.dropout(attention)
        
        # 5. 聚合
        out = torch.einsum('bhij,bhjo->bhio', attention, h)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        out = F.elu(out)
        
        # 优化：LayerNorm
        out = self.layer_norm(out)
        
        return out
