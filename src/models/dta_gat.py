# -*- coding: utf-8 -*-
"""
DTA-GAT: 动态时间感知图注意力网络
基于话题吸引力和动态时间感知的衍生话题传播预测模型

论文引用:
- 公式7: 归一化共现强度 W_VX = N(v,x) / max(N)
- 公式17: 加权平均共现时间 W_vx = Σ exp(-λ(h_current - h)) / |H_vx|
- 公式18: 时间敏感注意力 Attention(v,x) = a^T[MT_v || MT_x] × W_vx
- 公式19: 特征融合 T_m = [T_struct || T_attraction]
- 公式20: 注意力归一化 α_vx = softmax(LeakyReLU(Attention(v,x)))
- 公式21: 邻居聚合 T_m^new = σ(Σ α_mj M T_m)
- 公式22: 传播预测 B = W^(L) H' + b^(L)

任务: 预测话题的传播规模(数值预测)
评估指标: MAE (Mean Absolute Error) 和 RMSE (Root Mean Square Error)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math


class TopicPropagationModel(nn.Module):
    """
    话题传播预测模型
    
    核心组件:
    1. 话题吸引力表示 (Topic Attractiveness Representation)
       - 话题影响力 (Topic Influence)
       - 话题相关性 (Topic Relevance)
    
    2. 动态时间感知图注意力网络 (Dynamic Time-Aware Graph Attention Network, DTA-GAT)
       - 加权共现时间 (Weighted Co-occurrence Time)
       - 图注意力机制 (Graph Attention Mechanism)
    
    3. 传播规模预测 (Propagation Scale Prediction)
       - 融合结构特征和吸引力特征
       - 回归预测传播规模
    """
    
    @staticmethod
    def parse_model_args(parser):
        """
        添加模型特定参数
        
        Args:
            parser: argparse.ArgumentParser对象
        
        Returns:
            parser: 添加参数后的parser
        """
        parser.add_argument('--topic_dim', type=int, default=64,
                          help='话题嵌入维度')
        parser.add_argument('--hidden_dim', type=int, default=128,
                          help='隐藏层维度')
        parser.add_argument('--n_gat_layers', type=int, default=2,
                          help='GAT层数')
        parser.add_argument('--n_heads', type=int, default=4,
                          help='注意力头数')
        parser.add_argument('--alpha', type=float, default=0.2,
                          help='LeakyReLU的负斜率')
        parser.add_argument('--time_decay', type=float, default=0.1,
                          help='时间衰减系数')
        return parser
    
    def __init__(self, args, data_loader):
        """
        初始化DTA-GAT模型
        
        Args:
            args: 命令行参数
            data_loader: 数据加载器
        """
        super(TopicPropagationModel, self).__init__()
        
        self.topic_num = data_loader.topic_num  # 话题数量 |V|
        self.topic_dim = args.topic_dim  # 话题嵌入维度
        self.hidden_dim = args.hidden_dim  # 隐藏层维度
        self.n_heads = args.n_heads  # 注意力头数
        self.alpha = args.alpha  # LeakyReLU负斜率
        self.time_decay = args.time_decay  # 时间衰减系数λ (论文公式17)
        self.dropout = args.dropout
        
        # ========== 话题结构特征 (论文公式12) ==========
        # 如果data_loader提供了Node2vec嵌入,使用它;否则使用可学习嵌入
        if hasattr(data_loader, 'node2vec_embeddings') and data_loader.node2vec_embeddings is not None:
            # 使用预训练的Node2vec嵌入 (T_struct)
            self.use_pretrained_struct = True
            self.register_buffer('structural_features', data_loader.node2vec_embeddings)
            struct_dim = data_loader.node2vec_embeddings.size(1)
        else:
            # 使用可学习的嵌入
            self.use_pretrained_struct = False
            self.topic_embedding = nn.Embedding(self.topic_num, self.topic_dim)
            struct_dim = self.topic_dim
        
        # ========== 话题吸引力特征 (论文公式16) ==========
        # 如果data_loader提供了吸引力特征,使用它;否则计算
        if hasattr(data_loader, 'attractiveness_features') and data_loader.attractiveness_features is not None:
            self.use_pretrained_attract = True
            self.register_buffer('attractiveness_features', data_loader.attractiveness_features)
            attract_dim = 1  # 吸引力是标量
        else:
            self.use_pretrained_attract = False
            attract_dim = 1
        
        # ========== 特征融合层 (论文公式19) ==========
        # T_m = [T_struct || T_attraction]
        # 将结构特征和吸引力特征映射到统一维度
        self.struct_transform = nn.Linear(struct_dim, self.hidden_dim)
        self.attract_transform = nn.Linear(attract_dim, self.hidden_dim)
        
        # ========== 动态时间感知图注意力网络 (DTA-GAT) ==========
        # 多层GAT (论文公式18-21)
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
        
        # ========== 传播规模预测层 (论文公式22) ==========
        # B = W^(L) H' + b^(L)
        self.prediction_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.prediction_fc2 = nn.Linear(self.hidden_dim, 1)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        if not self.use_pretrained_struct:
            init.xavier_normal_(self.topic_embedding.weight)
        init.xavier_normal_(self.struct_transform.weight)
        init.xavier_normal_(self.attract_transform.weight)
        init.xavier_normal_(self.prediction_fc1.weight)
        init.xavier_normal_(self.prediction_fc2.weight)
    
    def forward(self, topic_ids, adj_matrix, time_matrix, attractiveness=None):
        """
        前向传播
        
        实现论文中的完整流程:
        1. 获取话题结构特征 T_struct (公式12)
        2. 获取话题吸引力特征 T_attraction (公式16)
        3. 特征融合 T_m = [T_struct || T_attraction] (公式19)
        4. 通过DTA-GAT层处理 (公式18-21)
        5. 预测传播规模 B (公式22)
        
        Args:
            topic_ids: 话题ID [batch_size]
            adj_matrix: 邻接矩阵 [batch_size, num_topics, num_topics] 或 [num_topics, num_topics]
            time_matrix: 时间矩阵 [batch_size, num_topics, num_topics] 或 [num_topics, num_topics]
            attractiveness: 话题吸引力 [batch_size] (可选)
        
        Returns:
            prediction: 传播规模预测 [batch_size, 1]
        """
        batch_size = topic_ids.size(0)
        device = topic_ids.device
        
        # ========== 1. 获取话题结构特征 (论文公式12: T_struct) ==========
        if self.use_pretrained_struct:
            # 使用预训练的Node2vec嵌入
            all_struct_features = self.structural_features  # [num_topics, struct_dim]
        else:
            # 使用可学习的嵌入
            all_struct_features = self.topic_embedding.weight  # [num_topics, topic_dim]
        
        # 映射到hidden_dim
        all_struct_features = self.struct_transform(all_struct_features)  # [num_topics, hidden_dim]
        
        # ========== 2. 获取话题吸引力特征 (论文公式16: T_attraction) ==========
        if attractiveness is not None:
            # 使用提供的吸引力特征
            attract_features = attractiveness.unsqueeze(-1)  # [batch_size, 1]
        elif self.use_pretrained_attract:
            # 使用预计算的吸引力特征
            attract_features = self.attractiveness_features[topic_ids].unsqueeze(-1)  # [batch_size, 1]
        else:
            # 使用默认值
            attract_features = torch.ones(batch_size, 1, device=device)
        
        # 映射到hidden_dim
        attract_features = self.attract_transform(attract_features)  # [batch_size, hidden_dim]
        
        # ========== 3. 特征融合 (论文公式19: T_m = [T_struct || T_attraction]) ==========
        # 扩展到batch维度
        all_struct_batch = all_struct_features.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_topics, hidden_dim]
        
        # 为每个话题添加吸引力特征
        # 注意:这里我们为所有话题使用相同的吸引力特征(简化版)
        # 实际应用中,每个话题应该有自己的吸引力
        node_features = all_struct_batch  # [batch, num_topics, hidden_dim]
        
        # 确保adj_matrix和time_matrix的维度正确
        if adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        if time_matrix.dim() == 2:
            time_matrix = time_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # ========== 4. 通过DTA-GAT层 (论文公式18-21) ==========
        for gat_layer in self.gat_layers:
            node_features = gat_layer(node_features, adj_matrix, time_matrix)
            node_features = self.dropout_layer(node_features)
        
        # 提取当前话题的特征
        topic_indices = topic_ids.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
        topic_features = torch.gather(node_features, 1, topic_indices).squeeze(1)  # [batch, hidden_dim]
        
        # 融合结构特征和吸引力特征
        combined_features = topic_features + attract_features  # [batch, hidden_dim]
        
        # ========== 5. 传播规模预测 (论文公式22: B = W^(L) H' + b^(L)) ==========
        hidden = F.relu(self.prediction_fc1(combined_features))
        hidden = self.dropout_layer(hidden)
        prediction = self.prediction_fc2(hidden)  # [batch_size, 1]
        
        return prediction
    
    def get_performance(self, topic_ids, adj_matrix, time_matrix, gold, attractiveness=None):
        """
        计算模型性能
        
        Args:
            topic_ids: 话题ID
            adj_matrix: 邻接矩阵
            time_matrix: 时间矩阵
            gold: 真实传播规模
            attractiveness: 话题吸引力 (可选)
        
        Returns:
            mae: 平均绝对误差 (论文公式23)
            rmse: 均方根误差 (论文公式24)
        """
        # 前向传播
        prediction = self.forward(topic_ids, adj_matrix, time_matrix, attractiveness)
        
        # 计算MAE (论文公式23)
        mae = F.l1_loss(prediction.squeeze(), gold)
        
        # 计算RMSE (论文公式24)
        mse = F.mse_loss(prediction.squeeze(), gold)
        rmse = torch.sqrt(mse)
        
        return mae, rmse



class TimeAwareGATLayer(nn.Module):
    """
    时间感知图注意力层 (Time-Aware Graph Attention Layer)
    
    实现论文公式17-21:
    - 公式17: 加权平均共现时间 W_vx
    - 公式18: 时间敏感注意力 Attention(v,x)
    - 公式20: 注意力归一化 α_vx
    - 公式21: 邻居聚合 T_m^new
    """
    
    def __init__(self, in_features, out_features, n_heads, alpha=0.2, dropout=0.3, time_decay=0.1):
        """
        初始化时间感知GAT层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            n_heads: 注意力头数
            alpha: LeakyReLU负斜率
            dropout: Dropout率
            time_decay: 时间衰减系数λ (论文公式17)
        """
        super(TimeAwareGATLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.alpha = alpha
        self.time_decay = time_decay
        
        # 每个头的输出维度
        self.head_dim = out_features // n_heads
        assert self.head_dim * n_heads == out_features, "out_features必须能被n_heads整除"
        
        # 多头注意力的权重矩阵 M (论文公式18中的M)
        self.W = nn.Parameter(torch.zeros(size=(n_heads, in_features, self.head_dim)))
        
        # 注意力系数计算的权重 a (论文公式18中的a)
        self.a = nn.Parameter(torch.zeros(size=(n_heads, 2 * self.head_dim, 1)))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # 初始化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def compute_weighted_cooccurrence_time(self, time_matrix, current_time=None):
        """
        计算加权平均共现时间 (论文公式17)
        
        W_vx = Σ exp(-λ(h_current - h)) / |H_vx|
        
        Args:
            time_matrix: 时间矩阵 [batch, num_nodes, num_nodes]
                        每个元素表示两个节点最近共现的时间
            current_time: 当前时间 (如果为None,使用time_matrix的最大值)
        
        Returns:
            time_weights: 时间权重 [batch, num_nodes, num_nodes]
        """
        if current_time is None:
            # 使用time_matrix中的最大值作为当前时间
            current_time = time_matrix.max()
        
        # 计算时间差
        time_diff = current_time - time_matrix  # h_current - h
        
        # 应用指数衰减 (论文公式17)
        # W_vx = exp(-λ * time_diff)
        time_weights = torch.exp(-self.time_decay * time_diff)
        
        return time_weights
    
    def forward(self, node_features, adj_matrix, time_matrix):
        """
        前向传播
        
        实现论文公式18-21:
        1. 计算加权平均共现时间 W_vx (公式17)
        2. 计算时间敏感注意力 Attention(v,x) (公式18)
        3. 归一化注意力系数 α_vx (公式20)
        4. 聚合邻居特征 T_m^new (公式21)
        
        Args:
            node_features: 节点特征 [batch_size, num_nodes, in_features]
            adj_matrix: 邻接矩阵 [batch_size, num_nodes, num_nodes]
            time_matrix: 时间矩阵 [batch_size, num_nodes, num_nodes]
        
        Returns:
            out: 输出特征 [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # ========== 1. 计算加权平均共现时间 W_vx (论文公式17) ==========
        time_weights = self.compute_weighted_cooccurrence_time(time_matrix)  # [batch, nodes, nodes]
        
        # ========== 2. 线性变换 MT_v (论文公式18) ==========
        # h = M * node_features
        # [batch, nodes, in_features] @ [heads, in_features, head_dim] -> [batch, heads, nodes, head_dim]
        h = torch.einsum('bni,hio->bhno', node_features, self.W)
        
        # ========== 3. 计算注意力分数 (论文公式18) ==========
        # Attention(v,x) = a^T [MT_v || MT_x] × W_vx
        
        # 拼接源节点和目标节点特征 [MT_v || MT_x]
        h_i = h.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1)  # [batch, heads, nodes, nodes, head_dim]
        h_j = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)  # [batch, heads, nodes, nodes, head_dim]
        
        # 拼接
        h_concat = torch.cat([h_i, h_j], dim=-1)  # [batch, heads, nodes, nodes, 2*head_dim]
        
        # 计算注意力能量 a^T [MT_v || MT_x]
        e = torch.einsum('bhijd,hdo->bhij', h_concat, self.a).squeeze(-1)  # [batch, heads, nodes, nodes]
        e = self.leakyrelu(e)
        
        # 乘以时间权重 W_vx (论文公式18)
        # Attention(v,x) = a^T[MT_v || MT_x] × W_vx
        time_weights_expanded = time_weights.unsqueeze(1)  # [batch, 1, nodes, nodes]
        e = e * time_weights_expanded  # [batch, heads, nodes, nodes]
        
        # ========== 4. 注意力归一化 (论文公式20) ==========
        # α_vx = softmax(LeakyReLU(Attention(v,x)))
        
        # 掩码(只关注邻接矩阵中的边)
        mask = adj_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [batch, heads, nodes, nodes]
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attention = F.softmax(e, dim=-1)  # [batch, heads, nodes, nodes]
        attention = self.dropout(attention)
        
        # ========== 5. 聚合邻居特征 (论文公式21) ==========
        # T_m^new = σ(Σ α_mj M T_m)
        
        # 聚合: [batch, heads, nodes, nodes] @ [batch, heads, nodes, head_dim] -> [batch, heads, nodes, head_dim]
        out = torch.einsum('bhij,bhjo->bhio', attention, h)
        
        # 拼接多头输出
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)  # [batch, nodes, out_features]
        
        # 应用激活函数 σ (论文公式21)
        out = F.elu(out)
        
        return out

