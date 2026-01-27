# -*- coding: utf-8 -*-
"""
Node2vec话题嵌入模块
基于论文公式8-12实现Node2vec算法,用于学习衍生话题共现网络的结构特征

论文引用:
- 公式8: 随机游走跳转概率 P(c_i = x | c_{i-1} = v)
- 公式9: 跳转概率计算 π_vx = α_pq(m,x) · w_vx
- 公式10: 修正因子 α_pq(m,x) 基于距离d_mx
- 公式11: Skip-gram优化目标
- 公式12: 话题结构特征表示 T_struct ∈ R^{|V| × d}
"""

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from collections import defaultdict
import logging


class Node2vecEmbedding:
    """
    Node2vec话题嵌入
    
    实现基于偏置随机游走的图嵌入算法,用于学习衍生话题共现网络的低维表示
    """
    
    def __init__(self, p=1.0, q=1.0, walk_length=80, num_walks=10, 
                 embedding_dim=64, window_size=10, workers=4, epochs=10):
        """
        初始化Node2vec参数
        
        Args:
            p: 返回参数,控制返回前一个节点的概率 (论文公式10)
            q: 进出参数,控制DFS/BFS倾向 (论文公式10)
            walk_length: 每次随机游走的长度
            num_walks: 每个节点的游走次数
            embedding_dim: 嵌入维度 (论文公式12中的d)
            window_size: Skip-gram窗口大小
            workers: 并行线程数
            epochs: Skip-gram训练轮数
        """
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.workers = workers
        self.epochs = epochs
        
        self.model = None
        self.embeddings = None
    
    def _compute_transition_probs(self, graph, edge_weights):
        """
        计算转移概率
        
        基于论文公式9-10计算随机游走的转移概率
        π_vx = α_pq(m,x) · w_vx
        
        Args:
            graph: 邻接列表 {node: [neighbors]}
            edge_weights: 边权重 {(u,v): weight}
        
        Returns:
            alias_nodes: 节点采样表
            alias_edges: 边采样表
        """
        alias_nodes = {}
        alias_edges = {}
        
        # 为每个节点构建采样表
        for node in graph:
            unnormalized_probs = []
            neighbors = graph[node]
            
            for neighbor in neighbors:
                weight = edge_weights.get((node, neighbor), 1.0)
                unnormalized_probs.append(weight)
            
            if len(unnormalized_probs) > 0:
                norm_const = sum(unnormalized_probs)
                normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
                alias_nodes[node] = self._create_alias_table(normalized_probs)
        
        # 为每条边构建采样表 (考虑p和q参数)
        for edge in edge_weights:
            src, dst = edge
            unnormalized_probs = []
            neighbors = graph[dst]
            
            for dst_nbr in neighbors:
                weight = edge_weights.get((dst, dst_nbr), 1.0)
                
                # 计算修正因子 α_pq(m,x) (论文公式10)
                if dst_nbr == src:
                    # 返回前一个节点, d_mx = 0
                    alpha = 1.0 / self.p
                elif dst_nbr in graph.get(src, []):
                    # 邻居节点, d_mx = 1
                    alpha = 1.0
                else:
                    # 非邻居节点, d_mx = 2
                    alpha = 1.0 / self.q
                
                # π_vx = α_pq(m,x) · w_vx (论文公式9)
                unnormalized_probs.append(alpha * weight)
            
            if len(unnormalized_probs) > 0:
                norm_const = sum(unnormalized_probs)
                normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
                alias_edges[edge] = self._create_alias_table(normalized_probs)
        
        return alias_nodes, alias_edges
    
    def _create_alias_table(self, probs):
        """
        创建Alias采样表 (Walker's Alias Method)
        用于O(1)时间复杂度的采样
        
        Args:
            probs: 概率分布
        
        Returns:
            (J, q): Alias表
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int32)
        
        smaller = []
        larger = []
        
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        
        return J, q
    
    def _alias_sample(self, J, q):
        """
        从Alias表中采样
        
        Args:
            J, q: Alias表
        
        Returns:
            采样的索引
        """
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
    
    def _node2vec_walk(self, graph, start_node, alias_nodes, alias_edges):
        """
        执行一次Node2vec随机游走
        
        基于论文公式8实现偏置随机游走:
        P(c_i = x | c_{i-1} = v) = π_vx / Z
        
        Args:
            graph: 邻接列表
            start_node: 起始节点
            alias_nodes: 节点采样表
            alias_edges: 边采样表
        
        Returns:
            walk: 游走序列
        """
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = graph.get(cur, [])
            
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    # 第一步,直接从邻居中采样
                    J, q = alias_nodes[cur]
                    next_node = cur_nbrs[self._alias_sample(J, q)]
                else:
                    # 后续步骤,考虑前一个节点
                    prev = walk[-2]
                    edge = (prev, cur)
                    if edge in alias_edges:
                        J, q = alias_edges[edge]
                        next_node = cur_nbrs[self._alias_sample(J, q)]
                    else:
                        # 如果边不存在,随机选择
                        next_node = np.random.choice(cur_nbrs)
                
                walk.append(next_node)
            else:
                break
        
        return walk
    
    def fit(self, edge_index, edge_weights=None, num_nodes=None):
        """
        训练Node2vec模型
        
        Args:
            edge_index: 边索引 [2, num_edges] 或 边列表 [(u,v), ...]
            edge_weights: 边权重 (可选)
            num_nodes: 节点数量 (可选)
        
        Returns:
            self
        """
        logging.info("开始训练Node2vec模型...")
        
        # 构建邻接列表和边权重字典
        graph = defaultdict(list)
        edge_weight_dict = {}
        
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()
        
        if edge_index.ndim == 2 and edge_index.shape[0] == 2:
            # [2, num_edges] 格式
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i], edge_index[1, i]
                graph[int(u)].append(int(v))
                
                if edge_weights is not None:
                    if isinstance(edge_weights, torch.Tensor):
                        weight = edge_weights[i].item()
                    else:
                        weight = edge_weights[i]
                    edge_weight_dict[(int(u), int(v))] = weight
                else:
                    edge_weight_dict[(int(u), int(v))] = 1.0
        else:
            # 边列表格式
            for edge in edge_index:
                u, v = edge
                graph[int(u)].append(int(v))
                edge_weight_dict[(int(u), int(v))] = 1.0
        
        # 确定节点集合
        if num_nodes is None:
            nodes = set()
            for u in graph:
                nodes.add(u)
                nodes.update(graph[u])
            nodes = sorted(list(nodes))
        else:
            nodes = list(range(num_nodes))
        
        logging.info(f"  图统计: {len(nodes)} 个节点, {len(edge_weight_dict)} 条边")
        
        # 计算转移概率
        logging.info(f"  计算转移概率 (p={self.p}, q={self.q})...")
        alias_nodes, alias_edges = self._compute_transition_probs(graph, edge_weight_dict)
        
        # 生成随机游走序列
        logging.info(f"  生成随机游走序列 (每个节点{self.num_walks}次, 长度{self.walk_length})...")
        walks = []
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                if node in graph:  # 只对有邻居的节点进行游走
                    walk = self._node2vec_walk(graph, node, alias_nodes, alias_edges)
                    walks.append([str(n) for n in walk])
        
        logging.info(f"  生成了 {len(walks)} 条游走序列")
        
        # 使用Skip-gram训练嵌入 (论文公式11)
        logging.info(f"  训练Skip-gram模型 (维度={self.embedding_dim})...")
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,  # Skip-gram
            workers=self.workers,
            epochs=self.epochs
        )
        
        # 提取嵌入矩阵 (论文公式12: T_struct ∈ R^{|V| × d})
        self.embeddings = np.zeros((len(nodes), self.embedding_dim))
        for i, node in enumerate(nodes):
            if str(node) in self.model.wv:
                self.embeddings[i] = self.model.wv[str(node)]
        
        logging.info(f"  Node2vec训练完成! 嵌入维度: {self.embeddings.shape}")
        
        return self
    
    def get_embeddings(self):
        """
        获取话题结构特征表示
        
        Returns:
            T_struct: 话题结构特征 [num_nodes, embedding_dim] (论文公式12)
        """
        if self.embeddings is None:
            raise ValueError("模型尚未训练,请先调用fit()方法")
        
        return torch.FloatTensor(self.embeddings)
    
    def get_embedding(self, node_id):
        """
        获取单个节点的嵌入
        
        Args:
            node_id: 节点ID
        
        Returns:
            embedding: 节点嵌入向量
        """
        if self.model is None:
            raise ValueError("模型尚未训练,请先调用fit()方法")
        
        if str(node_id) in self.model.wv:
            return torch.FloatTensor(self.model.wv[str(node_id)])
        else:
            return torch.zeros(self.embedding_dim)
