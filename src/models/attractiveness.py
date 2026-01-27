# -*- coding: utf-8 -*-
"""
话题吸引力量化模块
基于论文公式1-6, 13-16实现多维度话题吸引力计算

论文引用:
- 公式1: 总话题交互量 totalEngage(v)
- 公式2: 话题参与率 engageRate(v)
- 公式3: 话题增长率 growthRate(v)
- 公式4: 话题覆盖范围 reach(v)
- 公式5: 初始用户影响力指数 influIndex(v)
- 公式13: 标准化 S(y)
- 公式14: 话题影响力 I_impact
- 公式15: 话题相关性 I_relate (度中心性)
- 公式16: 话题吸引力 T_attraction = λ0 + λ1·I_impact + λ2·I_relate
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import logging


class TopicAttractiveness:
    """
    话题吸引力量化
    
    从影响力和相关性两个维度量化话题吸引力
    """
    
    def __init__(self):
        """初始化话题吸引力计算器"""
        self.scaler = MinMaxScaler()
        
        # 多元线性回归参数 (论文公式16)
        self.lambda_0 = None  # 截距
        self.lambda_1 = None  # 影响力权重
        self.lambda_2 = None  # 相关性权重
    
    def compute_total_engagement(self, likes, comments, shares, forwards):
        """
        计算总话题交互量 (论文公式1)
        
        totalEngage(v) = like(v) + comment(v) + share(v) + forward(v)
        
        Args:
            likes: 点赞数
            comments: 评论数
            shares: 分享数
            forwards: 转发数
        
        Returns:
            total_engagement: 总交互量
        """
        return likes + comments + shares + forwards
    
    def compute_engagement_rate(self, total_engagement, total_followers):
        """
        计算话题参与率 (论文公式2)
        
        engageRate(v) = totalEngage(v) / totalFollow(v)
        
        Args:
            total_engagement: 总交互量
            total_followers: 总关注数(浏览量)
        
        Returns:
            engagement_rate: 参与率
        """
        # 避免除零
        if total_followers == 0:
            return 0.0
        return total_engagement / total_followers
    
    def compute_growth_rate(self, current_engagement, previous_engagement):
        """
        计算话题增长率 (论文公式3)
        
        growthRate(v) = (totalEngage_cur(v) - totalEngage_pre(v)) / totalEngage_pre(v)
        
        Args:
            current_engagement: 当前时期总交互量
            previous_engagement: 前一时期总交互量
        
        Returns:
            growth_rate: 增长率
        """
        # 避免除零
        if previous_engagement == 0:
            return 1.0 if current_engagement > 0 else 0.0
        return (current_engagement - previous_engagement) / previous_engagement
    
    def compute_reach(self, direct_audience, indirect_audience):
        """
        计算话题覆盖范围 (论文公式4)
        
        reach(v) = audiCount_dir(v) + audiCount_indir(v)
        
        Args:
            direct_audience: 直接受众数
            indirect_audience: 间接受众数(通过分享、转发等)
        
        Returns:
            reach: 覆盖范围
        """
        return direct_audience + indirect_audience
    
    def compute_initial_user_influence(self, num_followers, historical_engagement_rate, 
                                      content_reach_rate):
        """
        计算初始用户影响力指数 (论文公式5)
        
        influIndex(v) = I_f(u) + I_h(u) + I_e(u)
        
        Args:
            num_followers: 粉丝数 I_f(u)
            historical_engagement_rate: 历史平均互动率 I_h(u)
            content_reach_rate: 内容触达率 I_e(u)
        
        Returns:
            influence_index: 影响力指数
        """
        return num_followers + historical_engagement_rate + content_reach_rate
    
    def standardize(self, values):
        """
        标准化指标 (论文公式13)
        
        S(y) = (y - min(y)) / (max(y) - min(y))
        
        将指标标准化到[0, 1]范围
        
        Args:
            values: 原始值数组
        
        Returns:
            standardized_values: 标准化后的值
        """
        values = np.array(values).reshape(-1, 1)
        
        # 避免所有值相同的情况
        if np.max(values) == np.min(values):
            return np.ones_like(values).flatten()
        
        standardized = self.scaler.fit_transform(values)
        return standardized.flatten()
    
    def compute_topic_influence(self, engagement_rates, growth_rates, reaches, 
                                initial_influences):
        """
        计算话题影响力 (论文公式14)
        
        I_impact = (Σ S(y_i)) / 4
        
        综合考虑四个因素:
        1. 参与率 (公式2)
        2. 增长率 (公式3)
        3. 覆盖范围 (公式4)
        4. 初始用户影响力 (公式5)
        
        Args:
            engagement_rates: 参与率数组
            growth_rates: 增长率数组
            reaches: 覆盖范围数组
            initial_influences: 初始用户影响力数组
        
        Returns:
            topic_influence: 话题影响力分数
        """
        # 标准化各个指标 (论文公式13)
        s_engagement = self.standardize(engagement_rates)
        s_growth = self.standardize(growth_rates)
        s_reach = self.standardize(reaches)
        s_influence = self.standardize(initial_influences)
        
        # 计算综合影响力 (论文公式14)
        # 分母4对应四个量化指标
        topic_influence = (s_engagement + s_growth + s_reach + s_influence) / 4.0
        
        return topic_influence
    
    def compute_topic_relevance(self, degrees, num_topics):
        """
        计算话题相关性 (论文公式15)
        
        I_relate = deg(v) / (|V| - 1)
        
        基于度中心性计算话题相关性
        
        Args:
            degrees: 节点度数组
            num_topics: 话题总数 |V|
        
        Returns:
            topic_relevance: 话题相关性分数
        """
        # 避免除零
        if num_topics <= 1:
            return np.zeros_like(degrees, dtype=float)
        
        # 度中心性 (论文公式15)
        topic_relevance = np.array(degrees) / (num_topics - 1)
        
        return topic_relevance
    
    def fit_attractiveness_model(self, topic_influence, topic_relevance, 
                                 propagation_scales):
        """
        训练话题吸引力模型 (论文公式16)
        
        使用多元线性回归拟合:
        T_attraction = λ0 + λ1·I_impact + λ2·I_relate + ε
        
        Args:
            topic_influence: 话题影响力数组
            topic_relevance: 话题相关性数组
            propagation_scales: 传播规模(目标变量)
        
        Returns:
            self
        """
        from sklearn.linear_model import LinearRegression
        
        # 构建特征矩阵
        X = np.column_stack([topic_influence, topic_relevance])
        y = np.array(propagation_scales)
        
        # 训练线性回归模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 保存参数 (论文公式16)
        self.lambda_0 = model.intercept_  # 截距
        self.lambda_1 = model.coef_[0]    # 影响力权重
        self.lambda_2 = model.coef_[1]    # 相关性权重
        
        logging.info(f"话题吸引力模型训练完成:")
        logging.info(f"  λ0 (截距) = {self.lambda_0:.4f}")
        logging.info(f"  λ1 (影响力权重) = {self.lambda_1:.4f}")
        logging.info(f"  λ2 (相关性权重) = {self.lambda_2:.4f}")
        
        return self
    
    def compute_attractiveness(self, topic_influence, topic_relevance):
        """
        计算话题吸引力 (论文公式16)
        
        T_attraction = λ0 + λ1·I_impact + λ2·I_relate
        
        Args:
            topic_influence: 话题影响力
            topic_relevance: 话题相关性
        
        Returns:
            attractiveness: 话题吸引力分数
        """
        if self.lambda_0 is None:
            # 如果未训练模型,使用默认权重
            logging.warning("话题吸引力模型未训练,使用默认权重")
            self.lambda_0 = 0.0
            self.lambda_1 = 0.5
            self.lambda_2 = 0.5
        
        # 论文公式16
        attractiveness = (self.lambda_0 + 
                         self.lambda_1 * topic_influence + 
                         self.lambda_2 * topic_relevance)
        
        return attractiveness
    
    def compute_full_attractiveness(self, topic_data, graph_degrees, num_topics):
        """
        计算完整的话题吸引力
        
        整合所有步骤:
        1. 计算话题影响力 (公式1-5, 13-14)
        2. 计算话题相关性 (公式15)
        3. 融合得到话题吸引力 (公式16)
        
        Args:
            topic_data: 话题数据字典,包含:
                - likes: 点赞数
                - comments: 评论数
                - shares: 分享数
                - forwards: 转发数
                - followers: 关注数
                - current_engagement: 当前交互量
                - previous_engagement: 前期交互量
                - direct_audience: 直接受众
                - indirect_audience: 间接受众
                - num_followers: 粉丝数
                - historical_engagement_rate: 历史互动率
                - content_reach_rate: 内容触达率
            graph_degrees: 图中各节点的度
            num_topics: 话题总数
        
        Returns:
            attractiveness: 话题吸引力张量
        """
        n_topics = len(topic_data['likes'])
        
        # 1. 计算各项指标
        engagement_rates = []
        growth_rates = []
        reaches = []
        initial_influences = []
        
        for i in range(n_topics):
            # 总交互量 (公式1)
            total_engage = self.compute_total_engagement(
                topic_data['likes'][i],
                topic_data['comments'][i],
                topic_data['shares'][i],
                topic_data['forwards'][i]
            )
            
            # 参与率 (公式2)
            engage_rate = self.compute_engagement_rate(
                total_engage,
                topic_data['followers'][i]
            )
            engagement_rates.append(engage_rate)
            
            # 增长率 (公式3)
            current_engage = topic_data.get('current_engagement', [total_engage] * n_topics)[i]
            previous_engage = topic_data.get('previous_engagement', [total_engage * 0.8] * n_topics)[i]
            growth_rate = self.compute_growth_rate(current_engage, previous_engage)
            growth_rates.append(growth_rate)
            
            # 覆盖范围 (公式4)
            direct_aud = topic_data.get('direct_audience', [topic_data['followers'][i]] * n_topics)[i]
            indirect_aud = topic_data.get('indirect_audience', [0] * n_topics)[i]
            reach = self.compute_reach(direct_aud, indirect_aud)
            reaches.append(reach)
            
            # 初始用户影响力 (公式5)
            num_follow = topic_data.get('num_followers', [1000] * n_topics)[i]
            hist_engage = topic_data.get('historical_engagement_rate', [0.1] * n_topics)[i]
            content_reach = topic_data.get('content_reach_rate', [0.05] * n_topics)[i]
            init_influence = self.compute_initial_user_influence(num_follow, hist_engage, content_reach)
            initial_influences.append(init_influence)
        
        # 2. 计算话题影响力 (公式14)
        topic_influence = self.compute_topic_influence(
            engagement_rates, growth_rates, reaches, initial_influences
        )
        
        # 3. 计算话题相关性 (公式15)
        topic_relevance = self.compute_topic_relevance(graph_degrees, num_topics)
        
        # 4. 计算话题吸引力 (公式16)
        attractiveness = np.array([
            self.compute_attractiveness(topic_influence[i], topic_relevance[i])
            for i in range(n_topics)
        ])
        
        return torch.FloatTensor(attractiveness)
