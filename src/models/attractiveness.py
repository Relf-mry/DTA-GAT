# -*- coding: utf-8 -*-
"""
话题吸引力量化模块
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import logging


class TopicAttractiveness:
    """话题吸引力量化"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lambda_0 = None
        self.lambda_1 = None
        self.lambda_2 = None
    
    def compute_total_engagement(self, likes, comments, shares, forwards):
        """计算总话题交互量"""
        return likes + comments + shares + forwards
    
    def compute_engagement_rate(self, total_engagement, total_followers):
        """计算话题参与率"""
        if total_followers == 0:
            return 0.0
        return total_engagement / total_followers
    
    def compute_growth_rate(self, current_engagement, previous_engagement):
        """计算话题增长率"""
        if previous_engagement == 0:
            return 1.0 if current_engagement > 0 else 0.0
        return (current_engagement - previous_engagement) / previous_engagement
    
    def compute_reach(self, direct_audience, indirect_audience):
        """计算话题覆盖范围"""
        return direct_audience + indirect_audience
    
    def compute_initial_user_influence(self, num_followers, historical_engagement_rate, 
                                      content_reach_rate):
        """计算初始用户影响力指数"""
        return num_followers + historical_engagement_rate + content_reach_rate
    
    def standardize(self, values):
        """标准化指标到[0,1]范围"""
        values = np.array(values).reshape(-1, 1)
        if np.max(values) == np.min(values):
            return np.ones_like(values).flatten()
        standardized = self.scaler.fit_transform(values)
        return standardized.flatten()
    
    def compute_topic_influence(self, engagement_rates, growth_rates, reaches, 
                                initial_influences):
        """
        计算话题影响力 
        """
        s_engagement = self.standardize(engagement_rates)
        s_growth = self.standardize(growth_rates)
        s_reach = self.standardize(reaches)
        s_influence = self.standardize(initial_influences)
        
        topic_influence = (s_engagement + s_growth + s_reach + s_influence) / 4.0
        
        return topic_influence
    
    def compute_topic_relevance(self, degrees, num_topics):
        """
        计算话题相关性
        """
        if num_topics <= 1:
            return np.zeros_like(degrees, dtype=float)
        
        topic_relevance = np.array(degrees) / (num_topics - 1)
        
        return topic_relevance
    
    def fit_attractiveness_model(self, topic_influence, topic_relevance, 
                                 propagation_scales):
        """
        训练话题吸引力模型 
        """
        from sklearn.linear_model import LinearRegression
        
        X = np.column_stack([topic_influence, topic_relevance])
        y = np.array(propagation_scales)
        
        model = LinearRegression()
        model.fit(X, y)
        
        self.lambda_0 = model.intercept_
        self.lambda_1 = model.coef_[0]
        self.lambda_2 = model.coef_[1]
        
        logging.info(f"话题吸引力模型训练完成:")
        logging.info(f"  λ0 (截距) = {self.lambda_0:.4f}")
        logging.info(f"  λ1 (影响力权重) = {self.lambda_1:.4f}")
        logging.info(f"  λ2 (相关性权重) = {self.lambda_2:.4f}")
        
        return self
    
    def compute_attractiveness(self, topic_influence, topic_relevance):
        """计算话题吸引力"""
        if self.lambda_0 is None:
            logging.warning("话题吸引力模型未训练,使用默认权重")
            self.lambda_0 = 0.0
            self.lambda_1 = 0.5
            self.lambda_2 = 0.5
        
        attractiveness = (self.lambda_0 + 
                         self.lambda_1 * topic_influence + 
                         self.lambda_2 * topic_relevance)
        
        return attractiveness
    
    def compute_full_attractiveness(self, topic_data, graph_degrees, num_topics):
        """计算完整的话题吸引力"""
        n_topics = len(topic_data['likes'])
        
        # 1. 计算各项指标
        engagement_rates = []
        growth_rates = []
        reaches = []
        initial_influences = []
        
        for i in range(n_topics):
            total_engage = self.compute_total_engagement(
                topic_data['likes'][i],
                topic_data['comments'][i],
                topic_data['shares'][i],
                topic_data['forwards'][i]
            )
            
            engage_rate = self.compute_engagement_rate(
                total_engage,
                topic_data['followers'][i]
            )
            engagement_rates.append(engage_rate)
            
            current_engage = topic_data.get('current_engagement', [total_engage] * n_topics)[i]
            previous_engage = topic_data.get('previous_engagement', [total_engage * 0.8] * n_topics)[i]
            growth_rate = self.compute_growth_rate(current_engage, previous_engage)
            growth_rates.append(growth_rate)
            
            direct_aud = topic_data.get('direct_audience', [topic_data['followers'][i]] * n_topics)[i]
            indirect_aud = topic_data.get('indirect_audience', [0] * n_topics)[i]
            reach = self.compute_reach(direct_aud, indirect_aud)
            reaches.append(reach)
            
            num_follow = topic_data.get('num_followers', [1000] * n_topics)[i]
            hist_engage = topic_data.get('historical_engagement_rate', [0.1] * n_topics)[i]
            content_reach = topic_data.get('content_reach_rate', [0.05] * n_topics)[i]
            init_influence = self.compute_initial_user_influence(num_follow, hist_engage, content_reach)
            initial_influences.append(init_influence)
        
        # 2. 计算话题影响力
        topic_influence = self.compute_topic_influence(
            engagement_rates, growth_rates, reaches, initial_influences
        )
        
        # 3. 计算话题相关性
        topic_relevance = self.compute_topic_relevance(graph_degrees, num_topics)
        
        # 4. 计算话题吸引力
        attractiveness = np.array([
            self.compute_attractiveness(topic_influence[i], topic_relevance[i])
            for i in range(n_topics)
        ])
        
        return torch.FloatTensor(attractiveness)
