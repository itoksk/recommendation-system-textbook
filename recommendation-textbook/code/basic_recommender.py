#!/usr/bin/env python3
"""
基本的な推薦システムの実装
高校生向けの教材用コード
"""

import json
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class SimpleRecommenderSystem:
    """シンプルな推薦システムの基本クラス"""
    
    def __init__(self):
        self.users = {}
        self.items = {}
        self.interactions = []
        
    def add_user(self, user_id: str, name: str, preferences: Dict = None):
        """ユーザーを追加"""
        self.users[user_id] = {
            'name': name,
            'preferences': preferences or {},
            'history': []
        }
        
    def add_item(self, item_id: str, title: str, category: str, features: Dict = None):
        """アイテムを追加"""
        self.items[item_id] = {
            'title': title,
            'category': category,
            'features': features or {},
            'popularity': 0
        }
        
    def record_interaction(self, user_id: str, item_id: str, rating: float, timestamp: datetime = None):
        """ユーザーのインタラクションを記録"""
        if timestamp is None:
            timestamp = datetime.now()
            
        interaction = {
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': timestamp
        }
        
        self.interactions.append(interaction)
        self.users[user_id]['history'].append(interaction)
        self.items[item_id]['popularity'] += 1
        
        return interaction


class PopularityBasedRecommender(SimpleRecommenderSystem):
    """人気度ベースの推薦システム"""
    
    def calculate_popularity_scores(self) -> Dict[str, float]:
        """各アイテムの人気度スコアを計算"""
        scores = {}
        
        for item_id, item in self.items.items():
            # 単純な人気度：インタラクション数
            interaction_count = item['popularity']
            
            # 評価の平均を計算
            ratings = [i['rating'] for i in self.interactions if i['item_id'] == item_id]
            avg_rating = np.mean(ratings) if ratings else 0
            
            # 複合スコア：人気度 × 平均評価
            scores[item_id] = interaction_count * (1 + avg_rating / 5)
            
        return scores
    
    def recommend(self, user_id: str, n: int = 5) -> List[Dict]:
        """人気のアイテムを推薦"""
        # ユーザーの履歴を取得
        user_history = set(i['item_id'] for i in self.users[user_id]['history'])
        
        # 人気度スコアを計算
        scores = self.calculate_popularity_scores()
        
        # 未視聴のアイテムのみフィルタリング
        candidates = [(item_id, score) for item_id, score in scores.items() 
                     if item_id not in user_history]
        
        # スコア順にソート
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 上位n個を返す
        recommendations = []
        for item_id, score in candidates[:n]:
            recommendations.append({
                'item_id': item_id,
                'title': self.items[item_id]['title'],
                'score': score,
                'reason': '人気のアイテム'
            })
            
        return recommendations


class ContentBasedRecommender(SimpleRecommenderSystem):
    """コンテンツベースの推薦システム"""
    
    def calculate_item_similarity(self, item1_id: str, item2_id: str) -> float:
        """2つのアイテムの類似度を計算"""
        item1 = self.items[item1_id]
        item2 = self.items[item2_id]
        
        similarity = 0.0
        
        # カテゴリの一致
        if item1['category'] == item2['category']:
            similarity += 0.5
            
        # 特徴の類似度
        features1 = set(item1['features'].keys())
        features2 = set(item2['features'].keys())
        
        if features1 and features2:
            jaccard = len(features1 & features2) / len(features1 | features2)
            similarity += 0.5 * jaccard
            
        return similarity
    
    def build_user_profile(self, user_id: str) -> Dict:
        """ユーザープロファイルを構築"""
        profile = {
            'categories': {},
            'features': {}
        }
        
        user_history = self.users[user_id]['history']
        
        for interaction in user_history:
            item = self.items[interaction['item_id']]
            rating = interaction['rating']
            
            # カテゴリの重み付け
            category = item['category']
            if category not in profile['categories']:
                profile['categories'][category] = []
            profile['categories'][category].append(rating)
            
            # 特徴の重み付け
            for feature, value in item['features'].items():
                if feature not in profile['features']:
                    profile['features'][feature] = []
                profile['features'][feature].append(rating * value)
                
        # 平均化
        for category in profile['categories']:
            profile['categories'][category] = np.mean(profile['categories'][category])
            
        for feature in profile['features']:
            profile['features'][feature] = np.mean(profile['features'][feature])
            
        return profile
    
    def recommend(self, user_id: str, n: int = 5) -> List[Dict]:
        """ユーザーの好みに基づいて推薦"""
        profile = self.build_user_profile(user_id)
        user_history = set(i['item_id'] for i in self.users[user_id]['history'])
        
        recommendations = []
        
        for item_id, item in self.items.items():
            if item_id in user_history:
                continue
                
            score = 0.0
            
            # カテゴリスコア
            if item['category'] in profile['categories']:
                score += profile['categories'][item['category']]
                
            # 特徴スコア
            for feature, value in item['features'].items():
                if feature in profile['features']:
                    score += profile['features'][feature] * value
                    
            recommendations.append({
                'item_id': item_id,
                'title': item['title'],
                'score': score,
                'reason': 'あなたの好みに基づく'
            })
            
        # スコア順にソート
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:n]


class CollaborativeFilteringRecommender(SimpleRecommenderSystem):
    """協調フィルタリングによる推薦システム"""
    
    def calculate_user_similarity(self, user1_id: str, user2_id: str) -> float:
        """2人のユーザーの類似度を計算"""
        user1_items = {i['item_id']: i['rating'] for i in self.users[user1_id]['history']}
        user2_items = {i['item_id']: i['rating'] for i in self.users[user2_id]['history']}
        
        # 共通アイテム
        common_items = set(user1_items.keys()) & set(user2_items.keys())
        
        if not common_items:
            return 0.0
            
        # ピアソン相関係数
        ratings1 = [user1_items[item] for item in common_items]
        ratings2 = [user2_items[item] for item in common_items]
        
        if len(ratings1) < 2:
            return 0.0
            
        correlation = np.corrcoef(ratings1, ratings2)[0, 1]
        
        # NaNの場合は0を返す
        if np.isnan(correlation):
            return 0.0
            
        return correlation
    
    def find_similar_users(self, user_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """類似ユーザーを見つける"""
        similarities = []
        
        for other_id in self.users:
            if other_id != user_id:
                similarity = self.calculate_user_similarity(user_id, other_id)
                if similarity > 0:
                    similarities.append((other_id, similarity))
                    
        # 類似度順にソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def recommend(self, user_id: str, n: int = 5) -> List[Dict]:
        """類似ユーザーの評価に基づいて推薦"""
        user_history = set(i['item_id'] for i in self.users[user_id]['history'])
        similar_users = self.find_similar_users(user_id)
        
        if not similar_users:
            return []
            
        item_scores = {}
        
        for similar_user_id, similarity in similar_users:
            similar_user_history = self.users[similar_user_id]['history']
            
            for interaction in similar_user_history:
                item_id = interaction['item_id']
                
                if item_id not in user_history:
                    if item_id not in item_scores:
                        item_scores[item_id] = 0.0
                    # 類似度で重み付けしたスコア
                    item_scores[item_id] += interaction['rating'] * similarity
                    
        # スコア順にソート
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in sorted_items[:n]:
            recommendations.append({
                'item_id': item_id,
                'title': self.items[item_id]['title'],
                'score': score,
                'reason': '似た人が高評価'
            })
            
        return recommendations


class HybridRecommender(SimpleRecommenderSystem):
    """ハイブリッド推薦システム"""
    
    def __init__(self):
        super().__init__()
        self.popularity_recommender = PopularityBasedRecommender()
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        
    def sync_data(self):
        """データを各推薦システムに同期"""
        for recommender in [self.popularity_recommender, 
                           self.content_recommender, 
                           self.collaborative_recommender]:
            recommender.users = self.users
            recommender.items = self.items
            recommender.interactions = self.interactions
            
    def recommend(self, user_id: str, n: int = 5, weights: Dict[str, float] = None) -> List[Dict]:
        """複数の手法を組み合わせて推薦"""
        self.sync_data()
        
        if weights is None:
            weights = {
                'popularity': 0.3,
                'content': 0.4,
                'collaborative': 0.3
            }
            
        all_recommendations = {}
        
        # 各推薦システムから結果を取得
        pop_recs = self.popularity_recommender.recommend(user_id, n * 2)
        for rec in pop_recs:
            item_id = rec['item_id']
            if item_id not in all_recommendations:
                all_recommendations[item_id] = {'scores': {}, 'title': rec['title']}
            all_recommendations[item_id]['scores']['popularity'] = rec['score']
            
        content_recs = self.content_recommender.recommend(user_id, n * 2)
        for rec in content_recs:
            item_id = rec['item_id']
            if item_id not in all_recommendations:
                all_recommendations[item_id] = {'scores': {}, 'title': rec['title']}
            all_recommendations[item_id]['scores']['content'] = rec['score']
            
        collab_recs = self.collaborative_recommender.recommend(user_id, n * 2)
        for rec in collab_recs:
            item_id = rec['item_id']
            if item_id not in all_recommendations:
                all_recommendations[item_id] = {'scores': {}, 'title': rec['title']}
            all_recommendations[item_id]['scores']['collaborative'] = rec['score']
            
        # スコアを正規化して統合
        final_recommendations = []
        
        for item_id, data in all_recommendations.items():
            final_score = 0.0
            
            for method, weight in weights.items():
                if method in data['scores']:
                    # 正規化（0-1の範囲に）
                    normalized_score = min(1.0, data['scores'][method] / 10)
                    final_score += normalized_score * weight
                    
            final_recommendations.append({
                'item_id': item_id,
                'title': data['title'],
                'score': final_score,
                'reason': 'ハイブリッド推薦'
            })
            
        # スコア順にソート
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return final_recommendations[:n]


def demo():
    """デモンストレーション"""
    print("🎬 推薦システムのデモ")
    print("=" * 50)
    
    # ハイブリッド推薦システムを作成
    recommender = HybridRecommender()
    
    # サンプルデータを追加
    # 映画を追加
    movies = [
        ('m1', 'アベンジャーズ', 'アクション', {'hero': 1, 'sf': 1}),
        ('m2', '君の名は', 'アニメ', {'romance': 1, 'fantasy': 0.5}),
        ('m3', 'インセプション', 'SF', {'sf': 1, 'thriller': 1}),
        ('m4', 'トイストーリー', 'アニメ', {'family': 1, 'comedy': 0.8}),
        ('m5', 'ダークナイト', 'アクション', {'hero': 1, 'crime': 1}),
        ('m6', 'タイタニック', 'ロマンス', {'romance': 1, 'drama': 1}),
        ('m7', 'マトリックス', 'SF', {'sf': 1, 'action': 1}),
    ]
    
    for movie_id, title, category, features in movies:
        recommender.add_item(movie_id, title, category, features)
        
    # ユーザーを追加
    recommender.add_user('user1', 'あなた')
    recommender.add_user('user2', 'ユーザーA')
    recommender.add_user('user3', 'ユーザーB')
    
    # インタラクションを記録
    # あなたの評価
    recommender.record_interaction('user1', 'm1', 5)  # アベンジャーズ
    recommender.record_interaction('user1', 'm3', 4)  # インセプション
    
    # ユーザーAの評価（SF好き）
    recommender.record_interaction('user2', 'm3', 5)  # インセプション
    recommender.record_interaction('user2', 'm7', 5)  # マトリックス
    recommender.record_interaction('user2', 'm1', 4)  # アベンジャーズ
    
    # ユーザーBの評価（アニメ好き）
    recommender.record_interaction('user3', 'm2', 5)  # 君の名は
    recommender.record_interaction('user3', 'm4', 4)  # トイストーリー
    
    # 推薦を取得
    print("\n📽️ あなたへのおすすめ:")
    recommendations = recommender.recommend('user1', n=5)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   スコア: {rec['score']:.2f}")
        print(f"   理由: {rec['reason']}")
        print()


if __name__ == "__main__":
    demo()