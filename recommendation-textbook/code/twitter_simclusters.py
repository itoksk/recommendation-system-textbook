#!/usr/bin/env python3
"""
TwitterのSimClustersアルゴリズムの簡易実装
教育用に簡略化したバージョン
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from typing import Dict, List, Set, Tuple
import networkx as nx
import matplotlib.pyplot as plt


class SimClusters:
    """
    TwitterのSimClustersアルゴリズムの教育用実装
    
    基本的な流れ：
    1. フォローグラフから生産者（Producer）と消費者（Consumer）の二部グラフを作成
    2. 生産者間の類似度を計算
    3. コミュニティ検出で生産者をクラスタリング
    4. 消費者の興味（InterestedIn）ベクトルを計算
    5. コンテンツ（ツイート）の埋め込みを生成
    """
    
    def __init__(self, n_clusters: int = 20):
        """
        Args:
            n_clusters: 検出するコミュニティ数（本番では約145,000）
        """
        self.n_clusters = n_clusters
        self.known_for_matrix = None  # 生産者×クラスタ行列
        self.interested_in_matrix = None  # 消費者×クラスタ行列
        self.producer_embeddings = None  # 生産者の埋め込み
        self.tweet_embeddings = {}  # ツイートの埋め込み
        
    def fit(self, follow_graph: Dict[str, List[str]]):
        """
        フォローグラフからSimClustersモデルを学習
        
        Args:
            follow_graph: {consumer_id: [producer_ids]} の辞書
        """
        print("📊 SimClustersの学習を開始...")
        
        # Step 1: 二部グラフ行列を作成
        adjacency_matrix, consumers, producers = self._create_bipartite_matrix(follow_graph)
        print(f"  ✓ 二部グラフ作成: {len(consumers)}消費者 × {len(producers)}生産者")
        
        # Step 2: 生産者間の類似度を計算
        producer_similarity = self._calculate_producer_similarity(adjacency_matrix)
        print(f"  ✓ 生産者類似度計算完了")
        
        # Step 3: コミュニティ検出（KnownFor）
        self.known_for_matrix = self._detect_communities(producer_similarity)
        print(f"  ✓ {self.n_clusters}個のコミュニティを検出")
        
        # Step 4: InterestedIn行列を計算
        self.interested_in_matrix = self._calculate_interested_in(
            adjacency_matrix, self.known_for_matrix
        )
        print(f"  ✓ InterestedIn行列を計算")
        
        # Step 5: Producer Embeddingsを計算
        self.producer_embeddings = self._calculate_producer_embeddings(
            adjacency_matrix, self.interested_in_matrix
        )
        print(f"  ✓ Producer Embeddings計算完了")
        
        # インデックスを保存
        self.consumers = consumers
        self.producers = producers
        
        print("✅ SimClustersの学習完了！")
        
    def _create_bipartite_matrix(self, follow_graph: Dict[str, List[str]]) -> Tuple:
        """フォローグラフから二部グラフ行列を作成"""
        consumers = list(follow_graph.keys())
        producers = list(set(p for follows in follow_graph.values() for p in follows))
        
        consumer_idx = {c: i for i, c in enumerate(consumers)}
        producer_idx = {p: i for i, p in enumerate(producers)}
        
        # 疎行列を作成
        rows, cols = [], []
        for consumer, follows in follow_graph.items():
            for producer in follows:
                if producer in producer_idx:
                    rows.append(consumer_idx[consumer])
                    cols.append(producer_idx[producer])
                    
        data = np.ones(len(rows))
        adjacency = csr_matrix((data, (rows, cols)), 
                              shape=(len(consumers), len(producers)))
        
        return adjacency, consumers, producers
    
    def _calculate_producer_similarity(self, adjacency_matrix: csr_matrix) -> np.ndarray:
        """生産者間のコサイン類似度を計算"""
        # 各生産者のフォロワーベクトル
        producer_vectors = adjacency_matrix.T
        
        # L2正規化
        producer_vectors_norm = normalize(producer_vectors, axis=1, norm='l2')
        
        # コサイン類似度行列
        similarity_matrix = producer_vectors_norm @ producer_vectors_norm.T
        
        return similarity_matrix.toarray()
    
    def _detect_communities(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """スペクトラルクラスタリングでコミュニティ検出"""
        # ノイズ除去：低い類似度をゼロに
        threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 20)
        similarity_matrix[similarity_matrix < threshold] = 0
        
        # スペクトラルクラスタリング
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        labels = clustering.fit_predict(similarity_matrix)
        
        # KnownFor行列を作成（生産者×クラスタ）
        n_producers = similarity_matrix.shape[0]
        known_for = np.zeros((n_producers, self.n_clusters))
        
        for producer_idx, cluster_id in enumerate(labels):
            known_for[producer_idx, cluster_id] = 1.0
            
        return known_for
    
    def _calculate_interested_in(self, adjacency_matrix: csr_matrix, 
                                 known_for_matrix: np.ndarray) -> np.ndarray:
        """InterestedIn行列を計算（消費者×クラスタ）"""
        # InterestedIn = AdjacencyMatrix × KnownFor
        interested_in = adjacency_matrix @ known_for_matrix
        
        # 正規化
        row_sums = interested_in.sum(axis=1)
        row_sums[row_sums == 0] = 1
        interested_in = interested_in / row_sums.reshape(-1, 1)
        
        return interested_in
    
    def _calculate_producer_embeddings(self, adjacency_matrix: csr_matrix,
                                      interested_in_matrix: np.ndarray) -> np.ndarray:
        """Producer Embeddingsを計算"""
        # ProducerEmbeddings = Adjacency.T × InterestedIn
        producer_embeddings = adjacency_matrix.T @ interested_in_matrix
        
        # 正規化
        producer_embeddings = normalize(producer_embeddings, axis=1, norm='l2')
        
        return producer_embeddings
    
    def update_tweet_embedding(self, tweet_id: str, user_likes: List[str]):
        """
        ツイートの埋め込みを更新（いいねされるたびに呼ばれる）
        
        Args:
            tweet_id: ツイートID
            user_likes: いいねしたユーザーのリスト
        """
        if tweet_id not in self.tweet_embeddings:
            self.tweet_embeddings[tweet_id] = np.zeros(self.n_clusters)
            
        for user_id in user_likes:
            if user_id in self.consumers:
                user_idx = self.consumers.index(user_id)
                # ユーザーのInterestedInベクトルを加算
                self.tweet_embeddings[tweet_id] += self.interested_in_matrix[user_idx]
                
        # 正規化
        norm = np.linalg.norm(self.tweet_embeddings[tweet_id])
        if norm > 0:
            self.tweet_embeddings[tweet_id] /= norm
            
    def get_similar_tweets(self, tweet_id: str, n: int = 5) -> List[Tuple[str, float]]:
        """類似ツイートを取得"""
        if tweet_id not in self.tweet_embeddings:
            return []
            
        target_embedding = self.tweet_embeddings[tweet_id]
        similarities = []
        
        for other_id, other_embedding in self.tweet_embeddings.items():
            if other_id != tweet_id:
                similarity = np.dot(target_embedding, other_embedding)
                similarities.append((other_id, similarity))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def get_user_recommendations(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """ユーザーに推薦するツイートを取得"""
        if user_id not in self.consumers:
            return []
            
        user_idx = self.consumers.index(user_id)
        user_interests = self.interested_in_matrix[user_idx]
        
        recommendations = []
        for tweet_id, tweet_embedding in self.tweet_embeddings.items():
            score = np.dot(user_interests, tweet_embedding)
            recommendations.append((tweet_id, score))
            
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]
    
    def visualize_clusters(self):
        """クラスタの可視化"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # KnownFor行列のヒートマップ
        ax1 = axes[0]
        im1 = ax1.imshow(self.known_for_matrix.T, aspect='auto', cmap='YlOrRd')
        ax1.set_xlabel('生産者')
        ax1.set_ylabel('クラスタ')
        ax1.set_title('KnownFor行列（生産者のクラスタ所属）')
        plt.colorbar(im1, ax=ax1)
        
        # InterestedIn行列のヒートマップ
        ax2 = axes[1]
        im2 = ax2.imshow(self.interested_in_matrix.T, aspect='auto', cmap='YlGnBu')
        ax2.set_xlabel('消費者')
        ax2.set_ylabel('クラスタ')
        ax2.set_title('InterestedIn行列（消費者の興味）')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()


def demo():
    """SimClustersのデモンストレーション"""
    print("🐦 SimClustersデモ（簡易版Twitter）")
    print("=" * 50)
    
    # サンプルのフォローグラフを作成
    follow_graph = {
        # テック系ユーザー
        'user_tech1': ['elon_musk', 'sundarpichai', 'satyanadella'],
        'user_tech2': ['elon_musk', 'jeffbezos', 'timcook'],
        'user_tech3': ['sundarpichai', 'satyanadella', 'markzuck'],
        
        # スポーツ系ユーザー
        'user_sports1': ['cristiano', 'messi', 'neymar'],
        'user_sports2': ['cristiano', 'messi', 'lebron'],
        'user_sports3': ['lebron', 'stephcurry', 'neymar'],
        
        # エンタメ系ユーザー
        'user_ent1': ['taylorswift', 'brunomars', 'drake'],
        'user_ent2': ['taylorswift', 'arianagrande', 'brunomars'],
        'user_ent3': ['drake', 'theweeknd', 'arianagrande'],
        
        # 混合ユーザー
        'user_mix1': ['elon_musk', 'cristiano', 'taylorswift'],
        'user_mix2': ['sundarpichai', 'messi', 'brunomars'],
    }
    
    # SimClustersモデルを学習
    model = SimClusters(n_clusters=5)
    model.fit(follow_graph)
    
    # ツイートにいいねを追加
    print("\n📝 ツイートの埋め込みを生成...")
    
    # テック系ツイート
    model.update_tweet_embedding('tweet_ai', ['user_tech1', 'user_tech2', 'user_mix1'])
    model.update_tweet_embedding('tweet_tesla', ['user_tech1', 'user_tech2', 'user_tech3'])
    
    # スポーツ系ツイート
    model.update_tweet_embedding('tweet_worldcup', ['user_sports1', 'user_sports2', 'user_mix2'])
    model.update_tweet_embedding('tweet_nba', ['user_sports2', 'user_sports3'])
    
    # エンタメ系ツイート
    model.update_tweet_embedding('tweet_concert', ['user_ent1', 'user_ent2'])
    model.update_tweet_embedding('tweet_album', ['user_ent2', 'user_ent3'])
    
    print("  ✓ 6個のツイート埋め込みを生成")
    
    # 類似ツイートを検索
    print("\n🔍 類似ツイート検索:")
    similar = model.get_similar_tweets('tweet_ai', n=3)
    print(f"  'tweet_ai'に似たツイート:")
    for tweet_id, score in similar:
        print(f"    - {tweet_id}: 類似度 {score:.3f}")
    
    # ユーザーへの推薦
    print("\n🎯 ユーザーへの推薦:")
    recommendations = model.get_user_recommendations('user_tech1', n=3)
    print(f"  'user_tech1'へのおすすめ:")
    for tweet_id, score in recommendations:
        print(f"    - {tweet_id}: スコア {score:.3f}")
    
    # クラスタ情報を表示
    print("\n📊 クラスタ分析:")
    for i in range(model.n_clusters):
        cluster_producers = []
        for j, producer in enumerate(model.producers):
            if model.known_for_matrix[j, i] > 0:
                cluster_producers.append(producer)
        if cluster_producers:
            print(f"  クラスタ{i}: {', '.join(cluster_producers[:3])}...")


class SimpleTwitterSimulator:
    """教育用のシンプルなTwitterシミュレーター"""
    
    def __init__(self):
        self.users = {}
        self.tweets = {}
        self.follows = {}  # user -> [followed_users]
        self.likes = {}    # tweet -> [users_who_liked]
        self.simclusters = None
        
    def add_user(self, user_id: str, name: str, interests: List[str]):
        """ユーザーを追加"""
        self.users[user_id] = {
            'name': name,
            'interests': interests,
            'tweets': [],
            'following': []
        }
        self.follows[user_id] = []
        
    def follow(self, follower_id: str, followed_id: str):
        """フォロー関係を追加"""
        if follower_id in self.follows:
            self.follows[follower_id].append(followed_id)
            self.users[follower_id]['following'].append(followed_id)
            
    def post_tweet(self, user_id: str, tweet_id: str, content: str):
        """ツイートを投稿"""
        self.tweets[tweet_id] = {
            'author': user_id,
            'content': content,
            'likes': 0
        }
        self.users[user_id]['tweets'].append(tweet_id)
        self.likes[tweet_id] = []
        
    def like_tweet(self, user_id: str, tweet_id: str):
        """ツイートにいいね"""
        if tweet_id in self.likes:
            self.likes[tweet_id].append(user_id)
            self.tweets[tweet_id]['likes'] += 1
            
    def train_simclusters(self, n_clusters: int = 10):
        """SimClustersモデルを学習"""
        self.simclusters = SimClusters(n_clusters=n_clusters)
        self.simclusters.fit(self.follows)
        
        # ツイートの埋め込みを更新
        for tweet_id, users_who_liked in self.likes.items():
            if users_who_liked:
                self.simclusters.update_tweet_embedding(tweet_id, users_who_liked)
                
    def get_timeline(self, user_id: str, n: int = 10) -> List[str]:
        """ユーザーのタイムラインを生成"""
        if self.simclusters is None:
            return []
            
        recommendations = self.simclusters.get_user_recommendations(user_id, n)
        timeline = []
        
        for tweet_id, score in recommendations:
            tweet = self.tweets[tweet_id]
            timeline.append(f"{tweet['author']}: {tweet['content']} (スコア: {score:.2f})")
            
        return timeline


if __name__ == "__main__":
    demo()