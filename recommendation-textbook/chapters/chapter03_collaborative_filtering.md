# Chapter 3: 協調フィルタリングを極める 🤝

## 📚 この章で学ぶこと

- 協調フィルタリングの数学的基礎
- メモリベース vs モデルベース
- TwitterのSimClustersアルゴリズム詳解
- 実装とパフォーマンス最適化

## 🎯 協調フィルタリングの本質

### なぜ「協調」なのか？

協調フィルタリング（Collaborative Filtering）は、**多くのユーザーの行動データを「協調」させて推薦を行う**手法です。

```python
# 基本的な考え方
def collaborative_filtering_concept():
    """
    「あなたと似た人が好きなものは、あなたも好きかもしれない」
    """
    # Step 1: 類似ユーザーを見つける
    similar_users = find_similar_users(target_user)
    
    # Step 2: 類似ユーザーの好みを集約
    recommendations = aggregate_preferences(similar_users)
    
    return recommendations
```

## 🔍 メモリベース協調フィルタリング

### ユーザーベース協調フィルタリング

```python
import numpy as np
from scipy.spatial.distance import cosine

class UserBasedCF:
    """ユーザーベース協調フィルタリング"""
    
    def __init__(self, min_common_items=2):
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.user_similarities = None
        
    def fit(self, ratings_data):
        """
        ratings_data: [(user_id, item_id, rating), ...]
        """
        # ユーザー×アイテム行列を作成
        self.user_item_matrix = self._create_matrix(ratings_data)
        
        # 全ユーザーペアの類似度を事前計算
        self.user_similarities = self._calculate_all_similarities()
        
    def _calculate_similarity(self, user1_vec, user2_vec):
        """コサイン類似度を計算"""
        # 両方が評価したアイテムのみ使用
        mask = (user1_vec > 0) & (user2_vec > 0)
        
        if mask.sum() < self.min_common_items:
            return 0.0
            
        # コサイン類似度
        return 1 - cosine(user1_vec[mask], user2_vec[mask])
    
    def predict(self, user_id, item_id):
        """評価を予測"""
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]
        
        # 類似ユーザーの評価を重み付け平均
        similarities = self.user_similarities[user_idx]
        ratings = self.user_item_matrix[:, item_idx]
        
        # 評価済みユーザーのみ
        mask = ratings > 0
        if not mask.any():
            return self.global_mean
            
        weighted_sum = np.sum(similarities[mask] * ratings[mask])
        weight_sum = np.sum(np.abs(similarities[mask]))
        
        if weight_sum == 0:
            return self.global_mean
            
        return weighted_sum / weight_sum
```

### アイテムベース協調フィルタリング

```python
class ItemBasedCF:
    """アイテムベース協調フィルタリング（より効率的）"""
    
    def __init__(self):
        self.item_similarities = None
        
    def fit(self, ratings_data):
        """アイテム間の類似度を事前計算"""
        # アイテム×ユーザー行列
        item_user_matrix = self._create_item_matrix(ratings_data)
        
        # アイテム類似度行列を計算
        self.item_similarities = self._calculate_item_similarities(item_user_matrix)
        
    def _calculate_item_similarities(self, matrix):
        """調整コサイン類似度（Adjusted Cosine Similarity）"""
        # ユーザーごとの平均を引く（個人の評価傾向を除去）
        user_means = matrix.mean(axis=0)
        adjusted_matrix = matrix - user_means
        
        # コサイン類似度を計算
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(adjusted_matrix)
        
        return similarities
    
    def recommend(self, user_history, n=10):
        """ユーザーの履歴から推薦"""
        scores = {}
        
        for item_id, rating in user_history.items():
            # このアイテムと類似したアイテムを探す
            similar_items = self.get_similar_items(item_id)
            
            for similar_item, similarity in similar_items:
                if similar_item not in user_history:
                    scores[similar_item] = scores.get(similar_item, 0)
                    scores[similar_item] += rating * similarity
                    
        # スコア順にソート
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
```

## 🧮 モデルベース協調フィルタリング

### 行列分解（Matrix Factorization）

```python
class MatrixFactorization:
    """
    特異値分解（SVD）による協調フィルタリング
    R ≈ P × Q^T
    """
    
    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.01):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        
    def fit(self, ratings_matrix, epochs=100):
        """確率的勾配降下法（SGD）で学習"""
        m, n = ratings_matrix.shape
        
        # ランダム初期化
        self.P = np.random.normal(0, 0.1, (m, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n, self.n_factors))
        
        for epoch in range(epochs):
            for i in range(m):
                for j in range(n):
                    if ratings_matrix[i, j] > 0:
                        # 誤差を計算
                        prediction = self.P[i] @ self.Q[j]
                        error = ratings_matrix[i, j] - prediction
                        
                        # 勾配更新
                        p_gradient = error * self.Q[j] - self.reg * self.P[i]
                        q_gradient = error * self.P[i] - self.reg * self.Q[j]
                        
                        self.P[i] += self.lr * p_gradient
                        self.Q[j] += self.lr * q_gradient
                        
    def predict(self, user_idx, item_idx):
        """予測評価を計算"""
        return self.P[user_idx] @ self.Q[item_idx]
```

## 🐦 TwitterのSimClusters深掘り

### SimClustersの革新性

```python
class TwitterSimClusters:
    """
    Twitterの実装に近いSimClustersアルゴリズム
    ポイント：スケーラビリティと解釈可能性の両立
    """
    
    def __init__(self, n_clusters=145000):
        self.n_clusters = n_clusters
        
    def build_producer_similarity_graph(self, follow_graph):
        """
        生産者の類似度グラフを構築
        類似度 = フォロワーの重複度
        """
        producer_followers = defaultdict(set)
        
        # 各生産者のフォロワーを収集
        for consumer, producers in follow_graph.items():
            for producer in producers:
                producer_followers[producer].add(consumer)
                
        # ペアワイズ類似度を計算
        similarity_edges = []
        producers = list(producer_followers.keys())
        
        for i, p1 in enumerate(producers):
            for p2 in producers[i+1:]:
                followers1 = producer_followers[p1]
                followers2 = producer_followers[p2]
                
                # Jaccard係数
                intersection = len(followers1 & followers2)
                union = len(followers1 | followers2)
                
                if union > 0:
                    similarity = intersection / union
                    if similarity > 0.01:  # 閾値
                        similarity_edges.append((p1, p2, similarity))
                        
        return similarity_edges
    
    def metropolis_hastings_clustering(self, similarity_graph, n_iterations=1000):
        """
        Metropolis-Hastingsサンプリングによるコミュニティ検出
        （Twitterの実装を簡略化）
        """
        import random
        
        # 初期クラスタ割り当て
        nodes = list(set(n for edge in similarity_graph for n in edge[:2]))
        clusters = {node: random.randint(0, self.n_clusters-1) for node in nodes}
        
        # エネルギー関数（モジュラリティ）
        def calculate_energy(clusters, graph):
            energy = 0
            for n1, n2, weight in graph:
                if clusters[n1] == clusters[n2]:
                    energy += weight
            return energy
        
        current_energy = calculate_energy(clusters, similarity_graph)
        
        for _ in range(n_iterations):
            # ランダムにノードを選択
            node = random.choice(nodes)
            old_cluster = clusters[node]
            new_cluster = random.randint(0, self.n_clusters-1)
            
            # クラスタを変更
            clusters[node] = new_cluster
            new_energy = calculate_energy(clusters, similarity_graph)
            
            # Metropolis基準
            delta = new_energy - current_energy
            if delta > 0 or random.random() < np.exp(delta):
                current_energy = new_energy
            else:
                clusters[node] = old_cluster  # 元に戻す
                
        return clusters
    
    def create_embeddings(self, known_for, interested_in):
        """
        最終的な埋め込みを作成
        """
        # Producer Embeddings: より豊かな表現
        # 単一クラスタ（KnownFor）ではなく、複数クラスタへの所属度
        producer_embeddings = {}
        
        for producer, cluster in known_for.items():
            embedding = np.zeros(self.n_clusters)
            
            # メインクラスタ
            embedding[cluster] = 1.0
            
            # 関連クラスタも考慮（実際はもっと複雑）
            # ...
            
            producer_embeddings[producer] = embedding
            
        return producer_embeddings
```

## 📊 パフォーマンス最適化

### 1. 近似最近傍探索（ANN）

```python
from annoy import AnnoyIndex

class FastSimilaritySearch:
    """高速な類似アイテム検索"""
    
    def __init__(self, n_dimensions, n_trees=10):
        self.index = AnnoyIndex(n_dimensions, 'angular')
        self.n_trees = n_trees
        
    def build_index(self, item_vectors):
        """インデックスを構築"""
        for i, vector in enumerate(item_vectors):
            self.index.add_item(i, vector)
            
        self.index.build(self.n_trees)
        
    def find_similar(self, query_vector, n=10):
        """類似アイテムを高速検索"""
        # O(log n)で検索
        similar_items = self.index.get_nns_by_vector(
            query_vector, n, include_distances=True
        )
        return similar_items
```

### 2. オンライン学習

```python
class OnlineCollaborativeFiltering:
    """リアルタイムで更新可能な協調フィルタリング"""
    
    def __init__(self, n_factors=50, learning_rate=0.01):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.user_factors = {}
        self.item_factors = {}
        
    def update(self, user_id, item_id, rating):
        """新しい評価でモデルを更新（SGD）"""
        # 因子を取得（なければ初期化）
        u_factors = self.user_factors.get(user_id, 
                                         np.random.normal(0, 0.1, self.n_factors))
        i_factors = self.item_factors.get(item_id,
                                         np.random.normal(0, 0.1, self.n_factors))
        
        # 予測と誤差
        prediction = np.dot(u_factors, i_factors)
        error = rating - prediction
        
        # SGD更新
        u_factors += self.lr * (error * i_factors - 0.01 * u_factors)
        i_factors += self.lr * (error * u_factors - 0.01 * i_factors)
        
        # 保存
        self.user_factors[user_id] = u_factors
        self.item_factors[item_id] = i_factors
```

## 🎯 実践演習

### 演習3-1: ハイブリッド協調フィルタリング

```python
def exercise_hybrid_cf():
    """
    メモリベースとモデルベースを組み合わせる
    """
    # TODO: 実装
    pass
```

### 演習3-2: コールドスタート問題への対処

```python
def exercise_cold_start():
    """
    新規ユーザー/アイテムへの推薦
    """
    # TODO: 実装
    pass
```

## 📈 評価指標

```python
def evaluate_cf_model(model, test_data):
    """協調フィルタリングの評価"""
    
    # RMSE（Root Mean Square Error）
    predictions = []
    actuals = []
    
    for user_id, item_id, rating in test_data:
        pred = model.predict(user_id, item_id)
        predictions.append(pred)
        actuals.append(rating)
        
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
    
    # MAP（Mean Average Precision）
    # Precision@K
    # Recall@K
    # NDCG（Normalized Discounted Cumulative Gain）
    
    return {
        'rmse': rmse,
        'map': map_score,
        'precision_at_10': precision,
        'recall_at_10': recall,
        'ndcg': ndcg_score
    }
```

## 💡 まとめ

- **メモリベース**: シンプルで解釈可能、但し計算量大
- **モデルベース**: 高速で汎化性能高い、但しブラックボックス
- **SimClusters**: 両者の良いとこ取り、スケーラブルで解釈可能
- **最適化**: ANN、オンライン学習、分散処理が鍵

[→ Chapter 4: 特徴量とランキングへ](chapter04_features_ranking.md)