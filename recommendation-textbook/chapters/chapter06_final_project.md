# Chapter 6: 実践プロジェクト - ミニTwitterを作ろう！ 🐦

## 🎯 この章のゴール

これまで学んだ全ての技術を統合して、実際に動く**ミニTwitter推薦システム**を構築します！

- SimClustersの実装
- Light/Heavy Rankerの構築
- リアルタイム処理
- A/Bテストの実装
- パフォーマンス最適化

## 🏗️ システムアーキテクチャ

```python
class MiniTwitterArchitecture:
    """
    ミニTwitterの全体アーキテクチャ
    
    コンポーネント：
    1. データ層（ユーザー、ツイート、インタラクション）
    2. 特徴量層（静的、動的、リアルタイム）
    3. 推薦層（SimClusters、ランキング）
    4. 配信層（タイムライン生成）
    5. 評価層（A/Bテスト、メトリクス）
    """
    
    def __init__(self):
        # データストア
        self.user_store = UserStore()
        self.tweet_store = TweetStore()
        self.interaction_store = InteractionStore()
        
        # 推薦システム
        self.simclusters = SimClustersEngine()
        self.light_ranker = LightRankingEngine()
        self.heavy_ranker = HeavyRankingEngine()
        
        # リアルタイム処理
        self.stream_processor = StreamProcessor()
        self.feature_store = FeatureStore()
        
        # 配信
        self.timeline_builder = TimelineBuilder()
        
        # 評価
        self.ab_test_manager = ABTestManager()
        self.metrics_collector = MetricsCollector()
```

## 📦 完全実装：ミニTwitterシステム

### 1. データモデル

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Set, Optional
import uuid

@dataclass
class User:
    """ユーザーモデル"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    name: str = ""
    bio: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    follower_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    verified: bool = False
    
    # 推薦用の属性
    simcluster_embedding: List[float] = field(default_factory=list)
    tweepcred_score: float = 50.0
    interests: List[str] = field(default_factory=list)
    
@dataclass
class Tweet:
    """ツイートモデル"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    author_id: str = ""
    content: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    # エンゲージメント
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    impression_count: int = 0
    
    # コンテンツ特徴
    has_media: bool = False
    has_url: bool = False
    language: str = "ja"
    
    # 推薦用
    simcluster_embedding: List[float] = field(default_factory=list)
    quality_score: float = 0.0
    
@dataclass
class Interaction:
    """インタラクションモデル"""
    user_id: str
    tweet_id: str
    type: str  # 'like', 'retweet', 'reply', 'impression'
    timestamp: datetime = field(default_factory=datetime.now)
    dwell_time_seconds: float = 0.0
```

### 2. SimClustersエンジン

```python
import numpy as np
from sklearn.cluster import SpectralClustering
from collections import defaultdict

class SimClustersEngine:
    """SimClusters実装"""
    
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.known_for = {}  # producer -> cluster
        self.interested_in = {}  # consumer -> cluster_vector
        self.tweet_clusters = {}  # tweet -> cluster_vector
        
    def train(self, follow_graph: Dict[str, Set[str]], interactions: List[Interaction]):
        """SimClustersモデルを学習"""
        print("🔨 SimClusters学習開始...")
        
        # Step 1: Producer類似度グラフを構築
        producer_similarity = self._build_producer_similarity(follow_graph)
        
        # Step 2: スペクトラルクラスタリング
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        producers = list(producer_similarity.keys())
        similarity_matrix = self._create_similarity_matrix(producer_similarity)
        
        labels = clustering.fit_predict(similarity_matrix)
        
        # Step 3: KnownFor行列
        for producer, label in zip(producers, labels):
            self.known_for[producer] = label
            
        # Step 4: InterestedIn行列
        for consumer, followed in follow_graph.items():
            interest_vector = np.zeros(self.n_clusters)
            for producer in followed:
                if producer in self.known_for:
                    cluster = self.known_for[producer]
                    interest_vector[cluster] += 1
                    
            # 正規化
            if interest_vector.sum() > 0:
                interest_vector /= interest_vector.sum()
            self.interested_in[consumer] = interest_vector
            
        # Step 5: Tweet埋め込み
        self._compute_tweet_embeddings(interactions)
        
        print(f"✅ SimClusters学習完了: {self.n_clusters}クラスタ")
        
    def _build_producer_similarity(self, follow_graph):
        """Producer間の類似度を計算"""
        producer_followers = defaultdict(set)
        
        for consumer, producers in follow_graph.items():
            for producer in producers:
                producer_followers[producer].add(consumer)
                
        similarities = {}
        producers = list(producer_followers.keys())
        
        for i, p1 in enumerate(producers):
            for p2 in producers[i+1:]:
                followers1 = producer_followers[p1]
                followers2 = producer_followers[p2]
                
                if followers1 and followers2:
                    jaccard = len(followers1 & followers2) / len(followers1 | followers2)
                    if jaccard > 0.01:
                        similarities[(p1, p2)] = jaccard
                        
        return similarities
    
    def _compute_tweet_embeddings(self, interactions):
        """ツイート埋め込みを計算"""
        tweet_likes = defaultdict(list)
        
        for interaction in interactions:
            if interaction.type == 'like':
                tweet_likes[interaction.tweet_id].append(interaction.user_id)
                
        for tweet_id, users in tweet_likes.items():
            embedding = np.zeros(self.n_clusters)
            
            for user_id in users:
                if user_id in self.interested_in:
                    embedding += self.interested_in[user_id]
                    
            # 正規化
            if embedding.sum() > 0:
                embedding /= embedding.sum()
                
            self.tweet_clusters[tweet_id] = embedding
    
    def get_recommendations(self, user_id: str, n: int = 20) -> List[tuple]:
        """ユーザーに推薦"""
        if user_id not in self.interested_in:
            return []
            
        user_interest = self.interested_in[user_id]
        recommendations = []
        
        for tweet_id, tweet_embedding in self.tweet_clusters.items():
            score = np.dot(user_interest, tweet_embedding)
            recommendations.append((tweet_id, score))
            
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]
```

### 3. ランキングエンジン

```python
class LightRankingEngine:
    """軽量ランキングエンジン"""
    
    def __init__(self):
        self.feature_extractors = [
            self._extract_engagement_features,
            self._extract_content_features,
            self._extract_temporal_features,
        ]
        
    def score(self, tweet: Tweet, user: User, context: Dict) -> float:
        """軽量スコアリング"""
        features = []
        
        for extractor in self.feature_extractors:
            features.extend(extractor(tweet, user, context))
            
        # 簡単な線形結合
        weights = [0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
        score = sum(f * w for f, w in zip(features, weights))
        
        return score
    
    def _extract_engagement_features(self, tweet, user, context):
        """エンゲージメント特徴"""
        return [
            tweet.like_count / (tweet.impression_count + 1),
            tweet.retweet_count / (tweet.impression_count + 1),
            np.log1p(tweet.like_count),
            np.log1p(tweet.retweet_count),
        ]
    
    def _extract_content_features(self, tweet, user, context):
        """コンテンツ特徴"""
        return [
            float(tweet.has_media),
            float(tweet.has_url),
            min(1.0, len(tweet.content) / 280),
        ]
    
    def _extract_temporal_features(self, tweet, user, context):
        """時間特徴"""
        age_hours = (context['current_time'] - tweet.created_at).total_seconds() / 3600
        return [
            np.exp(-age_hours / 24),  # 24時間減衰
            float(age_hours < 1),     # 1時間以内
        ]

class HeavyRankingEngine:
    """重量ランキングエンジン（詳細版）"""
    
    def __init__(self):
        self.light_ranker = LightRankingEngine()
        
    def score(self, tweet: Tweet, user: User, context: Dict) -> float:
        """詳細スコアリング"""
        # Light Rankerのスコア
        light_score = self.light_ranker.score(tweet, user, context)
        
        # 追加の複雑な特徴
        author_score = self._get_author_score(tweet, user)
        similarity_score = self._get_similarity_score(tweet, user)
        social_proof_score = self._get_social_proof(tweet, user, context)
        
        # 重み付き結合
        final_score = (
            light_score * 0.3 +
            author_score * 0.3 +
            similarity_score * 0.2 +
            social_proof_score * 0.2
        )
        
        return final_score
    
    def _get_author_score(self, tweet, user):
        """著者スコア（Tweepcred等）"""
        # 簡略版
        return tweet.author.tweepcred_score / 100
    
    def _get_similarity_score(self, tweet, user):
        """類似度スコア"""
        if len(tweet.simcluster_embedding) and len(user.simcluster_embedding):
            return np.dot(tweet.simcluster_embedding, user.simcluster_embedding)
        return 0.5
    
    def _get_social_proof(self, tweet, user, context):
        """ソーシャルプルーフ"""
        # フォローしている人がいいねしているか等
        friends_who_liked = context.get('friends_engagements', {}).get(tweet.id, 0)
        return min(1.0, friends_who_liked / 10)
```

### 4. タイムラインビルダー

```python
class TimelineBuilder:
    """タイムライン構築"""
    
    def __init__(self, simclusters: SimClustersEngine,
                 light_ranker: LightRankingEngine,
                 heavy_ranker: HeavyRankingEngine):
        self.simclusters = simclusters
        self.light_ranker = light_ranker
        self.heavy_ranker = heavy_ranker
        
    def build_timeline(self, user: User, algorithm: str = 'hybrid') -> List[Tweet]:
        """タイムラインを構築"""
        
        if algorithm == 'chronological':
            return self._build_chronological(user)
        elif algorithm == 'simclusters':
            return self._build_simclusters(user)
        elif algorithm == 'hybrid':
            return self._build_hybrid(user)
            
    def _build_hybrid(self, user: User) -> List[Tweet]:
        """ハイブリッドタイムライン（Twitterの"For You"）"""
        
        # Step 1: 候補生成（複数ソース）
        candidates = []
        
        # In-network（フォローしている人）
        in_network = self._get_in_network_tweets(user)
        candidates.extend(in_network[:500])
        
        # Out-of-network（SimClusters）
        simcluster_recs = self.simclusters.get_recommendations(user.id, 500)
        out_network = [self.get_tweet(tid) for tid, _ in simcluster_recs]
        candidates.extend(out_network)
        
        # Trending
        trending = self._get_trending_tweets()
        candidates.extend(trending[:100])
        
        # Step 2: Light Ranking
        context = self._create_context(user)
        light_scores = []
        
        for tweet in candidates:
            score = self.light_ranker.score(tweet, user, context)
            light_scores.append((tweet, score))
            
        # Top 500を選択
        light_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [tweet for tweet, _ in light_scores[:500]]
        
        # Step 3: Heavy Ranking（上位のみ）
        final_scores = []
        
        for tweet in top_candidates[:100]:  # 上位100件のみ
            score = self.heavy_ranker.score(tweet, user, context)
            final_scores.append((tweet, score))
            
        # Light Rankerの結果も含める
        for tweet, light_score in light_scores[100:500]:
            final_scores.append((tweet, light_score * 0.7))  # 重み下げ
            
        # Step 4: 最終ソート
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: 多様性フィルタ
        timeline = self._apply_diversity_filter(
            [tweet for tweet, _ in final_scores]
        )
        
        return timeline[:20]
    
    def _apply_diversity_filter(self, tweets: List[Tweet]) -> List[Tweet]:
        """多様性フィルタ"""
        filtered = []
        seen_authors = set()
        seen_topics = set()
        
        for tweet in tweets:
            # 同じ著者の制限
            if tweet.author_id in seen_authors:
                if len([t for t in filtered if t.author_id == tweet.author_id]) >= 2:
                    continue
                    
            # トピックの多様性
            # （簡略化）
            
            filtered.append(tweet)
            seen_authors.add(tweet.author_id)
            
            if len(filtered) >= 50:
                break
                
        return filtered
```

### 5. A/Bテストシステム

```python
import hashlib
from enum import Enum

class ExperimentGroup(Enum):
    CONTROL = "control"
    TREATMENT = "treatment"

class ABTestManager:
    """A/Bテスト管理"""
    
    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(lambda: {
            'control': {'impressions': 0, 'engagements': 0},
            'treatment': {'impressions': 0, 'engagements': 0}
        })
        
    def create_experiment(self, name: str, treatment_percentage: float = 0.1):
        """実験を作成"""
        self.experiments[name] = {
            'name': name,
            'treatment_percentage': treatment_percentage,
            'start_time': datetime.now(),
            'is_active': True
        }
        
    def get_user_group(self, user_id: str, experiment_name: str) -> ExperimentGroup:
        """ユーザーの実験グループを決定"""
        if experiment_name not in self.experiments:
            return ExperimentGroup.CONTROL
            
        experiment = self.experiments[experiment_name]
        if not experiment['is_active']:
            return ExperimentGroup.CONTROL
            
        # 安定したハッシュで割り当て
        hash_input = f"{user_id}:{experiment_name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        percentage = (hash_value % 100) / 100
        
        if percentage < experiment['treatment_percentage']:
            return ExperimentGroup.TREATMENT
        else:
            return ExperimentGroup.CONTROL
    
    def log_impression(self, user_id: str, tweet_id: str, experiment_name: str):
        """インプレッションを記録"""
        group = self.get_user_group(user_id, experiment_name)
        self.results[experiment_name][group.value]['impressions'] += 1
        
    def log_engagement(self, user_id: str, tweet_id: str, 
                       engagement_type: str, experiment_name: str):
        """エンゲージメントを記録"""
        group = self.get_user_group(user_id, experiment_name)
        self.results[experiment_name][group.value]['engagements'] += 1
        
    def get_results(self, experiment_name: str) -> Dict:
        """実験結果を取得"""
        if experiment_name not in self.results:
            return {}
            
        results = self.results[experiment_name]
        
        # CTR計算
        control_ctr = (results['control']['engagements'] / 
                      max(1, results['control']['impressions']))
        treatment_ctr = (results['treatment']['engagements'] / 
                        max(1, results['treatment']['impressions']))
        
        # 統計的有意性の計算（簡略版）
        lift = ((treatment_ctr - control_ctr) / max(0.001, control_ctr)) * 100
        
        return {
            'control': {
                'impressions': results['control']['impressions'],
                'engagements': results['control']['engagements'],
                'ctr': control_ctr
            },
            'treatment': {
                'impressions': results['treatment']['impressions'],
                'engagements': results['treatment']['engagements'],
                'ctr': treatment_ctr
            },
            'lift_percentage': lift,
            'is_significant': abs(lift) > 5  # 簡略化
        }
```

### 6. パフォーマンス最適化

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cachetools

class OptimizedTimelineBuilder:
    """最適化されたタイムラインビルダー"""
    
    def __init__(self):
        # キャッシュ
        self.timeline_cache = cachetools.TTLCache(maxsize=10000, ttl=60)
        self.feature_cache = cachetools.LRUCache(maxsize=100000)
        
        # 並列処理
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def build_timeline_optimized(self, user_id: str) -> List[Tweet]:
        """最適化されたタイムライン構築"""
        
        # キャッシュチェック
        cache_key = f"timeline:{user_id}"
        if cache_key in self.timeline_cache:
            return self.timeline_cache[cache_key]
            
        # 並列で候補取得
        tasks = [
            self._fetch_in_network_async(user_id),
            self._fetch_simclusters_async(user_id),
            self._fetch_trending_async()
        ]
        
        results = await asyncio.gather(*tasks)
        candidates = []
        for result in results:
            candidates.extend(result)
            
        # バッチスコアリング
        scored = await self._batch_score_async(candidates, user_id)
        
        # 結果をキャッシュ
        timeline = scored[:20]
        self.timeline_cache[cache_key] = timeline
        
        return timeline
    
    async def _batch_score_async(self, tweets: List[Tweet], user_id: str):
        """バッチスコアリング"""
        # 特徴量を並列抽出
        feature_tasks = []
        for tweet in tweets:
            task = self._extract_features_async(tweet, user_id)
            feature_tasks.append(task)
            
        features = await asyncio.gather(*feature_tasks)
        
        # バッチ予測
        scores = self._batch_predict(features)
        
        # ソート
        scored_tweets = list(zip(tweets, scores))
        scored_tweets.sort(key=lambda x: x[1], reverse=True)
        
        return [tweet for tweet, _ in scored_tweets]
```

## 🎮 統合テスト

```python
def run_integrated_test():
    """統合テスト"""
    print("🚀 ミニTwitter統合テスト開始")
    
    # システム初期化
    system = MiniTwitterArchitecture()
    
    # テストデータ作成
    users = create_test_users(100)
    tweets = create_test_tweets(1000)
    interactions = create_test_interactions(10000)
    
    # SimClusters学習
    follow_graph = create_follow_graph(users)
    system.simclusters.train(follow_graph, interactions)
    
    # A/Bテスト設定
    system.ab_test_manager.create_experiment(
        "heavy_ranker_v2",
        treatment_percentage=0.1
    )
    
    # タイムライン生成テスト
    for user in users[:10]:
        timeline = system.timeline_builder.build_timeline(user, 'hybrid')
        print(f"User {user.username}: {len(timeline)} tweets")
        
        # メトリクス記録
        for tweet in timeline:
            system.ab_test_manager.log_impression(
                user.id, tweet.id, "heavy_ranker_v2"
            )
    
    # 結果表示
    results = system.ab_test_manager.get_results("heavy_ranker_v2")
    print(f"A/Bテスト結果: {results}")
    
    print("✅ 統合テスト完了！")

if __name__ == "__main__":
    run_integrated_test()
```

## 🎯 最終課題

### 課題1: 独自の推薦アルゴリズム
あなた独自の推薦アルゴリズムを実装してください。

### 課題2: リアルタイム更新
WebSocketを使ってリアルタイムタイムラインを実装してください。

### 課題3: 可視化ダッシュボード
推薦システムの動作を可視化するダッシュボードを作成してください。

## 🎊 おめでとう！

これで、X(Twitter)レベルの推薦システムの基礎を全てマスターしました！

学んだこと：
- ✅ 推薦システムの基礎理論
- ✅ 協調フィルタリングとSimClusters
- ✅ 特徴量エンジニアリングとランキング
- ✅ スケーラブルな設計
- ✅ 実践的な実装

次のステップ：
- 実際のデータで試す
- 機械学習モデルを深める
- プロダクション環境へのデプロイ

頑張ってください！ 🚀