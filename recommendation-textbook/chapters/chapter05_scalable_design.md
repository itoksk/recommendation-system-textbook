# Chapter 5: スケーラブルな設計 🚀

## 📚 この章で学ぶこと

- 大規模システムの設計原則
- Product Mixerパターン
- 分散処理とキャッシング
- リアルタイム処理の実装

## 🏗️ Twitterスケールの課題

### 数字で見るTwitterのスケール

```python
class TwitterScale:
    """Twitterが扱うデータ規模"""
    
    STATS = {
        'daily_tweets': 500_000_000,      # 1日5億ツイート
        'active_users': 330_000_000,      # 月間アクティブユーザー
        'queries_per_second': 300_000,    # 秒間30万クエリ
        'timeline_loads': 200_000_000,    # 1日2億回のタイムライン表示
        'simclusters': 145_000,           # 14.5万クラスタ
        'features_per_tweet': 6_000,      # ツイートあたり6000特徴量
    }
    
    @classmethod
    def calculate_requirements(cls):
        """必要なリソースを計算"""
        # 1リクエストあたりの処理
        candidates_per_request = 100_000   # 候補ツイート
        light_ranker_time = 0.1           # ms/ツイート
        heavy_ranker_time = 10            # ms/ツイート
        
        # レイテンシ要件
        max_latency = 100  # ms
        
        # 必要な並列度
        parallel_factor = (candidates_per_request * light_ranker_time) / max_latency
        
        print(f"必要な並列度: {parallel_factor:.0f}倍")
        print(f"1秒あたりの処理量: {cls.STATS['queries_per_second'] * candidates_per_request:,}")
```

## 🎭 Product Mixerパターン

### Twitterのコンポーネントベース設計

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import asyncio

class Component(ABC):
    """基本コンポーネント"""
    
    @abstractmethod
    async def execute(self, request: Dict, state: Dict) -> Dict:
        pass

class Pipeline:
    """パイプライン：コンポーネントを組み合わせる"""
    
    def __init__(self, components: List[Component]):
        self.components = components
        
    async def execute(self, request: Dict) -> Dict:
        state = {'request': request}
        
        for component in self.components:
            try:
                state = await component.execute(request, state)
            except Exception as e:
                # エラーハンドリング
                state['error'] = str(e)
                break
                
        return state

class ProductMixer:
    """
    Product Mixer: Twitterのコア設計パターン
    - 再利用可能なコンポーネント
    - 宣言的なパイプライン定義
    - 自動的な監視とロギング
    """
    
    def __init__(self):
        self.pipelines = {}
        
    def register_pipeline(self, name: str, pipeline: Pipeline):
        """パイプラインを登録"""
        self.pipelines[name] = pipeline
        
    async def execute(self, pipeline_name: str, request: Dict) -> Dict:
        """パイプラインを実行"""
        pipeline = self.pipelines.get(pipeline_name)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_name} not found")
            
        # メトリクス収集
        start_time = asyncio.get_event_loop().time()
        
        result = await pipeline.execute(request)
        
        # レイテンシ記録
        latency = asyncio.get_event_loop().time() - start_time
        result['latency_ms'] = latency * 1000
        
        return result

# 実際のコンポーネント実装例

class CandidateSourceComponent(Component):
    """候補生成コンポーネント"""
    
    def __init__(self, sources: List[str]):
        self.sources = sources
        
    async def execute(self, request: Dict, state: Dict) -> Dict:
        # 並列で候補を取得
        tasks = []
        for source in self.sources:
            task = self.fetch_candidates(source, request['user_id'])
            tasks.append(task)
            
        candidates_lists = await asyncio.gather(*tasks)
        
        # 結果をマージ
        all_candidates = []
        for candidates in candidates_lists:
            all_candidates.extend(candidates)
            
        state['candidates'] = all_candidates
        return state
    
    async def fetch_candidates(self, source: str, user_id: str):
        """各ソースから候補を取得"""
        # 実際はRPCやDBアクセス
        await asyncio.sleep(0.01)  # シミュレート
        return [f"{source}_candidate_{i}" for i in range(100)]

class FeatureHydrationComponent(Component):
    """特徴量付与コンポーネント"""
    
    async def execute(self, request: Dict, state: Dict) -> Dict:
        candidates = state['candidates']
        
        # バッチで特徴量を取得
        features = await self.batch_fetch_features(candidates)
        
        # 候補に特徴量を付与
        for candidate, feature in zip(candidates, features):
            candidate['features'] = feature
            
        return state
    
    async def batch_fetch_features(self, candidates):
        """バッチで特徴量取得（効率化）"""
        # 実際は特徴量ストアから取得
        await asyncio.sleep(0.02)
        return [{'feature': i} for i in range(len(candidates))]

class ScoringComponent(Component):
    """スコアリングコンポーネント"""
    
    def __init__(self, model):
        self.model = model
        
    async def execute(self, request: Dict, state: Dict) -> Dict:
        candidates = state['candidates']
        
        # バッチ予測
        scores = await self.model.predict_batch(candidates)
        
        # スコアを付与
        for candidate, score in zip(candidates, scores):
            candidate['score'] = score
            
        # スコア順にソート
        state['candidates'] = sorted(
            candidates, 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        return state

class FilteringComponent(Component):
    """フィルタリングコンポーネント"""
    
    async def execute(self, request: Dict, state: Dict) -> Dict:
        candidates = state['candidates']
        user = request['user']
        
        filtered = []
        seen_authors = set()
        
        for candidate in candidates:
            # 多様性フィルタ
            if candidate['author_id'] in seen_authors:
                continue
                
            # ブロック/ミュートフィルタ
            if candidate['author_id'] in user['blocked_users']:
                continue
                
            filtered.append(candidate)
            seen_authors.add(candidate['author_id'])
            
            if len(filtered) >= 20:
                break
                
        state['candidates'] = filtered
        return state

# パイプラインの構築

def build_timeline_pipeline():
    """タイムラインパイプラインを構築"""
    
    pipeline = Pipeline([
        # 1. 候補生成
        CandidateSourceComponent(sources=[
            'in_network',      # フォローしている人
            'out_of_network',  # フォロー外
            'trending',        # トレンド
        ]),
        
        # 2. 特徴量付与
        FeatureHydrationComponent(),
        
        # 3. スコアリング
        ScoringComponent(model=LightRanker()),
        
        # 4. フィルタリング
        FilteringComponent(),
        
        # 5. 詳細スコアリング（上位のみ）
        ScoringComponent(model=HeavyRanker()),
        
        # 6. 最終調整
        FilteringComponent(),
    ])
    
    return pipeline
```

## ⚡ 分散処理の実装

### 並列処理とシャーディング

```python
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import grpc

class DistributedRanker:
    """分散ランキングシステム"""
    
    def __init__(self, n_shards=100):
        self.n_shards = n_shards
        self.executor = ThreadPoolExecutor(max_workers=50)
        self.shard_clients = self._init_shard_clients()
        
    def _init_shard_clients(self):
        """各シャードへのクライアントを初期化"""
        clients = {}
        for shard_id in range(self.n_shards):
            # gRPCクライアント
            channel = grpc.insecure_channel(f'shard-{shard_id}:50051')
            clients[shard_id] = RankerStub(channel)
        return clients
    
    def get_shard(self, item_id: str) -> int:
        """アイテムのシャードを決定"""
        hash_value = hashlib.md5(item_id.encode()).hexdigest()
        return int(hash_value, 16) % self.n_shards
    
    async def rank_distributed(self, candidates: List[str], user_context: Dict):
        """分散ランキング"""
        
        # 1. 候補をシャードごとに分割
        sharded_candidates = defaultdict(list)
        for candidate in candidates:
            shard_id = self.get_shard(candidate)
            sharded_candidates[shard_id].append(candidate)
            
        # 2. 並列でRPC
        futures = []
        for shard_id, shard_candidates in sharded_candidates.items():
            future = self.executor.submit(
                self._rank_on_shard,
                shard_id,
                shard_candidates,
                user_context
            )
            futures.append(future)
            
        # 3. 結果を収集
        all_results = []
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
            
        # 4. マージしてソート
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        return all_results[:100]
    
    def _rank_on_shard(self, shard_id: int, candidates: List, context: Dict):
        """特定のシャードでランキング"""
        client = self.shard_clients[shard_id]
        
        request = RankRequest(
            candidates=candidates,
            user_context=context
        )
        
        response = client.Rank(request)
        return response.results

class MapReduceRanker:
    """MapReduceスタイルのランキング"""
    
    def map_phase(self, chunk: List, user_context: Dict) -> List:
        """Map: 各チャンクでローカルランキング"""
        local_ranker = LightRanker()
        scored = []
        
        for item in chunk:
            score = local_ranker.score(item, user_context)
            scored.append((item, score))
            
        # ローカルでTop-K
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:100]
    
    def reduce_phase(self, mapped_results: List[List]) -> List:
        """Reduce: 結果をマージ"""
        all_results = []
        for chunk_results in mapped_results:
            all_results.extend(chunk_results)
            
        # グローバルでソート
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:20]
    
    def rank(self, candidates: List, user_context: Dict, chunk_size: int = 1000):
        """MapReduceでランキング"""
        # チャンクに分割
        chunks = [
            candidates[i:i+chunk_size] 
            for i in range(0, len(candidates), chunk_size)
        ]
        
        # Map phase（並列）
        with ThreadPoolExecutor() as executor:
            mapped = list(executor.map(
                lambda chunk: self.map_phase(chunk, user_context),
                chunks
            ))
            
        # Reduce phase
        final_results = self.reduce_phase(mapped)
        
        return final_results
```

## 💾 キャッシング戦略

### 多層キャッシュシステム

```python
import pickle
from functools import lru_cache
import memcache

class MultiLayerCache:
    """
    多層キャッシュ
    L1: プロセス内メモリ（最速）
    L2: Redis/Memcached（共有）
    L3: ディスク/DB（永続）
    """
    
    def __init__(self):
        self.l1_cache = {}  # プロセス内
        self.l2_cache = memcache.Client(['localhost:11211'])  # Memcached
        self.l3_cache = DiskCache()  # ディスク
        
        # TTL設定
        self.ttl_config = {
            'user_timeline': 60,      # 1分
            'tweet_features': 300,    # 5分
            'user_embeddings': 3600,  # 1時間
            'popular_tweets': 30,     # 30秒
        }
        
    def get(self, key: str, cache_type: str = 'default'):
        """キャッシュから取得"""
        
        # L1チェック（最速）
        if key in self.l1_cache:
            return self.l1_cache[key]
            
        # L2チェック
        value = self.l2_cache.get(key)
        if value:
            # L1に昇格
            self.l1_cache[key] = value
            return pickle.loads(value)
            
        # L3チェック
        value = self.l3_cache.get(key)
        if value:
            # L1, L2に昇格
            self.l2_cache.set(key, pickle.dumps(value), 
                            time=self.ttl_config.get(cache_type, 60))
            self.l1_cache[key] = value
            return value
            
        return None
    
    def set(self, key: str, value: Any, cache_type: str = 'default'):
        """キャッシュに保存"""
        ttl = self.ttl_config.get(cache_type, 60)
        
        # 全層に保存
        self.l1_cache[key] = value
        self.l2_cache.set(key, pickle.dumps(value), time=ttl)
        self.l3_cache.set(key, value)
        
    def invalidate(self, key: str):
        """キャッシュ無効化"""
        if key in self.l1_cache:
            del self.l1_cache[key]
        self.l2_cache.delete(key)
        self.l3_cache.delete(key)

class SmartCaching:
    """インテリジェントなキャッシング"""
    
    def __init__(self):
        self.cache = MultiLayerCache()
        self.hit_rate = defaultdict(lambda: {'hits': 0, 'misses': 0})
        
    def get_with_fallback(self, key: str, fetch_func, cache_type: str = 'default'):
        """キャッシュまたはフォールバック"""
        
        # キャッシュチェック
        value = self.cache.get(key, cache_type)
        
        if value is not None:
            self.hit_rate[cache_type]['hits'] += 1
            return value
            
        # キャッシュミス
        self.hit_rate[cache_type]['misses'] += 1
        
        # データ取得
        value = fetch_func()
        
        # キャッシュに保存
        if value is not None:
            self.cache.set(key, value, cache_type)
            
        return value
    
    def precompute_popular(self):
        """人気コンテンツを事前計算"""
        # 人気ツイートを事前計算
        popular_tweets = self.compute_popular_tweets()
        self.cache.set('popular_tweets', popular_tweets, 'popular_tweets')
        
        # 人気ユーザーのタイムラインを事前計算
        for user_id in self.get_popular_users():
            timeline = self.compute_timeline(user_id)
            self.cache.set(f'timeline:{user_id}', timeline, 'user_timeline')
```

## 🌊 ストリーム処理

### リアルタイムパイプライン

```python
from kafka import KafkaConsumer, KafkaProducer
import json

class StreamProcessor:
    """リアルタイムストリーム処理"""
    
    def __init__(self):
        self.consumer = KafkaConsumer(
            'tweet_events',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.processors = {
            'new_tweet': self.process_new_tweet,
            'like': self.process_like,
            'retweet': self.process_retweet,
            'follow': self.process_follow,
        }
        
    def start(self):
        """ストリーム処理を開始"""
        for message in self.consumer:
            event = message.value
            event_type = event['type']
            
            # 適切なプロセッサーを実行
            if event_type in self.processors:
                self.processors[event_type](event)
                
    def process_new_tweet(self, event):
        """新しいツイートを処理"""
        tweet = event['tweet']
        
        # 1. 特徴量を抽出
        features = self.extract_features(tweet)
        
        # 2. 初期スコアリング
        score = self.light_ranker.score(features)
        
        # 3. インデックスに追加
        if score > THRESHOLD:
            self.add_to_index(tweet, score)
            
        # 4. SimClusters埋め込みを初期化
        self.init_tweet_embedding(tweet)
        
        # 5. ファンアウト（フォロワーのタイムラインに追加）
        self.fanout_to_followers(tweet)
        
    def fanout_to_followers(self, tweet):
        """フォロワーにファンアウト"""
        author_id = tweet['author_id']
        followers = self.get_followers(author_id)
        
        # バッチ処理
        batch = []
        for follower_id in followers:
            update = {
                'user_id': follower_id,
                'tweet_id': tweet['id'],
                'timestamp': tweet['created_at']
            }
            batch.append(update)
            
            if len(batch) >= 100:
                self.producer.send('timeline_updates', batch)
                batch = []
                
        # 残りを送信
        if batch:
            self.producer.send('timeline_updates', batch)

class WindowedAggregation:
    """時間窓での集計"""
    
    def __init__(self, window_size_seconds=60):
        self.window_size = window_size_seconds
        self.windows = defaultdict(lambda: defaultdict(int))
        
    def add_event(self, event):
        """イベントを追加"""
        timestamp = event['timestamp']
        window_id = timestamp // self.window_size
        
        # 該当する窓に追加
        self.windows[window_id][event['type']] += 1
        
        # 古い窓を削除
        current_window = time.time() // self.window_size
        expired_windows = [w for w in self.windows if w < current_window - 10]
        for window in expired_windows:
            del self.windows[window]
            
    def get_aggregates(self, n_windows=5):
        """直近n個の窓の集計を取得"""
        current_window = time.time() // self.window_size
        
        aggregates = defaultdict(int)
        for i in range(n_windows):
            window_id = current_window - i
            if window_id in self.windows:
                for event_type, count in self.windows[window_id].items():
                    aggregates[event_type] += count
                    
        return dict(aggregates)
```

## 🎯 演習問題

### 演習5-1: カスタムコンポーネント作成
```python
def create_custom_component():
    """
    Product Mixer用のカスタムコンポーネントを作成
    例：スパムフィルター、言語フィルターなど
    """
    # TODO: 実装
    pass
```

### 演習5-2: キャッシュ効率の改善
```python
def optimize_cache_strategy():
    """
    キャッシュヒット率を改善する戦略を実装
    """
    # TODO: 実装
    pass
```

## 📚 まとめ

- **Product Mixer**: コンポーネントベースで柔軟な設計
- **分散処理**: シャーディングとMapReduceで並列化
- **キャッシング**: 多層キャッシュで高速化
- **ストリーム処理**: リアルタイムでの更新と集計
- **スケーラビリティ**: 水平スケールを前提とした設計

[→ Chapter 6: 実践プロジェクトへ](chapter06_final_project.md)