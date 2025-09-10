# Chapter 4: 特徴量エンジニアリングとランキング 🎯

## 📚 この章で学ぶこと

- TwitterのLight/Heavy Rankerアーキテクチャ
- 特徴量エンジニアリングの実践
- 機械学習モデルの導入
- リアルタイム特徴量の管理

## 🏗️ Twitterの2段階ランキングシステム

### なぜ2段階なのか？

```python
class TwoStageRankingSystem:
    """
    Twitterの2段階ランキングの考え方
    
    問題：
    - 候補が多すぎる（数十万ツイート）
    - Heavy Rankerは高精度だが遅い（1ツイート10ms）
    - リアルタイムレスポンスが必要（< 100ms）
    
    解決：
    1. Light Ranker: 高速フィルタリング（0.1ms/ツイート）
    2. Heavy Ranker: 精密ランキング（10ms/ツイート）
    """
    
    def __init__(self):
        self.light_ranker = LightRanker()
        self.heavy_ranker = HeavyRanker()
        
    def rank(self, candidates, user_context):
        # Stage 1: Light Ranking（数十万 → 1000）
        light_scored = self.light_ranker.score_batch(candidates)
        top_candidates = self.filter_top_k(light_scored, k=1000)
        
        # Stage 2: Heavy Ranking（1000 → 20）
        final_scores = self.heavy_ranker.score_batch(top_candidates, user_context)
        final_ranking = self.sort_by_score(final_scores)
        
        return final_ranking[:20]
```

## 🪶 Light Ranker: 高速スコアリング

### Light Rankerの実装

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class LightRanker:
    """
    Twitterの軽量ランカー
    - ロジスティック回帰（古いが高速）
    - 最小限の特徴量（〜100次元）
    - バッチ処理で高速化
    """
    
    def __init__(self):
        self.model = LogisticRegression()
        self.feature_extractors = [
            self.extract_engagement_features,
            self.extract_content_features,
            self.extract_author_features,
            self.extract_temporal_features
        ]
        
    def extract_features(self, tweet, user):
        """軽量な特徴量抽出"""
        features = []
        
        for extractor in self.feature_extractors:
            features.extend(extractor(tweet, user))
            
        return np.array(features)
    
    def extract_engagement_features(self, tweet, user):
        """エンゲージメント特徴（リアルタイム）"""
        return [
            tweet.like_count,
            tweet.retweet_count,
            tweet.reply_count,
            np.log1p(tweet.like_count),  # 対数変換
            np.log1p(tweet.retweet_count),
            tweet.like_count / (tweet.impression_count + 1),  # CTR
            tweet.retweet_count / (tweet.impression_count + 1),
        ]
    
    def extract_content_features(self, tweet, user):
        """コンテンツ特徴（静的）"""
        return [
            int(tweet.has_media),
            int(tweet.has_url),
            int(tweet.has_hashtag),
            int(tweet.is_reply),
            int(tweet.is_retweet),
            len(tweet.text),
            tweet.text.count('!'),
            tweet.text.count('?'),
            int('http' in tweet.text),
        ]
    
    def extract_author_features(self, tweet, user):
        """著者特徴"""
        author = tweet.author
        return [
            author.follower_count,
            author.following_count,
            np.log1p(author.follower_count),
            author.follower_count / (author.following_count + 1),
            author.verified,
            author.tweet_count,
            author.account_age_days,
            self.get_author_quality_score(author),  # Tweepcred
        ]
    
    def extract_temporal_features(self, tweet, user):
        """時間特徴"""
        age_minutes = (datetime.now() - tweet.created_at).total_seconds() / 60
        
        return [
            age_minutes,
            np.exp(-age_minutes / 1440),  # 24時間減衰
            int(tweet.created_at.hour),  # 投稿時間
            int(tweet.created_at.weekday()),  # 曜日
            int(age_minutes < 60),  # 1時間以内
            int(age_minutes < 1440),  # 24時間以内
        ]
    
    def score_batch(self, tweets, user, batch_size=1000):
        """バッチ処理で高速化"""
        all_scores = []
        
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i+batch_size]
            
            # ベクトル化
            feature_matrix = np.array([
                self.extract_features(tweet, user) for tweet in batch
            ])
            
            # 一括予測
            scores = self.model.predict_proba(feature_matrix)[:, 1]
            all_scores.extend(scores)
            
        return all_scores
```

## 🏋️ Heavy Ranker: 精密スコアリング

### ClemNetアーキテクチャ（Twitter実装）

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class HeavyRanker:
    """
    TwitterのHeavy Ranker（ClemNet）
    - 深層学習モデル
    - 6000+次元の特徴量
    - マルチタスク学習
    """
    
    def __init__(self, feature_dim=6000):
        self.feature_dim = feature_dim
        self.model = self.build_clemnet()
        
    def build_clemnet(self):
        """
        ClemNet: Twitterの独自アーキテクチャ
        Conv1D + Dense + Residual接続
        """
        inputs = layers.Input(shape=(self.feature_dim,))
        
        # 特徴量を3次元に変形（Conv1D用）
        x = layers.Reshape((self.feature_dim, 1))(inputs)
        
        # Block 1
        x = self.clemnet_block(x, filters=1024, kernel_size=3)
        x = self.clemnet_block(x, filters=512, kernel_size=3)
        x = self.clemnet_block(x, filters=256, kernel_size=3)
        x = self.clemnet_block(x, filters=128, kernel_size=3)
        
        # Flatten
        x = layers.Flatten()(x)
        
        # マルチタスク出力
        # 1. クリック予測
        click_output = layers.Dense(1, activation='sigmoid', name='click')(x)
        
        # 2. エンゲージメント予測
        engagement_output = layers.Dense(1, activation='sigmoid', name='engagement')(x)
        
        # 3. 滞在時間予測
        dwell_output = layers.Dense(1, activation='linear', name='dwell_time')(x)
        
        model = Model(
            inputs=inputs,
            outputs=[click_output, engagement_output, dwell_output]
        )
        
        # 重み付き損失
        model.compile(
            optimizer='adam',
            loss={
                'click': 'binary_crossentropy',
                'engagement': 'binary_crossentropy',
                'dwell_time': 'mse'
            },
            loss_weights={
                'click': 0.5,
                'engagement': 0.3,
                'dwell_time': 0.2
            }
        )
        
        return model
    
    def clemnet_block(self, x, filters, kernel_size):
        """ClemNetブロック"""
        residual = x
        
        # ChannelWise Dense
        x = layers.Dense(filters, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Conv1D
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Residual接続
        if residual.shape[-1] != filters:
            residual = layers.Conv1D(filters, 1)(residual)
        
        x = layers.Add()([x, residual])
        return x
    
    def extract_rich_features(self, tweet, user):
        """6000次元の豊富な特徴量"""
        features = []
        
        # 1. テキスト埋め込み（BERT等）
        features.extend(self.get_text_embedding(tweet.text))  # 768次元
        
        # 2. ユーザー埋め込み（SimClusters）
        features.extend(self.get_user_embedding(user))  # 145次元
        
        # 3. ツイート埋め込み（SimClusters）
        features.extend(self.get_tweet_embedding(tweet))  # 145次元
        
        # 4. グラフ特徴（Real Graph）
        features.extend(self.get_graph_features(user, tweet.author))  # 100次元
        
        # 5. 相互作用特徴
        features.extend(self.get_interaction_features(user, tweet))  # 500次元
        
        # 6. 統計特徴
        features.extend(self.get_statistical_features(tweet))  # 1000次元
        
        # ... 合計6000次元
        
        return np.array(features)
```

## 🔧 特徴量エンジニアリング実践

### 特徴量の種類と実装

```python
class FeatureEngineering:
    """包括的な特徴量エンジニアリング"""
    
    def __init__(self):
        self.feature_groups = {
            'static': self.extract_static_features,
            'dynamic': self.extract_dynamic_features,
            'user': self.extract_user_features,
            'context': self.extract_context_features,
            'interaction': self.extract_interaction_features,
            'embedding': self.extract_embedding_features
        }
        
    def extract_all_features(self, tweet, user, context):
        """全特徴量を抽出"""
        all_features = {}
        
        for group_name, extractor in self.feature_groups.items():
            features = extractor(tweet, user, context)
            all_features[group_name] = features
            
        return self.flatten_features(all_features)
    
    def extract_static_features(self, tweet, user, context):
        """静的特徴（変化しない）"""
        return {
            # コンテンツ
            'text_length': len(tweet.text),
            'word_count': len(tweet.text.split()),
            'has_media': int(tweet.media is not None),
            'media_type': self.encode_media_type(tweet.media),
            'has_url': int('http' in tweet.text),
            'url_count': tweet.text.count('http'),
            'hashtag_count': tweet.text.count('#'),
            'mention_count': tweet.text.count('@'),
            
            # 言語
            'language': self.encode_language(tweet.language),
            'is_english': int(tweet.language == 'en'),
            
            # タイプ
            'is_reply': int(tweet.in_reply_to is not None),
            'is_retweet': int(tweet.retweeted_status is not None),
            'is_quote': int(tweet.quoted_status is not None),
            
            # テキスト品質
            'text_entropy': self.calculate_entropy(tweet.text),
            'readability_score': self.calculate_readability(tweet.text),
            'sentiment_score': self.calculate_sentiment(tweet.text),
        }
    
    def extract_dynamic_features(self, tweet, user, context):
        """動的特徴（時間で変化）"""
        age_seconds = (context.current_time - tweet.created_at).total_seconds()
        
        return {
            # エンゲージメント
            'like_count': tweet.like_count,
            'retweet_count': tweet.retweet_count,
            'reply_count': tweet.reply_count,
            'quote_count': tweet.quote_count,
            
            # 正規化
            'likes_per_hour': tweet.like_count / (age_seconds / 3600 + 1),
            'retweets_per_hour': tweet.retweet_count / (age_seconds / 3600 + 1),
            
            # 比率
            'engagement_rate': (tweet.like_count + tweet.retweet_count) / (tweet.impression_count + 1),
            'like_rate': tweet.like_count / (tweet.impression_count + 1),
            'retweet_rate': tweet.retweet_count / (tweet.impression_count + 1),
            
            # 時間
            'age_seconds': age_seconds,
            'age_hours': age_seconds / 3600,
            'age_days': age_seconds / 86400,
            
            # 時間減衰
            'time_decay_1h': np.exp(-age_seconds / 3600),
            'time_decay_24h': np.exp(-age_seconds / 86400),
            'time_decay_7d': np.exp(-age_seconds / 604800),
        }
    
    def extract_user_features(self, tweet, user, context):
        """ユーザー特徴"""
        author = tweet.author
        
        return {
            # 基本統計
            'author_follower_count': author.follower_count,
            'author_following_count': author.following_count,
            'author_tweet_count': author.tweet_count,
            'author_listed_count': author.listed_count,
            
            # 比率
            'author_follower_following_ratio': author.follower_count / (author.following_count + 1),
            'author_tweets_per_day': author.tweet_count / (author.account_age_days + 1),
            
            # 品質指標
            'author_verified': int(author.verified),
            'author_has_bio': int(len(author.bio) > 0),
            'author_has_location': int(author.location is not None),
            'author_has_url': int(author.url is not None),
            
            # Tweepcred
            'author_reputation': self.get_tweepcred_score(author),
            
            # ユーザーとの関係
            'user_follows_author': int(author.id in user.following),
            'author_follows_user': int(user.id in author.following),
            'mutual_follow': int(author.id in user.following and user.id in author.following),
        }
    
    def extract_interaction_features(self, tweet, user, context):
        """相互作用特徴"""
        # ユーザーの過去の行動
        user_history = self.get_user_history(user)
        
        return {
            # 過去のインタラクション
            'user_liked_author_before': self.count_past_likes(user, tweet.author),
            'user_retweeted_author_before': self.count_past_retweets(user, tweet.author),
            'user_replied_author_before': self.count_past_replies(user, tweet.author),
            
            # トピックの一致
            'topic_similarity': self.calculate_topic_similarity(user_history, tweet),
            
            # 時間パターン
            'user_active_now': int(self.is_user_active_time(user, context.current_time)),
            'matches_user_schedule': self.matches_schedule(user, tweet.created_at),
            
            # ソーシャルプルーフ
            'friends_who_liked': self.count_friends_engagement(user, tweet, 'like'),
            'friends_who_retweeted': self.count_friends_engagement(user, tweet, 'retweet'),
        }
    
    def extract_embedding_features(self, tweet, user, context):
        """埋め込み特徴"""
        return {
            # SimClusters
            'simclusters_dot_product': np.dot(
                self.get_user_simclusters(user),
                self.get_tweet_simclusters(tweet)
            ),
            
            # セマンティック埋め込み
            'semantic_similarity': self.calculate_semantic_similarity(
                user.interests,
                tweet.content
            ),
            
            # グラフ埋め込み
            'graph_distance': self.get_graph_distance(user, tweet.author),
        }
```

## 🚀 リアルタイム特徴量管理

### 特徴量ストアの実装

```python
import redis
from datetime import datetime, timedelta

class RealtimeFeatureStore:
    """
    リアルタイム特徴量の管理
    Twitterは Memcached + Manhattan を使用
    """
    
    def __init__(self):
        self.redis = redis.Redis(decode_responses=True)
        self.ttl = 3600  # 1時間
        
    def update_engagement(self, tweet_id, engagement_type, user_id):
        """エンゲージメントをリアルタイム更新"""
        # カウンタをインクリメント
        key = f"tweet:{tweet_id}:{engagement_type}"
        self.redis.incr(key)
        
        # 時間窓での集計
        window_key = f"tweet:{tweet_id}:{engagement_type}:1h:{datetime.now().hour}"
        self.redis.incr(window_key)
        self.redis.expire(window_key, self.ttl)
        
        # ユーザー別の記録
        user_key = f"user:{user_id}:{engagement_type}:history"
        self.redis.lpush(user_key, tweet_id)
        self.redis.ltrim(user_key, 0, 999)  # 最新1000件
        
    def get_realtime_features(self, tweet_id):
        """リアルタイム特徴量を取得"""
        features = {}
        
        # 総計
        features['likes_total'] = int(self.redis.get(f"tweet:{tweet_id}:like") or 0)
        features['retweets_total'] = int(self.redis.get(f"tweet:{tweet_id}:retweet") or 0)
        
        # 時間窓
        current_hour = datetime.now().hour
        features['likes_1h'] = int(
            self.redis.get(f"tweet:{tweet_id}:like:1h:{current_hour}") or 0
        )
        
        # 速度（エンゲージメント/時間）
        features['engagement_velocity'] = self.calculate_velocity(tweet_id)
        
        # トレンド性
        features['is_trending'] = self.is_trending(tweet_id)
        
        return features
    
    def calculate_velocity(self, tweet_id):
        """エンゲージメント速度を計算"""
        # 過去1時間のエンゲージメント履歴
        history = []
        for i in range(60):  # 過去60分
            minute = (datetime.now() - timedelta(minutes=i)).minute
            key = f"tweet:{tweet_id}:engagement:minute:{minute}"
            count = int(self.redis.get(key) or 0)
            history.append(count)
            
        # 加速度を計算
        if len(history) > 1:
            recent = sum(history[:10])
            past = sum(history[10:20])
            velocity = (recent - past) / (past + 1)
            return velocity
        return 0
    
    def is_trending(self, tweet_id):
        """トレンド判定"""
        velocity = self.calculate_velocity(tweet_id)
        total_engagement = int(self.redis.get(f"tweet:{tweet_id}:engagement") or 0)
        
        # 急速に伸びている && 一定以上のエンゲージメント
        return velocity > 2.0 and total_engagement > 100
```

## 📊 学習とEvaluation

### オンライン学習パイプライン

```python
class OnlineLearningPipeline:
    """
    継続的な学習と改善
    """
    
    def __init__(self):
        self.model = HeavyRanker()
        self.feature_store = RealtimeFeatureStore()
        self.training_buffer = []
        
    def collect_training_data(self, impression):
        """インプレッションから学習データを収集"""
        features = self.extract_features(impression)
        
        # ラベル（ユーザーの行動）
        labels = {
            'clicked': impression.clicked,
            'engaged': impression.liked or impression.retweeted,
            'dwell_time': impression.dwell_time_seconds
        }
        
        self.training_buffer.append((features, labels))
        
        # バッファが満たされたら学習
        if len(self.training_buffer) >= 1000:
            self.train_batch()
            
    def train_batch(self):
        """バッチ学習"""
        X = np.array([f for f, _ in self.training_buffer])
        y = {
            'click': np.array([l['clicked'] for _, l in self.training_buffer]),
            'engagement': np.array([l['engaged'] for _, l in self.training_buffer]),
            'dwell_time': np.array([l['dwell_time'] for _, l in self.training_buffer])
        }
        
        # 学習
        self.model.model.fit(X, y, epochs=1, batch_size=32)
        
        # バッファをクリア
        self.training_buffer = []
        
    def evaluate(self, test_data):
        """モデル評価"""
        metrics = {
            'auc_click': self.calculate_auc(test_data, 'click'),
            'auc_engagement': self.calculate_auc(test_data, 'engagement'),
            'rmse_dwell': self.calculate_rmse(test_data, 'dwell_time'),
            'ndcg@10': self.calculate_ndcg(test_data, k=10)
        }
        
        return metrics
```

## 🎯 演習問題

### 演習4-1: カスタム特徴量の作成
```python
def create_custom_features(tweet, user):
    """
    独自の特徴量を3つ作成してください
    例：絵文字の数、質問の有無、など
    """
    # TODO: 実装
    pass
```

### 演習4-2: Light Rankerの最適化
```python
def optimize_light_ranker(training_data):
    """
    Light Rankerのハイパーパラメータを最適化
    """
    # TODO: 実装
    pass
```

## 📚 まとめ

- **2段階ランキング**: 速度と精度のトレードオフを解決
- **Light Ranker**: シンプルで高速、最小限の特徴量
- **Heavy Ranker**: 複雑で精密、豊富な特徴量
- **特徴量エンジニアリング**: 静的/動的/リアルタイムの組み合わせ
- **継続的学習**: オンラインで改善し続ける

[→ Chapter 5: スケーラブルな設計へ](chapter05_scalable_design.md)