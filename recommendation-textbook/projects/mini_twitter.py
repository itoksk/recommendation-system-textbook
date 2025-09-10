#!/usr/bin/env python3
"""
ミニTwitter - 教育用の推薦システム実装
X(Twitter)のアルゴリズムを簡略化して実装
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from collections import defaultdict, Counter
import pickle
import os


class MiniTwitter:
    """
    教育用のミニTwitterシステム
    実際のTwitterの主要コンポーネントを簡略化して実装
    """
    
    def __init__(self):
        # データストア
        self.users = {}
        self.tweets = {}
        self.follows = defaultdict(set)  # user -> {followed_users}
        self.followers = defaultdict(set)  # user -> {followers}
        self.likes = defaultdict(set)  # tweet -> {users}
        self.retweets = defaultdict(set)  # tweet -> {users}
        
        # 推薦システムコンポーネント
        self.simclusters = None
        self.tweepcred_scores = {}
        self.real_graph = defaultdict(lambda: defaultdict(float))
        
        # 統計情報
        self.stats = {
            'total_users': 0,
            'total_tweets': 0,
            'total_likes': 0,
            'total_retweets': 0
        }
        
    def create_user(self, username: str, name: str, bio: str = "") -> str:
        """新規ユーザーを作成"""
        user_id = f"u_{len(self.users)}"
        self.users[user_id] = {
            'username': username,
            'name': name,
            'bio': bio,
            'created_at': datetime.now(),
            'tweet_count': 0,
            'follower_count': 0,
            'following_count': 0,
            'tweepcred': 50.0  # 初期信頼度スコア
        }
        self.stats['total_users'] += 1
        print(f"✅ ユーザー @{username} を作成しました（ID: {user_id}）")
        return user_id
    
    def follow(self, follower_id: str, followed_id: str) -> bool:
        """フォロー関係を追加"""
        if follower_id == followed_id:
            print("❌ 自分自身はフォローできません")
            return False
            
        if followed_id in self.follows[follower_id]:
            print("❌ すでにフォローしています")
            return False
            
        self.follows[follower_id].add(followed_id)
        self.followers[followed_id].add(follower_id)
        
        self.users[follower_id]['following_count'] += 1
        self.users[followed_id]['follower_count'] += 1
        
        # Real Graphを更新
        self._update_real_graph(follower_id, followed_id, 'follow')
        
        return True
    
    def post_tweet(self, user_id: str, content: str, media: bool = False) -> str:
        """ツイートを投稿"""
        tweet_id = f"t_{len(self.tweets)}"
        
        self.tweets[tweet_id] = {
            'id': tweet_id,
            'author_id': user_id,
            'content': content,
            'created_at': datetime.now(),
            'has_media': media,
            'like_count': 0,
            'retweet_count': 0,
            'reply_count': 0,
            'features': self._extract_tweet_features(content, media)
        }
        
        self.users[user_id]['tweet_count'] += 1
        self.stats['total_tweets'] += 1
        
        print(f"📝 ツイートを投稿しました（ID: {tweet_id}）")
        return tweet_id
    
    def like_tweet(self, user_id: str, tweet_id: str) -> bool:
        """ツイートにいいね"""
        if user_id in self.likes[tweet_id]:
            print("❌ すでにいいねしています")
            return False
            
        self.likes[tweet_id].add(user_id)
        self.tweets[tweet_id]['like_count'] += 1
        self.stats['total_likes'] += 1
        
        # Real Graphを更新
        author_id = self.tweets[tweet_id]['author_id']
        self._update_real_graph(user_id, author_id, 'like')
        
        return True
    
    def retweet(self, user_id: str, tweet_id: str) -> bool:
        """リツイート"""
        if user_id in self.retweets[tweet_id]:
            print("❌ すでにリツイートしています")
            return False
            
        self.retweets[tweet_id].add(user_id)
        self.tweets[tweet_id]['retweet_count'] += 1
        self.stats['total_retweets'] += 1
        
        # Real Graphを更新
        author_id = self.tweets[tweet_id]['author_id']
        self._update_real_graph(user_id, author_id, 'retweet')
        
        return True
    
    def _extract_tweet_features(self, content: str, has_media: bool) -> Dict:
        """ツイートから特徴量を抽出"""
        features = {
            'length': len(content),
            'has_url': 'http' in content.lower(),
            'has_hashtag': '#' in content,
            'has_mention': '@' in content,
            'has_media': has_media,
            'word_count': len(content.split()),
            'question': '?' in content,
            'exclamation': '!' in content,
            'emoji_count': sum(1 for c in content if ord(c) > 127)
        }
        return features
    
    def _update_real_graph(self, user1: str, user2: str, interaction_type: str):
        """Real Graph（ユーザー間のインタラクション）を更新"""
        weight = {
            'follow': 1.0,
            'like': 0.3,
            'retweet': 0.5,
            'reply': 0.4
        }.get(interaction_type, 0.1)
        
        self.real_graph[user1][user2] += weight
    
    def calculate_tweepcred(self):
        """Tweepcred（ユーザー信頼度スコア）を計算"""
        print("📊 Tweepcredスコアを計算中...")
        
        for user_id, user_data in self.users.items():
            # 基本スコア
            follower_count = user_data['follower_count']
            following_count = user_data['following_count']
            tweet_count = user_data['tweet_count']
            
            # フォロワー/フォロー比率
            if following_count > 0:
                ratio = follower_count / following_count
            else:
                ratio = follower_count
                
            # PageRank風のスコア計算
            base_score = np.log1p(follower_count) * 10
            
            # 調整
            if ratio < 0.1:  # スパムの可能性
                base_score *= 0.3
            elif ratio > 10:  # 影響力のあるアカウント
                base_score *= 1.5
                
            # ツイート品質
            if tweet_count > 0:
                avg_engagement = sum(
                    self.tweets[tid]['like_count'] + self.tweets[tid]['retweet_count']
                    for tid in self.tweets if self.tweets[tid]['author_id'] == user_id
                ) / tweet_count
                base_score += avg_engagement * 2
                
            # 0-100にスケーリング
            self.tweepcred_scores[user_id] = min(100, max(0, base_score))
            self.users[user_id]['tweepcred'] = self.tweepcred_scores[user_id]
            
        print("✅ Tweepcredスコア計算完了")
    
    def build_simclusters(self, n_clusters: int = 10):
        """SimClustersモデルを構築"""
        print("🔨 SimClustersモデルを構築中...")
        
        # フォローグラフから二部グラフを作成
        follow_matrix = {}
        for follower, followed_set in self.follows.items():
            follow_matrix[follower] = list(followed_set)
            
        if not follow_matrix:
            print("❌ フォローデータがありません")
            return
            
        # 簡易版SimClusters実装
        self.simclusters = SimplifiedSimClusters(n_clusters)
        self.simclusters.fit(follow_matrix, self.likes)
        
        print(f"✅ {n_clusters}個のクラスタでSimClustersモデル構築完了")
    
    def get_timeline(self, user_id: str, algorithm: str = 'hybrid') -> List[Dict]:
        """
        タイムラインを生成
        
        Args:
            user_id: ユーザーID
            algorithm: 'chronological', 'popular', 'personalized', 'hybrid'
        """
        if algorithm == 'chronological':
            return self._get_chronological_timeline(user_id)
        elif algorithm == 'popular':
            return self._get_popular_timeline(user_id)
        elif algorithm == 'personalized':
            return self._get_personalized_timeline(user_id)
        else:  # hybrid
            return self._get_hybrid_timeline(user_id)
    
    def _get_chronological_timeline(self, user_id: str) -> List[Dict]:
        """時系列タイムライン"""
        following = self.follows[user_id]
        tweets = []
        
        for tweet_id, tweet in self.tweets.items():
            if tweet['author_id'] in following or tweet['author_id'] == user_id:
                tweets.append(tweet)
                
        # 時間順にソート
        tweets.sort(key=lambda x: x['created_at'], reverse=True)
        return tweets[:20]
    
    def _get_popular_timeline(self, user_id: str) -> List[Dict]:
        """人気順タイムライン"""
        tweets = []
        
        for tweet_id, tweet in self.tweets.items():
            # エンゲージメントスコア
            score = (tweet['like_count'] * 1 + 
                    tweet['retweet_count'] * 2 + 
                    tweet['reply_count'] * 3)
            
            # 時間減衰
            age_hours = (datetime.now() - tweet['created_at']).total_seconds() / 3600
            time_decay = 1 / (1 + age_hours / 24)
            
            tweet['popularity_score'] = score * time_decay
            tweets.append(tweet)
            
        # スコア順にソート
        tweets.sort(key=lambda x: x['popularity_score'], reverse=True)
        return tweets[:20]
    
    def _get_personalized_timeline(self, user_id: str) -> List[Dict]:
        """パーソナライズされたタイムライン"""
        if self.simclusters is None:
            return self._get_popular_timeline(user_id)
            
        # SimClustersで推薦
        recommendations = self.simclusters.recommend(user_id, n=20)
        
        tweets = []
        for tweet_id, score in recommendations:
            if tweet_id in self.tweets:
                tweet = self.tweets[tweet_id].copy()
                tweet['recommendation_score'] = score
                tweets.append(tweet)
                
        return tweets
    
    def _get_hybrid_timeline(self, user_id: str) -> List[Dict]:
        """ハイブリッドタイムライン（Twitterの"For You"に相当）"""
        # 複数のソースから候補を取得
        chronological = self._get_chronological_timeline(user_id)[:10]
        popular = self._get_popular_timeline(user_id)[:10]
        personalized = self._get_personalized_timeline(user_id)[:10]
        
        # 重複を除いて統合
        seen = set()
        combined = []
        
        # ラウンドロビンで混ぜる
        for tweets_list in [personalized, popular, chronological]:
            for tweet in tweets_list:
                if tweet['id'] not in seen:
                    combined.append(tweet)
                    seen.add(tweet['id'])
                    
        # Light Rankerでスコアリング
        for tweet in combined:
            tweet['final_score'] = self._light_ranker_score(tweet, user_id)
            
        # Heavy Ranker（簡易版）でトップを選択
        combined.sort(key=lambda x: x['final_score'], reverse=True)
        
        return combined[:20]
    
    def _light_ranker_score(self, tweet: Dict, user_id: str) -> float:
        """Light Ranker: 簡単な特徴量でスコアリング"""
        score = 0.0
        
        # エンゲージメント
        score += tweet['like_count'] * 0.1
        score += tweet['retweet_count'] * 0.2
        
        # コンテンツ特徴
        if tweet['features']['has_media']:
            score += 2.0
        if tweet['features']['has_url']:
            score += 1.0
            
        # 著者の信頼度
        author_id = tweet['author_id']
        if author_id in self.tweepcred_scores:
            score += self.tweepcred_scores[author_id] / 100 * 3
            
        # 時間減衰
        age_hours = (datetime.now() - tweet['created_at']).total_seconds() / 3600
        time_factor = np.exp(-age_hours / 48)  # 48時間で減衰
        score *= time_factor
        
        # Real Graphスコア（ユーザーとの関係性）
        if author_id in self.real_graph[user_id]:
            score += self.real_graph[user_id][author_id] * 2
            
        return score
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        stats = self.stats.copy()
        
        # 追加の統計
        if self.users:
            stats['avg_followers'] = np.mean([u['follower_count'] for u in self.users.values()])
            stats['avg_tweets_per_user'] = stats['total_tweets'] / stats['total_users']
            
        if self.tweets:
            stats['avg_likes_per_tweet'] = stats['total_likes'] / stats['total_tweets']
            stats['avg_retweets_per_tweet'] = stats['total_retweets'] / stats['total_tweets']
            
        return stats
    
    def save_state(self, filepath: str = 'mini_twitter_state.pkl'):
        """システムの状態を保存"""
        state = {
            'users': self.users,
            'tweets': self.tweets,
            'follows': dict(self.follows),
            'followers': dict(self.followers),
            'likes': dict(self.likes),
            'retweets': dict(self.retweets),
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 状態を {filepath} に保存しました")
    
    def load_state(self, filepath: str = 'mini_twitter_state.pkl'):
        """システムの状態を読み込み"""
        if not os.path.exists(filepath):
            print(f"❌ ファイル {filepath} が見つかりません")
            return False
            
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.users = state['users']
        self.tweets = state['tweets']
        self.follows = defaultdict(set, state['follows'])
        self.followers = defaultdict(set, state['followers'])
        self.likes = defaultdict(set, state['likes'])
        self.retweets = defaultdict(set, state['retweets'])
        self.stats = state['stats']
        
        print(f"📂 状態を {filepath} から読み込みました")
        return True


class SimplifiedSimClusters:
    """簡略化されたSimClustersアルゴリズム"""
    
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.user_clusters = {}
        self.tweet_clusters = {}
        
    def fit(self, follow_graph: Dict, likes: Dict):
        """モデルを学習"""
        # ユーザーをランダムにクラスタに割り当て（簡略化）
        users = list(set(follow_graph.keys()) | 
                    set(u for follows in follow_graph.values() for u in follows))
        
        for i, user in enumerate(users):
            # フォロー関係に基づいてクラスタを決定
            cluster_id = i % self.n_clusters
            self.user_clusters[user] = cluster_id
            
        # ツイートのクラスタを計算
        for tweet_id, likers in likes.items():
            cluster_scores = Counter()
            for user in likers:
                if user in self.user_clusters:
                    cluster_scores[self.user_clusters[user]] += 1
                    
            if cluster_scores:
                # 最も多いクラスタを割り当て
                self.tweet_clusters[tweet_id] = cluster_scores.most_common(1)[0][0]
                
    def recommend(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """ユーザーに推薦"""
        if user_id not in self.user_clusters:
            return []
            
        user_cluster = self.user_clusters[user_id]
        recommendations = []
        
        for tweet_id, tweet_cluster in self.tweet_clusters.items():
            if tweet_cluster == user_cluster:
                # 同じクラスタのツイートを推薦
                score = 1.0 + random.random()  # ランダムな変動を追加
                recommendations.append((tweet_id, score))
                
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]


class InteractiveTwitter:
    """対話型のTwitterシミュレーター"""
    
    def __init__(self):
        self.twitter = MiniTwitter()
        self.current_user = None
        self._setup_sample_data()
        
    def _setup_sample_data(self):
        """サンプルデータをセットアップ"""
        # サンプルユーザー
        users = [
            ("tech_guru", "Tech Guru", "AI・機械学習の専門家"),
            ("sports_fan", "Sports Fan", "スポーツ大好き！"),
            ("foodie", "Foodie", "美味しいものを求めて"),
            ("news_bot", "News Bot", "最新ニュースをお届け"),
            ("comedian", "Comedian", "笑いをお届けします"),
        ]
        
        for username, name, bio in users:
            self.twitter.create_user(username, name, bio)
            
        # フォロー関係を作成
        user_ids = list(self.twitter.users.keys())
        for i in range(len(user_ids)):
            for j in range(len(user_ids)):
                if i != j and random.random() < 0.3:
                    self.twitter.follow(user_ids[i], user_ids[j])
                    
        # サンプルツイート
        tweets = [
            (user_ids[0], "機械学習の新しいブレークスルー！TransformerがさらKに進化", False),
            (user_ids[0], "Pythonで推薦システムを実装する方法を解説しました", True),
            (user_ids[1], "今日の試合は最高だった！#スポーツ", False),
            (user_ids[2], "このレストランのパスタが絶品！", True),
            (user_ids[3], "速報：新しい技術が発表されました", False),
            (user_ids[4], "今日のジョーク：なぜプログラマーは...😄", False),
        ]
        
        for author, content, media in tweets:
            tweet_id = self.twitter.post_tweet(author, content, media)
            
            # ランダムにいいね
            for user_id in user_ids:
                if random.random() < 0.4:
                    self.twitter.like_tweet(user_id, tweet_id)
                if random.random() < 0.2:
                    self.twitter.retweet(user_id, tweet_id)
                    
        # Tweepcredを計算
        self.twitter.calculate_tweepcred()
        
        # SimClustersを構築
        self.twitter.build_simclusters()
        
    def run(self):
        """対話型セッションを開始"""
        print("🐦 ミニTwitterへようこそ！")
        print("=" * 50)
        
        while True:
            self._show_menu()
            choice = input("\n選択してください: ")
            
            if choice == '1':
                self._login()
            elif choice == '2':
                self._post_tweet()
            elif choice == '3':
                self._view_timeline()
            elif choice == '4':
                self._interact_with_tweet()
            elif choice == '5':
                self._follow_user()
            elif choice == '6':
                self._view_statistics()
            elif choice == '7':
                self._rebuild_models()
            elif choice == '8':
                print("👋 さようなら！")
                break
            else:
                print("❌ 無効な選択です")
                
    def _show_menu(self):
        """メニューを表示"""
        print("\n" + "=" * 50)
        if self.current_user:
            username = self.twitter.users[self.current_user]['username']
            print(f"ログイン中: @{username}")
        print("\n1. ログイン/ユーザー作成")
        print("2. ツイートする")
        print("3. タイムラインを見る")
        print("4. ツイートにアクション（いいね/RT）")
        print("5. ユーザーをフォロー")
        print("6. 統計を見る")
        print("7. 推薦モデルを再構築")
        print("8. 終了")
        
    def _login(self):
        """ログイン/ユーザー作成"""
        print("\n既存ユーザー:")
        for uid, user in self.twitter.users.items():
            print(f"  {uid}: @{user['username']} - {user['name']}")
            
        user_input = input("\nユーザーID入力 or 'new'で新規作成: ")
        
        if user_input == 'new':
            username = input("ユーザー名: ")
            name = input("名前: ")
            bio = input("自己紹介: ")
            self.current_user = self.twitter.create_user(username, name, bio)
        elif user_input in self.twitter.users:
            self.current_user = user_input
            print(f"✅ ログインしました")
        else:
            print("❌ ユーザーが見つかりません")
            
    def _post_tweet(self):
        """ツイートを投稿"""
        if not self.current_user:
            print("❌ ログインしてください")
            return
            
        content = input("ツイート内容: ")
        has_media = input("画像を追加？(y/n): ").lower() == 'y'
        
        self.twitter.post_tweet(self.current_user, content, has_media)
        
    def _view_timeline(self):
        """タイムラインを表示"""
        if not self.current_user:
            print("❌ ログインしてください")
            return
            
        print("\nアルゴリズムを選択:")
        print("1. 時系列")
        print("2. 人気順")
        print("3. パーソナライズ")
        print("4. ハイブリッド（For You）")
        
        algo_choice = input("選択: ")
        algo_map = {
            '1': 'chronological',
            '2': 'popular',
            '3': 'personalized',
            '4': 'hybrid'
        }
        
        algorithm = algo_map.get(algo_choice, 'hybrid')
        timeline = self.twitter.get_timeline(self.current_user, algorithm)
        
        print(f"\n📱 タイムライン ({algorithm}):")
        print("-" * 50)
        
        for i, tweet in enumerate(timeline[:10], 1):
            author = self.twitter.users[tweet['author_id']]
            print(f"{i}. @{author['username']}: {tweet['content']}")
            print(f"   ❤️ {tweet['like_count']} 🔄 {tweet['retweet_count']}")
            
            if 'final_score' in tweet:
                print(f"   スコア: {tweet['final_score']:.2f}")
            print()
            
    def _interact_with_tweet(self):
        """ツイートにアクション"""
        if not self.current_user:
            print("❌ ログインしてください")
            return
            
        # 最新ツイートを表示
        recent_tweets = sorted(self.twitter.tweets.values(), 
                              key=lambda x: x['created_at'], 
                              reverse=True)[:5]
        
        print("\n最近のツイート:")
        for tweet in recent_tweets:
            author = self.twitter.users[tweet['author_id']]
            print(f"{tweet['id']}: @{author['username']}: {tweet['content'][:50]}...")
            
        tweet_id = input("\nツイートID: ")
        if tweet_id not in self.twitter.tweets:
            print("❌ ツイートが見つかりません")
            return
            
        action = input("アクション (1: いいね, 2: リツイート): ")
        
        if action == '1':
            self.twitter.like_tweet(self.current_user, tweet_id)
            print("❤️ いいねしました")
        elif action == '2':
            self.twitter.retweet(self.current_user, tweet_id)
            print("🔄 リツイートしました")
            
    def _follow_user(self):
        """ユーザーをフォロー"""
        if not self.current_user:
            print("❌ ログインしてください")
            return
            
        print("\nユーザー一覧:")
        for uid, user in self.twitter.users.items():
            if uid != self.current_user:
                status = "✓ フォロー中" if uid in self.twitter.follows[self.current_user] else ""
                print(f"  {uid}: @{user['username']} - {user['name']} {status}")
                
        user_id = input("\nフォローするユーザーID: ")
        
        if self.twitter.follow(self.current_user, user_id):
            print("✅ フォローしました")
            
    def _view_statistics(self):
        """統計を表示"""
        stats = self.twitter.get_statistics()
        
        print("\n📊 システム統計:")
        print("-" * 30)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
                
        if self.current_user:
            user_data = self.twitter.users[self.current_user]
            print("\n👤 あなたの統計:")
            print(f"フォロワー: {user_data['follower_count']}")
            print(f"フォロー中: {user_data['following_count']}")
            print(f"ツイート数: {user_data['tweet_count']}")
            print(f"Tweepcred: {user_data['tweepcred']:.1f}")
            
    def _rebuild_models(self):
        """推薦モデルを再構築"""
        print("\n🔨 モデルを再構築中...")
        self.twitter.calculate_tweepcred()
        self.twitter.build_simclusters()
        print("✅ 完了！")


def main():
    """メイン関数"""
    app = InteractiveTwitter()
    app.run()


if __name__ == "__main__":
    main()