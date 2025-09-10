# Chapter 2: はじめての推薦システムを作ろう！ 🛠️

## 🎯 この章のゴール

実際に動く推薦システムを作ります！最終的には：
- 映画推薦システム
- 音楽推薦システム
- ニュース推薦システム

これらすべてを自分で作れるようになります。

## 📚 準備：必要なツール

```python
# 必要なライブラリをインストール
# ターミナルで実行:
# pip install pandas numpy matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random

print("準備完了！推薦システムを作る準備ができました 🚀")
```

## 🎬 プロジェクト1：映画推薦システム

### Step 1: データを準備しよう

```python
class MovieDatabase:
    """映画データベースクラス"""
    
    def __init__(self):
        self.movies = [
            {"id": 1, "title": "アベンジャーズ", "genre": ["アクション", "SF"], "year": 2012, "rating": 4.5},
            {"id": 2, "title": "君の名は", "genre": ["アニメ", "恋愛"], "year": 2016, "rating": 4.3},
            {"id": 3, "title": "パラサイト", "genre": ["ドラマ", "スリラー"], "year": 2019, "rating": 4.6},
            {"id": 4, "title": "トイ・ストーリー", "genre": ["アニメ", "家族"], "year": 1995, "rating": 4.2},
            {"id": 5, "title": "インセプション", "genre": ["SF", "アクション"], "year": 2010, "rating": 4.4},
            {"id": 6, "title": "千と千尋の神隠し", "genre": ["アニメ", "ファンタジー"], "year": 2001, "rating": 4.7},
            {"id": 7, "title": "ダークナイト", "genre": ["アクション", "犯罪"], "year": 2008, "rating": 4.8},
            {"id": 8, "title": "タイタニック", "genre": ["恋愛", "ドラマ"], "year": 1997, "rating": 4.1},
            {"id": 9, "title": "マトリックス", "genre": ["SF", "アクション"], "year": 1999, "rating": 4.3},
            {"id": 10, "title": "ララランド", "genre": ["恋愛", "ミュージカル"], "year": 2016, "rating": 4.0},
        ]
        
        self.users = []
        self.user_ratings = {}
    
    def add_user(self, user_name):
        """新しいユーザーを追加"""
        user_id = len(self.users) + 1
        self.users.append({"id": user_id, "name": user_name})
        self.user_ratings[user_id] = {}
        return user_id
    
    def rate_movie(self, user_id, movie_id, rating):
        """映画を評価"""
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {}
        self.user_ratings[user_id][movie_id] = rating
        print(f"✅ 評価を記録しました！")
    
    def get_movie_by_id(self, movie_id):
        """IDで映画を取得"""
        for movie in self.movies:
            if movie['id'] == movie_id:
                return movie
        return None

# データベースを作成
db = MovieDatabase()

# あなたのアカウントを作成
your_id = db.add_user("あなた")
print(f"ユーザーID {your_id} として登録されました！")
```

### Step 2: 人気ベースの推薦システム

```python
class PopularityRecommender:
    """人気度に基づく推薦システム"""
    
    def __init__(self, database):
        self.db = database
    
    def calculate_popularity(self):
        """各映画の人気度を計算"""
        popularity = {}
        
        for movie in self.db.movies:
            # 評価の平均と評価数を考慮
            ratings = []
            for user_ratings in self.db.user_ratings.values():
                if movie['id'] in user_ratings:
                    ratings.append(user_ratings[movie['id']])
            
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                num_ratings = len(ratings)
                # 人気度 = 平均評価 × 評価数の対数
                popularity[movie['id']] = avg_rating * np.log(num_ratings + 1)
            else:
                # デフォルトの評価を使用
                popularity[movie['id']] = movie['rating']
        
        return popularity
    
    def recommend(self, user_id, n=5):
        """人気の映画をn個推薦"""
        popularity = self.calculate_popularity()
        
        # ユーザーがまだ見ていない映画を取得
        watched = set(self.db.user_ratings.get(user_id, {}).keys())
        unwatched = [m for m in self.db.movies if m['id'] not in watched]
        
        # 人気度でソート
        unwatched.sort(key=lambda x: popularity[x['id']], reverse=True)
        
        recommendations = []
        for movie in unwatched[:n]:
            recommendations.append({
                'title': movie['title'],
                'genre': movie['genre'],
                'score': popularity[movie['id']]
            })
        
        return recommendations

# テスト：いくつか映画を評価してみよう
db.rate_movie(your_id, 1, 5)  # アベンジャーズを5点
db.rate_movie(your_id, 2, 4)  # 君の名はを4点

# 推薦を取得
pop_recommender = PopularityRecommender(db)
recommendations = pop_recommender.recommend(your_id)

print("\n🎬 あなたへのおすすめ映画（人気順）:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['title']} ({', '.join(rec['genre'])}) - スコア: {rec['score']:.2f}")
```

### Step 3: コンテンツベースの推薦システム

```python
class ContentBasedRecommender:
    """コンテンツの類似性に基づく推薦システム"""
    
    def __init__(self, database):
        self.db = database
    
    def calculate_similarity(self, movie1, movie2):
        """2つの映画の類似度を計算"""
        # ジャンルの重複度
        genres1 = set(movie1['genre'])
        genres2 = set(movie2['genre'])
        genre_similarity = len(genres1 & genres2) / len(genres1 | genres2) if genres1 | genres2 else 0
        
        # 年代の近さ（10年以内なら類似）
        year_diff = abs(movie1['year'] - movie2['year'])
        year_similarity = max(0, 1 - year_diff / 20)
        
        # 総合的な類似度
        similarity = 0.7 * genre_similarity + 0.3 * year_similarity
        return similarity
    
    def get_user_profile(self, user_id):
        """ユーザーの好みのプロファイルを作成"""
        user_ratings = self.db.user_ratings.get(user_id, {})
        
        # 高評価の映画からプロファイルを作成
        liked_genres = {}
        avg_year = 0
        count = 0
        
        for movie_id, rating in user_ratings.items():
            if rating >= 4:  # 4点以上を「好き」とする
                movie = self.db.get_movie_by_id(movie_id)
                if movie:
                    for genre in movie['genre']:
                        liked_genres[genre] = liked_genres.get(genre, 0) + 1
                    avg_year += movie['year']
                    count += 1
        
        if count > 0:
            avg_year = avg_year / count
        else:
            avg_year = 2010  # デフォルト
        
        return {
            'liked_genres': liked_genres,
            'preferred_year': avg_year
        }
    
    def recommend(self, user_id, n=5):
        """ユーザーの好みに基づいて推薦"""
        profile = self.get_user_profile(user_id)
        watched = set(self.db.user_ratings.get(user_id, {}).keys())
        
        recommendations = []
        
        for movie in self.db.movies:
            if movie['id'] not in watched:
                # プロファイルとの類似度を計算
                score = 0
                
                # ジャンルの一致度
                for genre in movie['genre']:
                    if genre in profile['liked_genres']:
                        score += profile['liked_genres'][genre]
                
                # 年代の近さ
                year_diff = abs(movie['year'] - profile['preferred_year'])
                year_score = max(0, 1 - year_diff / 20)
                score += year_score * 2
                
                recommendations.append({
                    'movie': movie,
                    'score': score
                })
        
        # スコアでソート
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        result = []
        for rec in recommendations[:n]:
            result.append({
                'title': rec['movie']['title'],
                'genre': rec['movie']['genre'],
                'year': rec['movie']['year'],
                'score': rec['score']
            })
        
        return result

# もう少し映画を評価
db.rate_movie(your_id, 5, 5)  # インセプション（SF）を5点
db.rate_movie(your_id, 9, 4)  # マトリックス（SF）を4点

# コンテンツベースの推薦
content_recommender = ContentBasedRecommender(db)
recommendations = content_recommender.recommend(your_id)

print("\n🎯 あなたの好みに基づくおすすめ映画:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['title']} ({rec['year']}) - {', '.join(rec['genre'])} - スコア: {rec['score']:.2f}")
```

### Step 4: 協調フィルタリング

```python
class CollaborativeRecommender:
    """協調フィルタリングによる推薦システム"""
    
    def __init__(self, database):
        self.db = database
        self._add_sample_users()
    
    def _add_sample_users(self):
        """サンプルユーザーとその評価を追加"""
        # ユーザーA: アクション好き
        user_a = self.db.add_user("ユーザーA")
        self.db.rate_movie(user_a, 1, 5)  # アベンジャーズ
        self.db.rate_movie(user_a, 5, 5)  # インセプション
        self.db.rate_movie(user_a, 7, 5)  # ダークナイト
        self.db.rate_movie(user_a, 9, 4)  # マトリックス
        
        # ユーザーB: アニメ好き
        user_b = self.db.add_user("ユーザーB")
        self.db.rate_movie(user_b, 2, 5)  # 君の名は
        self.db.rate_movie(user_b, 4, 4)  # トイ・ストーリー
        self.db.rate_movie(user_b, 6, 5)  # 千と千尋
        
        # ユーザーC: SF好き（あなたと似ている）
        user_c = self.db.add_user("ユーザーC")
        self.db.rate_movie(user_c, 1, 4)  # アベンジャーズ
        self.db.rate_movie(user_c, 5, 5)  # インセプション
        self.db.rate_movie(user_c, 9, 5)  # マトリックス
        self.db.rate_movie(user_c, 7, 4)  # ダークナイト（あなたは未視聴）
    
    def calculate_user_similarity(self, user1_id, user2_id):
        """2人のユーザーの類似度を計算"""
        ratings1 = self.db.user_ratings.get(user1_id, {})
        ratings2 = self.db.user_ratings.get(user2_id, {})
        
        # 共通して評価した映画
        common_movies = set(ratings1.keys()) & set(ratings2.keys())
        
        if not common_movies:
            return 0
        
        # ピアソン相関係数を計算
        sum1 = sum([ratings1[m] for m in common_movies])
        sum2 = sum([ratings2[m] for m in common_movies])
        
        sum1_sq = sum([ratings1[m]**2 for m in common_movies])
        sum2_sq = sum([ratings2[m]**2 for m in common_movies])
        
        sum_product = sum([ratings1[m] * ratings2[m] for m in common_movies])
        
        n = len(common_movies)
        
        # 相関係数の計算
        numerator = sum_product - (sum1 * sum2 / n)
        denominator = np.sqrt((sum1_sq - sum1**2/n) * (sum2_sq - sum2**2/n))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def find_similar_users(self, user_id, n=3):
        """類似ユーザーを見つける"""
        similarities = []
        
        for other_user in self.db.users:
            if other_user['id'] != user_id:
                similarity = self.calculate_user_similarity(user_id, other_user['id'])
                if similarity > 0:
                    similarities.append({
                        'user': other_user,
                        'similarity': similarity
                    })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:n]
    
    def recommend(self, user_id, n=5):
        """類似ユーザーの評価に基づいて推薦"""
        similar_users = self.find_similar_users(user_id)
        
        if not similar_users:
            return []
        
        # 推薦スコアを計算
        watched = set(self.db.user_ratings.get(user_id, {}).keys())
        movie_scores = {}
        
        for sim_user in similar_users:
            user_ratings = self.db.user_ratings.get(sim_user['user']['id'], {})
            
            for movie_id, rating in user_ratings.items():
                if movie_id not in watched:
                    if movie_id not in movie_scores:
                        movie_scores[movie_id] = 0
                    # 類似度で重み付けしたスコア
                    movie_scores[movie_id] += rating * sim_user['similarity']
        
        # スコアの高い順にソート
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie_id, score in sorted_movies[:n]:
            movie = self.db.get_movie_by_id(movie_id)
            if movie:
                recommendations.append({
                    'title': movie['title'],
                    'genre': movie['genre'],
                    'score': score
                })
        
        return recommendations

# 協調フィルタリングで推薦
collab_recommender = CollaborativeRecommender(db)

# 類似ユーザーを表示
similar_users = collab_recommender.find_similar_users(your_id)
print("\n👥 あなたと似ているユーザー:")
for sim_user in similar_users:
    print(f"- {sim_user['user']['name']}: 類似度 {sim_user['similarity']:.2f}")

# 推薦を取得
recommendations = collab_recommender.recommend(your_id)
print("\n🌟 似た人が好きな映画:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['title']} ({', '.join(rec['genre'])}) - スコア: {rec['score']:.2f}")
```

## 🎨 可視化：推薦システムの動作を見る

```python
class RecommenderVisualizer:
    """推薦システムの可視化"""
    
    def __init__(self, database):
        self.db = database
    
    def plot_user_ratings_matrix(self):
        """ユーザー×映画の評価行列を可視化"""
        # 行列を作成
        users = self.db.users
        movies = self.db.movies
        
        matrix = np.zeros((len(users), len(movies)))
        
        for i, user in enumerate(users):
            ratings = self.db.user_ratings.get(user['id'], {})
            for j, movie in enumerate(movies):
                if movie['id'] in ratings:
                    matrix[i][j] = ratings[movie['id']]
        
        # ヒートマップを描画
        plt.figure(figsize=(12, 6))
        plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='評価')
        
        # ラベルを設定
        plt.yticks(range(len(users)), [u['name'] for u in users])
        plt.xticks(range(len(movies)), [m['title'][:10] for m in movies], rotation=45, ha='right')
        
        plt.title('ユーザー×映画 評価行列')
        plt.xlabel('映画')
        plt.ylabel('ユーザー')
        plt.tight_layout()
        plt.show()
    
    def plot_genre_distribution(self, user_id):
        """ユーザーの好みのジャンル分布"""
        ratings = self.db.user_ratings.get(user_id, {})
        
        genre_scores = {}
        for movie_id, rating in ratings.items():
            movie = self.db.get_movie_by_id(movie_id)
            if movie:
                for genre in movie['genre']:
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append(rating)
        
        # 平均スコアを計算
        genres = []
        scores = []
        for genre, ratings_list in genre_scores.items():
            genres.append(genre)
            scores.append(sum(ratings_list) / len(ratings_list))
        
        # グラフを描画
        plt.figure(figsize=(10, 6))
        bars = plt.bar(genres, scores, color='skyblue')
        
        # 色を評価に応じて変更
        for bar, score in zip(bars, scores):
            if score >= 4.5:
                bar.set_color('green')
            elif score >= 3.5:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        plt.title('あなたの好みのジャンル')
        plt.xlabel('ジャンル')
        plt.ylabel('平均評価')
        plt.ylim(0, 5)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

# 可視化を実行
visualizer = RecommenderVisualizer(db)
# visualizer.plot_user_ratings_matrix()  # 評価行列を表示
# visualizer.plot_genre_distribution(your_id)  # ジャンル分布を表示
```

## 🎯 演習問題

### 演習1: ハイブリッド推薦システム
3つの推薦手法を組み合わせたハイブリッドシステムを作ってください。

```python
class HybridRecommender:
    """複数の推薦手法を組み合わせる"""
    
    def __init__(self, database):
        self.db = database
        self.popularity = PopularityRecommender(database)
        self.content = ContentBasedRecommender(database)
        self.collaborative = CollaborativeRecommender(database)
    
    def recommend(self, user_id, n=5):
        """3つの手法を組み合わせて推薦"""
        # TODO: 実装してください
        # ヒント:
        # 1. 各推薦システムから候補を取得
        # 2. スコアを正規化して組み合わせる
        # 3. 重み付けは自由に決める（例: 人気30%, コンテンツ40%, 協調30%）
        pass

# テスト
hybrid = HybridRecommender(db)
# recommendations = hybrid.recommend(your_id)
```

### 演習2: 時間を考慮した推薦
新しい映画を優先的に推薦するシステムを作ってください。

```python
def time_weighted_recommendation(movies, current_year=2024):
    """新しい映画に高いスコアを与える"""
    # TODO: 実装してください
    # ヒント: 
    # - 映画の年齢 = current_year - movie['year']
    # - 重み = 1 / (1 + 年齢/10)
    pass
```

### 演習3: 評価の信頼度
評価数が少ないユーザーの推薦は信頼度を下げる仕組みを追加してください。

```python
def calculate_confidence(user_ratings):
    """ユーザーの評価の信頼度を計算"""
    # TODO: 実装してください
    # ヒント:
    # - 評価数が5個未満: 信頼度0.5
    # - 評価数が10個以上: 信頼度1.0
    # - その間は線形補間
    pass
```

## 🚀 チャレンジ課題：ミニNetflixを作ろう！

```python
class MiniNetflix:
    """総合的な映画推薦アプリケーション"""
    
    def __init__(self):
        self.db = MovieDatabase()
        self.current_user = None
        self.recommenders = {
            'popularity': PopularityRecommender(self.db),
            'content': ContentBasedRecommender(self.db),
            'collaborative': CollaborativeRecommender(self.db)
        }
    
    def run(self):
        """アプリケーションを実行"""
        print("🎬 ミニNetflixへようこそ！")
        
        while True:
            print("\n" + "="*50)
            print("1. ログイン/新規登録")
            print("2. 映画を評価する")
            print("3. おすすめを見る")
            print("4. 統計を見る")
            print("5. 終了")
            
            choice = input("\n選択してください (1-5): ")
            
            if choice == '1':
                self.login()
            elif choice == '2':
                self.rate_movies()
            elif choice == '3':
                self.show_recommendations()
            elif choice == '4':
                self.show_statistics()
            elif choice == '5':
                print("👋 またのご利用をお待ちしています！")
                break
    
    def login(self):
        """ユーザーログイン"""
        name = input("お名前を入力してください: ")
        self.current_user = self.db.add_user(name)
        print(f"✅ {name}さん、ログインしました！")
    
    def rate_movies(self):
        """映画を評価"""
        if not self.current_user:
            print("❌ まずログインしてください")
            return
        
        print("\n映画リスト:")
        for movie in self.db.movies:
            print(f"{movie['id']}. {movie['title']} ({movie['year']})")
        
        movie_id = int(input("\n評価する映画の番号: "))
        rating = int(input("評価 (1-5): "))
        
        self.db.rate_movie(self.current_user, movie_id, rating)
    
    def show_recommendations(self):
        """推薦を表示"""
        if not self.current_user:
            print("❌ まずログインしてください")
            return
        
        print("\n推薦方法を選択:")
        print("1. 人気順")
        print("2. あなたの好みに基づく")
        print("3. 似た人の評価に基づく")
        
        method = input("選択 (1-3): ")
        
        if method == '1':
            recs = self.recommenders['popularity'].recommend(self.current_user)
        elif method == '2':
            recs = self.recommenders['content'].recommend(self.current_user)
        elif method == '3':
            recs = self.recommenders['collaborative'].recommend(self.current_user)
        else:
            return
        
        print("\n🌟 おすすめ映画:")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['title']} - スコア: {rec['score']:.2f}")
    
    def show_statistics(self):
        """統計情報を表示"""
        print("\n📊 統計情報")
        print(f"登録ユーザー数: {len(self.db.users)}")
        print(f"映画数: {len(self.db.movies)}")
        
        total_ratings = sum(len(ratings) for ratings in self.db.user_ratings.values())
        print(f"総評価数: {total_ratings}")

# アプリを起動
# app = MiniNetflix()
# app.run()
```

## 📚 まとめ

この章で学んだこと：
- ✅ 3つの基本的な推薦手法の実装
- ✅ 実際に動く推薦システムの構築
- ✅ データの管理と処理
- ✅ ユーザーインターフェースの作成

次の章では、より高度な技術を学びます：
- 機械学習の導入
- リアルタイム処理
- 大規模データへの対応

[→ Chapter 3: 協調フィルタリングを極めるへ](chapter03_collaborative_filtering.md)