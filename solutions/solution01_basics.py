#!/usr/bin/env python3
"""
演習問題1の解答例
"""

from datetime import datetime, timedelta
from collections import Counter
import numpy as np

# ===============================
# 演習1-1の解答
# ===============================
def solution_1_1():
    """人気ランキングの解答"""
    
    movies = [
        {"title": "映画A", "views": 1000},
        {"title": "映画B", "views": 2500},
        {"title": "映画C", "views": 1800},
        {"title": "映画D", "views": 500},
        {"title": "映画E", "views": 3000},
    ]
    
    # 視聴数で降順ソート
    top3 = sorted(movies, key=lambda x: x['views'], reverse=True)[:3]
    
    print("Top 3 人気映画:")
    for i, movie in enumerate(top3, 1):
        print(f"{i}. {movie['title']} ({movie['views']}回視聴)")
    
    return top3


# ===============================
# 演習1-2の解答
# ===============================
def solution_1_2():
    """類似度計算の解答"""
    
    movie1 = {
        "title": "アベンジャーズ",
        "genres": ["アクション", "SF", "ヒーロー"]
    }
    
    movie2 = {
        "title": "スパイダーマン",
        "genres": ["アクション", "ヒーロー", "青春"]
    }
    
    # ジャンルを集合に変換
    genres1 = set(movie1['genres'])
    genres2 = set(movie2['genres'])
    
    # Jaccard係数を計算
    intersection = genres1 & genres2  # 共通ジャンル
    union = genres1 | genres2  # 全ジャンル
    
    similarity = len(intersection) / len(union) if union else 0
    
    print(f"共通ジャンル: {intersection}")
    print(f"全ジャンル: {union}")
    print(f"類似度: {similarity:.2f}")
    
    return similarity


# ===============================
# 演習1-3の解答
# ===============================
def solution_1_3():
    """ユーザープロファイルの解答"""
    
    watch_history = [
        {"title": "映画A", "genres": ["アクション", "SF"]},
        {"title": "映画B", "genres": ["アクション", "コメディ"]},
        {"title": "映画C", "genres": ["SF"]},
        {"title": "映画D", "genres": ["アクション"]},
    ]
    
    # ジャンルごとの視聴回数を集計
    genre_count = {}
    
    for movie in watch_history:
        for genre in movie['genres']:
            genre_count[genre] = genre_count.get(genre, 0) + 1
    
    # または、Counterを使う方法
    # genre_count = Counter()
    # for movie in watch_history:
    #     genre_count.update(movie['genres'])
    
    # ランキング形式で表示
    print("好きなジャンルランキング:")
    sorted_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
    for i, (genre, count) in enumerate(sorted_genres, 1):
        print(f"{i}. {genre}: {count}回")
    
    return genre_count


# ===============================
# 演習1-4の解答
# ===============================
def solution_1_4():
    """簡単な推薦の解答"""
    
    user_preferences = ["SF", "アクション"]
    watched = ["映画A", "映画C"]
    
    movies = [
        {"title": "映画A", "genres": ["SF", "アクション"]},
        {"title": "映画B", "genres": ["コメディ"]},
        {"title": "映画C", "genres": ["SF"]},
        {"title": "映画D", "genres": ["アクション", "犯罪"]},
        {"title": "映画E", "genres": ["SF", "アクション", "冒険"]},
    ]
    
    recommendations = []
    
    for movie in movies:
        # 視聴済みはスキップ
        if movie['title'] in watched:
            continue
        
        # 共通ジャンルを計算
        common_genres = set(movie['genres']) & set(user_preferences)
        
        if common_genres:
            recommendations.append({
                'movie': movie,
                'common_count': len(common_genres),
                'common_genres': common_genres
            })
    
    # 共通ジャンル数で降順ソート
    recommendations.sort(key=lambda x: x['common_count'], reverse=True)
    
    print("おすすめ映画:")
    for rec in recommendations:
        movie = rec['movie']
        print(f"  - {movie['title']} (共通: {rec['common_genres']})")
    
    return recommendations


# ===============================
# 演習1-5の解答
# ===============================
def solution_1_5():
    """評価の重み付けの解答"""
    
    ratings = [
        {"movie": "映画A", "rating": 5, "date": datetime.now() - timedelta(days=7)},
        {"movie": "映画B", "rating": 3, "date": datetime.now() - timedelta(days=3)},
        {"movie": "映画C", "rating": 4, "date": datetime.now() - timedelta(days=1)},
        {"movie": "映画D", "rating": 5, "date": datetime.now()},
    ]
    
    weighted_sum = 0
    weight_total = 0
    
    for rating_data in ratings:
        # 経過日数を計算
        days_ago = (datetime.now() - rating_data['date']).days
        
        # 重みを計算（新しいほど重い）
        weight = 1 / (days_ago + 1)
        
        # 加重和を計算
        weighted_sum += rating_data['rating'] * weight
        weight_total += weight
        
        print(f"{rating_data['movie']}: 評価{rating_data['rating']}, "
              f"{days_ago}日前, 重み{weight:.3f}")
    
    weighted_average = weighted_sum / weight_total
    print(f"\n加重平均評価: {weighted_average:.2f}")
    
    # 比較: 単純平均
    simple_average = sum(r['rating'] for r in ratings) / len(ratings)
    print(f"単純平均評価: {simple_average:.2f}")
    
    return weighted_average


# ===============================
# チャレンジ問題の解答
# ===============================
class MovieRecommender:
    """完全な映画推薦システム"""
    
    def __init__(self):
        self.movies = {}
        self.users = {}
        self.ratings = {}  # {user_id: {movie_title: rating}}
        
    def add_movie(self, title, genres):
        """映画を追加"""
        self.movies[title] = {
            'title': title,
            'genres': genres,
            'total_rating': 0,
            'rating_count': 0
        }
        
    def add_user(self, user_id, name):
        """ユーザーを追加"""
        self.users[user_id] = {
            'name': name,
            'ratings': {}
        }
        self.ratings[user_id] = {}
        
    def rate_movie(self, user_id, movie_title, rating):
        """映画を評価"""
        if user_id not in self.users:
            print(f"ユーザー {user_id} が存在しません")
            return
            
        if movie_title not in self.movies:
            print(f"映画 {movie_title} が存在しません")
            return
            
        # 評価を記録
        self.ratings[user_id][movie_title] = rating
        self.users[user_id]['ratings'][movie_title] = rating
        
        # 映画の統計を更新
        self.movies[movie_title]['total_rating'] += rating
        self.movies[movie_title]['rating_count'] += 1
        
    def recommend_popular(self, user_id, n=5):
        """人気ベースの推薦"""
        # 平均評価を計算
        movie_scores = []
        
        for title, movie in self.movies.items():
            if title in self.ratings.get(user_id, {}):
                continue  # 既に視聴済み
                
            if movie['rating_count'] > 0:
                avg_rating = movie['total_rating'] / movie['rating_count']
                popularity = avg_rating * np.log1p(movie['rating_count'])
                movie_scores.append((title, popularity))
        
        # スコア順にソート
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [title for title, _ in movie_scores[:n]]
    
    def recommend_content_based(self, user_id, n=5):
        """コンテンツベースの推薦"""
        if user_id not in self.ratings:
            return []
            
        # ユーザーの好みのジャンルを分析
        genre_scores = Counter()
        
        for movie_title, rating in self.ratings[user_id].items():
            if movie_title in self.movies:
                for genre in self.movies[movie_title]['genres']:
                    genre_scores[genre] += rating
        
        # 未視聴映画をスコアリング
        recommendations = []
        
        for title, movie in self.movies.items():
            if title in self.ratings[user_id]:
                continue
                
            score = sum(genre_scores.get(genre, 0) for genre in movie['genres'])
            if score > 0:
                recommendations.append((title, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [title for title, _ in recommendations[:n]]
    
    def recommend_collaborative(self, user_id, n=5):
        """協調フィルタリング"""
        if user_id not in self.ratings:
            return []
            
        # 類似ユーザーを見つける
        similarities = []
        
        for other_id in self.users:
            if other_id == user_id:
                continue
                
            # 共通評価映画での相関を計算
            common_movies = (set(self.ratings[user_id].keys()) & 
                           set(self.ratings.get(other_id, {}).keys()))
            
            if len(common_movies) >= 2:
                ratings1 = [self.ratings[user_id][m] for m in common_movies]
                ratings2 = [self.ratings[other_id][m] for m in common_movies]
                
                correlation = np.corrcoef(ratings1, ratings2)[0, 1]
                if not np.isnan(correlation):
                    similarities.append((other_id, correlation))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 類似ユーザーの高評価映画を推薦
        movie_scores = Counter()
        
        for similar_user, similarity in similarities[:3]:
            for movie, rating in self.ratings[similar_user].items():
                if movie not in self.ratings[user_id] and rating >= 4:
                    movie_scores[movie] += rating * similarity
        
        return list(movie_scores.most_common(n))
    
    def recommend_hybrid(self, user_id, n=5):
        """ハイブリッド推薦"""
        # 各手法の結果を取得
        popular = self.recommend_popular(user_id, n * 2)
        content = self.recommend_content_based(user_id, n * 2)
        collab = self.recommend_collaborative(user_id, n * 2)
        
        # スコアを統合
        movie_scores = Counter()
        
        # 重み付け
        for i, movie in enumerate(popular):
            movie_scores[movie] += (n * 2 - i) * 0.3
            
        for i, movie in enumerate(content):
            movie_scores[movie] += (n * 2 - i) * 0.4
            
        for i, movie in enumerate(collab):
            movie_scores[movie] += (n * 2 - i) * 0.3
        
        # 上位n個を返す
        return [movie for movie, _ in movie_scores.most_common(n)]


def test_recommender():
    """推薦システムのテスト"""
    
    # システムを初期化
    recommender = MovieRecommender()
    
    # 映画を追加
    recommender.add_movie("アベンジャーズ", ["アクション", "SF", "ヒーロー"])
    recommender.add_movie("君の名は", ["アニメ", "恋愛", "ファンタジー"])
    recommender.add_movie("インセプション", ["SF", "サスペンス"])
    recommender.add_movie("トイストーリー", ["アニメ", "家族", "冒険"])
    recommender.add_movie("ダークナイト", ["アクション", "犯罪", "ヒーロー"])
    recommender.add_movie("千と千尋", ["アニメ", "ファンタジー", "冒険"])
    
    # ユーザーを追加
    recommender.add_user("user1", "太郎")
    recommender.add_user("user2", "花子")
    recommender.add_user("user3", "次郎")
    
    # 評価を追加
    recommender.rate_movie("user1", "アベンジャーズ", 5)
    recommender.rate_movie("user1", "インセプション", 4)
    recommender.rate_movie("user1", "ダークナイト", 5)
    
    recommender.rate_movie("user2", "君の名は", 5)
    recommender.rate_movie("user2", "千と千尋", 5)
    recommender.rate_movie("user2", "トイストーリー", 4)
    
    recommender.rate_movie("user3", "アベンジャーズ", 4)
    recommender.rate_movie("user3", "君の名は", 3)
    recommender.rate_movie("user3", "インセプション", 5)
    
    # 推薦を取得
    print("\n🎬 user1への推薦:")
    print("人気ベース:", recommender.recommend_popular("user1", 3))
    print("コンテンツベース:", recommender.recommend_content_based("user1", 3))
    print("協調フィルタリング:", recommender.recommend_collaborative("user1", 3))
    print("ハイブリッド:", recommender.recommend_hybrid("user1", 3))


# ===============================
# メイン実行
# ===============================
if __name__ == "__main__":
    print("📚 演習問題1の解答")
    print("=" * 50)
    
    print("\n解答1-1: 人気ランキング")
    solution_1_1()
    
    print("\n" + "=" * 50)
    print("解答1-2: 類似度計算")
    solution_1_2()
    
    print("\n" + "=" * 50)
    print("解答1-3: ユーザープロファイル")
    solution_1_3()
    
    print("\n" + "=" * 50)
    print("解答1-4: 簡単な推薦")
    solution_1_4()
    
    print("\n" + "=" * 50)
    print("解答1-5: 評価の重み付け")
    solution_1_5()
    
    print("\n" + "=" * 50)
    print("🏆 チャレンジ問題の解答")
    test_recommender()