#!/usr/bin/env python3
"""
演習問題1: 推薦システムの基礎
高校生向けの練習問題
"""

# ===============================
# 演習1-1: 人気ランキング
# ===============================
def exercise_1_1():
    """
    問題: 映画の視聴数データから、最も人気のある映画Top3を見つけてください。
    
    ヒント:
    - sorted()関数を使う
    - key引数で並び替えの基準を指定
    - reverse=Trueで降順にする
    """
    
    movies = [
        {"title": "映画A", "views": 1000},
        {"title": "映画B", "views": 2500},
        {"title": "映画C", "views": 1800},
        {"title": "映画D", "views": 500},
        {"title": "映画E", "views": 3000},
    ]
    
    # TODO: ここにコードを書く
    # top3 = ...
    
    # 答えを確認
    # print("Top 3 人気映画:")
    # for i, movie in enumerate(top3, 1):
    #     print(f"{i}. {movie['title']} ({movie['views']}回視聴)")


# ===============================
# 演習1-2: 類似度計算
# ===============================
def exercise_1_2():
    """
    問題: 2つの映画のジャンルの類似度を計算してください。
    類似度 = 共通ジャンル数 / 全ジャンル数
    
    ヒント:
    - set()を使って集合演算
    - & で積集合（共通部分）
    - | で和集合（全体）
    """
    
    movie1 = {
        "title": "アベンジャーズ",
        "genres": ["アクション", "SF", "ヒーロー"]
    }
    
    movie2 = {
        "title": "スパイダーマン",
        "genres": ["アクション", "ヒーロー", "青春"]
    }
    
    # TODO: 類似度を計算
    # similarity = ...
    
    # print(f"類似度: {similarity:.2f}")


# ===============================
# 演習1-3: ユーザープロファイル
# ===============================
def exercise_1_3():
    """
    問題: ユーザーの視聴履歴から、好きなジャンルのランキングを作成してください。
    
    期待される出力:
    1. アクション: 3回
    2. SF: 2回
    3. コメディ: 1回
    """
    
    watch_history = [
        {"title": "映画A", "genres": ["アクション", "SF"]},
        {"title": "映画B", "genres": ["アクション", "コメディ"]},
        {"title": "映画C", "genres": ["SF"]},
        {"title": "映画D", "genres": ["アクション"]},
    ]
    
    # TODO: ジャンルごとの視聴回数を集計
    genre_count = {}
    
    # ここにコードを書く
    
    # 結果を表示
    # print("好きなジャンランキング:")
    # for genre, count in sorted(genre_count.items(), key=lambda x: x[1], reverse=True):
    #     print(f"  {genre}: {count}回")


# ===============================
# 演習1-4: 簡単な推薦
# ===============================
def exercise_1_4():
    """
    問題: ユーザーが好きなジャンルに基づいて、映画を推薦してください。
    
    推薦ルール:
    - ユーザーが好きなジャンルを含む映画を推薦
    - まだ見ていない映画のみ
    - 共通ジャンルが多い順に並べる
    """
    
    # ユーザーの好きなジャンル
    user_preferences = ["SF", "アクション"]
    
    # 視聴済み映画
    watched = ["映画A", "映画C"]
    
    # 映画リスト
    movies = [
        {"title": "映画A", "genres": ["SF", "アクション"]},  # 視聴済み
        {"title": "映画B", "genres": ["コメディ"]},
        {"title": "映画C", "genres": ["SF"]},  # 視聴済み
        {"title": "映画D", "genres": ["アクション", "犯罪"]},
        {"title": "映画E", "genres": ["SF", "アクション", "冒険"]},
    ]
    
    # TODO: 推薦リストを作成
    recommendations = []
    
    # ここにコードを書く
    
    # 結果を表示
    # print("おすすめ映画:")
    # for movie in recommendations:
    #     print(f"  - {movie['title']}")


# ===============================
# 演習1-5: 評価の重み付け
# ===============================
def exercise_1_5():
    """
    問題: ユーザーの評価に基づいて、加重平均を計算してください。
    
    加重平均の計算:
    - 新しい評価ほど重みを大きくする
    - 重み = 1 / (現在日 - 評価日 + 1)
    """
    
    from datetime import datetime, timedelta
    
    ratings = [
        {"movie": "映画A", "rating": 5, "date": datetime.now() - timedelta(days=7)},
        {"movie": "映画B", "rating": 3, "date": datetime.now() - timedelta(days=3)},
        {"movie": "映画C", "rating": 4, "date": datetime.now() - timedelta(days=1)},
        {"movie": "映画D", "rating": 5, "date": datetime.now()},
    ]
    
    # TODO: 加重平均を計算
    weighted_sum = 0
    weight_total = 0
    
    # ここにコードを書く
    
    # weighted_average = weighted_sum / weight_total
    # print(f"加重平均評価: {weighted_average:.2f}")


# ===============================
# チャレンジ問題
# ===============================
def challenge_problem():
    """
    チャレンジ: 完全な映画推薦システムを作る
    
    要件:
    1. ユーザーは映画を評価できる（1-5点）
    2. 3つの推薦方法を実装:
       - 人気ベース
       - ユーザーの好みベース
       - 似たユーザーベース
    3. ハイブリッド推薦（3つを組み合わせ）
    """
    
    class MovieRecommender:
        def __init__(self):
            self.movies = []
            self.users = {}
            self.ratings = {}
            
        def add_movie(self, title, genres):
            # TODO: 実装
            pass
            
        def add_user(self, user_id, name):
            # TODO: 実装
            pass
            
        def rate_movie(self, user_id, movie_title, rating):
            # TODO: 実装
            pass
            
        def recommend_popular(self, user_id, n=5):
            # TODO: 人気ベースの推薦を実装
            pass
            
        def recommend_content_based(self, user_id, n=5):
            # TODO: コンテンツベースの推薦を実装
            pass
            
        def recommend_collaborative(self, user_id, n=5):
            # TODO: 協調フィルタリングを実装
            pass
            
        def recommend_hybrid(self, user_id, n=5):
            # TODO: ハイブリッド推薦を実装
            pass
    
    # システムをテスト
    # recommender = MovieRecommender()
    # テストコードを書く


# ===============================
# メイン実行
# ===============================
if __name__ == "__main__":
    print("📚 推薦システムの基礎 - 演習問題")
    print("=" * 50)
    
    print("\n演習1-1: 人気ランキング")
    exercise_1_1()
    
    print("\n演習1-2: 類似度計算")
    exercise_1_2()
    
    print("\n演習1-3: ユーザープロファイル")
    exercise_1_3()
    
    print("\n演習1-4: 簡単な推薦")
    exercise_1_4()
    
    print("\n演習1-5: 評価の重み付け")
    exercise_1_5()
    
    print("\n🏆 チャレンジ問題")
    print("challenge_problem()を実装してみてください！")