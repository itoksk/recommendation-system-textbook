# Chapter 1: 推薦システムって何？ 🤔

## 📱 まずは身近な例から考えてみよう

### あなたも毎日使っている推薦システム

朝起きてから寝るまで、私たちは無意識に推薦システムを使っています：

- **YouTube**: 「次の動画」が自動で提案される
- **Spotify**: 「あなたのためのプレイリスト」
- **Instagram**: フィードに表示される投稿の順番
- **Amazon**: 「この商品を見た人はこちらも」
- **Netflix**: 「あなたにおすすめ」の映画

これらはすべて**推薦システム**（レコメンドシステム）の例です！

## 🎯 なぜ推薦システムが必要なの？

### 情報爆発問題

```python
# 1日に生まれるコンテンツの量を見てみよう
content_per_day = {
    "YouTubeの動画": 720_000,  # 時間
    "Twitterの投稿": 500_000_000,  # ツイート
    "Instagramの写真": 95_000_000,  # 枚
    "ブログ記事": 7_500_000,  # 記事
}

# 1人が1日に消費できるコンテンツ
human_capacity = {
    "動画視聴": 3,  # 時間
    "SNS投稿閲覧": 200,  # 投稿
    "記事を読む": 10,  # 記事
}

# 問題：どうやって自分に合うものを見つける？
print("YouTubeの動画を全部見るには...")
print(f"{content_per_day['YouTubeの動画'] / 24 / 365:.0f}年かかります！")
```

### 推薦システムの役割

推薦システムは、この膨大な選択肢から**あなたが好きそうなもの**を選んでくれる**フィルター**です。

## 🔬 推薦システムの3つの基本戦略

### 1. 人気ベース（Popularity-based）
「みんなが好きなものは、あなたも好きかも」

```python
def recommend_popular(items):
    """最も人気のあるアイテムを推薦"""
    # いいね数でソート
    sorted_items = sorted(items, key=lambda x: x['likes'], reverse=True)
    return sorted_items[:10]

# 例：今週の人気ツイート
tweets = [
    {"id": 1, "text": "おはよう！", "likes": 100},
    {"id": 2, "text": "プログラミング楽しい", "likes": 500},
    {"id": 3, "text": "今日のランチ", "likes": 50},
]

popular = recommend_popular(tweets)
print(f"人気No.1: {popular[0]['text']} ({popular[0]['likes']}いいね)")
```

### 2. コンテンツベース（Content-based）
「過去に好きだったものと似たものを推薦」

```python
def recommend_similar(liked_item, all_items):
    """好きなアイテムと似たものを推薦"""
    similar_items = []
    
    for item in all_items:
        # タグが共通しているか確認
        common_tags = set(liked_item['tags']) & set(item['tags'])
        if common_tags and item != liked_item:
            similar_items.append({
                'item': item,
                'similarity': len(common_tags)
            })
    
    # 類似度の高い順にソート
    similar_items.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_items[:5]

# 例：動画の推薦
liked_video = {
    "title": "Python入門",
    "tags": ["プログラミング", "Python", "初心者"]
}

all_videos = [
    {"title": "JavaScript入門", "tags": ["プログラミング", "JavaScript", "初心者"]},
    {"title": "Python応用", "tags": ["プログラミング", "Python", "上級"]},
    {"title": "料理の基本", "tags": ["料理", "初心者"]},
]

recommendations = recommend_similar(liked_video, all_videos)
print(f"おすすめ: {recommendations[0]['item']['title']}")
```

### 3. 協調フィルタリング（Collaborative Filtering）
「あなたと似た人が好きなものを推薦」

```python
def find_similar_users(target_user, all_users):
    """似た趣味の人を見つける"""
    similar_users = []
    
    for user in all_users:
        if user['id'] != target_user['id']:
            # 共通の「いいね」を数える
            common_likes = set(target_user['likes']) & set(user['likes'])
            if common_likes:
                similar_users.append({
                    'user': user,
                    'similarity': len(common_likes)
                })
    
    return sorted(similar_users, key=lambda x: x['similarity'], reverse=True)

# 例：音楽の推薦
you = {
    "id": "user1",
    "likes": ["曲A", "曲B", "曲C"]
}

other_users = [
    {"id": "user2", "likes": ["曲A", "曲B", "曲D"]},  # 2曲共通
    {"id": "user3", "likes": ["曲E", "曲F"]},         # 0曲共通
    {"id": "user4", "likes": ["曲A", "曲C", "曲G"]},  # 2曲共通
]

similar = find_similar_users(you, other_users)
print(f"最も似ているユーザー: {similar[0]['user']['id']}")
print(f"そのユーザーが好きな新しい曲を推薦します！")
```

## 🐦 X(Twitter)の推薦システム

Twitterは、これらすべての方法を組み合わせた**ハイブリッド型**推薦システムを使っています。

### Twitterのタイムラインができるまで

```python
class TwitterTimeline:
    def __init__(self):
        self.stages = [
            "1. 候補集め（数十万ツイート）",
            "2. 軽い採点（数千に絞る）",
            "3. 詳細な採点（順位を決める）",
            "4. 最終調整（多様性を確保）",
            "5. 表示！"
        ]
    
    def build_timeline(self, user):
        print(f"📱 {user}さんのタイムラインを作成中...")
        
        for stage in self.stages:
            print(f"  {stage}")
            # 実際の処理（簡略化）
            import time
            time.sleep(0.5)
        
        print("✅ タイムライン完成！")
        return ["ツイート1", "ツイート2", "ツイート3"]

# 実行してみよう
timeline_builder = TwitterTimeline()
your_timeline = timeline_builder.build_timeline("あなた")
```

## 🎯 練習問題

### 問題1：人気ランキングを作ろう
以下のデータから、最も人気のある映画Top3を表示するプログラムを書いてください。

```python
movies = [
    {"title": "映画A", "views": 1000, "rating": 4.5},
    {"title": "映画B", "views": 2000, "rating": 3.8},
    {"title": "映画C", "views": 1500, "rating": 4.2},
    {"title": "映画D", "views": 500, "rating": 4.9},
    {"title": "映画E", "views": 3000, "rating": 3.5},
]

# ここにあなたのコードを書く
def get_top_movies(movies, n=3):
    # TODO: 実装する
    pass

# テスト
top3 = get_top_movies(movies)
for i, movie in enumerate(top3, 1):
    print(f"{i}位: {movie['title']} (視聴数: {movie['views']})")
```

### 問題2：似ているものを見つけよう
あなたが「アクション映画」が好きだとして、似たジャンルの映画を推薦するプログラムを書いてください。

```python
your_favorite = {
    "title": "アベンジャーズ",
    "genres": ["アクション", "SF", "ヒーロー"]
}

movie_list = [
    {"title": "スパイダーマン", "genres": ["アクション", "ヒーロー"]},
    {"title": "ラブストーリー", "genres": ["恋愛", "ドラマ"]},
    {"title": "スターウォーズ", "genres": ["SF", "冒険"]},
    {"title": "アイアンマン", "genres": ["アクション", "SF", "ヒーロー"]},
]

# ここにあなたのコードを書く
def find_similar_movies(favorite, movies):
    # TODO: 実装する
    pass

# テスト
recommendations = find_similar_movies(your_favorite, movie_list)
print("あなたにおすすめの映画:")
for movie in recommendations:
    print(f"- {movie}")
```

## 💡 考えてみよう

1. **なぜYouTubeは、あなたが好きそうな動画を知っているの？**
   - ヒント：視聴履歴、いいね、視聴時間...

2. **推薦システムの良い点と悪い点は？**
   - 良い点：便利、新しい発見...
   - 悪い点：フィルターバブル、プライバシー...

3. **もしあなたが推薦システムを作るなら？**
   - 何を推薦する？（音楽、本、友達...）
   - どんなデータを使う？
   - どうやって「良い推薦」を判断する？

## 🚀 次のステップ

次の章では、実際にPythonで**動く推薦システム**を作ります！
- データの準備方法
- ユーザーの好みを記録する仕組み
- 簡単な推薦アルゴリズムの実装

準備はいいですか？ Let's build! 🛠️

---

### 📝 まとめ

- 推薦システムは、膨大な選択肢から**あなたに合うもの**を選ぶ
- 3つの基本戦略：**人気**、**コンテンツ**、**協調フィルタリング**
- Twitterは複数の方法を組み合わせた**ハイブリッド型**
- 推薦システムは私たちの生活を便利にしているが、課題もある

[→ Chapter 2: はじめての推薦システムへ](chapter02_first_recommender.md)