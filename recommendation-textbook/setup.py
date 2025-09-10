#!/usr/bin/env python3
"""
推薦システム教科書のセットアップスクリプト
必要な環境を整えて、デモを実行できるようにします
"""

import os
import sys
import subprocess
import platform


def check_python_version():
    """Pythonバージョンをチェック"""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8以上が必要です")
        return False
    
    print("✅ Pythonバージョン OK")
    return True


def install_requirements():
    """必要なパッケージをインストール"""
    print("\n📦 パッケージをインストール中...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ パッケージインストール完了")
        return True
    except subprocess.CalledProcessError:
        print("❌ パッケージインストールに失敗しました")
        print("手動で以下を実行してください:")
        print("  pip install -r requirements.txt")
        return False


def create_directories():
    """必要なディレクトリを作成"""
    dirs = [
        "data/sample",
        "data/output",
        "visualizations/output",
        "projects/output"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ ディレクトリ作成完了")


def run_tests():
    """基本的なテストを実行"""
    print("\n🧪 動作テスト中...")
    
    # 基本的な推薦システムをテスト
    try:
        from code.basic_recommender import SimpleRecommenderSystem
        
        system = SimpleRecommenderSystem()
        system.add_item("item1", "テストアイテム", "カテゴリA")
        system.add_user("user1", "テストユーザー")
        
        print("✅ 基本推薦システム OK")
        return True
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def run_demo():
    """デモを実行"""
    print("\n🎮 デモを起動しますか？")
    choice = input("1. 基本デモ\n2. ミニTwitter\n3. スキップ\n選択 (1-3): ")
    
    if choice == "1":
        print("\n" + "=" * 50)
        print("基本的な推薦システムのデモ")
        print("=" * 50)
        
        try:
            from code.basic_recommender import demo
            demo()
        except Exception as e:
            print(f"エラー: {e}")
            
    elif choice == "2":
        print("\n" + "=" * 50)
        print("ミニTwitterを起動")
        print("=" * 50)
        
        try:
            from projects.mini_twitter import InteractiveTwitter
            app = InteractiveTwitter()
            app.run()
        except Exception as e:
            print(f"エラー: {e}")


def print_welcome():
    """ウェルカムメッセージ"""
    print("""
    ╔════════════════════════════════════════════╗
    ║                                            ║
    ║   🚀 推薦システム完全マスター教科書 🚀      ║
    ║                                            ║
    ║   X(Twitter)のアルゴリズムで学ぶ           ║
    ║   実践的推薦技術                           ║
    ║                                            ║
    ╚════════════════════════════════════════════╝
    """)
    
    print("📚 学習内容:")
    print("  • 推薦システムの基礎")
    print("  • 協調フィルタリング")
    print("  • SimClustersアルゴリズム")
    print("  • リアルタイム推薦")
    print("  • ミニTwitterの実装")
    print()


def print_next_steps():
    """次のステップを表示"""
    print("\n" + "=" * 50)
    print("📖 次のステップ")
    print("=" * 50)
    
    print("""
1. 教科書を読む:
   📂 chapters/
   - chapter01_introduction.md    : はじめに
   - chapter02_first_recommender.md : 最初の推薦システム
   
2. コードを実行:
   📂 code/
   - python code/basic_recommender.py
   - python code/twitter_simclusters.py
   
3. 演習問題に挑戦:
   📂 exercises/
   - python exercises/exercise01_basics.py
   
4. プロジェクトを作る:
   📂 projects/
   - python projects/mini_twitter.py
   
5. 解答を確認:
   📂 solutions/
   - python solutions/solution01_basics.py
    """)
    
    print("\n💡 ヒント:")
    print("  • Jupyter Notebookで対話的に学習:")
    print("    jupyter notebook")
    print("  • VSCodeで快適な開発環境:")
    print("    code .")
    print()
    print("頑張ってください！ 🎯")


def main():
    """メイン処理"""
    print_welcome()
    
    # 環境チェック
    print("🔍 環境をチェック中...")
    print("-" * 30)
    
    if not check_python_version():
        return
    
    # セットアップ
    if not install_requirements():
        print("\n⚠️ 一部のセットアップが失敗しましたが続行します")
    
    create_directories()
    
    if run_tests():
        print("\n✅ セットアップ完了！")
    else:
        print("\n⚠️ 一部のテストが失敗しました")
    
    # デモ実行
    run_demo()
    
    # 次のステップ
    print_next_steps()


if __name__ == "__main__":
    main()