# GitHubへのプッシュ手順

## 1. GitHub CLIでログイン（推奨）

```bash
# GitHub CLIでログイン
gh auth login

# Webブラウザでの認証を選択し、指示に従ってください
```

## 2. リポジトリを作成してプッシュ

### オプションA: GitHub CLI使用（ログイン後）

```bash
# リポジトリ作成
gh repo create itoksk/recommendation-system-textbook \
  --public \
  --description "X(Twitter)のアルゴリズムで学ぶ推薦システム完全マスター教科書" \
  --source=. \
  --push
```

### オプションB: 手動でGitHubウェブサイトから作成

1. https://github.com/new にアクセス
2. Repository name: `recommendation-system-textbook`
3. Description: `X(Twitter)のアルゴリズムで学ぶ推薦システム完全マスター教科書`
4. Public を選択
5. "Create repository" をクリック
6. 以下のコマンドを実行:

```bash
# リモートリポジトリを追加
git remote add origin https://github.com/itoksk/recommendation-system-textbook.git

# mainブランチに切り替え（必要な場合）
git branch -M main

# プッシュ
git push -u origin main
```

## 3. 確認

ブラウザで https://github.com/itoksk/recommendation-system-textbook にアクセスして確認

## トラブルシューティング

### 認証エラーの場合

Personal Access Token を使用:
1. https://github.com/settings/tokens/new にアクセス
2. "Generate new token (classic)" を選択
3. スコープで "repo" を選択
4. トークンを生成してコピー
5. `git push` 時のパスワードとして使用

### 既にリポジトリが存在する場合

```bash
# 既存のリモートを削除
git remote remove origin

# 新しいリモートを追加
git remote add origin https://github.com/itoksk/recommendation-system-textbook.git

# 強制プッシュ（注意：既存のコンテンツを上書き）
git push -u origin main --force
```