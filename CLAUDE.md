# voice-auth-engine

音声認証エンジン。

## 技術スタック

- Python
- パッケージ管理: uv

## コマンド

| コマンド | 説明 |
|---|---|
| `make build-dev` | 開発用 Docker イメージをビルド |
| `make build-prod` | 本番用 Docker イメージをビルド |
| `make build` | 開発用・本番用の両方をビルド |
| `make run-dev` | 開発コンテナを起動（バインドマウント） |
| `make run-prod` | 本番コンテナを起動 |
| `make test` | テスト実行（pytest） |
| `make lint` | Lint（ruff check） |
| `make lint-fix` | Lint 自動修正（ruff check --fix） |
| `make format` | フォーマット（ruff format） |
| `make shell` | 開発コンテナのシェルを開く |
| `make help` | 利用可能なコマンド一覧を表示 |
