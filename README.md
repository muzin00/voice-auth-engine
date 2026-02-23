# voice-auth-engine

音声バイオメトリクスによる話者認証エンジン。声紋（話者埋め込み）とパスフレーズの音素照合を組み合わせた二重認証を提供する。

## 特徴

- **話者照合**: CAM++ モデルによる声紋ベースの話者認証
- **パスフレーズ検証**: 発話内容の音素照合による追加認証レイヤー
- **音素多様性チェック**: 登録時にパスフレーズの音素バリエーションを検証
- **複数音声フォーマット対応**: WAV, MP3, OGG, WebM, AAC, FLAC, M4A
- **音声区間検出**: Silero VAD による発話区間の自動検出

## セットアップ

**必要環境:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
# 依存パッケージのインストール
uv sync

# ML モデルのダウンロード（初回のみ）
uv run python scripts/download_models.py
```

## 使用モデル

| モデル | 用途 |
|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) | 音声区間検出 |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | 音声認識（日本語対応） |
| [CAM++ (3D-Speaker)](https://github.com/modelscope/3D-Speaker) | 話者埋め込み抽出 |

すべて [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 経由で ONNX 形式のモデルを使用。

## 開発

```bash
uv run pytest                          # テスト実行
uv run ruff check .                    # Lint
uv run ruff check --fix .              # Lint 自動修正
uv run ruff format .                   # フォーマット
uv run python scripts/download_models.py  # モデルダウンロード
```

## 外部プロジェクトからの利用

GitHub Releases に公開された wheel から直接インストールできる。

```bash
uv add "voice-auth-engine @ https://github.com/muzin00/voice-auth-engine/releases/download/v0.1.0/voice_auth_engine-0.1.0-py3-none-any.whl"
```

## リリース手順

1. `pyproject.toml` の `version` を更新する

   ```toml
   version = "0.2.0"
   ```

2. 変更をコミットして main にマージする

   ```bash
   git add pyproject.toml
   git commit -m "chore: bump version to 0.2.0"
   git push
   ```

3. バージョンに対応するタグを作成して push する

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

   タグ push により CI が自動実行され、テスト通過後に GitHub Release が作成される。
   ビルド成果物（wheel / sdist）が Release に添付される。

   > タグ名（`v0.2.0`）と `pyproject.toml` の version（`0.2.0`）が一致しない場合、リリースは失敗する。
