# ============================================================
# base: 共通レイヤー（本番依存のみインストール）
# ============================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# 依存定義ファイルのみ先にコピーしてレイヤーキャッシュを最適化
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# アプリケーションコードをコピーしてプロジェクト自体をインストール
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ============================================================
# development: dev 依存を含む開発用イメージ
# ============================================================
FROM base AS development

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

CMD ["uv", "run", "python", "main.py"]

# ============================================================
# production: 最小限の本番用イメージ（非 root ユーザー）
# ============================================================
FROM python:3.13-slim-bookworm AS production

ENV PYTHONUNBUFFERED=1

# venv を base ステージからコピー
COPY --from=base /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --from=base /app .

# 非 root ユーザーで実行
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --no-create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py"]
