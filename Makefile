IMAGE_NAME := voice-auth-engine
DEV_TAG := $(IMAGE_NAME):dev
PROD_TAG := $(IMAGE_NAME):latest

# ============================================================
# Build
# ============================================================

.PHONY: build-dev
build-dev: ## Build development image
	docker build --target development -t $(DEV_TAG) .

.PHONY: build-prod
build-prod: ## Build production image
	docker build --target production -t $(PROD_TAG) .

.PHONY: build
build: build-dev build-prod ## Build both images

# ============================================================
# Run
# ============================================================

.PHONY: run-dev
run-dev: ## Run development container (bind mount)
	docker run --rm -v "$$(pwd):/app" $(DEV_TAG) uv run python main.py

.PHONY: run-prod
run-prod: ## Run production container
	docker run --rm $(PROD_TAG)

# ============================================================
# Test / Lint / Format
# ============================================================

.PHONY: test
test: ## Run pytest in development container
	docker run --rm -v "$$(pwd):/app" $(DEV_TAG) uv run pytest

.PHONY: lint
lint: ## Run ruff check in development container
	docker run --rm -v "$$(pwd):/app" $(DEV_TAG) uv run ruff check .

.PHONY: lint-fix
lint-fix: ## Run ruff check --fix in development container
	docker run --rm -v "$$(pwd):/app" $(DEV_TAG) uv run ruff check --fix .

.PHONY: format
format: ## Run ruff format in development container
	docker run --rm -v "$$(pwd):/app" $(DEV_TAG) uv run ruff format .

# ============================================================
# Utilities
# ============================================================

.PHONY: shell
shell: ## Open a shell in development container
	docker run --rm -it -v "$$(pwd):/app" $(DEV_TAG) bash

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
