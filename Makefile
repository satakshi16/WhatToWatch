.DEFAULT_GOAL := help
.PHONY: help install lint fmt typecheck test api ui ingest open-content embed neighbours bench-recs migrate seed-demo

UV ?= uv
HOST ?= 127.0.0.1
PORT ?= 8000

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-13s %s\n", $$1, $$2}'

install: ## Create venv, install deps (+dev), git hooks; .env from .env.example (random JWT key)
	$(UV) sync
	@[ -f .env ] || ($(UV) run python -c "import re, secrets, pathlib; t = pathlib.Path('.env.example').read_text(); pathlib.Path('.env').write_text(re.sub(r'(?m)^JWT_SECRET_KEYS=change-me', 'JWT_SECRET_KEYS=' + secrets.token_hex(32), t))" \
		&& echo "Created .env from .env.example (with a random JWT_SECRET_KEYS)")
	@if [ -d .git ]; then $(UV) run pre-commit install; fi

lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## Auto-fix lint issues and format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## mypy --strict on app/
	$(UV) run mypy

RECSYS_COV_MIN ?= 60

test: ## Run test suite (ui group for AppTest) + coverage gate on app/recsys (§5.6)
	$(UV) run --group ui pytest --cov=app/recsys --cov-report=term --cov-fail-under=$(RECSYS_COV_MIN)

api: migrate ## Run FastAPI dev server (with the ml group: seeds are embedded on the fly)
	$(UV) run --group ml uvicorn app.main:app --reload --host $(HOST) --port $(PORT) --no-access-log

ui: ## Run the Streamlit UI on http://localhost:8501 (needs `make api`)
	$(UV) run --group ui streamlit run ui/streamlit_app.py

ingest: migrate ## Ingest TMDB catalog (flags via ARGS="--limit 200")
	$(UV) run python -m app.ingest.ingest_catalog $(ARGS)

open-content: ## Wikidata + Wikipedia content for catalog titles (ARGS="--limit 200")
	$(UV) run python -m app.ingest.open_content $(ARGS)

embed: ## Build title embeddings from open content (ARGS="--incremental")
	$(UV) run --group ml python -m app.ingest.build_embeddings $(ARGS)

neighbours: ## Print nearest neighbours for known titles (embedding sanity check)
	$(UV) run python -m scripts.embedding_neighbours $(ARGS)

bench-recs: ## Recommender latency on a synthetic 15k catalog (ARGS="--titles 15000 --runs 50")
	$(UV) run python -m scripts.bench_recommender $(ARGS)

migrate: ## Apply DB migrations (alembic upgrade head; also run by ingest / api / seed-demo)
	$(UV) run alembic upgrade head

seed-demo: migrate ## Demo users Priya / Marcus / Ji-woo via the API (ARGS="--api-url http://127.0.0.1:8000")
	$(UV) run --group ml python -m scripts.seed_demo $(ARGS)
