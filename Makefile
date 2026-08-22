.PHONY: install install-dev install-embeddings docs seed api frontend evaluate simulate test lint docker-build docker-up docker-down reset-store

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

install-embeddings:
	pip install -r requirements-embeddings.txt
	pip install -e .

docs:
	python scripts/make_sample_documents.py

seed: docs
	python scripts/seed_documents.py

api:
	uvicorn sourceiq.api.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

evaluate:
	python scripts/evaluate.py

simulate:
	python scripts/simulate_queries.py --n 25

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

docker-build:
	docker build -t sourceiq-api:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

reset-store:
	python scripts/reset_store.py --yes
