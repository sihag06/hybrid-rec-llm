PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate
APP_NAME ?= llm-recommender

.PHONY: setup install config-check lint test docker-build docker-run clean

setup:
	$(PYTHON) -m venv $(VENV)

install: setup
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt

config-check:
	$(ACTIVATE) && PYTHONPATH=. $(PYTHON) config.py --print

lint:
	@echo "Add linting tools (ruff/black/flake8) here"

test:
	$(ACTIVATE) && PYTHONPATH=. pytest

docker-build:
	docker build -t $(APP_NAME):dev .

docker-run:
	docker run --rm -it -p 8000:8000 -p 3000:3000 --env-file .env.example $(APP_NAME):dev

clean:
	rm -rf $(VENV) __pycache__ */__pycache__
