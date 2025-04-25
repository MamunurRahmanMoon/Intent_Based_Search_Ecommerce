venv:
	python3 -m venv venv

activate:
	@echo "Please run 'source venv/bin/activate' in your shell"

decactivate:
	deactivate

install:
	pip install -r requirements.txt

dev:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

