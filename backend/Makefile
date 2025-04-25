activate:
	python3 -m venv venv
	source venv/bin/activate

deactivate:
	deactivate

install:
	pip install -r requirements.txt

dev:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

