# Intent_Based_Search

## Local Setup
#### Start database
- docker compose -f docker-compose.database.yml up -d

#### change terminal directory
- cd backend

#### create virtual environment
- make venv && source venv/bin/activate

#### install dependencies
- make install

#### run app
- make dev

#### deactivate virtual environment
- deactivate

## fine-tuning
- python fine_tune.py

<!-- #### data store in qdrant - text -> embedding -> vector database
- python init_data.py -->