# Intent_Based_Search

## Local Setup
#### Start database
- docker-compose -f docker-compose.database.yml up -d

#### create virtual environment
- make activate

#### install dependencies
- make install

#### run app
- make run

#### deactivate virtual environment
- make deactivate

## fine-tuning
- python fine_tune.py

<!-- #### data store in qdrant - text -> embedding -> vector database
- python init_data.py -->