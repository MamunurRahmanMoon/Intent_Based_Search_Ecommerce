# E-commerce Semantic Search System

This project is a semantic search system for e-commerce platforms. It enables natural language search in Banglish (Bengali-English mixed language) using dense vector embeddings and a vector database.

---

## Features
- **Semantic Search**: Supports natural language queries.
- **Banglish Support**: Handles mixed Bengali-English queries.
- **Vector Database**: Uses Qdrant for storing embeddings.
- **Relational Database**: Uses MySQL for storing raw product data.
- **FastAPI Backend**: Provides a RESTful API for search functionality.

---

## Project Structure
```
.
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Dockerfile for FastAPI app
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── data/                    # Data folder
│   ├── processed/           # Processed data
│   └── raw/                 # Raw data (e.g., product.csv)
├── logs/                    # Logs folder
│   └── app.log              # Application logs
└── src/                     # Source code
    ├── api.py               # FastAPI application
    ├── data_loader.py       # Data loading and processing
    ├── embedding_model.py   # Embedding generation
    ├── initi_db.py          # Database initialization script
    ├── logger.py            # Centralized logging configuration
    ├── mysql_database.py    # MySQL database integration
    ├── vector_database.py   # Qdrant integration
```

---

## Setup Guide

### **With Docker (Recommended)**

1. **Prerequisites**:
   - Install [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/install/).

2. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd ecommerce-semantic-search
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory with the following content:
     ```properties
     # MySQL Configuration
     MYSQL_HOST=mysql
     MYSQL_USER=root
     MYSQL_PASSWORD=1234

     # Qdrant Configuration
     QDRANT_URL=http://qdrant:6333
     QDRANT_API_KEY=  # Leave empty if not using an API key
     QDRANT_COLLECTION=ecommerce
     ```

4. **Start the Services**:
   - Build and start all services (MySQL, Qdrant, and FastAPI app):
     ```bash
     docker-compose up --build
     ```

5. **Verify the Setup**:
   - **FastAPI**: Open [http://localhost:8000/docs](http://localhost:8000/docs) to access the API documentation.
   - **MySQL**: Ensure the database is accessible on port `3306`.
   - **Qdrant**: Verify it is running on [http://localhost:6333](http://localhost:6333).

6. **Run the Initialization Script**:
   - Open a new terminal and run the following command to initialize the database and load data:
     ```bash
     docker exec -it fastapi_app python src/initi_db.py
     ```

7. **Test the API**:
   - Use the `/search` endpoint to perform semantic search:
     ```bash
     curl -X POST "http://localhost:8000/search" \
          -H "Content-Type: application/json" \
          -d '{"query": "comfortable T-shirt", "top_k": 3}'
     ```

---

### **Without Docker (Manual Setup)**

1. **Prerequisites**:
   - Install the following:
     - [Python 3.8+](https://www.python.org/downloads/)
     - [MySQL](https://dev.mysql.com/downloads/)
     - [Docker](https://www.docker.com/) (for Qdrant)

2. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd ecommerce-semantic-search
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory with the following content:
     ```properties
     # MySQL Configuration
     MYSQL_HOST=localhost
     MYSQL_USER=root
     MYSQL_PASSWORD=1234

     # Qdrant Configuration
     QDRANT_URL=http://localhost:6333
     QDRANT_API_KEY=  # Leave empty if not using an API key
     QDRANT_COLLECTION=ecommerce
     ```

4. **Set Up MySQL**:
   - Start the MySQL server.
   - Create the `ecommerce` database manually or let the script handle it during initialization.

5. **Start Qdrant Using Docker**:
   - Run the following command to start Qdrant:
     ```bash
     docker run -d -p 6333:6333 qdrant/qdrant:v1.3.0
     ```

6. **Set Up Python Environment**:
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     venv\Scripts\activate  # On Windows
     source venv/bin/activate  # On macOS/Linux
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

7. **Run the Initialization Script**:
   - Initialize the database and load data:
     ```bash
     python src/initi_db.py
     ```

8. **Start the FastAPI Server**:
   - Run the FastAPI app:
     ```bash
     uvicorn src.api:app --reload
     ```

9. **Verify the Setup**:
   - **FastAPI**: Open [http://localhost:8000/docs](http://localhost:8000/docs) to access the API documentation.
   - **MySQL**: Ensure the database is accessible on port `3306`.
   - **Qdrant**: Verify it is running on [http://localhost:6333](http://localhost:6333).

10. **Test the API**:
    - Use the `/search` endpoint to perform semantic search:
      ```bash
      curl -X POST "http://localhost:8000/search" \
           -H "Content-Type: application/json" \
           -d '{"query": "comfortable T-shirt", "top_k": 3}'
      ```

---

## Comparison: With Docker vs. Without Docker

| Feature                  | With Docker                     | Without Docker                  |
|--------------------------|----------------------------------|---------------------------------|
| **Ease of Setup**        | Single command (`docker-compose up`) | Manual installation required   |
| **Environment Consistency** | Fully consistent across systems | May vary based on local setup   |
| **Portability**          | Highly portable                 | Less portable                  |
| **Dependency Management**| Handled by Docker               | Requires manual management      |
| **Flexibility**          | Easy to scale and extend        | More effort to scale and extend |

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.# Intent_Based_Search_Ecommerce
