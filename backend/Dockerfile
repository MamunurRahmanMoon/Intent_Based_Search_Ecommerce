# Use an official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]