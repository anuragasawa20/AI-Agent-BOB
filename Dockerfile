FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy the rest of the application
COPY . .

# Create directory for persistent embeddings
RUN mkdir -p /app/whisky_embeddings

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080
ENV PERSIST_DIR=/app/whisky_embeddings
ENV DATA_PATH=/app/501\ Bottle\ Dataset.csv

# Expose the port
EXPOSE ${PORT}

# Run the application with gunicorn
CMD gunicorn --bind 0.0.0.0:${PORT} deployment:app 