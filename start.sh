#!/bin/bash

echo "Starting Gemma3 FastAPI service..."

# Start Ollama in the background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
max_attempts=30
attempt=0
while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    attempt=$((attempt+1))
    if [ $attempt -ge $max_attempts ]; then
        echo "ERROR: Ollama failed to start after $max_attempts attempts."
        exit 1
    fi
    echo "Waiting for Ollama to become available... (Attempt $attempt/$max_attempts)"
    sleep 5
done
echo "Ollama is running!"

# Pull the Gemma 3 model
DEFAULT_MODEL=${DEFAULT_MODEL:-"gemma3:27b"}
echo "Downloading model: $DEFAULT_MODEL..."
ollama pull $DEFAULT_MODEL

# Verify model is accessible
echo "Verifying model access..."
ollama list

# Start FastAPI application
echo "Starting FastAPI application..."
cd /workspace/gemma
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info &
FASTAPI_PID=$!

echo "Service is up and running!"
echo "Access the API at port 8000 and Ollama directly at port 11434"
echo "FastAPI PID: $FASTAPI_PID"
echo "Ollama PID: $OLLAMA_PID"

# Keep the container running
wait $FASTAPI_PID

# If FastAPI exits, kill Ollama
kill $OLLAMA_PID
exit 0
