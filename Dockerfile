FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and required dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create working directory
WORKDIR /workspace/gemma

# Copy application files
COPY requirements.txt .
COPY main.py .
COPY start.sh .
COPY docker-compose.yml .
COPY gemma_UI_with_formated.html .
COPY runpod.json .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Make start script executable
RUN chmod +x start.sh

# Expose ports
EXPOSE 8000 11434 8800

# Set environment variables
ENV OLLAMA_HOST=localhost
ENV DEFAULT_MODEL=gemma3:27b

# Command to run the server
CMD ["/workspace/gemma/start.sh"]
