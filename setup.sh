#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "Starting services..."
chmod +x start.sh
./start.sh
