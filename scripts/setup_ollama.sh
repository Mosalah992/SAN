#!/bin/bash
# Sancta LLM Setup Script (Linux/Mac)
# Installs Ollama and pulls the llama3.2 model for local LLM integration.

echo "=== Sancta LLM Setup Script ==="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✓ Ollama already installed"
fi

# Pull the model
echo ""
echo "Pulling llama3.2 model (this may take several minutes)..."
ollama pull llama3.2

# Test the installation
echo ""
echo "Testing Ollama..."
ollama list

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Start Ollama server: ollama serve"
echo "2. Update .env with: USE_LOCAL_LLM=true"
echo "3. Start SIEM server: python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787"
echo "4. Run tests: pytest tests/test_llm_integration.py -v"
echo ""
