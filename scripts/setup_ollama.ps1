# Sancta LLM Setup Script (Windows)
# Installs Ollama and pulls the llama3.2 model for local LLM integration.

Write-Host "=== Sancta LLM Setup Script ===" -ForegroundColor Cyan
Write-Host ""

# Check if Ollama is in PATH
$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaPath) {
    Write-Host "Ollama not found in PATH." -ForegroundColor Yellow
    Write-Host "Install from: https://ollama.com/download"
    Write-Host "Or run: winget install Ollama.Ollama"
    Write-Host ""
    exit 1
}

Write-Host "Ollama found: $($ollamaPath.Source)" -ForegroundColor Green
Write-Host ""

# Pull the model
Write-Host "Pulling llama3.2 model (this may take several minutes)..." -ForegroundColor Yellow
& ollama pull llama3.2
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to pull model." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
& ollama list

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Start Ollama server (if not running): ollama serve"
Write-Host "2. Copy .env.example to .env"
Write-Host "3. Set USE_LOCAL_LLM=true in .env"
Write-Host "4. Start SIEM: python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787"
Write-Host "5. Run tests: pytest tests/test_llm_integration.py -v"
Write-Host ""
