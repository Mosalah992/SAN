# Rebuild ChatML corpus + QLoRA fine-tune (Llama 3.2 1B by default from .env SANCTA_BASE_MODEL)
# + merge + optional GGUF for Ollama model name: sancta-analyst (see docs/FINETUNE_GUIDE.md)
#
# Prerequisites: CUDA torch, pip install -r requirements-training.txt, HF_TOKEN for gated Llama
#
# Usage:
#   .\train_sancta.ps1
#   .\train_sancta.ps1 -Epochs 5
#   .\train_sancta.ps1 -SkipCorpus -SkipGGUF

param(
    [int] $Epochs = 3,
    [switch] $SkipCorpus,
    [switch] $SkipGGUF
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $SkipCorpus) {
    python backend/build_training_corpus.py
}

$pyArgs = @("train_sancta_llm.py", "--epochs", "$Epochs", "--merge")
if (-not $SkipGGUF) { $pyArgs += "--export-gguf" }
python @pyArgs

Write-Host ""
Write-Host "Ollama: create Modelfile FROM ./sancta-analyst-q4_k_m.gguf then: ollama create sancta-analyst -f Modelfile"
Write-Host ".env:   USE_LOCAL_LLM=true  LOCAL_MODEL=sancta-analyst"
