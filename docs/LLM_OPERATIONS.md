# Sancta LLM Operations Guide

## Starting services

```bash
# Terminal 1
ollama serve

# Terminal 2
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
```

## Monitoring

```bash
curl http://localhost:11434/api/tags
curl http://127.0.0.1:8787/api/model/info
```

## Model management

```bash
ollama list
ollama pull llama3.2
ollama pull qwen2.5:14b
ollama rm llama3.1:70b
```

Set model in `.env`:

```env
LOCAL_MODEL=qwen2.5:14b
```

## Troubleshooting

### Cannot connect to Ollama

```bash
curl http://localhost:11434/api/tags
ollama serve
```

### Slow responses

- Reduce context: set `num_ctx` lower in `backend/sancta_conversational.py` (OllamaLLMEngine.generate_chat), or try `OLLAMA_NUM_CTX=32000` if supported.
- Use a smaller model, e.g. `llama3.2:3b`.
- Check system resources: `htop` / Task Manager; `nvidia-smi` for GPU.

### Recovery

```bash
pkill ollama    # Linux/Mac — or close Ollama on Windows
ollama serve
```

---

## Additional sections

### High Memory Usage

Ollama keeps models in memory. To free:

```bash
# Stop Ollama
pkill ollama   # Linux/Mac
# or close the Ollama process on Windows

# Restart
ollama serve
```

### Slow Responses

1. Check system resources: `htop` (Linux/Mac) or Task Manager (Windows); `nvidia-smi` for GPU
2. Reduce context size in `backend/sancta_conversational.py`: set `num_ctx` lower in `generate_chat`
3. Use a smaller model: `ollama pull llama3.2:3b` and set `LOCAL_MODEL=llama3.2:3b`

### Connection Errors

Check Ollama status:

```bash
curl http://localhost:11434/api/tags
```

If it fails, restart Ollama: `ollama serve`

Check port conflicts:

```bash
# Linux/Mac
lsof -i :11434
```

## Backup and Recovery

### Backup Model Configurations

```bash
ollama list > models_backup.txt
cp .env .env.backup
```

### Restore

Reinstall models from backup:

```bash
# Manually pull each model from models_backup.txt
ollama pull llama3.2
```

## Performance Tuning

### GPU Acceleration

Ollama automatically uses GPU if available. Verify:

```bash
ollama run llama3.2 "test" --verbose
```

Output should indicate GPU device when available.

### Context Window Tuning

For faster responses, reduce context in `OllamaLLMEngine.generate_chat`:

```python
"num_ctx": 32000  # Instead of 128000
```

Trade-off: less context = faster but may miss details in large incident logs.

## Security Considerations

- Ollama listens on localhost only by default (no external access)
- No authentication required (trusted network only)
- Logs stored in: `~/.ollama/logs`
- Models stored in: `~/.ollama/models`

To restrict access:

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

## Maintenance Schedule

**Daily:** Monitor response times and system resources

**Weekly:** Review error logs; test fallback to rule-based mode

**Monthly:** Check for model updates; evaluate performance; consider model upgrades
