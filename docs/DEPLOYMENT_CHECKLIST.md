# Sancta LLM Integration — Deployment Checklist

## Prerequisites

- [ ] Ubuntu/Linux or Windows with 16GB+ RAM
- [ ] Python 3.10+
- [ ] Network access (for initial model download)
- [ ] Admin/sudo privileges (for Ollama install)

## Installation Steps

### 1. Install Ollama

- [ ] Run: `curl -fsSL https://ollama.com/install.sh | sh` (Linux/Mac) or download from [ollama.com](https://ollama.com)
- [ ] Verify: `ollama --version`

### 2. Download Model

- [ ] Run: `ollama pull llama3.2`
- [ ] Verify: `ollama list` shows llama3.2

### 3. Configure Environment

- [ ] Copy: `cp .env.example .env`
- [ ] Set: `USE_LOCAL_LLM=true`
- [ ] Set: `OLLAMA_URL=http://localhost:11434`
- [ ] Set: `LOCAL_MODEL=llama3.2`

### 4. Install Dependencies

- [ ] Run: `pip install -r requirements.txt`
- [ ] Verify: `requests` is installed

### 5. Test Integration

- [ ] Start Ollama: `ollama serve` (separate terminal)
- [ ] Start SIEM: `python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787`
- [ ] Run tests: `pytest tests/test_llm_integration.py -v`
- [ ] Check UI: model status indicator in SIEM header shows "llama3.2 ready" or similar

### 6. Validate Functionality

- [ ] SIEM chat responds to queries
- [ ] Model status shows "connected"
- [ ] Response time acceptable (typically <30s for simple queries)
- [ ] Simulator endpoint works (`/simulator`)
- [ ] Fallback mode works: stop Ollama, verify chat still responds with rule-based reply

## Post-Deployment

### Performance Baseline

- [ ] Record average response time
- [ ] Note memory usage (Task Manager / htop / top)
- [ ] Document any slowness issues

### Documentation

- [ ] Team briefed on LLM capabilities
- [ ] Troubleshooting guide accessible: `docs/LLM_OPERATIONS.md`
- [ ] Model management procedures documented

### Monitoring

- [ ] Set up alerts for Ollama downtime
- [ ] Monitor response time trends
- [ ] Track error rates in logs

## Rollback Plan

If issues occur:

1. Set `USE_LOCAL_LLM=false` in `.env`
2. Restart SIEM server
3. System falls back to Anthropic (if key set) or rule-based responses
4. Investigate and fix Ollama issues
5. Re-enable when resolved

## Success Criteria

- [ ] Ollama running and stable
- [ ] SIEM chat provides intelligent responses
- [ ] Response time <30s for typical queries
- [ ] No memory leaks or crashes
- [ ] Fallback mode functional
- [ ] Team trained and confident

## Notes

Hardware specs: ___________

Model chosen: ___________

Average response time: ___________

Issues encountered: ___________
