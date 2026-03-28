# Local LLM Integration — Alignment with Codebase

## Summary vs. Actual Implementation

| Item | Changelog / Plan | Our Codebase | Notes |
|------|-----------------|--------------|-------|
| Conversational module | "new sancta_conversational.py" | **Extended** `backend/sancta_conversational.py` | We added OllamaLLMEngine; kept Anthropic and templates |
| SIEM backend | siem_dashboard/server.py | `backend/siem_server.py` | Different path — we use FastAPI in backend/ |
| POST /api/chat | ✓ | ✓ | Uses `message`, `session_id`, optional `incident_logs` |
| POST /api/simulator/generate | `{prompt, context}` | `{system, messages, max_tokens}` | Simulator expects Anthropic-style payload |
| GET /api/model/info | ✓ | ✓ | No auth; returns status, model, ollama_url |
| Frontend status | ✓ | ✓ | Header indicator + 30s polling |
| Config | ✓ | ✓ | requests, .env.example Ollama vars |
| Scripts | setup_ollama.sh | setup_ollama.sh + setup_ollama.ps1 | We have both |
| Tests | test_llm_integration.py | ✓ | Uses pytest |

## Corrected Test Commands

**Note:** `siem_dashboard/server.py` does not exist. Use `backend.siem_server` (uvicorn).

### 1. Syntax check (no server needed)
```powershell
python -m py_compile backend/sancta_conversational.py backend/siem_server.py tests/test_llm_integration.py
```

### 2. Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Start SIEM (Terminal 1)
```powershell
cd "e:\CODE PROKECTS\sancta-main\sancta-main"
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
```

### 4. Start Ollama (Terminal 2 — if using local LLM)
```powershell
ollama serve
```

### 5. API tests (Terminal 3 — server must be running)
```powershell
# Model info (no auth)
curl -sS http://127.0.0.1:8787/api/model/info

# Chat (add -H "Authorization: Bearer YOUR_TOKEN" if SIEM_AUTH_TOKEN is set)
curl -sS -X POST http://127.0.0.1:8787/api/chat -H "Content-Type: application/json" -d "{\"message\":\"Analyze failed login attempts\"}"

# Simulator — use system/messages/max_tokens (NOT prompt/context)
curl -sS -X POST http://127.0.0.1:8787/api/simulator/generate -H "Content-Type: application/json" -d "{\"system\":\"You are a security analyst.\",\"messages\":[{\"role\":\"user\",\"content\":\"Generate a phishing incident scenario\"}],\"max_tokens\":50}"
```

### 6. Pytest
```powershell
pytest tests/test_llm_integration.py -v
# Or: python tests/test_llm_integration.py
```
