# Sancta SIEM — Debug Report & Fixes

**Date:** 2025-03-19  
**Scope:** Full codebase scan, frontend/backend integration, ease-of-use improvements

---

## 1. Bugs Fixed

### High Severity

| Bug | Location | Description | Fix |
|-----|----------|-------------|-----|
| Chat scroll container | `js/tabs/chat.js` | Auto-scroll targeted `#chat-messages` instead of scroll parent `#chat-messages-wrap`; new messages did not scroll into view | Use `scrollChat()` on `chat-messages-wrap` |
| Chat null safety | `js/tabs/chat.js` | `chatSend.disabled`, `messagesEl.scrollTop` could throw when DOM elements missing | Added guards: `chatSend?.disabled`, `scrollChat()` helper with null check |
| Lab pipeline 422 | `js/tabs/lab.js`, `api.js` | `runPipelinePhase('redteam')` sent string; backend expects `phase: int` 1–7 | Mapped `redteam`→6 in `PHASE_MAP`; `runPipelinePhase` accepts string or number |
| _tail_jsonl_sync NameError | `siem_server.py` | Non-`OSError` exceptions left `data` unset before `data.decode()` | Initialize `data = b""`; add `except Exception: return []` |

### Medium Severity

| Bug | Location | Description | Fix |
|-----|----------|-------------|-----|
| inner_circle / recruited | `js/app.js` | Backend returns `inner_circle_count`, `recruited_count`; frontend expected `inner_circle`, `recruited` | `applyStatus`: use `m.inner_circle ?? m.inner_circle_count`, `m.recruited ?? m.recruited_count` |
| onMetrics state mutation | `js/app.js` | `Object.assign(S, msg.agent)` could overwrite S with arbitrary keys | Whitelist agent fields: only `running`→`agentRunning`, `pid`→`agentPid`, `suspended`→`agentSuspended` |
| WebSocket metrics handling | `js/websocket.js` | `msg.agent !== undefined` treated any agent field as metrics; malformed payloads could break logic | Validate `msg.metrics` or `msg.type === 'metrics'`; require `typeof msg.agent === 'object'` when handling agent |
| Epidemic runSim state | `js/tabs/epidemic.js` | `Object.assign(S, status)` overwrote `S.seir` with object instead of string | Merge only epidemic fields: `seir`, `driftScore`, `alertLevel`, `epidSignals`, `epidParams` |
| color-mix fallback | `styles/chrome.css` | `color-mix()` unsupported in older browsers | Added `@supports` block; solid `var(--bg-panel)` fallback |
| Control mode select | `js/tabs/control.js` | Empty `ctrlModeSelect?.value` could send `''` | `getMode()` returns `(value || 'passive').trim() || 'passive'` |

### Low Severity

| Bug | Location | Description | Fix |
|-----|----------|-------------|-----|
| Lab error message | `js/tabs/lab.js` | `data?.error` sometimes absent | Use `data?.error ?? data?.detail ?? 'unknown'` |
| Lab result text | `js/tabs/lab.js` | Phase 6 runs eval, not custom payload; label was misleading | Result text: "Evaluation completed" + `data.detail` |
| Epistemic guard | `js/app.js` | `Object.assign(S.epistemic, data.epistemic)` when non-object | Added `typeof data.epistemic === 'object'` check (pre-existing) |
| Backend snapshot alias | `siem_server.py` | Frontend could use either naming | Added `inner_circle`, `recruited` as aliases in `_agent_state_extras()` |
| Epidemic SVG collapse | `styles/layout.css` | `#epid-network-svg` could collapse when parent empty | Added `min-height: 150px` |
| Reduced motion | `styles/animations.css` | No `prefers-reduced-motion` support | Added media query to disable/reduce animations |

---

## 2. Ease-of-Use Features Added

| Feature | Location | Description |
|---------|----------|-------------|
| Toast notifications | `app.js`, `chrome.css`, `index.html` | `showToast(msg, type)` for user feedback; Control tab errors now surface as toasts |
| Control error feedback | `control.js` | API failures (e.g. 401, 500) show toast instead of only console |
| Chat clear confirm | `chat.js` | Confirm before clearing messages when any exist |
| Kill/Restart confirm | `control.js` | Confirm before Stop or Restart when agent is running |
| Scroll helper | `chat.js` | `scrollChat()` centralizes scroll-to-bottom for new messages |

---

## 3. Summary Statistics

| Severity | Found | Fixed |
|----------|-------|------|
| High | 4 | 4 |
| Medium | 6 | 6 |
| Low | 6 | 6 |
| **Total** | **16** | **16** |

---

## 4. Roadmap — Features & Fixes for Onwards

### High Priority

1. **Dedicated Red-Team Injection API** — Lab "Run" with custom payload should call a live injection test endpoint, not pipeline phase 6 (eval).
2. **Knowledge base integration** — Control tab shows "—" for entries/patterns; wire to real knowledge API if available.
3. **WebSocket reconnection UX** — Show connection status (e.g. "Reconnecting…") in bottom bar when WS drops.
4. ~~**Confirm before Kill Agent**~~ — Done: Stop and Restart now show confirmation when agent is running.

### Medium Priority

5. **API error detail** — Surface HTTP status and `detail` for 4xx/5xx in toasts.
6. **Chat session persistence** — Persist chat history across page refresh (e.g. localStorage).
7. **Phenomenology endpoint** — Wire Lab phenomenology "Analyze" to backend when available.
8. **Live event sort order** — Verify `/api/live-events` newest-first vs newest-last; align poll dedup with order.
9. **Security pattern bars** — Populate `sec-patterns-body` from actual pattern data.

### Low Priority

10. **Refresh stale indicator** — Subtle badge when data is older than N seconds.
11. **Auth modal focus trap** — Trap focus inside modal for accessibility.
12. **Keyboard shortcut for Control** — e.g. `Ctrl+Shift+K` for Kill.
13. **Export logs** — Button to download activity/threat logs as file.
14. **Responsive layout** — `responsive.css` already in plan; ensure mobile works.

---

## 5. Files Modified

**Frontend**
- `frontend/siem/js/app.js` — applyStatus, onMetrics, showToast
- `frontend/siem/js/api.js` — runPipelinePhase phase mapping
- `frontend/siem/js/websocket.js` — message validation
- `frontend/siem/js/tabs/chat.js` — scroll, null guards, clear confirm
- `frontend/siem/js/tabs/control.js` — toast, getMode
- `frontend/siem/js/tabs/epidemic.js` — status merge fix
- `frontend/siem/js/tabs/lab.js` — error message, result text
- `frontend/siem/styles/chrome.css` — color-mix fallback, toast
- `frontend/siem/styles/animations.css` — reduced-motion
- `frontend/siem/styles/layout.css` — epid-network-svg min-height
- `frontend/siem/dist/index.html` — toast container

**Backend**
- `backend/siem_server.py` — _tail_jsonl_sync, inner_circle/recruited aliases, epidemic.log
- `backend/sancta_launcher.py` — Autoscroll button shows ↓ AUTO / ↓ MANUAL state

---

## 6. Epidemic Logging

Epidemic operations write to `logs/epidemic.log`:

| Event | Level | Example |
|-------|-------|---------|
| `api_epidemic_status` AgentEpidemicModel error | WARNING | Model import/eval failure |
| `api_epidemic_simulation` read error | WARNING | JSON parse / file not found |
| `builtin sim complete` | INFO | type=deterministic, health=susceptible, agents=4 |
| `builtin sim failed` | ERROR | Full traceback |
| `script launched` | INFO | type=llm, script=ollama_agents.py, pid=1234 |
| `script launch failed` | ERROR | subprocess error |
| `no script found, using builtin` | INFO | Fallback when infection_sim.py / ollama_agents.py missing |

**Tail epidemic log (PowerShell):**
```powershell
# From project root (e:\...\sancta-main\sancta-main):
Get-Content logs\epidemic.log -Tail 20 -Wait

# Or from anywhere with full path:
Get-Content "E:\CODE PROKECTS\sancta-main\sancta-main\logs\epidemic.log" -Tail 20 -Wait
```

**Check for errors:**
```powershell
Select-String -Path "logs\epidemic.log" -Pattern "ERROR|WARNING"
```

*Note: epidemic.log is created when the SIEM server starts. Restart the server if it does not exist.*

---

## 7. Verification

After deploying:

1. **Chat** — Send messages; verify auto-scroll to bottom. Clear and confirm dialog.
2. **Lab** — Click Run; verify no 422 and "Evaluation completed" result.
3. **Control** — Trigger an error (e.g. Kill when agent not running); verify toast appears.
4. **Dashboard** — Verify inner circle and recruited counts update when agent runs.
5. **Boot** — Confirm no console errors during startup.
