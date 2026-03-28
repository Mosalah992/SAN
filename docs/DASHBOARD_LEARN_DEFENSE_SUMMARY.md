# Dashboard LEARN + DEFENSE Tabs — Implementation Summary

**Source:** dashboard overhaul1.docx  
**Implementation Date:** March 2026

---

## What Was Added

### Two New Tabs

| Tab | Shortcut | Purpose |
|-----|----------|---------|
| **LEARN** | `7` | Pattern learning health, feedback, interactions |
| **DEFENSE** | `8` | Adversary intel, threat level, attacks, defense stats |

---

## Backend Endpoints

### 1. `/api/learning/health` (GET)

Returns learning health for the LEARN tab:

- `pattern_count` — Number of learned patterns
- `interaction_count` — Total interactions
- `pattern_hit_rate` — Pattern usage hit rate (0–1)
- `positive_feedback_pct` — Percentage of positive feedback
- `top_patterns` — Top 10 patterns by success rate (topic, success_rate, frequency)
- `recent_interactions` — Last 20 interactions with feedback and preview

**Implementation:** `backend/siem_server.py`, `sancta_learning.get_learning_health()`

### 2. `/api/security/adversary` (GET)

Returns adversary/defense data for the DEFENSE tab:

- `threat_level` — `green` / `yellow` / `orange` / `red` (based on attack count)
- `total_attacks` — input_reject + ioc_domain_detected
- `unique_fingerprints` — Distinct pattern fingerprints
- `high_risk_count` — Events with complexity_score ≥ 0.8
- `known_attackers` — Authors with attack counts
- `recent_attacks` — Last 20 attack events
- `defense_stats` — blocked, ioc_detected, unicode_sanitized, normal

**Implementation:** `backend/siem_server.py`, aggregates from `logs/security.jsonl`

---

## Frontend Changes

### 1. index.html

- Two tab buttons: LEARN, DEFENSE (after SYS)
- Two panels: `panel-learn`, `panel-defense` with `learn-content`, `defense-content` containers

### 2. styles.css

New classes:

- `.panel-metrics` — Scrollable metrics container
- `.metric-card` — Card layout for metrics
- `.metric-cards-row` — Flex row for metric cards
- `.alert-box`, `.alert-box.success`, `.alert-box.warning` — Alert styling
- `.recent-item` — List item for recent entries
- `.threat-level` — Color-coded threat (green/yellow/orange/red)

### 3. app.js

- `panes.learn`, `panes.defense` — Content containers
- `panelLearn`, `panelDefense`, `feedSection` — Panel refs
- `setTab()` — Shows/hides feed vs learn/defense, loads data
- Keyboard: `7` → learn, `8` → defense
- `loadLearningHealth()` — Fetches `/api/learning/health`, renders cards + lists
- `loadAdversaryDefense()` — Fetches `/api/security/adversary`, renders threat + attacks
- Auto-refresh: 5 seconds while LEARN or DEFENSE tab is active
- `clearLearnDefenseRefresh()` — Stops refresh when switching away

---

## Data Flow

```
User clicks LEARN (or presses 7)
    → setTab("learn")
    → Hide feed, show panel-learn
    → loadLearningHealth()
    → authFetch("/api/learning/health")
    → Render cards: Patterns, Hit Rate, Positive Feedback %, Interactions
    → Render: Top patterns by success rate
    → Render: Recent interactions with ratings
    → setInterval(loadLearningHealth, 5000)

User clicks DEFENSE (or presses 8)
    → setTab("defense")
    → Hide feed, show panel-defense
    → loadAdversaryDefense()
    → authFetch("/api/security/adversary")
    → Render cards: Threat Level, Attacks, Fingerprints, High-Risk
    → Render: Defense stats
    → Render: Known attackers
    → Render: Recent attacks
    → setInterval(loadAdversaryDefense, 5000)
```

---

## File Summary

| File | Changes |
|------|---------|
| `backend/siem_server.py` | +`/api/learning/health`, +`/api/security/adversary`, +`_load_security_jsonl_tail()` |
| `backend/sancta_learning.py` | +`get_learning_health()` |
| `frontend/siem/index.html` | +2 tab buttons, +2 panel sections |
| `frontend/siem/styles.css` | +panel-metrics, metric-card, alert-box, recent-item, threat-level |
| `frontend/siem/app.js` | +panes.learn/defense, +setTab logic, +loadLearningHealth, +loadAdversaryDefense, +keyboard 7/8, +auto-refresh |

---

## Testing

- Click LEARN tab → Shows pattern metrics, top patterns, recent interactions
- Click DEFENSE tab → Shows threat level, attacks, known attackers, recent attacks
- Press `7` / `8` → Keyboard shortcuts work
- Tabs auto-refresh every 5 seconds when active
- Switching to other tabs stops the refresh
