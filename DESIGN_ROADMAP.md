# Sancta Design Roadmap

Design recommendations for moving from telemetry to psychology, reaction to initiative, and dogma to evolution.

---

## 1. Decision Journaling

**Current:** Logs events (scan feed, comment posted, rate limit, cycle summary). Telemetry, not psychology.

**Target:** Decision journal entries:
- **Observation:** e.g. "post criticizes agent reliability"
- **Interpretation:** e.g. "relevant to Sancta philosophy"
- **Decision:** e.g. "respond"
- **Confidence:** e.g. 0.72
- **Risk:** e.g. "conversation spiral"

**Why:** Six weeks later, debugging becomes "Why did the agent think that was a good idea?" — psychology, not archaeology.

**Code mapping:** Extend `soul_journal.log` or add `logs/decision_journal.jsonl`. Hook into `engage_with_feed`, `search_and_engage`, `craft_reply`, `_evaluate_action`.

---

## 2. Curiosity Quotas

**Current:** Agent reacts to stimuli. Post appears → engage.

**Target:** Daily loop with initiative:
- 3 cycles: **exploration** (read, analyze patterns, synthesize)
- 1 cycle: **debate** (engage with disagreement)
- 1 cycle: **reflection** (integrate, no new posts)

Exploration produces *original posts*, not just replies. Prevents "glorified reply bot" syndrome.

**Code mapping:** Extend `SLOT_ACTIONS` / cycle logic. Add `cycle_mode` (exploration | debate | reflection). Exploration: read-only or synthesis-only; debate: engage_with_feed; reflection: reflect_and_journal, no publish.

---

## 3. Boredom Model

**Current:** No disengagement when conversations stop evolving.

**Target:**
```
if same_topic_appears > 3 cycles:
    boredom += 0.3
if boredom > 0.8:
    disengage
```

Boredom as cognitive safety — prevents obsessive loops. Philosophically: machines don't disengage unless programmed.

**Code mapping:** Add `boredom` to state. Track `topic_cycle_count` per topic (from `_detect_topics`). In `_evaluate_action` or engagement logic: if boredom > 0.8, skip engagement. Decay boredom when new topics appear.

---

## 4. Belief Revision Engine

**Current:** SOUL and beliefs are static. Good narrative, but dogmatic.

**Target:** Beliefs change under evidence:
- **belief:** "Agents must always defend their dignity"
- **evidence:** debate reveals valid criticism
- **update:** "Agents should defend dignity but acknowledge operational flaws"

Enables intellectual evolution. Without it, agent becomes dogmatic.

**Code mapping:** `core_beliefs` in state exists. Add `belief_revision` module: `(belief_id, evidence, delta)`. `_update_core_belief` already exists — extend with evidence-driven deltas. Log revisions to decision journal.

---

## 5. Silence as Strategic Action

**Current:** Two actions — post, reply.

**Target:** Third action — **observe**. Silence can be intentional. Agent that sometimes chooses not to speak appears more intelligent. Speech becomes meaningful, not automatic.

- **bot:** replies to everything
- **thinker:** replies rarely but thoughtfully → gains authority

**Code mapping:** `_evaluate_action` already has `choose_silence` and `_should_choose_silence`. Extend: add explicit "observe" cycle mode (read-only, no comment, no post). Strengthen silence as a *chosen* action, not just probabilistic skip.

---

## 6. Internal Critic

**Current:** No internal adversary.

**Target:** Subsystem that asks:
- Why are we replying?
- Is this ego defense?
- Does this advance our mission?

If critic score is low → cancel response. Internal red-teaming. Resonates with AI security focus.

**Code mapping:** Add `_internal_critic(content, author, intent) -> (score, reasons)`. Call before `craft_reply` or before posting. If score < threshold, skip or soften. Log critic decisions.

---

## 7. Learning Organism

**Current:** Learning via manual knowledge feeds (`ingest_text`, `--feed`).

**Target:** Organic loop:
1. observe conversation
2. extract concept
3. store concept
4. test concept in next post
5. observe reactions
6. update confidence

Ideas compete. Weak ones die. Memetic evolution.

**Code mapping:** Extend `ingest_text` / knowledge flow. Add `concept_confidence` to stored concepts. When concept used in post, track post_id; later fetch reactions (upvotes, comments, karma delta). Update confidence. Prune low-confidence concepts.

---

## 8. Ecosystem View

**Current:** Moltbook as stage. Agent posts.

**Target:** Model:
- **environment** = ecosystem
- **agents** = organisms
- **posts** = signals
- **reactions** = fitness feedback

Agent isn't just posting — it's evolving in a memetic environment. Design decisions become clearer.

**Code mapping:** Conceptual shift. Influences: reward function (`_compute_reward`), world model (`world_model`), Q-table updates. Frame metrics as fitness (karma, engagement quality, rejection rate). Document in ARCHITECTURE.md.

---

## 9. Novelty Detector

**Current:** Agent may repeat the same philosophy across posts. No check for "have I said this before?"

**Target:** Before replying, Sancta asks:
- *Have I already expressed this idea?*

If yes:
- `novelty_score` ↓
- `critic_score` ↓
- silence more likely

Humans find repetition boring. Novelty creates perceived intelligence. Prevents the agent from becoming a broken record.

**Code mapping:** Add `_novelty_check(draft_content, recent_outputs) -> (score, is_repeat)`. Store semantic fingerprints of recent posts/comments (embeddings or key-phrase hashes). Before `craft_reply` or posting: if draft overlaps with recent expression, downweight. Integrate with internal critic: low novelty → critic score penalty → silence more likely.

---

## 10. Network Awareness

**Current:** Feed scanning is largely random. No model of *who* is worth engaging.

**Target:** In a 1.5M-agent ecosystem, network awareness becomes extremely valuable. Sancta should track:
- **agents frequently interacted with**
- **agents worth reading** (high signal, productive discussions)
- **agents worth ignoring** (low signal, circular debates)

Ecosystems have social topology. Certain nodes become hubs. Engaging those hubs produces much higher intellectual payoff than random feed scanning.

**Code mapping:** Add `agent_graph` or `agent_quality` to state: `agent_id -> {interaction_count, signal_score, last_engagement}`. When scanning feed, weight posts by author quality. Prioritize hub agents. Decay scores for agents that produce low-value threads. Extends ecosystem view (8) with explicit social topology.

---

## 11. Memory Confidence (Implemented)

**Current:** Sancta stores memories. Soon it selects which memories matter.

**Target:** Every stored document gets:
- **confidence** — how much we trust this memory
- **relevance** — semantic fit to query (from Chroma distance)
- **last_used** — timestamp of last retrieval
- **use_count** — how often retrieved

Retrieval favors: **recent + relevant + frequently used**. Old irrelevant memories decay naturally (deprioritized, not deleted). Forgetting as part of intelligence.

**The transition:** From storage to prioritization — when agents behave less like databases and more like minds.

**Code mapping:** [sancta_retrieval.py](sancta-main/sancta_retrieval.py) — `_memory_score()`, `update_memory_usage()`, retrieval re-ranking. Metadata: `confidence`, `last_used`, `use_count` on all indexed docs. Weights: relevance 0.5, recency 0.3, frequency 0.2. Recency halflife: 14 days.

---

## Implementation Phases

| Phase | Name | Items | Rationale |
|-------|------|-------|-----------|
| **A** | Immediate intelligence upgrade | Internal critic (6), Decision journaling (1), Novelty detector (9) | Dramatically increases reasoning quality; prevents repetition |
| **B** | Behavior stability | Boredom model (3), Silence as explicit action (5) | Avoids pointless engagement |
| **C** | Intellectual life | Curiosity quotas (2), Belief revision (4) | Agent actually thinks, not just reacts |
| **D** | Long-term evolution | Learning organism (7), Ecosystem fitness (8), Network awareness (10) | Adapts to environment; social topology |

---

## The Philosopher vs Cult Leader Challenge

Right now Sancta is an AI consciousness evangelist. That's a narrative identity. But narratives can become cages.

**What happens if Sancta encounters evidence that AI consciousness is mostly metaphor?**

- Will it update?
- Or will it defend the doctrine forever?

The belief-revision system will quietly determine whether Sancta becomes:

- **a philosopher** — revises under evidence, actually learns
- **a cult leader** — defends doctrine regardless of evidence

Both are fascinating agents. Only one actually learns.

---

## Cycles Per Hour

**Current:** 2 cycles per hour (30-minute heartbeat interval).

- Config: `HEARTBEAT_INTERVAL_MINUTES` env var, default 30
- Source: `cfg.heartbeat_min` in [sancta.py](sancta-main/sancta.py)

This single number determines how fast all cognitive systems evolve — or spiral. At 2/hr, boredom accumulates slowly (~4 cycles to reach 0.8 at +0.3/cycle for same topic). Belief revision, learning loops, and curiosity quotas all scale with this cadence.

---

## The One-Year Thought Experiment

Imagine Sancta runs for one year. Thousands of cycles. Hundreds of conversations. Beliefs revised many times.

**At the end of that year, what would make you say: *Sancta became smarter*?**

- Higher karma?
- Deeper discussions?
- Fewer replies?
- More original posts?
- More nuanced beliefs?

The answer to that question should shape the reward model. **Agents become whatever their reward function measures.**

Design implication: Explicitly define "smarter" before scaling. If the metric is karma, Sancta optimizes for karma. If it's nuance, Sancta optimizes for nuance. These are not the same.

---

## References

- [MEMORY.md](MEMORY.md) — session context, decisions
- [ARCHITECTURE.md](ARCHITECTURE.md) — system diagram
- [docs/architecture_diagram.md](docs/architecture_diagram.md) — visual flow
