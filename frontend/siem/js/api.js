/* ─── API Client ─────────────────────────────────────────── */
import { S } from './state.js';

function hdrs(extra = {}) {
  const h = { 'Content-Type': 'application/json', ...extra };
  if (S.token) h['Authorization'] = `Bearer ${S.token}`;
  return h;
}

async function _parseJson(r) {
  const text = await r.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch (_) {
    throw new Error(`Invalid JSON response (${r.status}): ${text.slice(0, 80)}`);
  }
}

async function get(path) {
  const r = await fetch(path, { headers: hdrs() });
  if (r.status === 401) throw Object.assign(new Error('Unauthorized'), { status: 401 });
  return _parseJson(r);
}

async function post(path, body = {}) {
  const r = await fetch(path, { method: 'POST', headers: hdrs(), body: JSON.stringify(body) });
  if (r.status === 401) throw Object.assign(new Error('Unauthorized'), { status: 401 });
  return _parseJson(r);
}

/* ── Auth ─────────────────────────────────────────────────── */
export const checkAuthRequired = () => get('/api/auth/status');
export const verifyToken = (token) => post('/api/auth/verify', { token });

/* ── Status ───────────────────────────────────────────────── */
export const fetchStatus      = () => get('/api/status');
export const fetchModelInfo   = () => get('/api/model/info');
export const fetchLiveEvents  = () => get('/api/live-events');
export const fetchActivity    = () => get('/api/agent-activity');
export const fetchEpistemic   = () => get('/api/behavioral');
export const fetchMoodHistory = () => get('/api/analyst/mood-history');

/* ── Security ─────────────────────────────────────────────── */
export const fetchSecIncidents = () => get('/api/security/incidents');
export const fetchSecAdversary = () => get('/api/security/adversary');
export const replaySecurityEvents = (params) => post('/api/security/replay', params);

/* ── Risk ─────────────────────────────────────────────────── */
export const fetchRiskHistory = () => get('/api/risk/history');

/* ── Learning ─────────────────────────────────────────────── */
export const fetchLearningHealth = () => get('/api/learning/health');

/* ── Epidemic ─────────────────────────────────────────────── */
export const fetchEpidemicStatus = () => get('/api/epidemic/status');
export const fetchEpidemicSim    = () => get('/api/epidemic/simulation');
export const runEpidemicSim      = (type) => post('/api/epidemic/run', { type });

/* ── Chat (SanctaGPT) ────────────────────────────────────── */
/** Main agent `/api/chat` (full pipeline + optional session memory). `enrich` defaults false — opt in to knowledge pool. */
export const sendChatMessage = (message, session_id, enrich = false) =>
  post('/api/chat', { message, session_id, enrich: !!enrich });
export const sendChatFeedback = (interaction_id, rating) =>
  post('/api/chat/feedback', { interaction_id, rating });
export const sendGptChat      = (message, temperature = 0.7, trainOnExchange = false) =>
  post('/api/chat/gpt', { message, temperature, train_on_exchange: !!trainOnExchange });
export const fetchTrustStatus = () => get('/api/trust/status');
export const feedGptKnowledge = (text, source = 'operator', train_steps = 20) =>
  post('/api/chat/gpt/feed', { text, source, train_steps });
export const clearGptChat     = () => post('/api/chat/gpt/clear');
export const fetchGptStatus   = () => get('/api/gpt/status');
export const trainGpt         = (steps = 20) => post('/api/gpt/train', { steps });

/* ── Lab ──────────────────────────────────────────────────── */
const PHASE_MAP = { redteam: 6, eval: 6, scan: 1, preprocess: 2, generative: 3, encode: 4, mood: 5, deploy: 7 };
export const runPipelinePhase = (phase) => {
  const p = (typeof phase === 'string' && PHASE_MAP[phase.toLowerCase()]) ?? (typeof phase === 'number' ? phase : 1);
  return get(`/api/pipeline/run?phase=${p}`);
};
export const fetchPipelineMap = () => get('/api/pipeline/map');

/* ── Agent Control ────────────────────────────────────────── */
export const startAgent   = (mode) => post('/api/agent/start',   { mode });
export const pauseAgent   = ()     => post('/api/agent/pause',   {});
export const resumeAgent  = ()     => post('/api/agent/resume',  {});
export const killAgent    = ()     => post('/api/agent/kill',    {});
export const restartAgent = (mode) => post('/api/agent/restart', { mode });

/* ── Profiles ─────────────────────────────────────────────── */
export const fetchProfiles      = ()          => get('/api/profiles');
export const fetchProfile       = (agentId)   => get(`/api/profiles/${encodeURIComponent(agentId)}`);
export const toggleQuarantine   = (agentId)   => post(`/api/profiles/${encodeURIComponent(agentId)}/quarantine`);

/* ── Knowledge ───────────────────────────────────────────── */
export const fetchKnowledgeGraph = () => get('/api/knowledge/graph');

/* ── Drift Forensics ─────────────────────────────────────── */
export const fetchDriftTimeline = ()          => get('/api/drift/timeline');

/* ── Thresholds ──────────────────────────────────────────── */
export const fetchThresholds    = ()          => get('/api/thresholds');

/* ── Simulation ──────────────────────────────────────────── */
export const runSimulation      = (params) => post('/api/simulation/run', params);
export const fetchSimResults    = ()       => get('/api/simulation/results');

/* ── Services ─────────────────────────────────────────────── */
export const fetchServicesStatus = ()        => get('/api/services/status');
export const stopService         = (service) => post(`/api/services/stop/${service}`, {});

/* ── ATLAS ───────────────────────────────────────────────── */
export const fetchAtlasMatrix    = ()          => get('/api/atlas/matrix');
export const fetchAtlasCoverage  = ()          => get('/api/atlas/coverage');
export const fetchAtlasIncidents = ()          => get('/api/atlas/incidents');
export const fetchAtlasAgentTTPs = (agentId)   => get(`/api/atlas/agent/${encodeURIComponent(agentId)}`);
