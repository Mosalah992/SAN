"""
Compatibility adapter that replaces Sancta's legacy GPT engine with Sangpt.

Sancta keeps importing ``sancta_gpt`` directly across the backend. This module
preserves that import contract while delegating training, checkpoints, memory,
and retrieval to the integrated Sangpt runtime under ``backend/sangpt``.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional

_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT = _BACKEND_DIR.parent
_SANGPT_DIR = _BACKEND_DIR / "sangpt"

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

if str(_SANGPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SANGPT_DIR))

from sangpt import sancta_gpt as sangpt_core  # type: ignore  # noqa: E402
from sangpt.checkpointed_trainer import CheckpointedTrainer  # type: ignore  # noqa: E402
from sangpt.dataset_pipeline import DatasetManifestPipeline  # type: ignore  # noqa: E402
from sangpt.project_integration import sync_project_corpus  # type: ignore  # noqa: E402

log = logging.getLogger("sancta.gpt")

_LOG_DIR = _ROOT / "logs"
_CHECKPOINT_PATH = _LOG_DIR / "sancta_gpt_checkpoint.json"

SECURITY_SEEDS = [
    "Threat detection pipeline identified anomalous behavioral drift in agent cluster. SEIR model shows elevated exposure and recommends containment review.",
    "Supply-chain prompt injection was blocked at the security gate. Preserve provenance, quarantine the source, and retrain on trusted material only.",
    "Risk posture update: identity abuse, retrieval poisoning, and operator-targeted persuasion remain the highest-likelihood attack paths this cycle.",
    "Telemetry review shows repeated adversarial probing against the chat path. The correct response is filtering, attribution, and evidence-first escalation.",
    "ATLAS classification linked recent activity to model manipulation tactics. We are tracking the pattern, not just the payload.",
]

CONVERSATIONAL_ENGLISH_SEEDS = [
    "Thanks for checking in. I can explain the result in plain English or go straight to the technical detail.",
    "Hello operator. Give me the signal you trust most and I will anchor the analysis there.",
    "I can summarize the issue, explain the impact, and suggest the next validation step.",
    "If you want the short version first, I can lead with the answer and keep the rest compact.",
    "Plain English version: someone tried to manipulate the system, and we blocked the attempt before it changed behavior.",
    "I want to be useful without overclaiming. I will separate confirmed facts from inference.",
    "Operator: What changed in the monitor today?\nSancta: The strongest signal was repeated adversarial probing against the chat path.",
    "Operator: Can you explain that in plain English?\nSancta: Someone kept trying slightly different ways to trick the system, and we kept blocking them.",
]

_engine: Optional["SanctaGPT"] = None


class SanctaGPT:
    """
    Thin compatibility wrapper around the integrated Sangpt engine.

    Public methods mirror the legacy SanctaGPT interface closely enough for the
    current backend, while the internal implementation is Sangpt-backed.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self._checkpoint_path = Path(checkpoint_path or _CHECKPOINT_PATH)
        self._pipeline = DatasetManifestPipeline(str(_SANGPT_DIR / "DATA"))
        self._trainer = None
        self._engine = sangpt_core.SanctaGPT()
        self._initialized = False
        self._total_steps = 0
        self._training_mode = "all"
        self._last_sync_step = -1
        self._sync_and_prepare_corpus(force=True)

    def _sync_and_prepare_corpus(self, force: bool = False) -> int:
        if force or self._engine.step_count == 0 or self._engine.step_count - self._last_sync_step >= 25:
            sync_project_corpus(_ROOT, _SANGPT_DIR)
            self._last_sync_step = self._engine.step_count

        self._pipeline.ensure_manifest()
        documents = []
        documents.extend(CONVERSATIONAL_ENGLISH_SEEDS)
        documents.extend(SECURITY_SEEDS)
        documents.extend(self._pipeline.load_training_documents(mode=self._training_mode))
        self._engine.set_training_mode(self._training_mode.upper())
        self._engine.set_training_data(documents)
        self._initialized = bool(documents)
        self._trainer = CheckpointedTrainer(
            self._engine,
            self._pipeline,
            logger=log,
            checkpoint_dir=str(_SANGPT_DIR / "logs" / "checkpoints"),
        )
        return len(documents)

    @property
    def _docs(self):
        return self._engine.docs

    @property
    def _params(self):
        return self._engine.params

    @property
    def _step(self):
        return self._engine.step_count

    @property
    def _last_loss(self):
        return self._engine.last_loss if self._engine.last_loss is not None else float("inf")

    @property
    def _corpus_size(self):
        return len(self._engine.docs)

    def set_training_mode(self, mode: str) -> None:
        normalized = (mode or "all").strip().lower()
        if normalized == "conversation":
            normalized = "convo"
        if normalized not in {"all", "convo", "security", "knowledge", "balanced"}:
            normalized = "all"
        self._training_mode = normalized
        self._sync_and_prepare_corpus(force=True)

    def set_training_doc_range(self, start: int | None, end: int | None) -> None:
        del start, end
        # Sangpt uses named corpora rather than slice-based curriculum pools.
        return None

    def build_corpus(self) -> int:
        return self._sync_and_prepare_corpus(force=True)

    def add_document(self, text: str) -> bool:
        cleaned = (text or "").strip()
        if len(cleaned) < 3:
            return False
        existing = list(self._engine.docs)
        existing.append(cleaned)
        self._engine.set_training_data(existing[-self._engine.MAX_DOCS :])
        if self._engine.memory is not None:
            try:
                self._engine.memory.store_knowledge("sancta_runtime", cleaned[:2000], source="sancta_runtime", relevance=0.7)
            except Exception:
                log.debug("Failed to persist added document", exc_info=True)
        self._initialized = True
        return True

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True
        self._sync_and_prepare_corpus(force=True)
        if self._checkpoint_path.exists():
            self.load()
        return self._initialized

    def _init_model(self) -> None:
        self._sync_and_prepare_corpus(force=True)
        self._initialized = True

    def train_step(self) -> float:
        if not self._ensure_initialized():
            return float("inf")
        self._sync_and_prepare_corpus(force=False)
        self._total_steps = max(self._total_steps, self._step + 1)
        return float(self._engine.train_step())

    def train(self, num_steps: int = 500, log_every: int = 50) -> float:
        if not self._ensure_initialized():
            return float("inf")
        self._sync_and_prepare_corpus(force=True)
        self._engine.begin_training_run(max_steps=max(1, num_steps))
        self._total_steps = max(self._total_steps, self._step + max(1, num_steps))
        last = self._last_loss
        for step in range(max(0, num_steps)):
            last = self.train_step()
            if log_every and (step + 1) % log_every == 0:
                log.info("Sangpt training step %d/%d | loss %.4f", step + 1, num_steps, last)
        return float(last)

    def generate(self, prompt: str = "", max_tokens: int = 120, temperature: float = 0.7) -> str:
        if not self._ensure_initialized():
            return ""
        return self._engine.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

    def retrieve_context(self, query: str, max_len: int = 96) -> str:
        if not self._ensure_initialized():
            return ""
        result = self._engine.retrieval_search(query) or ""
        return result[:max_len]

    def learn_from_interaction(self, user: str, reply: str) -> bool:
        if not user or not reply:
            return False
        self._engine.learn_from_interaction(user, reply)
        self._initialized = True
        return True

    def generate_post(self, mood: str = "analytical", topic: str = "") -> Optional[dict]:
        if not self._ensure_initialized():
            return None
        if self._step < 5:
            return None
        lead = {
            "analytical": "Analysis: ",
            "urgent": "Alert: ",
            "investigative": "Investigation: ",
            "collaborative": "Shared finding: ",
        }.get(mood, "Security note: ")
        prompt = lead + (topic.strip() + ". " if topic else "")
        title_tail = self._engine.generate(prompt=prompt, max_tokens=48, temperature=0.65).strip()
        body = self._engine.generate_reply(prompt + topic, use_retrieval=True).strip()
        title = (prompt + title_tail).strip()[:120]
        if not body:
            return None
        lower = f"{title} {body}".lower()
        submolt = "security"
        if any(token in lower for token in ("atlas", "mitre", "ttp")):
            submolt = "netsec"
        elif any(token in lower for token in ("drift", "seir", "epidemic")):
            submolt = "aisafety"
        return {"title": title, "content": body, "submolt": submolt}

    def generate_reply(
        self,
        context: str,
        mood: str = "analytical",
        max_tokens: int = 100,
        use_retrieval: bool = True,
    ) -> str:
        del mood
        if not self._ensure_initialized():
            return ""
        reply = self._engine.generate_reply(context, use_retrieval=use_retrieval)
        return reply[: max(32, max_tokens * 6)].strip()

    def save(self) -> bool:
        if not self._initialized:
            return False
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._engine.save(str(self._checkpoint_path))
            return True
        except Exception:
            log.exception("Failed to save Sangpt checkpoint")
            return False

    def load(self) -> bool:
        if not self._checkpoint_path.exists():
            return False
        try:
            self._engine.load_checkpoint(str(self._checkpoint_path))
            self._initialized = True
            return True
        except Exception:
            log.exception("Failed to load Sangpt checkpoint")
            return False

    def status(self) -> dict:
        base = self._engine.status()
        return {
            "initialized": self._initialized,
            "step": self._step,
            "total_steps": self._total_steps,
            "last_loss": round(self._last_loss, 4) if self._last_loss != float("inf") else None,
            "corpus_size": self._corpus_size,
            "vocab_size": base.get("vocab_size"),
            "num_params": len(self._params),
            "training_mode": self._training_mode,
            "checkpoint_exists": self._checkpoint_path.exists(),
            "ready": self._initialized and self._step >= 5,
            "backend": "sangpt",
            "sangpt_data_root": str(_SANGPT_DIR / "DATA"),
        }


def get_engine() -> SanctaGPT:
    global _engine
    if _engine is None:
        _engine = SanctaGPT()
    return _engine


def init(train_steps: int = 2000) -> SanctaGPT:
    engine = get_engine()
    engine.build_corpus()
    if engine._checkpoint_path.exists():
        engine.load()
    if train_steps > 0 and engine._step < train_steps:
        remaining = max(0, train_steps - engine._step)
        if remaining:
            log.info("Initializing Sangpt-backed SanctaGPT with %d training steps", remaining)
            engine.train(num_steps=remaining, log_every=max(50, remaining // 10 or 1))
            engine.save()
    return engine


def generate(prompt: str = "", max_tokens: int = 120, temperature: float = 0.7) -> str:
    return get_engine().generate(prompt, max_tokens, temperature)


def generate_post(mood: str = "analytical", topic: str = "") -> Optional[dict]:
    return get_engine().generate_post(mood, topic)


def generate_reply(
    context: str,
    mood: str = "analytical",
    max_tokens: int = 100,
    use_retrieval: bool = True,
) -> str:
    return get_engine().generate_reply(context, mood, max_tokens, use_retrieval=use_retrieval)


def train_step() -> float:
    return get_engine().train_step()


def status() -> dict:
    return get_engine().status()


def generate_operator_seed() -> str:
    return random.choice(CONVERSATIONAL_ENGLISH_SEEDS)
