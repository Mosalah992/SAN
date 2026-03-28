"""
SANCTA-GPT Engine — nanoGPT-backed implementation.

Replaces the original pure-Python autograd engine with a proper PyTorch
transformer (Karpathy's nanoGPT architecture). Falls back to the legacy
pure-Python engine when PyTorch is not available.

Core reference: https://github.com/karpathy/nanoGPT
License: MIT

The public interface is identical to the legacy engine so the compatibility
adapter (backend/sancta_gpt.py) requires zero changes.
"""

import os
import json
import math
import random
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger("sancta_gpt")

# ─── Optional dependencies ───────────────────────────────────────────────────

_HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    logger.info("PyTorch not available — using legacy pure-Python engine")

# Import memory manager for persistent context
try:
    from memory_manager import get_memory, MemoryManager
except ImportError:
    get_memory = None
    MemoryManager = None

# Seed for reproducibility
random.seed(42)

# Log directory
_LOG_DIR = Path("./logs")
_LOG_DIR.mkdir(exist_ok=True)


# ─── NanoGPT engine (requires PyTorch) ────────────────────────────────────────

if _HAS_TORCH:
    from nano_model import NanoGPT, GPTConfig, create_model
    from nano_tokenizer import Tokenizer, get_tokenizer, has_tiktoken


class SanctaGPT:
    """
    GPT engine for the SANCTA system.

    When PyTorch is available, uses a real nanoGPT transformer with BPE
    tokenization (tiktoken) or character-level fallback. Training uses
    AdamW with cosine LR schedule and gradient clipping — the same recipe
    as Karpathy's nanoGPT.

    When PyTorch is NOT available, transparently delegates to the legacy
    pure-Python autograd engine (sancta_gpt_legacy.py).
    """

    # Maximum documents to keep in memory (rolling window).
    MAX_DOCS = 8000
    BLOCK_SIZE = 256  # alias for adapter

    # Training hyperparameters
    WARMUP_STEPS = 50
    MAX_GRAD_NORM = 1.0
    MIN_LR_RATIO = 0.1

    def __init__(self, n_layer=4, n_embd=128, block_size=256, n_head=4, vocab_size=None):
        """Initialize the GPT engine."""
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.step_count = 0
        self.last_loss = None
        self.training_mode = "OPERATOR"

        # Document storage
        self.docs = []
        self.pools = {"convo": [], "security": [], "knowledge": [], "all": []}

        # Legacy compatibility attributes
        self.uchars = []
        self.char2id = {}
        self.id2char = {}
        self.BOS = None

        # Memory manager
        self.memory = None
        if get_memory is not None:
            try:
                self.memory = get_memory()
            except Exception as e:
                logger.warning("Memory manager initialization failed: %s", e)

        if _HAS_TORCH:
            self._backend = "nanogpt"
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = get_tokenizer("auto")
            actual_vocab = vocab_size or self._tokenizer.vocab_size

            self._config = GPTConfig(
                block_size=block_size,
                vocab_size=actual_vocab,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                dropout=0.1,
            )
            self._model = NanoGPT(self._config).to(self._device)
            self.vocab_size = actual_vocab

            # Optimizer
            self._learning_rate = 3e-4
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self._learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
            self._scheduler = None
            self._token_stream = []  # flat encoded corpus

            # params: len() must return total scalar parameter count (legacy compat)
            num_params = self._model.get_num_params()
            self.params = range(num_params)
            logger.info("SanctaGPT [nanoGPT] initialized: %d parameters, device=%s, tokenizer=%s",
                        num_params, self._device, self._tokenizer.backend)
        else:
            self._backend = "legacy"
            # Import and delegate to legacy engine
            from sancta_gpt_legacy import SanctaGPT as LegacyGPT
            self._legacy = LegacyGPT(
                n_layer=max(1, n_layer),
                n_embd=min(32, n_embd),
                block_size=block_size,
                n_head=min(4, n_head),
                vocab_size=vocab_size or 128,
            )
            self.vocab_size = self._legacy.vocab_size
            self.params = self._legacy.params
            logger.info("SanctaGPT [legacy pure-Python] initialized: %d parameters",
                        len(self.params))

    # ─── Training mode ────────────────────────────────────────────────────────

    def set_training_mode(self, mode: str):
        """Set the current operational mode."""
        self.training_mode = mode
        if self._backend == "legacy":
            self._legacy.set_training_mode(mode)
        logger.info("Engine mode changed to: %s", mode)

    def set_training_data(self, docs: List[str]):
        """Set the training documents and encode corpus."""
        self.docs = [doc for doc in docs if doc]
        self.pools["all"] = list(self.docs)

        if self._backend == "nanogpt":
            # Build char-level vocab for legacy compat attributes
            self.uchars = sorted(set("".join(self.docs)))
            self.BOS = len(self.uchars)
            self.char2id = {ch: i for i, ch in enumerate(self.uchars)}
            self.id2char = {i: ch for i, ch in enumerate(self.uchars)}

            # Build tokenized corpus for char backend
            if self._tokenizer.backend == "char":
                self._tokenizer.build_vocab_from_corpus(self.docs)
                self.vocab_size = self._tokenizer.vocab_size
                # Rebuild model if vocab changed
                if self.vocab_size != self._config.vocab_size:
                    self._config.vocab_size = self.vocab_size
                    old_step = self.step_count
                    old_loss = self.last_loss
                    self._model = NanoGPT(self._config).to(self._device)
                    self._optimizer = torch.optim.AdamW(
                        self._model.parameters(),
                        lr=self._learning_rate,
                        betas=(0.9, 0.999),
                        weight_decay=0.01,
                    )
                    self.params = range(self._model.get_num_params())
                    self.step_count = old_step
                    self.last_loss = old_loss

            self._token_stream = self._tokenizer.encode_corpus(self.docs)
            self._build_idf_index()
            logger.debug("Corpus encoded: %d tokens from %d documents (tokenizer=%s)",
                         len(self._token_stream), len(self.docs), self._tokenizer.backend)
        else:
            self._legacy.set_training_data(docs)
            self.uchars = self._legacy.uchars
            self.char2id = self._legacy.char2id
            self.id2char = self._legacy.id2char
            self.BOS = self._legacy.BOS
            self.vocab_size = self._legacy.vocab_size

    # ─── Training ─────────────────────────────────────────────────────────────

    def begin_training_run(self, max_steps: int = 300):
        """Reset the LR schedule for a new training run."""
        self._run_step_offset = self.step_count
        self._run_max_steps = max(1, max_steps)

        if self._backend == "nanogpt":
            # Cosine annealing scheduler — reset for each training run
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=max_steps,
                eta_min=self._learning_rate * self.MIN_LR_RATIO,
            )
        elif hasattr(self._legacy, "begin_training_run"):
            self._legacy.begin_training_run(max_steps)

        logger.info("Training run started: offset=%d, max_steps=%d",
                     self.step_count, max_steps)

    def _get_batch(self, batch_size: int = 4) -> Tuple:
        """Get a random batch from the token stream."""
        if not self._token_stream or len(self._token_stream) <= self.block_size:
            return None, None

        stream = self._token_stream
        max_start = len(stream) - self.block_size - 1
        if max_start <= 0:
            return None, None

        ix = [random.randint(0, max_start) for _ in range(batch_size)]
        x = torch.stack([torch.tensor(stream[i:i + self.block_size], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(stream[i + 1:i + 1 + self.block_size], dtype=torch.long) for i in ix])
        return x.to(self._device), y.to(self._device)

    def train_step(self, doc: Optional[str] = None) -> float:
        """Perform one training step. Returns the loss value."""
        if self._backend == "legacy":
            loss = self._legacy.train_step(doc)
            self.step_count = self._legacy.step_count
            self.last_loss = self._legacy.last_loss
            return loss

        # nanoGPT training step
        if not self._token_stream:
            if doc:
                # Encode the single document as a training batch
                tokens = self._tokenizer.encode_document(doc)
                if len(tokens) <= 1:
                    return 0.0
                self._token_stream = tokens
            elif self.docs:
                self._token_stream = self._tokenizer.encode_corpus(self.docs)
            else:
                logger.warning("train_step called with no data — skipping")
                return 0.0

        x, y = self._get_batch(batch_size=min(4, max(1, len(self._token_stream) // self.block_size)))
        if x is None:
            # Corpus too small for block_size — train on what we have
            tokens = self._token_stream[:self.block_size + 1]
            if len(tokens) < 2:
                return 0.0
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=self._device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=self._device)

        self._model.train()
        logits, loss = self._model(x, y)

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.MAX_GRAD_NORM)

        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()

        self.step_count += 1
        self.last_loss = loss.item()
        return self.last_loss

    def train(self, steps: int = 100) -> float:
        """Run a multi-step training loop."""
        self.begin_training_run(max_steps=steps)
        loss = None
        for _ in range(max(0, steps)):
            loss = self.train_step()
        return loss if loss is not None else 0.0

    # ─── Generation ───────────────────────────────────────────────────────────

    def _ensure_vocabulary(self):
        """Ensure vocabulary is initialized with defaults if empty."""
        if self._backend == "legacy":
            self._legacy._ensure_vocabulary()
            return

        # For nanoGPT, tokenizer always has a vocab
        if self._tokenizer.vocab_size == 0 and self._tokenizer.backend == "char":
            self._tokenizer.build_vocab_from_corpus(
                ["abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'-\"()[]{}@#$%&*+=\n\t"]
            )

    def generate(self, prompt: str = "", max_tokens: int = 120, temperature: float = 0.7) -> str:
        """Generate text, optionally starting with a prompt context."""
        if self._backend == "legacy":
            return self._legacy.generate(prompt, max_tokens, temperature)

        self._ensure_vocabulary()
        self._model.eval()

        with torch.no_grad():
            # Encode prompt
            if prompt:
                tokens = self._tokenizer.encode(prompt)
            else:
                tokens = [self._tokenizer.eot_token]

            # Truncate to block_size
            tokens = tokens[-(self.block_size - max_tokens):]
            idx = torch.tensor([tokens], dtype=torch.long, device=self._device)

            generated = self._model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-8),
                top_k=40,
            )

            # Decode only the newly generated tokens
            new_tokens = generated[0, len(tokens):].tolist()

            # Stop at EOT if present
            eot = self._tokenizer.eot_token
            if eot in new_tokens:
                new_tokens = new_tokens[:new_tokens.index(eot)]

            return self._tokenizer.decode(new_tokens)

    def sample_batch(self, n: int = 5, temperature: float = 0.7) -> List[str]:
        """Generate n sample outputs for inspection."""
        samples = []
        prompts = ["Security", "The system", "Alert:", "Analysis:", ""]
        for i in range(n):
            prompt = prompts[i % len(prompts)]
            try:
                text = self.generate(prompt=prompt, max_tokens=60, temperature=temperature)
                samples.append(text.strip())
            except Exception:
                samples.append("")
        return samples

    # ─── Interaction learning ─────────────────────────────────────────────────

    def learn_from_interaction(self, prompt: str, response: str):
        """Learn from a prompt-response pair."""
        if self._backend == "legacy":
            self._legacy.learn_from_interaction(prompt, response)
            self.step_count = self._legacy.step_count
            return

        combined = f"{prompt} {response}"
        self.pools["all"].append(combined)
        self.docs.append(combined)

        # Cap document lists
        if len(self.docs) > self.MAX_DOCS:
            self.docs = self.docs[-self.MAX_DOCS:]
        if len(self.pools["all"]) > self.MAX_DOCS:
            self.pools["all"] = self.pools["all"][-self.MAX_DOCS:]

        # Update token stream incrementally
        new_tokens = self._tokenizer.encode_document(combined)
        self._token_stream.extend(new_tokens)

        if self.memory is not None:
            try:
                self.memory.store_knowledge(
                    "interaction", combined[:1000],
                    source="interaction", relevance=0.55,
                )
            except Exception as exc:
                logger.warning("Failed to persist learned interaction: %s", exc)

        # Do a quick training step on this interaction
        self.train_step()

    # ─── Retrieval: TF-IDF with n-grams and stemming ─────────────────────────

    _idf_cache: Dict[str, float] = {}
    _bigram_idf_cache: Dict[str, float] = {}

    @staticmethod
    def _stem(word: str) -> str:
        """Simple suffix-stripping stemmer for security domain text."""
        if len(word) <= 3:
            return word
        for suffix in ("ation", "ment", "ness", "ible", "able", "ial", "ity",
                        "ing", "ous", "ive", "ion", "ed", "ly", "er", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    def _tokenize_query(self, text: str) -> Counter:
        """Tokenize text into stemmed, lowercased word counts."""
        stop = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
                "for", "on", "and", "or", "it", "i", "you", "that", "this", "my",
                "be", "do", "does", "did", "has", "have", "had", "been", "will",
                "can", "could", "would", "should", "may", "also", "however", "but",
                "what", "how", "why", "when", "where", "who", "which", "me",
                "we", "they", "he", "she", "its", "our", "your", "their",
                "am", "not", "no", "so", "if", "then", "about", "with", "from",
                "tell", "explain", "describe", "show", "give", "help", "let",
                "say", "ask", "know", "think", "want", "need", "like", "make",
                "get", "see", "try", "go", "come", "take", "use", "look", "find",
                "more", "very", "just", "some", "any", "many", "much", "most"}
        words = ["".join(c for c in w if c.isalnum()) for w in text.lower().split()]
        stemmed = [self._stem(w) for w in words if len(w) > 1 and w not in stop]
        return Counter(stemmed)

    def _extract_bigrams(self, text: str) -> Counter:
        """Extract bigrams for phrase matching."""
        words = ["".join(c for c in w if c.isalnum()) for w in text.lower().split()]
        stemmed = [self._stem(w) for w in words if len(w) > 1]
        bigrams = [f"{stemmed[i]}_{stemmed[i+1]}" for i in range(len(stemmed) - 1)]
        return Counter(bigrams)

    def _build_idf_index(self):
        """Compute IDF for all terms and bigrams across the document corpus."""
        from collections import defaultdict
        doc_freq = defaultdict(int)
        bigram_doc_freq = defaultdict(int)
        total_docs = max(len(self.docs), 1)

        for doc in self.docs:
            unique_terms = set(self._tokenize_query(doc).keys())
            for term in unique_terms:
                doc_freq[term] += 1
            unique_bigrams = set(self._extract_bigrams(doc).keys())
            for bg in unique_bigrams:
                bigram_doc_freq[bg] += 1

        self._idf_cache = {
            term: math.log(total_docs / (1 + freq))
            for term, freq in doc_freq.items()
        }
        self._bigram_idf_cache = {
            bg: math.log(total_docs / (1 + freq))
            for bg, freq in bigram_doc_freq.items()
        }
        logger.debug("IDF index built: %d terms, %d bigrams",
                      len(self._idf_cache), len(self._bigram_idf_cache))

    def _tfidf_cosine(self, tokens_a: Counter, tokens_b: Counter,
                       bigrams_a: Counter = None, bigrams_b: Counter = None) -> float:
        """Compute TF-IDF weighted cosine similarity between two token sets."""
        def _weighted(tokens):
            return {t: count * self._idf_cache.get(t, 1.0) for t, count in tokens.items()}

        wa = _weighted(tokens_a)
        wb = _weighted(tokens_b)
        matched_terms = set(wa) & set(wb)
        overlap = sum(wa[t] * wb[t] for t in matched_terms)
        norm_a = math.sqrt(sum(v * v for v in wa.values())) or 1.0
        norm_b = math.sqrt(sum(v * v for v in wb.values())) or 1.0
        score = overlap / (norm_a * norm_b)

        if len(tokens_a) > 1:
            coverage = len(matched_terms) / len(tokens_a)
            if coverage < 0.6:
                score *= coverage

        if bigrams_a and bigrams_b:
            shared = set(bigrams_a) & set(bigrams_b)
            if shared:
                bonus = sum(self._bigram_idf_cache.get(bg, 1.0) for bg in shared)
                score += 0.15 * min(bonus / max(len(bigrams_a), 1), 1.0)

        return score

    def retrieval_search(self, query: str, top_k: int = 1) -> Optional[str]:
        """Search stored docs for the best-matching response using TF-IDF cosine."""
        if not self._idf_cache and self.docs:
            self._build_idf_index()

        query_tokens = self._tokenize_query(query)
        query_bigrams = self._extract_bigrams(query)
        if not query_tokens:
            return None

        best_score = 0.0
        best_response = None

        for doc in self.docs:
            parts = None
            for sep in ("\nASSISTANT: ", "\nA: ", "\nAnswer: ", "\n"):
                if sep in doc:
                    idx = doc.index(sep)
                    parts = (doc[:idx], doc[idx + len(sep):])
                    break
            if parts is None:
                continue

            q_part, a_part = parts
            if q_part.upper().startswith("USER:"):
                q_part = q_part[len("USER:"):].strip()
            doc_tokens = self._tokenize_query(q_part)

            score = 0.0
            if doc_tokens:
                doc_bigrams = self._extract_bigrams(q_part)
                score = self._tfidf_cosine(query_tokens, doc_tokens, query_bigrams, doc_bigrams)

            full_tokens = self._tokenize_query(doc)
            if full_tokens:
                full_coverage = len(set(query_tokens) & set(full_tokens)) / max(len(query_tokens), 1)
                if full_coverage > 0.6:
                    full_score = self._tfidf_cosine(
                        query_tokens, full_tokens, query_bigrams, self._extract_bigrams(doc))
                    score = max(score, full_score * 0.85)

            if score > best_score:
                best_score = score
                best_response = a_part.strip()

        # Also search knowledge stored in memory
        if self.memory is not None:
            try:
                keywords = list(query_tokens.keys())[:5]
                knowledge_results = self.memory.search_knowledge(keywords, limit=5)
                for entry in knowledge_results:
                    content = entry.get("content", "")
                    doc_tokens = self._tokenize_query(content)
                    if not doc_tokens:
                        continue
                    score = self._tfidf_cosine(query_tokens, doc_tokens)
                    if score > best_score:
                        best_score = score
                        best_response = content.strip()
            except Exception:
                pass

        # Adaptive threshold
        if len(query_tokens) >= 3:
            threshold = 0.30
        elif len(query_tokens) == 2:
            threshold = 0.40
        else:
            term = list(query_tokens.keys())[0] if query_tokens else ""
            threshold = 0.20 if term in self._idf_cache else 0.50

        if best_score >= threshold and best_response:
            return best_response
        return None

    # ─── Reply generation (retrieval-first) ───────────────────────────────────

    @staticmethod
    def _is_coherent(text: str, min_words: int = 3) -> bool:
        """Check if generated text looks like coherent English rather than gibberish."""
        if not text or not text.strip():
            return False
        words = text.split()
        if len(words) < min_words:
            return False
        clean_chars = sum(1 for c in text if c.isalnum() or c in " \n\t.,!?;:'-\"()")
        if clean_chars / max(len(text), 1) < 0.70:
            return False
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len < 1.5 or avg_len > 12:
            return False
        long_words = sum(1 for w in words if len(w) > 20)
        if long_words / len(words) > 0.3:
            return False
        alpha = [c for c in text.lower() if c.isalpha()]
        if not alpha:
            return False
        vowel_ratio = sum(1 for c in alpha if c in "aeiou") / len(alpha)
        if vowel_ratio < 0.20 or vowel_ratio > 0.65:
            return False
        return True

    _FALLBACK_RESPONSE = (
        "I don't have enough information to answer that. "
        "Try asking about AI security, adversarial robustness, or threat modeling."
    )

    def generate_reply(self, prompt: str, use_retrieval: bool = True) -> str:
        """Generate a reply using retrieval-first strategy."""
        response = None

        # Step 1: Corpus retrieval
        if use_retrieval:
            retrieved = self.retrieval_search(prompt)
            if retrieved:
                if retrieved.upper().startswith("ASSISTANT:"):
                    retrieved = retrieved[len("ASSISTANT:"):].strip()
                response = retrieved
                logger.debug("Used retrieval for response")

        # Step 2: Generative model
        if not response:
            generation_prompt = prompt
            if use_retrieval and self.memory is not None:
                try:
                    context_snippets = self.memory.retrieve_context(prompt, limit=2)
                    clean_context = []
                    for snippet in (context_snippets or []):
                        answer_part = snippet
                        if "\nA: " in snippet:
                            answer_part = snippet.split("\nA: ", 1)[1].strip()
                        if "Q: " not in answer_part and self._is_coherent(answer_part):
                            clean_context.append(answer_part)
                    if clean_context:
                        generation_prompt = clean_context[0][:200] + "\n" + prompt
                except Exception as exc:
                    logger.warning("Context retrieval failed: %s", exc)

            generated = self.generate(prompt=generation_prompt, max_tokens=60)
            if self._is_coherent(generated):
                response = generated
                logger.debug("Used generative model for response")

        # Step 3: Static fallback
        if not response:
            response = self._FALLBACK_RESPONSE
            logger.debug("Used static fallback response")

        # Persist
        if self.memory is not None and response != self._FALLBACK_RESPONSE:
            try:
                self.memory.store_conversation(
                    prompt, response, context_type="chat", relevance=0.75)
            except Exception as exc:
                logger.warning("Conversation persistence failed: %s", exc)

        return response

    # ─── Status & serialization ───────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Get current model status."""
        if self._backend == "legacy":
            st = self._legacy.status()
            st["backend"] = "legacy"
            return st

        num_params = self._model.get_num_params()
        return {
            "step": self.step_count,
            "last_loss": self.last_loss,
            "vocab_size": self.vocab_size,
            "corpus_size": len(self.docs),
            "train_data_size": len(self.docs),
            "params": num_params,
            "mode": self.training_mode,
            "backend": "nanogpt",
            "device": self._device,
            "tokenizer": self._tokenizer.backend if hasattr(self, "_tokenizer") else "unknown",
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "block_size": self.block_size,
            "token_stream_length": len(self._token_stream) if hasattr(self, "_token_stream") else 0,
        }

    CHECKPOINT_VERSION = 3  # v3 = nanoGPT format

    def export_state(self) -> Dict[str, Any]:
        """Export model state for JSON checkpoints (metadata only — weights go to .pt)."""
        state = {
            "version": self.CHECKPOINT_VERSION,
            "backend": self._backend,
            "config": {
                "n_layer": self.n_layer,
                "n_embd": self.n_embd,
                "block_size": self.block_size,
                "n_head": self.n_head,
                "vocab_size": self.vocab_size,
            },
            "training": {
                "step_count": self.step_count,
                "last_loss": self.last_loss,
                "training_mode": self.training_mode,
                "docs": self.docs[-500:],  # cap to prevent huge checkpoints
                "pools": {k: v[-200:] for k, v in self.pools.items()},
                "uchars": self.uchars,
                "BOS": self.BOS,
            },
        }
        if self._backend == "legacy":
            state["legacy_state"] = self._legacy.export_state()
        return state

    def load_state(self, payload: Dict[str, Any]):
        """Load a checkpoint payload into the current engine."""
        ckpt_version = payload.get("version", 1)
        ckpt_backend = payload.get("backend", "legacy")

        config = payload.get("config", {})
        self.n_layer = config.get("n_layer", self.n_layer)
        self.n_embd = config.get("n_embd", self.n_embd)
        self.block_size = config.get("block_size", self.block_size)
        self.n_head = config.get("n_head", self.n_head)
        self.vocab_size = config.get("vocab_size", self.vocab_size)

        training = payload.get("training", {})
        self.step_count = training.get("step_count", self.step_count)
        self.last_loss = training.get("last_loss", self.last_loss)
        self.training_mode = training.get("training_mode", self.training_mode)
        self.docs = training.get("docs", [])
        self.pools = training.get("pools", self.pools)
        self.uchars = training.get("uchars", self.uchars)
        self.BOS = training.get("BOS", self.BOS)
        self.char2id = {ch: i for i, ch in enumerate(self.uchars)}
        self.id2char = {i: ch for i, ch in enumerate(self.uchars)}

        # If loading a legacy checkpoint into nanoGPT engine, restore docs
        if self._backend == "nanogpt" and self.docs:
            self.set_training_data(self.docs)

        if self._backend == "legacy" and "legacy_state" in payload:
            self._legacy.load_state(payload["legacy_state"])
        elif self._backend == "legacy" and ckpt_version <= 2:
            # V1/V2 checkpoints are legacy format — pass entire payload
            self._legacy.load_state(payload)

    def save_checkpoint(self, path: str, metadata: Dict[str, Any] = None):
        """Save a full checkpoint: JSON metadata + .pt weights."""
        json_payload = {
            "saved_at": Path(path).name,
            "metadata": metadata or {},
            "engine": self.export_state(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(json_payload, handle)

        if self._backend == "nanogpt":
            pt_path = str(path).rsplit(".", 1)[0] + ".pt"
            torch.save({
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "config": {
                    "block_size": self._config.block_size,
                    "vocab_size": self._config.vocab_size,
                    "n_layer": self._config.n_layer,
                    "n_head": self._config.n_head,
                    "n_embd": self._config.n_embd,
                    "dropout": self._config.dropout,
                },
                "step_count": self.step_count,
                "last_loss": self.last_loss,
            }, pt_path)
            logger.info("Checkpoint saved: %s + %s", path, pt_path)
        else:
            logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> dict:
        """Load a full checkpoint."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.load_state(payload.get("engine", {}))

        # Try to load PyTorch weights
        if self._backend == "nanogpt":
            pt_path = str(path).rsplit(".", 1)[0] + ".pt"
            if os.path.exists(pt_path):
                try:
                    checkpoint = torch.load(pt_path, map_location=self._device, weights_only=False)

                    # Rebuild model if config differs
                    ckpt_config = checkpoint.get("config", {})
                    if (ckpt_config.get("vocab_size") and
                            ckpt_config["vocab_size"] != self._config.vocab_size):
                        self._config = GPTConfig(**ckpt_config)
                        self._model = NanoGPT(self._config).to(self._device)
                        self._optimizer = torch.optim.AdamW(
                            self._model.parameters(),
                            lr=self._learning_rate,
                            betas=(0.9, 0.999),
                            weight_decay=0.01,
                        )

                    self._model.load_state_dict(checkpoint["model_state_dict"])
                    self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    self.step_count = checkpoint.get("step_count", self.step_count)
                    self.last_loss = checkpoint.get("last_loss", self.last_loss)
                    self.params = range(self._model.get_num_params())
                    logger.info("PyTorch weights loaded from %s", pt_path)
                except Exception as exc:
                    logger.warning("Failed to load .pt weights: %s", exc)

        logger.info("Checkpoint loaded from %s", path)
        return payload

    def save(self, path: str = None):
        """Save full model state to disk."""
        if self._backend == "legacy":
            self._legacy.save(path)
            return

        if path is None:
            path = str(_LOG_DIR / "sancta_model_latest.json")
        try:
            self.save_checkpoint(path, metadata={"save_type": "manual"})
        except Exception as e:
            logger.error("Failed to save: %s", e)


# =====================================================
# GLOBAL ENGINE INSTANCE
# =====================================================
_engine = None


def get_engine() -> SanctaGPT:
    """Provides a singleton instance of the engine."""
    global _engine
    if _engine is None:
        _engine = SanctaGPT()
    return _engine


def init(train_steps: int = 0, n_layer: int = 4, n_embd: int = 128) -> SanctaGPT:
    """Initializes the engine with specific parameters."""
    global _engine
    _engine = SanctaGPT(n_layer=n_layer, n_embd=n_embd)
    return _engine
