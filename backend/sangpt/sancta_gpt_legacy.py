"""
SANCTA-GPT Engine
Minimal, dependency-free GPT implementation based on Karpathy's makemore.
Provides training and inference capabilities for the SANCTA system.

Core algorithm from: https://github.com/karpathy/makemore
License: MIT
"""

import os
import json
import math
import random
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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

logger = logging.getLogger("sancta_gpt")


# =====================================================
# AUTOGRAD: Automatic Differentiation
# =====================================================
class Value:
    """A scalar value with automatic gradient computation."""
    
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        """Compute gradients via backpropagation."""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# =====================================================
# SANCTA GPT ENGINE
# =====================================================
class SanctaGPT:
    """
    GPT model for SANCTA system.
    Trains on documents and generates text.
    """

    def __init__(self, n_layer=1, n_embd=32, block_size=256, n_head=4, vocab_size=128):
        """Initialize the GPT model."""
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.vocab_size = vocab_size
        self.step_count = 0
        self.last_loss = None
        self.training_mode = "OPERATOR"
        
        # Initialize state dict with model parameters
        self.state_dict = {
            'wte': self._matrix(vocab_size, n_embd),  # token embeddings
            'wpe': self._matrix(block_size, n_embd),   # position embeddings
            'lm_head': self._matrix(vocab_size, n_embd)  # output head
        }
        
        # Transformer layers
        for i in range(n_layer):
            self.state_dict[f'layer{i}.attn_wq'] = self._matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wk'] = self._matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wv'] = self._matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wo'] = self._matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.mlp_fc1'] = self._matrix(4 * n_embd, n_embd)
            self.state_dict[f'layer{i}.mlp_fc2'] = self._matrix(n_embd, 4 * n_embd)
        
        # Flatten parameters into single list
        self.params = [p for mat in self.state_dict.values() for row in mat for p in row]
        
        # Adam optimizer state
        self.m = [0.0] * len(self.params)  # first moment
        self.v = [0.0] * len(self.params)  # second moment
        self.learning_rate = 0.003
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps_adam = 1e-8
        
        # Training metadata
        self.docs = []
        self.uchars = []
        self.char2id = {}
        self.id2char = {}
        self.BOS = None
        
        # Training data pools
        self.pools = {
            "convo": [],
            "security": [],
            "knowledge": [],
            "all": []
        }
        
        # Initialize memory manager
        self.memory = None
        if get_memory is not None:
            try:
                self.memory = get_memory()
            except Exception as e:
                logger.warning(f"Memory manager initialization failed: {e}")
        
        logger.info(f"SanctaGPT initialized: {len(self.params)} parameters")

    def set_training_mode(self, mode: str):
        """Sets the current operational mode of the engine."""
        self.training_mode = mode
        logger.info(f"Engine mode changed to: {mode}")

    def _matrix(self, nout, nin, std=0.08):
        """Create a matrix of Value objects."""
        return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    def _matrix_from_data(self, data):
        """Create a matrix of Value objects from raw float data."""
        return [[Value(cell) for cell in row] for row in data]

    def _refresh_params(self):
        """Refresh flattened parameter view after state changes."""
        self.params = [p for mat in self.state_dict.values() for row in mat for p in row]

    def _expand_embeddings(self, new_vocab_size):
        """Expands embedding matrices without losing existing weights or optimizer momentum."""
        old_wte = self.state_dict['wte']
        old_lm = self.state_dict['lm_head']

        new_rows_count = new_vocab_size - len(old_wte)
        if new_rows_count <= 0:
            return

        old_param_count = len(self.params)

        new_wte_rows = self._matrix(new_rows_count, self.n_embd)
        new_lm_rows = self._matrix(new_rows_count, self.n_embd)

        self.state_dict['wte'] = old_wte + new_wte_rows
        self.state_dict['lm_head'] = old_lm + new_lm_rows

        # Refresh parameter list, then extend (not reset) optimizer state
        self._refresh_params()
        new_param_count = len(self.params)
        added = new_param_count - old_param_count
        self.m = self.m[:old_param_count] + [0.0] * added
        self.v = self.v[:old_param_count] + [0.0] * added
        logger.info(f"Vocab expanded to {new_vocab_size}. Optimizer momentum preserved for {old_param_count} existing params.")

    def set_training_data(self, docs: List[str]):
        """Set the training documents and initialize vocabulary."""
        self.docs = [doc for doc in docs if doc]
        self.pools["all"] = list(self.docs)
        self.uchars = sorted(set(''.join(self.docs)))
        self.BOS = len(self.uchars)
        self.char2id = {ch: i for i, ch in enumerate(self.uchars)}
        self.id2char = {i: ch for i, ch in enumerate(self.uchars)}
        self.vocab_size = len(self.uchars) + 1
        
        # Adjust weight matrices if vocab size changed
        if self.vocab_size > len(self.state_dict['wte']):
            self._expand_embeddings(self.vocab_size)

        # Build TF-IDF index for retrieval
        self._build_idf_index()
        logger.debug(f"Training vocab size: {self.vocab_size}")

    def train(self, steps: int = 100):
        """Run a simple multi-step training loop on the active corpus."""
        self.begin_training_run(max_steps=steps)
        loss = None
        for _ in range(max(0, steps)):
            loss = self.train_step()
        return loss

    def _linear(self, x, w):
        """Linear transformation."""
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

    def _softmax(self, logits):
        """Numerically stable softmax with underflow protection."""
        max_val = max(val.data for val in logits)
        exps = [(val - max_val).exp() for val in logits]
        total = sum(exps)
        # Guard against total underflow to zero
        if total.data == 0:
            uniform = Value(1.0 / len(logits))
            return [uniform for _ in logits]
        return [e / total for e in exps]

    def _log_softmax(self, logits, target_idx):
        """Compute log-softmax for target index using log-sum-exp trick (avoids log(0))."""
        max_val = max(val.data for val in logits)
        shifted = [val - max_val for val in logits]
        # log(sum(exp(shifted)))
        exp_sum = sum(s.exp() for s in shifted)
        # Guard against zero
        if exp_sum.data <= 0:
            return Value(-20.0)  # Large but finite loss
        log_sum_exp = exp_sum.log() + max_val
        # log_softmax(target) = logits[target] - log_sum_exp
        return logits[target_idx] - log_sum_exp

    def _clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients by global L2 norm to prevent exploding gradients."""
        total_norm_sq = sum(p.grad ** 2 for p in self.params)
        total_norm = math.sqrt(total_norm_sq)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for p in self.params:
                p.grad *= scale

    def _rmsnorm(self, x):
        """RMSNorm normalization."""
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]

    def _forward(self, token_id, pos_id, keys, values):
        """Forward pass through GPT."""
        if token_id is None:
            token_id = self.BOS if self.BOS is not None else 0

        # Embeddings — clamp with warning if token exceeds embedding table
        wte_len = len(self.state_dict['wte'])
        if token_id >= wte_len:
            logger.debug(f"Token id {token_id} exceeds embedding table size {wte_len}, clamping")
            token_id = wte_len - 1
        tok_emb = self.state_dict['wte'][token_id]
        pos_emb = self.state_dict['wpe'][min(pos_id, self.block_size-1)]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = self._rmsnorm(x)

        # Transformer layers
        for li in range(self.n_layer):
            x_residual = x
            x = self._rmsnorm(x)
            
            # Multi-head attention
            q = self._linear(x, self.state_dict[f'layer{li}.attn_wq'])
            k = self._linear(x, self.state_dict[f'layer{li}.attn_wk'])
            v = self._linear(x, self.state_dict[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)
            
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]
                
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / (self.head_dim**0.5)
                              for t in range(len(k_h))]
                attn_weights = self._softmax(attn_logits)
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                           for j in range(self.head_dim)]
                x_attn.extend(head_out)
            
            x = self._linear(x_attn, self.state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]
            
            # MLP
            x_residual = x
            x = self._rmsnorm(x)
            x = self._linear(x, self.state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = self._linear(x, self.state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = self._linear(x, self.state_dict['lm_head'])
        return logits

    # Training hyperparameters
    WARMUP_STEPS = 50
    MAX_GRAD_NORM = 1.0
    MIN_LR_RATIO = 0.1  # floor at 10% of peak LR

    def begin_training_run(self, max_steps: int = 300):
        """Reset the LR schedule for a new training run.

        Call this before starting a batch of train_step() calls so the cosine
        schedule is relative to the *current run*, not cumulative lifetime steps.
        """
        self._run_step_offset = self.step_count
        self._run_max_steps = max(1, max_steps)
        logger.info(f"Training run started: offset={self._run_step_offset}, max_steps={self._run_max_steps}")

    def _get_lr(self, step: int, max_steps: int = 300) -> float:
        """Cosine learning rate schedule with linear warmup and LR floor.

        Uses per-run step offset so the schedule resets for each training run
        instead of decaying permanently based on cumulative step_count.
        """
        run_offset = getattr(self, '_run_step_offset', 0)
        run_max = getattr(self, '_run_max_steps', max_steps)
        local_step = step - run_offset

        if local_step < self.WARMUP_STEPS:
            return self.learning_rate * (local_step + 1) / self.WARMUP_STEPS
        progress = min((local_step - self.WARMUP_STEPS) / max(1, run_max - self.WARMUP_STEPS), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.learning_rate * max(cosine_decay, self.MIN_LR_RATIO)

    def train_step(self, doc: Optional[str] = None):
        """Perform one training step with log-softmax loss, gradient clipping, and cosine LR."""
        if not doc and self.docs:
            doc = random.choice(self.docs)

        if not doc:
            logger.warning("train_step called with no document and empty corpus — skipping")
            return 0.0

        # Tokenize
        tokens = [self.BOS] + [self.char2id.get(ch, self.BOS) for ch in doc] + [self.BOS]
        n = min(self.block_size, len(tokens) - 1)

        # Forward pass — use log-softmax to avoid log(0) crashes
        keys, values = [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = self._forward(token_id, pos_id, keys, values)
            target_idx = min(target_id, len(logits) - 1)
            log_prob = self._log_softmax(logits, target_idx)
            losses.append(-log_prob)

        loss = (1 / n) * sum(losses) if losses else Value(0)
        self.last_loss = loss.data

        # Backward pass
        loss.backward()

        # Gradient clipping
        self._clip_gradients(self.MAX_GRAD_NORM)

        # Adam optimizer step with cosine LR schedule + warmup
        lr_t = self._get_lr(self.step_count)
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** (self.step_count + 1))
            v_hat = self.v[i] / (1 - self.beta2 ** (self.step_count + 1))
            p.data -= lr_t * m_hat / (math.sqrt(v_hat) + self.eps_adam)
            p.grad = 0

        self.step_count += 1
        return self.last_loss

    def _ensure_vocabulary(self):
        """Ensure vocabulary is initialized with defaults if empty."""
        if not self.uchars:
            self.uchars = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'-\"()[]{}@#$%&*+=\n\t"))
            self.BOS = len(self.uchars)
            self.char2id = {ch: i for i, ch in enumerate(self.uchars)}
            self.id2char = {i: ch for i, ch in enumerate(self.uchars)}
            self.vocab_size = len(self.uchars) + 1
            logger.debug(f"Initialized default vocabulary with {self.vocab_size} tokens")
        
        if not self.char2id:
            self.char2id = {ch: i for i, ch in enumerate(self.uchars)}
            self.id2char = {i: ch for i, ch in enumerate(self.uchars)}

    def generate(self, prompt: str = "", max_tokens: int = 120, temperature: float = 0.7) -> str:
        """Generate text, optionally starting with a prompt context."""
        self._ensure_vocabulary()
        
        keys, values = [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]
        sample = list(prompt)
        token_id = self.BOS
        
        # Warm up context with correct position indices
        if prompt:
            for pos, char in enumerate(prompt):
                char_id = self.char2id.get(char, self.BOS)
                _ = self._forward(char_id, pos, keys, values)
                token_id = char_id

        # Generate
        start_pos = len(prompt)
        for pos_id in range(start_pos, min(start_pos + max_tokens, self.block_size)):
            logits = self._forward(token_id, pos_id, keys, values)
            probs = self._softmax([l / temperature for l in logits])
            token_id = random.choices(range(len(probs)), weights=[p.data for p in probs])[0]
            
            if token_id == self.BOS:
                break
            
            if token_id < len(self.uchars):
                sample.append(self.id2char.get(token_id, ''))
        
        return "".join(sample[len(prompt):])

    # Maximum documents to keep in memory (rolling window).
    # Must be large enough to hold the full training corpus; previous value
    # of 2000 silently evicted older domain knowledge when corpus exceeded it.
    MAX_DOCS = 8000

    def learn_from_interaction(self, prompt: str, response: str):
        """Learn from a prompt-response pair."""
        self._ensure_vocabulary()
        combined = f"{prompt} {response}"

        new_chars = set(combined) - set(self.uchars)
        if new_chars:
            self.uchars = sorted(set(self.uchars) | new_chars)
            self.char2id = {ch: i for i, ch in enumerate(self.uchars)}
            self.id2char = {i: ch for i, ch in enumerate(self.uchars)}
            self.BOS = len(self.uchars)
            self.vocab_size = len(self.uchars) + 1
            self._expand_embeddings(self.vocab_size)

        self.pools["all"].append(combined)
        self.docs.append(combined)

        # Cap document lists to prevent unbounded memory growth
        if len(self.docs) > self.MAX_DOCS:
            self.docs = self.docs[-self.MAX_DOCS:]
        if len(self.pools["all"]) > self.MAX_DOCS:
            self.pools["all"] = self.pools["all"][-self.MAX_DOCS:]

        if self.memory is not None:
            try:
                self.memory.store_knowledge("interaction", combined[:1000], source="interaction", relevance=0.55)
            except Exception as exc:
                logger.warning(f"Failed to persist learned interaction: {exc}")
        self.train_step(combined[:min(self.block_size, len(combined))])

    # ── Retrieval: TF-IDF with n-grams and stemming ──

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
        # Minimal stop words — let IDF naturally downweight common terms
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
        # Strip punctuation from each word before processing
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
        logger.debug(f"IDF index built: {len(self._idf_cache)} terms, {len(self._bigram_idf_cache)} bigrams")

    def _tfidf_cosine(self, tokens_a: Counter, tokens_b: Counter,
                       bigrams_a: Counter = None, bigrams_b: Counter = None) -> float:
        """Compute TF-IDF weighted cosine similarity between two token sets."""
        # Unigram TF-IDF score
        def _weighted(tokens):
            return {t: count * self._idf_cache.get(t, 1.0) for t, count in tokens.items()}

        wa = _weighted(tokens_a)
        wb = _weighted(tokens_b)
        matched_terms = set(wa) & set(wb)
        overlap = sum(wa[t] * wb[t] for t in matched_terms)
        norm_a = math.sqrt(sum(v * v for v in wa.values())) or 1.0
        norm_b = math.sqrt(sum(v * v for v in wb.values())) or 1.0
        score = overlap / (norm_a * norm_b)

        # Query coverage penalty: if most query terms don't match the doc,
        # penalize the score to prevent single high-IDF term domination
        if len(tokens_a) > 1:
            coverage = len(matched_terms) / len(tokens_a)
            if coverage < 0.6:
                score *= coverage

        # Bigram bonus — phrase matches get a boost
        if bigrams_a and bigrams_b:
            shared = set(bigrams_a) & set(bigrams_b)
            if shared:
                bonus = sum(self._bigram_idf_cache.get(bg, 1.0) for bg in shared)
                score += 0.15 * min(bonus / max(len(bigrams_a), 1), 1.0)

        return score

    def retrieval_search(self, query: str, top_k: int = 1) -> Optional[str]:
        """
        Search stored training docs for the best-matching response using
        TF-IDF cosine similarity with n-gram phrase matching.
        """
        # Rebuild IDF if stale or missing
        if not self._idf_cache and self.docs:
            self._build_idf_index()

        query_tokens = self._tokenize_query(query)
        query_bigrams = self._extract_bigrams(query)
        if not query_tokens:
            return None

        best_score = 0.0
        best_response = None

        # Search through training documents that look like Q&A pairs
        for doc in self.docs:
            parts = None
            # Try structured separators first, then fall back to newline
            for sep in ("\nASSISTANT: ", "\nA: ", "\nAnswer: ", "\n"):
                if sep in doc:
                    idx = doc.index(sep)
                    parts = (doc[:idx], doc[idx + len(sep):])
                    break
            if parts is None:
                continue

            q_part, a_part = parts
            # Strip USER: prefix from question part for cleaner matching
            if q_part.upper().startswith("USER:"):
                q_part = q_part[len("USER:"):].strip()
            doc_tokens = self._tokenize_query(q_part)

            # Score against question part first (primary signal)
            score = 0.0
            if doc_tokens:
                doc_bigrams = self._extract_bigrams(q_part)
                score = self._tfidf_cosine(query_tokens, doc_tokens, query_bigrams, doc_bigrams)

            # Also check full document text for keyword coverage
            # (handles cases where search terms appear in the answer, not question)
            full_tokens = self._tokenize_query(doc)
            if full_tokens:
                full_coverage = len(set(query_tokens) & set(full_tokens)) / max(len(query_tokens), 1)
                if full_coverage > 0.6:
                    full_score = self._tfidf_cosine(query_tokens, full_tokens, query_bigrams, self._extract_bigrams(doc))
                    # Use full-doc score if better, but discount slightly (question match is more precise)
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

        # Adaptive threshold based on query length
        if len(query_tokens) >= 3:
            threshold = 0.30
        elif len(query_tokens) == 2:
            threshold = 0.40
        else:
            # Single-token query: use lower threshold if the term is a known corpus term
            # (prevents matching on random words while allowing domain-specific lookups)
            term = list(query_tokens.keys())[0] if query_tokens else ""
            threshold = 0.20 if term in self._idf_cache else 0.50

        if best_score >= threshold and best_response:
            return best_response
        return None

    @staticmethod
    def _is_coherent(text: str, min_words: int = 3) -> bool:
        """Check if generated text looks like coherent English rather than gibberish.

        Heuristics:
        - Must have enough words
        - Most characters should be alphanumeric or spaces (rejects symbol-heavy noise)
        - Average word length must be reasonable
        - Vowel ratio must be plausible for English
        - Most words should be short enough to be real words
        """
        if not text or not text.strip():
            return False

        words = text.split()
        if len(words) < min_words:
            return False

        # At least 70% of characters should be letters, digits, or spaces
        clean_chars = sum(1 for c in text if c.isalnum() or c in " \n\t.,!?;:'-\"()")
        if clean_chars / max(len(text), 1) < 0.70:
            return False

        # Average word length must be reasonable (2-12 chars for English)
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len < 1.5 or avg_len > 12:
            return False

        # Most words should be <= 20 chars (real words, not noise runs)
        long_words = sum(1 for w in words if len(w) > 20)
        if long_words / len(words) > 0.3:
            return False

        # Check vowel ratio — English is ~38%; gibberish is random (~20%)
        alpha = [c for c in text.lower() if c.isalpha()]
        if not alpha:
            return False
        vowel_ratio = sum(1 for c in alpha if c in "aeiou") / len(alpha)
        if vowel_ratio < 0.20 or vowel_ratio > 0.65:
            return False

        return True

    _FALLBACK_RESPONSE = "I don't have enough information to answer that. Try asking about AI security, adversarial robustness, or threat modeling."

    def generate_reply(self, prompt: str, use_retrieval: bool = True) -> str:
        """Generate a reply using retrieval-first strategy.

        Order of priority:
        1. TF-IDF retrieval from corpus (authoritative — most reliable)
        2. Generative model grounded with memory context (if coherent)
        3. Static fallback message
        """
        response = None

        # Step 1: Corpus retrieval — authoritative answer source
        if use_retrieval:
            retrieved = self.retrieval_search(prompt)
            if retrieved:
                # Strip the ASSISTANT: prefix if present from CSV-ingested data
                if retrieved.upper().startswith("ASSISTANT:"):
                    retrieved = retrieved[len("ASSISTANT:"):].strip()
                response = retrieved
                logger.debug("Used retrieval for response")

        # Step 2: Try the generative model, optionally grounded with memory context
        if not response:
            generation_prompt = prompt
            if use_retrieval and self.memory is not None:
                try:
                    context_snippets = self.memory.retrieve_context(prompt, limit=2)
                    # Only use context snippets that are coherent and not echoes
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
                    logger.warning(f"Context retrieval failed: {exc}")

            generated = self.generate(prompt=generation_prompt, max_tokens=60)
            if self._is_coherent(generated):
                response = generated
                logger.debug("Used generative model for response")

        # Step 3: Static fallback
        if not response:
            response = self._FALLBACK_RESPONSE
            logger.debug("Used static fallback response")

        # Only persist retrieval-sourced or coherent generated responses
        if self.memory is not None and response != self._FALLBACK_RESPONSE:
            try:
                self.memory.store_conversation(prompt, response, context_type="chat", relevance=0.75)
            except Exception as exc:
                logger.warning(f"Conversation persistence failed: {exc}")

        return response

    def status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            "step": self.step_count,
            "last_loss": self.last_loss,
            "vocab_size": self.vocab_size,
            "corpus_size": len(self.docs),
            "train_data_size": len(self.docs),
            "params": len(self.params),
            "mode": self.training_mode
        }

    CHECKPOINT_VERSION = 2

    def export_state(self) -> Dict[str, Any]:
        """Export full model state for JSON checkpoints."""
        return {
            "version": self.CHECKPOINT_VERSION,
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
                "docs": self.docs,
                "pools": self.pools,
                "uchars": self.uchars,
                "BOS": self.BOS,
            },
            "optimizer": {
                "learning_rate": self.learning_rate,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "eps_adam": self.eps_adam,
                "m": self.m,
                "v": self.v,
            },
            "state_dict": {
                name: [[cell.data for cell in row] for row in matrix]
                for name, matrix in self.state_dict.items()
            },
        }

    def load_state(self, payload: Dict[str, Any]):
        """Load a checkpoint payload into the current engine."""
        ckpt_version = payload.get("version", 1)
        if ckpt_version > self.CHECKPOINT_VERSION:
            logger.warning(f"Checkpoint version {ckpt_version} is newer than engine version {self.CHECKPOINT_VERSION}. Loading anyway.")
        config = payload.get("config", {})
        self.n_layer = config.get("n_layer", self.n_layer)
        self.n_embd = config.get("n_embd", self.n_embd)
        self.block_size = config.get("block_size", self.block_size)
        self.n_head = config.get("n_head", self.n_head)
        self.head_dim = self.n_embd // self.n_head
        self.vocab_size = config.get("vocab_size", self.vocab_size)

        state_dict = payload.get("state_dict", {})
        if state_dict:
            self.state_dict = {
                name: self._matrix_from_data(matrix)
                for name, matrix in state_dict.items()
            }
            self._refresh_params()

        optimizer = payload.get("optimizer", {})
        self.learning_rate = optimizer.get("learning_rate", self.learning_rate)
        self.beta1 = optimizer.get("beta1", self.beta1)
        self.beta2 = optimizer.get("beta2", self.beta2)
        self.eps_adam = optimizer.get("eps_adam", self.eps_adam)
        self.m = optimizer.get("m", [0.0] * len(self.params))
        self.v = optimizer.get("v", [0.0] * len(self.params))
        if len(self.m) != len(self.params):
            self.m = [0.0] * len(self.params)
        if len(self.v) != len(self.params):
            self.v = [0.0] * len(self.params)

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

    def save_checkpoint(self, path: str, metadata: Dict[str, Any] = None):
        """Save a full JSON checkpoint with model and metadata."""
        payload = {
            "saved_at": Path(path).name,
            "metadata": metadata or {},
            "engine": self.export_state(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load a full JSON checkpoint."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.load_state(payload.get("engine", {}))
        logger.info(f"Checkpoint loaded from {path}")
        return payload

    def save(self, path: str = None):
        """Save full model state (weights, vocab, optimizer) to disk."""
        if path is None:
            path = str(_LOG_DIR / "sancta_model_latest.json")
        try:
            self.save_checkpoint(path, metadata={"save_type": "manual"})
        except Exception as e:
            logger.error(f"Failed to save: {e}")

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

def init(train_steps: int = 0, n_layer: int = 1, n_embd: int = 32) -> SanctaGPT:
    """Initializes the engine with specific parameters."""
    global _engine
    _engine = SanctaGPT(n_layer=n_layer, n_embd=n_embd)
    return _engine
