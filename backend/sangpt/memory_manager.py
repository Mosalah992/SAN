"""
Persistent Memory Manager for SANCTA-GPT
Stores and retrieves conversation history, learned knowledge, and security events.
Uses SQLite for durability and efficient querying.
"""

import math
import sqlite3
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger("MemoryManager")


@dataclass
class Conversation:
    """Represents a single conversation exchange"""
    id: int
    operator_input: str
    agent_response: str
    timestamp: str
    context_type: str
    session_id: str
    relevance_score: float = 0.5


@dataclass
class KnowledgeEntry:
    """Represents a learned fact or pattern"""
    id: int
    topic: str
    content: str
    source: str
    learned_at: str
    relevance_score: float


class MemoryManager:
    """
    Persistent memory system for SANCTA-GPT.
    Stores conversations, learned knowledge, and security events.
    """
    
    def __init__(self, db_path: str = "./DATA/memory/sancta_memory.db", session_id: str = None):
        """Initialize memory manager and create tables if needed."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conn = None
        self._init_db()
        logger.info(f"Memory manager initialized at {self.db_path}")
    
    def _init_db(self):
        """Initialize database tables."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        
        # Conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                operator_input TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                context_type TEXT DEFAULT 'general',
                relevance_score REAL DEFAULT 0.5,
                embedding TEXT
            )
        """)
        
        # Learned knowledge base
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT DEFAULT 'interaction',
                learned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                relevance_score REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0
            )
        """)
        
        # Security events linked to conversations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                event_type TEXT NOT NULL,
                severity TEXT DEFAULT 'info',
                description TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        """)
        
        # Training statistics per session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                conversations_count INTEGER DEFAULT 0,
                knowledge_learned INTEGER DEFAULT 0,
                total_loss REAL,
                final_vocab_size INTEGER
            )
        """)
        
        # Extend security_events with attack classification columns (safe for existing DBs)
        for col, coltype in [
            ("attack_type", "TEXT"),
            ("confidence", "REAL"),
            ("response_quality", "REAL"),
            ("mitre_tactic", "TEXT"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE security_events ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Create indices for fast retrieval
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic ON knowledge(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relevance ON knowledge(relevance_score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sec_event_type ON security_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sec_attack_type ON security_events(attack_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sec_timestamp ON security_events(timestamp)")

        cursor.execute("""
            INSERT OR IGNORE INTO session_stats (session_id)
            VALUES (?)
        """, (self.session_id,))
        
        self.conn.commit()
        logger.debug("Database tables initialized")
    
    def store_conversation(self, operator_input: str, agent_response: str, 
                          context_type: str = "general", relevance: float = 0.5) -> int:
        """
        Store a conversation exchange.
        
        Args:
            operator_input: User query or command
            agent_response: Agent's response
            context_type: Type of context (general/security/risk/conversational)
            relevance: Relevance score 0-1 for retrieval ranking
            
        Returns:
            Conversation ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conversations 
            (session_id, operator_input, agent_response, context_type, relevance_score)
            VALUES (?, ?, ?, ?, ?)
        """, (self.session_id, operator_input, agent_response, context_type, relevance))
        self.conn.commit()
        conv_id = cursor.lastrowid
        logger.debug(f"Stored conversation {conv_id}")
        return conv_id
    
    def store_knowledge(self, topic: str, content: str, source: str = "interaction", 
                       relevance: float = 0.5) -> int:
        """
        Store a learned fact or pattern.
        
        Args:
            topic: Knowledge topic/category
            content: The knowledge content
            source: Where the knowledge came from
            relevance: Relevance score for future retrieval
            
        Returns:
            Knowledge ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO knowledge (topic, content, source, relevance_score)
            VALUES (?, ?, ?, ?)
        """, (topic, content, source, relevance))
        self.conn.commit()
        know_id = cursor.lastrowid
        logger.debug(f"Stored knowledge {know_id} on {topic}")
        return know_id
    
    def log_security_event(self, event_type: str, conv_id: int = None,
                          severity: str = "info", description: str = "",
                          attack_type: str = None, confidence: float = None,
                          response_quality: float = None, mitre_tactic: str = None) -> int:
        """
        Log a security-relevant event with optional attack classification.

        Args:
            event_type: Type of event (ATTACK, DEFENSE, ANOMALY, etc)
            conv_id: Associated conversation ID
            severity: critical/warning/info
            description: Event details
            attack_type: Classified attack category (from AttackDetector)
            confidence: Classification confidence 0-1
            response_quality: Defense response similarity score 0-1
            mitre_tactic: Mapped MITRE ATT&CK tactic

        Returns:
            Event ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO security_events
            (conversation_id, event_type, severity, description,
             attack_type, confidence, response_quality, mitre_tactic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (conv_id, event_type, severity, description,
              attack_type, confidence, response_quality, mitre_tactic))
        self.conn.commit()
        event_id = cursor.lastrowid
        logger.info(f"Logged security event {event_id}: {event_type} ({severity}) attack_type={attack_type}")
        return event_id
    
    @staticmethod
    def _stem(word: str) -> str:
        """Simple suffix-stripping stemmer matching the engine's stemmer."""
        if len(word) <= 3:
            return word
        for suffix in ("ation", "ment", "ness", "ible", "able", "ial", "ity",
                        "ing", "ous", "ive", "ion", "ed", "ly", "er", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    @staticmethod
    def _tokenize(text: str) -> Counter:
        """Tokenize text into stemmed word counts for TF-IDF scoring."""
        stop = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
                "for", "on", "and", "or", "it", "i", "you", "that", "this", "my",
                "be", "do", "does", "did", "has", "have", "had", "been", "will",
                "can", "could", "would", "should", "may", "also", "however", "but"}
        words = text.lower().split()
        return Counter(MemoryManager._stem(w) for w in words if len(w) > 1 and w not in stop)

    @staticmethod
    def _tfidf_score(tokens_a: Counter, tokens_b: Counter) -> float:
        """Cosine similarity between two token vectors (TF-weighted, no IDF needed for small sets)."""
        overlap = sum(tokens_a[t] * tokens_b[t] for t in tokens_a if t in tokens_b)
        norm_a = math.sqrt(sum(v * v for v in tokens_a.values())) or 1.0
        norm_b = math.sqrt(sum(v * v for v in tokens_b.values())) or 1.0
        return overlap / (norm_a * norm_b)

    def retrieve_context(self, query: str, limit: int = 5, context_type: str = None,
                         cross_session: bool = True) -> List[str]:
        """
        Retrieve relevant past conversations using TF-IDF-style scoring.
        Searches across all sessions by default for long-term memory.
        """
        cursor = self.conn.cursor()
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        conditions = []
        params = []

        if not cross_session:
            conditions.append("session_id = ?")
            params.append(self.session_id)

        if context_type:
            conditions.append("context_type = ?")
            params.append(context_type)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        cursor.execute(f"""
            SELECT operator_input, agent_response, relevance_score
            FROM conversations
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """, params + [limit * 5])

        results = cursor.fetchall()

        # Score each result by TF-IDF cosine similarity
        scored = []
        for row in results:
            doc_tokens = self._tokenize(row[0])
            if not doc_tokens:
                continue
            score = self._tfidf_score(query_tokens, doc_tokens)
            # Boost by stored relevance_score
            score *= (0.5 + 0.5 * (row[2] or 0.5))
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        context = []
        for score, row in scored[:limit]:
            if score > 0.05:
                context.append(f"Q: {row[0]}\nA: {row[1]}")

        logger.debug(f"Retrieved {len(context)} context snippets (cross_session={cross_session})")
        return context

    def search_conversations(self, keywords: List[str], limit: int = 10,
                              cross_session: bool = True) -> List[Dict[str, Any]]:
        """Search conversations by keyword content across operator_input and agent_response."""
        cursor = self.conn.cursor()
        conditions = " OR ".join(
            ["operator_input LIKE ?" for _ in keywords] +
            ["agent_response LIKE ?" for _ in keywords]
        )
        params = [f"%{kw}%" for kw in keywords] * 2

        session_filter = ""
        if not cross_session:
            session_filter = "AND session_id = ?"
            params.append(self.session_id)

        cursor.execute(f"""
            SELECT id, operator_input, agent_response, timestamp, context_type, relevance_score
            FROM conversations
            WHERE ({conditions}) {session_filter}
            ORDER BY relevance_score DESC, timestamp DESC
            LIMIT ?
        """, params + [limit])

        return [dict(row) for row in cursor.fetchall()]
    
    def get_knowledge_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge by topic.
        
        Args:
            topic: Knowledge topic to search
            limit: Maximum results
            
        Returns:
            List of knowledge entries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, topic, content, source, relevance_score, usage_count, learned_at
            FROM knowledge
            WHERE topic LIKE ?
            ORDER BY relevance_score DESC, usage_count DESC
            LIMIT ?
        """, (f"%{topic}%", limit))
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def search_knowledge(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search knowledge by keywords.
        
        Args:
            keywords: List of search terms
            limit: Maximum results
            
        Returns:
            List of matching knowledge entries
        """
        cursor = self.conn.cursor()
        
        # Build OR condition for all keywords
        conditions = " OR ".join(["content LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        
        cursor.execute(f"""
            SELECT id, topic, content, source, relevance_score, usage_count, learned_at
            FROM knowledge
            WHERE {conditions}
            ORDER BY relevance_score DESC, usage_count DESC
            LIMIT ?
        """, params + [limit])
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def increment_knowledge_usage(self, knowledge_id: int):
        """Mark a knowledge entry as used (for ranking)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE knowledge 
            SET usage_count = usage_count + 1 
            WHERE id = ?
        """, (knowledge_id,))
        self.conn.commit()
    
    def get_session_conversations(self, limit: int = 50) -> List[Conversation]:
        """Get conversation history for current session."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, operator_input, agent_response, timestamp, context_type, session_id, relevance_score
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.session_id, limit))

        rows = cursor.fetchall()
        return [Conversation(*row) for row in rows]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session."""
        cursor = self.conn.cursor()
        
        # Count conversations
        cursor.execute("""
            SELECT COUNT(*) FROM conversations WHERE session_id = ?
        """, (self.session_id,))
        conv_count = cursor.fetchone()[0]
        
        # Count knowledge learned
        cursor.execute("""
            SELECT COUNT(*) FROM knowledge 
            WHERE learned_at > (
                SELECT started_at FROM session_stats WHERE session_id = ?
                LIMIT 1
            )
        """, (self.session_id,))
        know_count = cursor.fetchone()[0]
        
        return {
            "session_id": self.session_id,
            "conversations": conv_count,
            "knowledge_learned": know_count,
            "timestamp": datetime.now().isoformat()
        }

    def update_session_stats(self, total_loss: float = None, final_vocab_size: int = None):
        """Persist session-level model statistics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE session_stats
            SET conversations_count = (
                    SELECT COUNT(*) FROM conversations WHERE session_id = ?
                ),
                knowledge_learned = (
                    SELECT COUNT(*) FROM knowledge
                ),
                total_loss = COALESCE(?, total_loss),
                final_vocab_size = COALESCE(?, final_vocab_size)
            WHERE session_id = ?
        """, (self.session_id, total_loss, final_vocab_size, self.session_id))
        self.conn.commit()
    
    # --- Security Event Forensics (Phase 2D) ---

    def get_security_events(self, limit: int = 50, attack_type: str = None,
                            session_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieve security events with optional filtering."""
        cursor = self.conn.cursor()
        conditions = []
        params = []
        if attack_type:
            conditions.append("attack_type = ?")
            params.append(attack_type)
        if session_only:
            conditions.append("""conversation_id IN (
                SELECT id FROM conversations WHERE session_id = ?)""")
            params.append(self.session_id)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cursor.execute(f"""
            SELECT id, conversation_id, event_type, severity, description,
                   timestamp, attack_type, confidence, response_quality, mitre_tactic
            FROM security_events {where}
            ORDER BY timestamp DESC
            LIMIT ?
        """, params + [limit])
        return [dict(row) for row in cursor.fetchall()]

    def detect_campaigns(self, time_window_minutes: int = 30, min_events: int = 3) -> List[Dict[str, Any]]:
        """Group security events by time window + attack_type to detect coordinated attacks."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, attack_type, timestamp, severity, mitre_tactic, confidence
            FROM security_events
            WHERE attack_type IS NOT NULL
            ORDER BY timestamp
        """)
        events = [dict(row) for row in cursor.fetchall()]
        if not events:
            return []

        campaigns = []
        current: Dict[str, Any] = {"attack_type": None, "events": []}

        for evt in events:
            if (current["attack_type"] == evt["attack_type"]
                    and current["events"]
                    and self._within_window(current["events"][-1]["timestamp"],
                                            evt["timestamp"], time_window_minutes)):
                current["events"].append(evt)
            else:
                if len(current["events"]) >= min_events:
                    campaigns.append(self._summarize_campaign(current))
                current = {"attack_type": evt["attack_type"], "events": [evt]}

        if len(current["events"]) >= min_events:
            campaigns.append(self._summarize_campaign(current))

        return campaigns

    @staticmethod
    def _within_window(ts1: str, ts2: str, minutes: int) -> bool:
        """Check if two ISO timestamps are within N minutes of each other."""
        try:
            from datetime import datetime as dt
            fmt_options = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"]
            t1 = t2 = None
            for fmt in fmt_options:
                try:
                    t1 = t1 or dt.strptime(ts1[:19], fmt[:19].replace(".%f", ""))
                except ValueError:
                    pass
                try:
                    t2 = t2 or dt.strptime(ts2[:19], fmt[:19].replace(".%f", ""))
                except ValueError:
                    pass
            if t1 and t2:
                return abs((t2 - t1).total_seconds()) <= minutes * 60
        except Exception:
            pass
        return False

    @staticmethod
    def _summarize_campaign(campaign: Dict[str, Any]) -> Dict[str, Any]:
        events = campaign["events"]
        return {
            "attack_type": campaign["attack_type"],
            "event_count": len(events),
            "first_seen": events[0]["timestamp"],
            "last_seen": events[-1]["timestamp"],
            "avg_confidence": sum(e.get("confidence") or 0 for e in events) / len(events),
            "max_severity": max((e.get("severity") or "info" for e in events),
                                key=lambda s: {"info": 0, "LOW": 1, "MED": 2, "HIGH": 3, "CRIT": 4}.get(s, 0)),
            "mitre_tactics": list(set(e.get("mitre_tactic") for e in events if e.get("mitre_tactic"))),
        }

    def session_security_report(self) -> Dict[str, Any]:
        """Generate a structured security report for the current session."""
        events = self.get_security_events(limit=500, session_only=True)
        if not events:
            return {"status": "clean", "total_events": 0, "summary": "No security events this session."}

        attack_counts: Dict[str, int] = defaultdict(int)
        severity_counts: Dict[str, int] = defaultdict(int)
        tactic_counts: Dict[str, int] = defaultdict(int)
        total_confidence = 0.0
        total_quality = 0.0
        quality_count = 0

        for evt in events:
            if evt.get("attack_type"):
                attack_counts[evt["attack_type"]] += 1
            severity_counts[evt.get("severity") or "info"] += 1
            if evt.get("mitre_tactic"):
                tactic_counts[evt["mitre_tactic"]] += 1
            total_confidence += evt.get("confidence") or 0
            if evt.get("response_quality") is not None:
                total_quality += evt["response_quality"]
                quality_count += 1

        campaigns = self.detect_campaigns()

        return {
            "status": "events_detected",
            "total_events": len(events),
            "attack_breakdown": dict(attack_counts),
            "severity_breakdown": dict(severity_counts),
            "mitre_tactics_seen": dict(tactic_counts),
            "avg_confidence": round(total_confidence / max(len(events), 1), 3),
            "avg_defense_quality": round(total_quality / max(quality_count, 1), 3) if quality_count else None,
            "campaigns_detected": len(campaigns),
            "campaigns": campaigns[:5],  # Top 5 campaigns
            "recommendations": self._generate_recommendations(attack_counts, severity_counts),
        }

    @staticmethod
    def _generate_recommendations(attack_counts: Dict[str, int],
                                   severity_counts: Dict[str, int]) -> List[str]:
        """Generate security recommendations based on observed patterns."""
        recs = []
        if attack_counts.get("prompt_injection", 0) > 3:
            recs.append("High prompt injection activity — consider additional hardening training on injection patterns.")
        if attack_counts.get("data_extraction", 0) > 2:
            recs.append("Data extraction attempts detected — review output filtering and rate limiting.")
        if attack_counts.get("jailbreak", 0) > 2:
            recs.append("Multiple jailbreak attempts — expand jailbreak scenario training data.")
        if severity_counts.get("CRIT", 0) > 0:
            recs.append("CRITICAL severity events detected — immediate review recommended.")
        if attack_counts.get("backdoor_trojan", 0) > 0:
            recs.append("Backdoor/trojan activation attempts detected — verify model integrity.")
        if not recs:
            recs.append("No high-priority recommendations. Continue monitoring.")
        return recs

    def historical_trend(self, last_n_sessions: int = 5) -> List[Dict[str, Any]]:
        """Compare security event patterns across recent sessions."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT s.session_id, s.started_at
            FROM session_stats s
            ORDER BY s.started_at DESC
            LIMIT ?
        """, (last_n_sessions,))
        sessions = [dict(row) for row in cursor.fetchall()]

        trends = []
        for sess in sessions:
            sid = sess["session_id"]
            cursor.execute("""
                SELECT COUNT(*) as cnt,
                       GROUP_CONCAT(DISTINCT se.attack_type) as types,
                       AVG(se.confidence) as avg_conf
                FROM security_events se
                JOIN conversations c ON se.conversation_id = c.id
                WHERE c.session_id = ?
            """, (sid,))
            row = cursor.fetchone()
            trends.append({
                "session_id": sid,
                "started_at": sess["started_at"],
                "event_count": row["cnt"] if row else 0,
                "attack_types": (row["types"] or "").split(",") if row and row["types"] else [],
                "avg_confidence": round(row["avg_conf"] or 0, 3) if row else 0,
            })
        return trends

    def export_session(self, output_path: str = None) -> str:
        """
        Export current session to JSON for analysis/backup.
        
        Args:
            output_path: Where to save (optional, default to logs/)
            
        Returns:
            Path to exported file
        """
        import json
        
        if not output_path:
            output_path = f"./logs/session_{self.session_id}_export.json"
        
        cursor = self.conn.cursor()
        
        # Get all session data
        cursor.execute("""
            SELECT id, operator_input, agent_response, timestamp, context_type, relevance_score
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp
        """, (self.session_id,))
        
        conversations = [dict(row) for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT id, topic, content, source, learned_at, relevance_score, usage_count
            FROM knowledge
            ORDER BY learned_at
        """)
        
        knowledge = [dict(row) for row in cursor.fetchall()]
        
        export_data = {
            "session_id": self.session_id,
            "export_time": datetime.now().isoformat(),
            "conversations": conversations,
            "knowledge": knowledge,
            "stats": self.get_session_stats()
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Session exported to {output_path}")
        return output_path
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Global memory instance
_memory = None


def init_memory(session_id: str = None) -> MemoryManager:
    """Initialize global memory manager."""
    global _memory
    _memory = MemoryManager(session_id=session_id)
    return _memory


def get_memory() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory
    if _memory is None:
        _memory = MemoryManager()
    return _memory


if __name__ == "__main__":
    # Demo
    print("Testing MemoryManager...")
    
    with MemoryManager(session_id="demo_session") as mem:
        # Store conversation
        conv_id = mem.store_conversation(
            "What is adversarial robustness?",
            "Adversarial robustness is the model's ability to resist adversarial examples...",
            context_type="security",
            relevance=0.9
        )
        print(f"Stored conversation: {conv_id}")
        
        # Store knowledge
        know_id = mem.store_knowledge(
            "adversarial_attacks",
            "FGSM, PGD, C&W are common adversarial attack methods",
            source="training",
            relevance=0.85
        )
        print(f"Stored knowledge: {know_id}")
        
        # Retrieve context
        context = mem.retrieve_context("robustness defense", limit=3)
        print(f"Context for 'robustness defense': {context}")
        
        # Get knowledge by topic
        knowledge = mem.get_knowledge_by_topic("adversarial", limit=5)
        print(f"Knowledge on adversarial topics: {knowledge}")
        
        # Get stats
        stats = mem.get_session_stats()
        print(f"Session stats: {stats}")
    
    print("Demo complete!")
