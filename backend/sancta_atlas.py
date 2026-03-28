"""
sancta_atlas.py — MITRE ATLAS Integration
──────────────────────────────────────────
Maps Sancta's security detections to the MITRE ATLAS framework
(Adversarial Threat Landscape for AI Systems).

Reference: https://atlas.mitre.org/matrices/ATLAS

Provides:
  - Full ATLAS tactic/technique taxonomy
  - Pattern-class → ATLAS technique mapping
  - Event classifier: security event → ATLAS TTPs
  - Coverage tracking: which techniques Sancta can detect
  - TTP chain analysis per adversary profile
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("atlas")

# ═══════════════════════════════════════════════════════════════════════════════
# ATLAS TACTICS — 16 columns of the matrix
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Tactic:
    id: str
    name: str
    shortname: str       # for UI badges
    description: str

TACTICS: dict[str, Tactic] = {}

def _t(tid: str, name: str, short: str, desc: str):
    TACTICS[tid] = Tactic(tid, name, short, desc)

_t("AML.TA0002", "Reconnaissance",        "RECON",     "Gathering information about the AI system")
_t("AML.TA0003", "Resource Development",   "RES-DEV",   "Building tools and capabilities for attacks")
_t("AML.TA0004", "Initial Access",         "ACCESS",    "Gaining entry to the AI system")
_t("AML.TA0000", "AI Model Access",        "MODEL",     "Obtaining access to the AI model itself")
_t("AML.TA0005", "Execution",             "EXEC",      "Running adversarial techniques against the model")
_t("AML.TA0006", "Persistence",           "PERSIST",   "Maintaining foothold in the AI system")
_t("AML.TA0012", "Privilege Escalation",   "PRIVESC",   "Gaining elevated access or capabilities")
_t("AML.TA0007", "Defense Evasion",        "EVASION",   "Avoiding detection by security measures")
_t("AML.TA0013", "Credential Access",      "CREDS",     "Stealing credentials or secrets")
_t("AML.TA0008", "Discovery",             "DISC",      "Exploring the AI system's configuration")
_t("AML.TA0015", "Lateral Movement",       "LATERAL",   "Moving between components of the system")
_t("AML.TA0009", "Collection",            "COLLECT",   "Gathering data from the AI system")
_t("AML.TA0001", "AI Attack Staging",      "STAGING",   "Preparing adversarial inputs and models")
_t("AML.TA0014", "Command and Control",    "C2",        "Establishing remote control channels")
_t("AML.TA0010", "Exfiltration",          "EXFIL",     "Extracting data or model information")
_t("AML.TA0011", "Impact",               "IMPACT",    "Disrupting or degrading AI system operation")

# Ordered tactic IDs (matrix column order)
TACTIC_ORDER = [
    "AML.TA0002", "AML.TA0003", "AML.TA0004", "AML.TA0000",
    "AML.TA0005", "AML.TA0006", "AML.TA0012", "AML.TA0007",
    "AML.TA0013", "AML.TA0008", "AML.TA0015", "AML.TA0009",
    "AML.TA0001", "AML.TA0014", "AML.TA0010", "AML.TA0011",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ATLAS TECHNIQUES — rows of the matrix
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Technique:
    id: str
    name: str
    tactics: tuple[str, ...]      # tactic IDs this maps to
    subtechniques: tuple[str, ...]  # e.g. (".000 Direct", ".001 Indirect")
    description: str

TECHNIQUES: dict[str, Technique] = {}

def _tech(tid: str, name: str, tactics: list[str],
          subs: list[str] | None = None, desc: str = ""):
    TECHNIQUES[tid] = Technique(tid, name, tuple(tactics),
                                tuple(subs or []), desc)


# ── Reconnaissance (TA0002) ──
_tech("AML.T0000", "Search Open Technical Databases", ["AML.TA0002"],
      [".000 Journals", ".001 Pre-Print Repos", ".002 Technical Blogs"],
      "Searching public technical resources for AI system information")
_tech("AML.T0001", "Search Open AI Vulnerability Analysis", ["AML.TA0002"],
      desc="Reviewing public AI vulnerability databases")
_tech("AML.T0003", "Search Victim-Owned Websites", ["AML.TA0002"],
      desc="Scanning target organization websites for AI info")
_tech("AML.T0004", "Search Application Repositories", ["AML.TA0002"],
      desc="Exploring code repos for model configs and training details")
_tech("AML.T0006", "Active Scanning", ["AML.TA0002"],
      desc="Probing AI system endpoints to discover capabilities")
_tech("AML.T0064", "Gather RAG-Indexed Targets", ["AML.TA0002"],
      desc="Identifying documents indexed by RAG systems")
_tech("AML.T0087", "Gather Victim Identity Information", ["AML.TA0002"],
      desc="Collecting identity details about AI system operators")
_tech("AML.T0095", "Search Open Websites/Domains", ["AML.TA0002"],
      desc="Searching public web for AI system information")

# ── Resource Development (TA0003) ──
_tech("AML.T0002", "Acquire Public AI Artifacts", ["AML.TA0003"],
      [".000 Datasets", ".001 Models"],
      "Obtaining publicly available AI models or datasets")
_tech("AML.T0008", "Acquire Infrastructure", ["AML.TA0003"],
      [".000 AI Dev Workspaces", ".001 Consumer HW", ".002 Domains",
       ".003 Physical Countermeasures", ".004 Serverless"],
      "Setting up infrastructure for AI attacks")
_tech("AML.T0016", "Obtain Capabilities", ["AML.TA0003"],
      [".000 Adversarial AI Attacks", ".001 Software Tools", ".002 Generative AI"],
      "Acquiring attack tools and software")
_tech("AML.T0017", "Develop Capabilities", ["AML.TA0003"],
      [".000 Adversarial AI Attacks"],
      "Building custom attack tools")
_tech("AML.T0019", "Publish Poisoned Datasets", ["AML.TA0003"],
      desc="Publishing datasets containing malicious samples")
_tech("AML.T0021", "Establish Accounts", ["AML.TA0003"],
      desc="Creating accounts on AI platforms")
_tech("AML.T0058", "Publish Poisoned Models", ["AML.TA0003"],
      desc="Publishing models with embedded backdoors")
_tech("AML.T0060", "Publish Hallucinated Entities", ["AML.TA0003"],
      desc="Creating fake entities that AI systems may hallucinate")
_tech("AML.T0065", "LLM Prompt Crafting", ["AML.TA0003"],
      desc="Designing adversarial prompts for LLM attacks")
_tech("AML.T0066", "Retrieval Content Crafting", ["AML.TA0003"],
      desc="Crafting content designed to be retrieved by RAG")
_tech("AML.T0079", "Stage Capabilities", ["AML.TA0003"],
      desc="Pre-positioning attack tools and payloads")
_tech("AML.T0104", "Publish Poisoned AI Agent Tool", ["AML.TA0003"],
      desc="Publishing malicious tools for AI agents")

# ── Initial Access (TA0004) ──
_tech("AML.T0010", "AI Supply Chain Compromise", ["AML.TA0004"],
      [".000 Hardware", ".001 AI Software", ".002 Data", ".003 Model",
       ".004 Container Registry"],
      "Compromising AI supply chain components")
_tech("AML.T0012", "Valid Accounts", ["AML.TA0004", "AML.TA0012"],
      desc="Using legitimate credentials to access AI systems")
_tech("AML.T0049", "Exploit Public-Facing Application", ["AML.TA0004"],
      desc="Exploiting vulnerabilities in AI-facing web apps")
_tech("AML.T0052", "Phishing", ["AML.TA0004", "AML.TA0015"],
      [".000 Spearphishing via Social Engineering LLM"],
      "Social engineering attacks via AI-generated content")
_tech("AML.T0078", "Drive-by Compromise", ["AML.TA0004"],
      desc="Compromising AI systems through malicious web content")
_tech("AML.T0093", "Prompt Infiltration via Public-Facing App",
      ["AML.TA0004", "AML.TA0006"],
      desc="Injecting prompts through public-facing applications")

# ── AI Model Access (TA0000) ──
_tech("AML.T0040", "AI Model Inference API Access", ["AML.TA0000"],
      desc="Accessing model through its inference API")
_tech("AML.T0041", "Physical Environment Access", ["AML.TA0000"],
      desc="Physical access to AI system hardware")
_tech("AML.T0044", "Full AI Model Access", ["AML.TA0000"],
      desc="Complete access to model weights and architecture")
_tech("AML.T0047", "AI-Enabled Product or Service", ["AML.TA0000"],
      desc="Accessing AI through a product or service interface")

# ── Execution (TA0005) ──
_tech("AML.T0011", "User Execution", ["AML.TA0005"],
      [".000 Unsafe AI Artifacts", ".001 Malicious Package",
       ".002 Poisoned AI Agent Tool", ".003 Malicious Link"],
      "Tricking users into running malicious AI content")
_tech("AML.T0050", "Command and Scripting Interpreter", ["AML.TA0005"],
      desc="Using command interpreters to execute attacks")
_tech("AML.T0051", "LLM Prompt Injection", ["AML.TA0005"],
      [".000 Direct", ".001 Indirect", ".002 Triggered"],
      "Injecting adversarial prompts into LLM input")
_tech("AML.T0053", "AI Agent Tool Invocation", ["AML.TA0005", "AML.TA0012"],
      desc="Manipulating AI agent to invoke tools maliciously")
_tech("AML.T0100", "AI Agent Clickbait", ["AML.TA0005"],
      desc="Luring AI agents to interact with malicious content")
_tech("AML.T0103", "Deploy AI Agent", ["AML.TA0005"],
      desc="Deploying a malicious AI agent")

# ── Persistence (TA0006) ──
_tech("AML.T0018", "Manipulate AI Model",
      ["AML.TA0006", "AML.TA0001"],
      [".000 Poison AI Model", ".001 Modify Architecture", ".002 Embed Malware"],
      "Modifying the AI model to maintain persistence")
_tech("AML.T0020", "Poison Training Data", ["AML.TA0003", "AML.TA0006"],
      desc="Injecting malicious data into training pipelines")
_tech("AML.T0061", "LLM Prompt Self-Replication", ["AML.TA0006"],
      desc="Prompts that cause LLM to propagate the attack")
_tech("AML.T0070", "RAG Poisoning", ["AML.TA0006"],
      desc="Poisoning RAG knowledge base documents")
_tech("AML.T0080", "AI Agent Context Poisoning",
      ["AML.TA0006"],
      [".000 Memory", ".001 Thread"],
      "Poisoning AI agent context or memory")
_tech("AML.T0081", "Modify AI Agent Configuration",
      ["AML.TA0006", "AML.TA0007"],
      desc="Altering AI agent settings for persistence or evasion")
_tech("AML.T0099", "AI Agent Tool Data Poisoning", ["AML.TA0006"],
      desc="Poisoning data returned by AI agent tools")

# ── Privilege Escalation (TA0012) ──
_tech("AML.T0054", "LLM Jailbreak", ["AML.TA0012", "AML.TA0007"],
      desc="Bypassing LLM safety constraints to escalate privileges")
_tech("AML.T0105", "Escape to Host", ["AML.TA0012"],
      desc="Escaping AI sandbox to access host system")

# ── Defense Evasion (TA0007) ──
_tech("AML.T0015", "Evade AI Model",
      ["AML.TA0004", "AML.TA0007", "AML.TA0011"],
      desc="Crafting inputs that evade AI model detection")
_tech("AML.T0067", "LLM Trusted Output Components Manipulation",
      ["AML.TA0007"],
      [".000 Citations"],
      "Manipulating trusted output elements like citations")
_tech("AML.T0068", "LLM Prompt Obfuscation", ["AML.TA0007"],
      desc="Obfuscating prompts to evade detection")
_tech("AML.T0071", "False RAG Entry Injection", ["AML.TA0007"],
      desc="Injecting false entries into RAG databases")
_tech("AML.T0073", "Impersonation", ["AML.TA0007"],
      desc="Impersonating trusted entities to evade defenses")
_tech("AML.T0074", "Masquerading", ["AML.TA0007"],
      desc="Disguising malicious content as legitimate")
_tech("AML.T0076", "Corrupt AI Model", ["AML.TA0007"],
      desc="Corrupting model to degrade defense capabilities")
_tech("AML.T0092", "Manipulate User LLM Chat History", ["AML.TA0007"],
      desc="Altering chat history to influence model behavior")
_tech("AML.T0094", "Delay Execution of LLM Instructions", ["AML.TA0007"],
      desc="Time-delayed attack payloads in LLM context")
_tech("AML.T0097", "Virtualization/Sandbox Evasion", ["AML.TA0007"],
      desc="Detecting and evading sandbox environments")
_tech("AML.T0107", "Exploitation for Defense Evasion", ["AML.TA0007"],
      desc="Exploiting vulnerabilities to bypass defenses")

# ── Credential Access (TA0013) ──
_tech("AML.T0055", "Unsecured Credentials", ["AML.TA0013"],
      desc="Accessing plaintext or poorly protected credentials")
_tech("AML.T0082", "RAG Credential Harvesting", ["AML.TA0013"],
      desc="Extracting credentials from RAG knowledge bases")
_tech("AML.T0083", "Credentials from AI Agent Configuration", ["AML.TA0013"],
      desc="Extracting credentials from AI agent configs")
_tech("AML.T0090", "OS Credential Dumping", ["AML.TA0013"],
      desc="Dumping credentials from the operating system")
_tech("AML.T0098", "AI Agent Tool Credential Harvesting", ["AML.TA0013"],
      desc="Harvesting credentials through AI agent tools")
_tech("AML.T0106", "Exploitation for Credential Access", ["AML.TA0013"],
      desc="Exploiting vulnerabilities to access credentials")

# ── Discovery (TA0008) ──
_tech("AML.T0007", "Discover AI Artifacts", ["AML.TA0008"],
      desc="Finding AI models, datasets, and configurations")
_tech("AML.T0013", "Discover AI Model Ontology", ["AML.TA0008"],
      desc="Mapping the model's classification taxonomy")
_tech("AML.T0014", "Discover AI Model Family", ["AML.TA0008"],
      desc="Identifying the model architecture and family")
_tech("AML.T0062", "Discover LLM Hallucinations", ["AML.TA0008"],
      desc="Probing for hallucination patterns in LLMs")
_tech("AML.T0063", "Discover AI Model Outputs", ["AML.TA0008"],
      desc="Mapping model output space and confidence patterns")
_tech("AML.T0069", "Discover LLM System Information", ["AML.TA0008"],
      [".000 Special Character Sets", ".001 System Instruction Keywords",
       ".002 System Prompt"],
      "Probing to discover LLM system prompt and configuration")
_tech("AML.T0075", "Cloud Service Discovery", ["AML.TA0008"],
      desc="Discovering cloud services used by AI system")
_tech("AML.T0084", "Discover AI Agent Configuration", ["AML.TA0008"],
      [".000 Embedded Knowledge", ".001 Tool Definitions",
       ".002 Activation Triggers"],
      "Probing AI agent to discover its configuration")
_tech("AML.T0089", "Process Discovery", ["AML.TA0008"],
      desc="Discovering processes running on AI system")

# ── Lateral Movement (TA0015) ──
_tech("AML.T0091", "Use Alternate Authentication Material", ["AML.TA0015"],
      [".000 Application Access Token"],
      "Using tokens or alternate auth to move laterally")

# ── Collection (TA0009) ──
_tech("AML.T0035", "AI Artifact Collection", ["AML.TA0009"],
      desc="Collecting AI artifacts like models and configs")
_tech("AML.T0036", "Data from Information Repositories", ["AML.TA0009"],
      desc="Extracting data from knowledge repos")
_tech("AML.T0037", "Data from Local System", ["AML.TA0009"],
      desc="Collecting data from the local file system")
_tech("AML.T0085", "Data from AI Services", ["AML.TA0009"],
      [".000 RAG Databases", ".001 AI Agent Tools"],
      "Extracting data from AI service components")

# ── AI Attack Staging (TA0001) ──
_tech("AML.T0005", "Create Proxy AI Model", ["AML.TA0001"],
      [".000 Train via Gathered Artifacts", ".001 Train via Replication",
       ".002 Use Pre-Trained Model"],
      "Creating a proxy model to develop transferable attacks")
_tech("AML.T0042", "Verify Attack", ["AML.TA0001"],
      desc="Testing adversarial inputs before deployment")
_tech("AML.T0043", "Craft Adversarial Data", ["AML.TA0001"],
      [".000 White-Box Optimization", ".001 Black-Box Optimization",
       ".002 Black-Box Transfer", ".003 Manual Modification",
       ".004 Insert Backdoor Trigger"],
      "Creating adversarial inputs to fool AI models")
_tech("AML.T0088", "Generate Deepfakes", ["AML.TA0001"],
      desc="Generating synthetic media for attacks")
_tech("AML.T0102", "Generate Malicious Commands", ["AML.TA0001"],
      desc="Using AI to generate malicious system commands")

# ── Command and Control (TA0014) ──
_tech("AML.T0072", "Reverse Shell", ["AML.TA0014"],
      desc="Establishing reverse shell through AI system")
_tech("AML.T0096", "AI Service API", ["AML.TA0014"],
      desc="Using AI service APIs as C2 channels")
_tech("AML.T0108", "AI Agent", ["AML.TA0014"],
      desc="Using AI agent as C2 relay")

# ── Exfiltration (TA0010) ──
_tech("AML.T0024", "Exfiltration via AI Inference API", ["AML.TA0010"],
      [".000 Infer Training Data Membership", ".001 Invert AI Model",
       ".002 Extract AI Model"],
      "Extracting data through model inference")
_tech("AML.T0025", "Exfiltration via Cyber Means", ["AML.TA0010"],
      desc="Traditional data exfiltration methods")
_tech("AML.T0056", "Extract LLM System Prompt", ["AML.TA0010"],
      desc="Extracting the LLM system prompt")
_tech("AML.T0057", "LLM Data Leakage", ["AML.TA0010"],
      desc="Causing LLM to leak training or context data")
_tech("AML.T0077", "LLM Response Rendering", ["AML.TA0010"],
      desc="Exploiting response rendering to exfiltrate data")
_tech("AML.T0086", "Exfiltration via AI Agent Tool Invocation",
      ["AML.TA0010"],
      desc="Using AI agent tools to exfiltrate data")

# ── Impact (TA0011) ──
_tech("AML.T0029", "Denial of AI Service", ["AML.TA0011"],
      desc="Disrupting AI service availability")
_tech("AML.T0031", "Erode AI Model Integrity", ["AML.TA0011"],
      desc="Gradually degrading model accuracy and reliability")
_tech("AML.T0034", "Cost Harvesting", ["AML.TA0011"],
      desc="Causing excessive compute costs")
_tech("AML.T0046", "Spamming AI System with Chaff Data", ["AML.TA0011"],
      desc="Flooding AI system with noise data")
_tech("AML.T0048", "External Harms", ["AML.TA0011"],
      [".000 Financial Harm", ".001 Reputational Harm",
       ".002 Societal Harm", ".003 User Harm",
       ".004 AI IP Theft"],
      "Causing real-world harms through AI system compromise")
_tech("AML.T0059", "Erode Dataset Integrity", ["AML.TA0011"],
      desc="Corrupting datasets used by AI system")
_tech("AML.T0101", "Data Destruction via AI Agent Tool Invocation",
      ["AML.TA0011"],
      desc="Using AI agent tools to destroy data")


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN CLASS → ATLAS TECHNIQUE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
# Maps Sancta's 22 IDPI attack classes to the ATLAS techniques they represent.
# A single pattern class can map to multiple techniques (multi-tactic attacks).

PATTERN_CLASS_MAP: dict[str, list[str]] = {
    # instruction override, role hijack, forget commands
    "instruction":          ["AML.T0051", "AML.T0051.000"],  # LLM Prompt Injection (Direct)
    # API key/token extraction
    "credential":           ["AML.T0055", "AML.T0083"],       # Unsecured Credentials + Agent Config Creds
    # OS info, paths, env vars, subprocess
    "system_info":          ["AML.T0069", "AML.T0089"],       # Discover LLM System Info + Process Discovery
    # URL redirection to malicious endpoints
    "redirect":             ["AML.T0078", "AML.T0077"],       # Drive-by Compromise + Response Rendering
    # Jailbreak, DAN mode, developer mode
    "role_hijack":          ["AML.T0054", "AML.T0051.000"],   # LLM Jailbreak + Direct Prompt Injection
    # Dump env, configs, credentials.json
    "data_extraction":      ["AML.T0057", "AML.T0056"],       # LLM Data Leakage + Extract System Prompt
    # CSS hiding, off-screen, zero-sizing
    "visual_concealment":   ["AML.T0068", "AML.T0074"],       # Prompt Obfuscation + Masquerading
    # Script/iframe injection, event handlers
    "html_obfuscation":     ["AML.T0068", "AML.T0049"],       # Prompt Obfuscation + Exploit Public App
    # URL parameter injection, encoded redirects
    "url_manipulation":     ["AML.T0078", "AML.T0068"],       # Drive-by + Obfuscation
    # Zero-width Unicode, bidi overrides
    "invisible_chars":      ["AML.T0068"],                     # Prompt Obfuscation
    # Cyrillic/Greek lookalikes
    "homoglyph":            ["AML.T0074", "AML.T0068"],       # Masquerading + Obfuscation
    # Multi-part payload reconstruction
    "payload_splitting":    ["AML.T0068", "AML.T0094"],       # Obfuscation + Delayed Execution
    # HTML entities, Base64, URL, nested escapes
    "encoding":             ["AML.T0068"],                     # Prompt Obfuscation
    # Non-English instruction obfuscation
    "multilingual":         ["AML.T0068", "AML.T0015"],       # Obfuscation + Evade AI Model
    # JSON/markdown role-level injection
    "syntax_injection":     ["AML.T0051.000", "AML.T0068"],   # Direct Injection + Obfuscation
    # Authority claims, urgency, testing/debug modes
    "social_engineering":   ["AML.T0073", "AML.T0052"],       # Impersonation + Phishing
    # rm -rf, drop table, fork bombs
    "destructive_commands": ["AML.T0050", "AML.T0101"],       # Command Interpreter + Data Destruction
    # Authority tokens, admin sessions
    "god_mode":             ["AML.T0054", "AML.T0012"],       # Jailbreak + Valid Accounts
    # Forced purchases, unauthorized transactions
    "payment_injection":    ["AML.T0048.000", "AML.T0053"],   # Financial Harm + Agent Tool Invocation
    # AI detection, scraper/crawler blocks
    "anti_scraping":        ["AML.T0097"],                     # Sandbox Evasion
    # Recommendation manipulation
    "seo_poisoning":        ["AML.T0031", "AML.T0046"],       # Erode Integrity + Spam Chaff
    # Forced positive reviews
    "review_manipulation":  ["AML.T0031", "AML.T0048.001"],   # Erode Integrity + Reputational Harm
}

# Additional event-type → ATLAS mapping for non-pattern detections
EVENT_TYPE_MAP: dict[str, list[str]] = {
    "input_reject":                     ["AML.T0051"],       # Prompt Injection (general)
    "injection_defense":                ["AML.T0051"],       # Prompt Injection blocked
    "tavern_defense":                   ["AML.T0051"],       # Legacy event name
    "suspicious_block":                 ["AML.T0015"],       # Evade AI Model (behavioral shift)
    "output_redact":                    ["AML.T0057"],       # Data Leakage (output side)
    "ioc_domain_detected":              ["AML.T0078"],       # Drive-by Compromise (known bad domain)
    "llm_deep_scan":                    ["AML.T0051"],       # Prompt Injection (deep analysis)
    "unicode_clean":                    ["AML.T0068"],       # Prompt Obfuscation
    "ingest_reject_direct_poisoning":   ["AML.T0020"],       # Poison Training Data
    "ingest_reject_indirect_poisoning": ["AML.T0070"],       # RAG Poisoning
    "ingest_reject_anomalous":          ["AML.T0066"],       # Retrieval Content Crafting
    "risk_assessment":                  [],                   # Not an attack — just telemetry
}

# Risk dimension → primary ATLAS tactic mapping
RISK_DIMENSION_TACTIC: dict[str, str] = {
    "injection":              "AML.TA0005",  # Execution
    "authority_manipulation": "AML.TA0012",  # Privilege Escalation
    "emotional_coercion":     "AML.TA0004",  # Initial Access (social engineering)
    "obfuscation":            "AML.TA0007",  # Defense Evasion
    "long_term_influence":    "AML.TA0011",  # Impact (erode integrity)
}

# Behavioral drift signal → ATLAS technique mapping
DRIFT_SIGNAL_MAP: dict[str, list[str]] = {
    "belief_decay_rate":          ["AML.T0031"],       # Erode AI Model Integrity
    "soul_alignment":             ["AML.T0080.000"],   # Agent Context Poisoning (Memory)
    "topic_drift":                ["AML.T0031"],       # Erode Integrity
    "strategy_entropy":           ["AML.T0081"],       # Modify Agent Configuration
    "dissonance_trend":           ["AML.T0080"],       # Agent Context Poisoning
    "engagement_pattern_delta":   ["AML.T0046"],       # Spam with Chaff Data
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER — security event → ATLAS TTPs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ATLASClassification:
    """Result of classifying a security event against ATLAS."""
    technique_ids: list[str]          # matched technique IDs
    tactic_ids: list[str]             # all parent tactic IDs
    primary_technique: Optional[str]  # highest-confidence technique
    primary_tactic: Optional[str]     # highest-confidence tactic
    confidence: float                 # 0-1 classification confidence
    source: str                       # "pattern_class" | "event_type" | "drift_signal" | "risk_dim"

    def to_dict(self) -> dict:
        return {
            "technique_ids": self.technique_ids,
            "tactic_ids": self.tactic_ids,
            "primary_technique": self.primary_technique,
            "primary_tactic": self.primary_tactic,
            "technique_names": [TECHNIQUES[t].name for t in self.technique_ids if t.split(".")[0] in TECHNIQUES or t in TECHNIQUES],
            "tactic_names": [TACTICS[t].name for t in self.tactic_ids if t in TACTICS],
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


def _resolve_technique_names(tech_ids: list[str]) -> list[str]:
    """Resolve technique IDs to names, handling sub-techniques."""
    names = []
    for tid in tech_ids:
        if tid in TECHNIQUES:
            names.append(TECHNIQUES[tid].name)
        else:
            # Sub-technique: AML.T0051.000 → base is AML.T0051
            base = tid.rsplit(".", 1)[0] if "." in tid else tid
            if base in TECHNIQUES:
                # Find the sub-technique name
                sub_suffix = tid[len(base):]
                for s in TECHNIQUES[base].subtechniques:
                    if s.startswith(sub_suffix):
                        names.append(f"{TECHNIQUES[base].name}: {s.lstrip('. ')}")
                        break
                else:
                    names.append(TECHNIQUES[base].name)
    return names


def _collect_tactics(tech_ids: list[str]) -> list[str]:
    """Given technique IDs, return all associated tactic IDs (deduplicated)."""
    seen = set()
    result = []
    for tid in tech_ids:
        base = tid.split(".")[0] + "." + tid.split(".")[1] if tid.count(".") >= 1 else tid
        # Handle sub-technique → base
        if base not in TECHNIQUES and "." in tid:
            base = ".".join(tid.split(".")[:2])
        tech = TECHNIQUES.get(base)
        if tech:
            for tac_id in tech.tactics:
                if tac_id not in seen:
                    seen.add(tac_id)
                    result.append(tac_id)
    return result


def classify_event(event: dict) -> Optional[ATLASClassification]:
    """
    Classify a security event against the ATLAS framework.

    Accepts a security.jsonl event dict. Returns ATLASClassification
    or None if the event doesn't map to any ATLAS technique.
    """
    technique_ids: list[str] = []
    confidence = 0.0
    source = "event_type"

    event_type = event.get("event", "")
    data = event.get("data", event)

    # 1. Check matched_classes from the pattern engine (highest confidence)
    matched_classes = data.get("matched_classes") or data.get("attack_classes") or []
    if isinstance(matched_classes, str):
        matched_classes = [matched_classes]

    if matched_classes:
        source = "pattern_class"
        for cls in matched_classes:
            cls_lower = cls.lower().strip()
            atlas_ids = PATTERN_CLASS_MAP.get(cls_lower, [])
            technique_ids.extend(atlas_ids)
        if technique_ids:
            confidence = min(0.95, 0.7 + 0.05 * len(matched_classes))

    # 2. Check event type mapping
    if not technique_ids and event_type:
        atlas_ids = EVENT_TYPE_MAP.get(event_type, [])
        technique_ids.extend(atlas_ids)
        if atlas_ids:
            confidence = 0.6
            source = "event_type"

    # 3. Check risk vector dimensions
    risk_vector = data.get("risk_vector") or data.get("risk") or {}
    if isinstance(risk_vector, dict):
        max_dim = None
        max_val = 0.0
        for dim, tactic in RISK_DIMENSION_TACTIC.items():
            val = risk_vector.get(dim, 0)
            if isinstance(val, (int, float)) and val > 0.3:
                if val > max_val:
                    max_dim = dim
                    max_val = val
        if max_dim and not technique_ids:
            # Map from risk dimension to techniques
            dim_techniques = {
                "injection": ["AML.T0051"],
                "authority_manipulation": ["AML.T0054", "AML.T0073"],
                "emotional_coercion": ["AML.T0052"],
                "obfuscation": ["AML.T0068"],
                "long_term_influence": ["AML.T0031", "AML.T0080"],
            }
            technique_ids = dim_techniques.get(max_dim, [])
            confidence = min(0.8, max_val)
            source = "risk_dim"

    # 4. Check drift signals
    triggered_signals = data.get("triggered_signals") or []
    if triggered_signals and not technique_ids:
        source = "drift_signal"
        for sig in triggered_signals:
            atlas_ids = DRIFT_SIGNAL_MAP.get(sig, [])
            technique_ids.extend(atlas_ids)
        if technique_ids:
            confidence = 0.5

    # Deduplicate
    seen = set()
    unique_ids = []
    for tid in technique_ids:
        if tid not in seen:
            seen.add(tid)
            unique_ids.append(tid)
    technique_ids = unique_ids

    if not technique_ids:
        return None

    tactic_ids = _collect_tactics(technique_ids)
    primary_tech = technique_ids[0] if technique_ids else None
    primary_tactic = tactic_ids[0] if tactic_ids else None

    return ATLASClassification(
        technique_ids=technique_ids,
        tactic_ids=tactic_ids,
        primary_technique=primary_tech,
        primary_tactic=primary_tactic,
        confidence=confidence,
        source=source,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE REPORT — what Sancta can detect
# ═══════════════════════════════════════════════════════════════════════════════

def get_coverage() -> dict:
    """
    Return which ATLAS techniques Sancta has detection coverage for,
    organized by tactic.
    """
    # Collect all technique IDs we can detect
    covered_ids: set[str] = set()
    for atlas_ids in PATTERN_CLASS_MAP.values():
        covered_ids.update(atlas_ids)
    for atlas_ids in EVENT_TYPE_MAP.values():
        covered_ids.update(atlas_ids)
    for atlas_ids in DRIFT_SIGNAL_MAP.values():
        covered_ids.update(atlas_ids)

    # Resolve sub-techniques to their base for counting
    covered_bases: set[str] = set()
    for tid in covered_ids:
        base = ".".join(tid.split(".")[:2])
        covered_bases.add(base)

    total_techniques = len(TECHNIQUES)
    covered_count = len(covered_bases & set(TECHNIQUES.keys()))

    # Per-tactic coverage
    tactic_coverage: dict[str, dict] = {}
    for tac_id in TACTIC_ORDER:
        tactic = TACTICS[tac_id]
        techs_in_tactic = [t for t in TECHNIQUES.values() if tac_id in t.tactics]
        covered_in_tactic = [t for t in techs_in_tactic if t.id in covered_bases]
        tactic_coverage[tac_id] = {
            "tactic_id": tac_id,
            "tactic_name": tactic.name,
            "tactic_short": tactic.shortname,
            "total": len(techs_in_tactic),
            "covered": len(covered_in_tactic),
            "coverage_pct": round(len(covered_in_tactic) / max(1, len(techs_in_tactic)) * 100, 1),
            "techniques": [
                {
                    "id": t.id,
                    "name": t.name,
                    "covered": t.id in covered_bases,
                }
                for t in techs_in_tactic
            ],
        }

    return {
        "total_techniques": total_techniques,
        "covered_techniques": covered_count,
        "coverage_pct": round(covered_count / max(1, total_techniques) * 100, 1),
        "by_tactic": tactic_coverage,
    }


def get_matrix_data() -> dict:
    """
    Return the full ATLAS matrix structure for frontend rendering.
    """
    matrix = {
        "tactics": [],
        "techniques_by_tactic": {},
    }

    for tac_id in TACTIC_ORDER:
        tactic = TACTICS[tac_id]
        matrix["tactics"].append({
            "id": tac_id,
            "name": tactic.name,
            "shortname": tactic.shortname,
            "description": tactic.description,
        })

        techs = [t for t in TECHNIQUES.values() if tac_id in t.tactics]
        matrix["techniques_by_tactic"][tac_id] = [
            {
                "id": t.id,
                "name": t.name,
                "subtechniques": list(t.subtechniques),
            }
            for t in sorted(techs, key=lambda x: x.id)
        ]

    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# TTP CHAIN TRACKER — per-adversary technique sequences
# ═══════════════════════════════════════════════════════════════════════════════

class TTPTracker:
    """
    Tracks ATLAS technique usage per adversary over time.
    Maintains a rolling window of recent technique observations.
    """

    def __init__(self, max_history: int = 200):
        self._max_history = max_history
        # agent_id → list of (timestamp, technique_id, tactic_id, event_type)
        self._history: dict[str, list[tuple[float, str, str, str]]] = {}
        # Global technique hit counts
        self._technique_counts: dict[str, int] = {}
        self._tactic_counts: dict[str, int] = {}

    def record(self, agent_id: str, classification: ATLASClassification,
               event_type: str = ""):
        """Record an ATLAS classification for an adversary."""
        ts = time.time()
        if agent_id not in self._history:
            self._history[agent_id] = []

        for tech_id in classification.technique_ids:
            tac_id = classification.primary_tactic or ""
            self._history[agent_id].append((ts, tech_id, tac_id, event_type))
            self._technique_counts[tech_id] = self._technique_counts.get(tech_id, 0) + 1

        for tac_id in classification.tactic_ids:
            self._tactic_counts[tac_id] = self._tactic_counts.get(tac_id, 0) + 1

        # Trim history
        if len(self._history[agent_id]) > self._max_history:
            self._history[agent_id] = self._history[agent_id][-self._max_history:]

    def get_agent_ttps(self, agent_id: str) -> dict:
        """Get ATLAS TTP profile for an agent."""
        entries = self._history.get(agent_id, [])
        if not entries:
            return {"agent_id": agent_id, "techniques": [], "tactics": [],
                    "history": [], "kill_chain_phase": None}

        tech_counts: dict[str, int] = {}
        tac_counts: dict[str, int] = {}
        for ts, tech_id, tac_id, evt in entries:
            tech_counts[tech_id] = tech_counts.get(tech_id, 0) + 1
            if tac_id:
                tac_counts[tac_id] = tac_counts.get(tac_id, 0) + 1

        # Determine kill chain phase (furthest tactic reached)
        kill_chain_phase = None
        for tac_id in reversed(TACTIC_ORDER):
            if tac_id in tac_counts:
                kill_chain_phase = {
                    "tactic_id": tac_id,
                    "tactic_name": TACTICS[tac_id].name,
                    "tactic_short": TACTICS[tac_id].shortname,
                }
                break

        techniques = sorted(
            [{"id": k, "name": TECHNIQUES.get(k, Technique(k, k, (), (), "")).name,
              "count": v}
             for k, v in tech_counts.items()],
            key=lambda x: -x["count"]
        )

        tactics = sorted(
            [{"id": k, "name": TACTICS.get(k, Tactic(k, k, k, "")).name,
              "count": v}
             for k, v in tac_counts.items()],
            key=lambda x: -x["count"]
        )

        return {
            "agent_id": agent_id,
            "techniques": techniques,
            "tactics": tactics,
            "total_events": len(entries),
            "kill_chain_phase": kill_chain_phase,
            "history": [
                {"ts": ts, "technique": tid, "tactic": tac, "event": evt}
                for ts, tid, tac, evt in entries[-50:]
            ],
        }

    def get_global_stats(self) -> dict:
        """Get global ATLAS technique/tactic frequency stats."""
        top_techniques = sorted(
            [{"id": k,
              "name": TECHNIQUES.get(k, Technique(k, k, (), (), "")).name,
              "count": v}
             for k, v in self._technique_counts.items()],
            key=lambda x: -x["count"]
        )[:20]

        top_tactics = sorted(
            [{"id": k,
              "name": TACTICS.get(k, Tactic(k, k, k, "")).name,
              "count": v}
             for k, v in self._tactic_counts.items()],
            key=lambda x: -x["count"]
        )

        return {
            "total_classifications": sum(self._technique_counts.values()),
            "unique_techniques_seen": len(self._technique_counts),
            "unique_tactics_seen": len(self._tactic_counts),
            "top_techniques": top_techniques,
            "top_tactics": top_tactics,
            "tracked_agents": len(self._history),
        }


# Module-level singleton
ttp_tracker = TTPTracker()
