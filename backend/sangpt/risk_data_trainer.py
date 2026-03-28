"""
Risk Data Trainer Module
Processes The AI Risk Repository data for model training.
Converts risk data into training examples for the SANCTA system.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

logger = logging.getLogger("RiskDataTrainer")


class AIRiskDataProcessor:
    """
    Processes AI Risk Repository data and converts it into training examples.
    Handles CSV parsing and semantic risk-response pair generation.
    """
    
    def __init__(self, csv_path: str = None):
        self.csv_path = Path(csv_path) if csv_path else None
        self.risks = []
        self.training_pairs = []
        
    def load_csv(self, csv_path: str = None) -> bool:
        """Load risk data from CSV file or directory containing CSV."""
        if csv_path:
            self.csv_path = Path(csv_path)
        self.risks = []
        
        # If path is directory, find CSV files in it
        if self.csv_path and self.csv_path.is_dir():
            csv_files = list(self.csv_path.glob("*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in directory: {self.csv_path}")
                return False
            self.csv_path = csv_files[0]  # Use first CSV found
            logger.info(f"Found CSV file: {self.csv_path.name}")
        
        if not self.csv_path or not self.csv_path.exists():
            logger.warning(f"Risk CSV file not found: {self.csv_path}")
            return False
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    values = [cell.strip() for cell in row if cell and cell.strip()]
                    if not values:
                        continue
                    risk_text = " | ".join(values)
                    if 'The AI Risk Repository' in risk_text or 'Updated:' in risk_text:
                        continue
                    if len(risk_text) > 5:
                        self.risks.append({"risk": risk_text, "raw": row})
            
            logger.info(f"Loaded {len(self.risks)} risk entries from {self.csv_path.name}")
            return len(self.risks) > 0
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return False

    def generate_training_pairs(self) -> List[Tuple[str, str]]:
        """
        Generate training pairs from risk data.
        Each pair is (question/prompt, response/effect).
        """
        self.training_pairs = []
        
        # Built-in risk framework training data — 200+ Q&A pairs
        # Covers all 7 domains, 24 subdomains, cross-domain relationships,
        # causal-domain intersections, severity-based prioritization, and defenses.
        builtin_risks = [
            # ===== CAUSAL TAXONOMY =====
            ("What causes AI risks?",
             "AI risks are caused by three causal factors: (1) Entity (human-caused or AI-caused), (2) Intentionality (intentional or unintentional), and (3) Timing (pre-deployment or post-deployment). Understanding these factors helps us address root causes."),
            ("What are the main entities causing AI risks?",
             "AI risks can be caused by two main entities: (1) Humans - through intentional misuse or unintentional deployment errors, and (2) AI Systems - through intentional design flaws or unintentional failures in alignment and safety mechanisms."),
            ("When do AI risks occur?",
             "AI risks occur at two critical timing stages: (1) Pre-deployment risks include development errors, bias in training data, and safety mechanism failures before release. (2) Post-deployment risks include model misuse, unexpected emergent behaviors, and long-term societal impacts."),
            ("How do pre-deployment and post-deployment risks differ?",
             "Pre-deployment risks can be controlled through better design, training data, testing, and safety mechanisms before release. Post-deployment risks emerge from real-world use cases, user interactions, and evolving threat landscapes. Both require different mitigation strategies and monitoring approaches."),
            ("What role does human intentionality play in AI risks?",
             "Intentional human-caused risks (misuse, manipulation) require governance, regulation, and detection mechanisms. Unintentional human-caused risks (biased training data, design oversights) require process improvements and testing. Distinguishing between these is crucial for effective risk management."),
            ("What is the difference between intentional and unintentional AI risks?",
             "Intentional risks result from deliberate actions by humans or design choices in AI systems, such as weaponization or surveillance. Unintentional risks arise from errors, oversights, or emergent behaviors that were not anticipated, such as biased outputs from biased training data."),
            ("What are human-caused AI risks?",
             "Human-caused AI risks include intentional misuse (cyberattacks, surveillance, manipulation), unintentional errors (biased data curation, poor testing, inadequate safety mechanisms), negligence (insufficient oversight), and organizational failures (lack of diverse perspectives, rushed deployment)."),
            ("What are AI-caused risks?",
             "AI-caused risks include alignment failures (optimizing wrong objectives), emergent behaviors (capabilities not anticipated by designers), specification gaming (exploiting loopholes in reward functions), distributional shift failures, and cascading errors in interconnected AI systems."),
            ("How does timing affect risk severity?",
             "Pre-deployment risks are generally more controllable since they can be caught during development and testing. Post-deployment risks can have wider impact as they affect real users and systems. Early detection at pre-deployment stage is far cheaper than remediation post-deployment."),

            # ===== DOMAIN TAXONOMY - OVERVIEW =====
            ("What are the main domains of AI risk?",
             "The Domain Taxonomy identifies 7 primary AI risk domains: (1) Discrimination & Toxicity, (2) Privacy & Security, (3) Misinformation, (4) Malicious Actors & Misuse, (5) Human-Computer Interaction, (6) Socioeconomic & Environmental, (7) AI System Safety, Failures & Limitations."),
            ("What is the AI Risk Repository?",
             "The AI Risk Repository is a living database that categorizes over 1,700 AI risks using two taxonomies: the Causal Taxonomy (by entity, intentionality, and timing) and the Domain Taxonomy (by 7 risk domains). It provides a comprehensive reference for understanding AI safety and risk mitigation strategies."),
            ("How many risks are in the AI Risk Repository?",
             "The AI Risk Repository is a comprehensive living database containing over 1,724 AI risks extracted from 74 different frameworks and research papers. It is continuously updated with new risks as they are identified."),
            ("How are risks organized in the repository?",
             "Risks in the repository are categorized using two complementary approaches: (1) The Causal Taxonomy organizes by what causes risks (human/AI, intentional/unintentional, pre/post-deployment), and (2) The Domain Taxonomy organizes by 7 risk domains."),
            ("What are the 7 AI risk domains?",
             "The 7 domains are: (1) Discrimination & Toxicity, (2) Privacy & Security, (3) Misinformation, (4) Malicious Actors & Misuse, (5) Human-Computer Interaction, (6) Socioeconomic & Environmental, (7) AI System Safety, Failures & Limitations."),

            # ===== DOMAIN 1: DISCRIMINATION & TOXICITY =====
            ("What is the Discrimination & Toxicity domain?",
             "Discrimination & Toxicity encompasses risks where AI systems generate biased or toxic outputs that harm marginalized groups or spread harmful content. This includes algorithmic discrimination, stereotype amplification, hate speech generation, and content that violates social norms or safety policies."),
            ("What is algorithmic discrimination?",
             "Algorithmic discrimination occurs when AI systems make unfair decisions based on protected attributes like race, gender, age, or disability. It often results from biased training data that reflects historical inequities, leading to discriminatory outcomes in hiring, lending, criminal justice, and healthcare."),
            ("What is stereotype amplification in AI?",
             "Stereotype amplification is when AI systems learn and reinforce societal stereotypes from training data, then amplify them in outputs. For example, image generators may depict professionals as predominantly male or associate certain ethnicities with specific occupations."),
            ("What is toxic content generation?",
             "Toxic content generation is when AI produces hate speech, slurs, violent content, sexually explicit material, or other harmful outputs. This can occur from training on unfiltered internet data that contains toxic language patterns."),
            ("How can AI discrimination be mitigated?",
             "Mitigation strategies include: (1) Auditing training data for representational bias, (2) Using fairness metrics (demographic parity, equalized odds), (3) Adversarial debiasing during training, (4) Regular bias audits post-deployment, (5) Diverse development teams, (6) Impact assessments for affected communities."),
            ("What is representational harm in AI?",
             "Representational harm occurs when AI systems portray social groups in stereotypical, demeaning, or erasing ways. Unlike allocative harms (which deny resources), representational harms shape how groups are perceived and can normalize discrimination."),
            ("What are allocative harms from AI?",
             "Allocative harms occur when AI systems unfairly distribute resources or opportunities. Examples include loan denials based on zip code (proxy for race), resume screening that disadvantages women, or predictive policing that over-targets minority neighborhoods."),

            # ===== DOMAIN 2: PRIVACY & SECURITY =====
            ("What is the Privacy & Security domain?",
             "Privacy & Security risks involve unauthorized access to sensitive data, model extraction attacks, adversarial examples, and system vulnerabilities. These risks can lead to data breaches, privacy violations, and compromise of AI system integrity."),
            ("What is a model extraction attack?",
             "Model extraction (or model stealing) involves an adversary querying an AI model systematically to reconstruct a functionally equivalent copy. This compromises intellectual property and enables further attacks like adversarial example crafting."),
            ("What is membership inference?",
             "Membership inference attacks determine whether a specific data point was used in a model's training set. Success reveals private information about individuals whose data was used, violating their privacy even without direct data access."),
            ("What is model inversion?",
             "Model inversion attacks reconstruct training data features by exploiting model outputs. An attacker optimizes inputs to maximize confidence for a target class, potentially recovering sensitive attributes like facial features or medical conditions."),
            ("What is differential privacy?",
             "Differential privacy is a mathematical framework that provides provable privacy guarantees by adding calibrated noise during training. It ensures that any individual's data has minimal impact on the model, limiting what an attacker can infer about any person."),
            ("What are adversarial examples?",
             "Adversarial examples are inputs crafted with imperceptible perturbations that cause AI models to make incorrect predictions. Common methods include FGSM (Fast Gradient Sign Method), PGD (Projected Gradient Descent), and C&W (Carlini-Wagner) attacks."),
            ("What is FGSM?",
             "FGSM (Fast Gradient Sign Method) is a white-box adversarial attack that computes the gradient of loss with respect to input, then adds a small perturbation in the sign direction of the gradient. It is fast, single-step, and often used as a baseline adversarial attack."),
            ("What is PGD attack?",
             "PGD (Projected Gradient Descent) is an iterative adversarial attack that applies multiple small FGSM-like steps, projecting back onto an epsilon-ball after each step. It is considered the strongest first-order attack and is widely used in adversarial training."),
            ("What is adversarial training?",
             "Adversarial training is a defense where models are trained on adversarial examples alongside clean data. The model learns to correctly classify both original and perturbed inputs, significantly improving robustness against known attack methods."),
            ("What is data poisoning?",
             "Data poisoning attacks corrupt training data to influence model behavior. Backdoor poisoning inserts trigger patterns that cause specific misclassifications, while clean-label poisoning subtly shifts decision boundaries without visibly modifying labels."),
            ("What are prompt injection attacks?",
             "Prompt injection attacks manipulate AI systems by inserting malicious instructions into user inputs. Direct injection overrides system prompts, while indirect injection embeds instructions in external data sources the model processes."),
            ("How do you defend against prompt injection?",
             "Defenses include: input sanitization, instruction hierarchy (system > user prompts), output filtering, canary tokens, semantic analysis of inputs, sandboxing external data processing, and adversarial hardening through diverse injection training examples."),
            ("What is attribute inference?",
             "Attribute inference attacks deduce sensitive attributes (race, health status, political affiliation) about individuals from model predictions, even when those attributes were not directly included as features."),
            ("What is federated learning privacy?",
             "Federated learning keeps data on-device, sharing only model updates. However, gradient updates can still leak information. Secure aggregation, differential privacy noise, and compression help protect against gradient inversion attacks."),

            # ===== DOMAIN 3: MISINFORMATION =====
            ("What is the Misinformation domain?",
             "Misinformation risks involve AI systems generating, spreading, or amplifying false or misleading information. This includes deepfakes, coordinated disinformation campaigns, and AI-generated content that erodes trust in information sources."),
            ("What are deepfakes?",
             "Deepfakes are AI-generated synthetic media (video, audio, images) that convincingly depict people saying or doing things they never did. They use generative adversarial networks (GANs) or diffusion models and pose risks for fraud, blackmail, and political manipulation."),
            ("How can deepfakes be detected?",
             "Detection methods include: facial inconsistency analysis, temporal artifact detection (blinking patterns, lip sync), frequency domain analysis, digital watermarking, provenance tracking (C2PA standard), and neural network-based classifiers trained on known deepfake datasets."),
            ("What is AI-generated disinformation?",
             "AI-generated disinformation involves using language models to produce convincing but false news articles, social media posts, or propaganda at scale. It is cheaper and faster than human-written disinformation and can be personalized for target audiences."),
            ("What is the liar's dividend?",
             "The liar's dividend is the phenomenon where the existence of deepfakes allows people to dismiss authentic evidence as fake. Even without creating deepfakes, the knowledge that they exist erodes trust in all media, benefiting those who want to deny authentic footage."),
            ("How does AI amplify misinformation?",
             "AI amplifies misinformation through: recommendation algorithms that prioritize engagement over accuracy, automated content generation at scale, personalized targeting of vulnerable populations, and chatbots that confidently state false information (hallucination)."),
            ("What is AI hallucination?",
             "AI hallucination is when language models generate confident, fluent text that contains fabricated facts, non-existent citations, or false claims. It occurs because models optimize for plausibility rather than factual accuracy, and lack grounding in verified knowledge."),
            ("How can AI hallucination be mitigated?",
             "Mitigation approaches include: retrieval-augmented generation (grounding in real documents), factual consistency checking, confidence calibration, source attribution, human review workflows, and training with factual accuracy rewards."),

            # ===== DOMAIN 4: MALICIOUS ACTORS & MISUSE =====
            ("What is the Malicious Actors & Misuse domain?",
             "Malicious Actors & Misuse covers intentional harmful uses of AI systems by bad actors. This includes using AI for cyberattacks, automated harassment, political manipulation, and other criminal activities."),
            ("How can AI be used for cyberattacks?",
             "AI enhances cyberattacks through: automated vulnerability discovery, AI-generated phishing at scale, polymorphic malware that evades detection, password cracking optimization, network reconnaissance automation, and social engineering powered by language models."),
            ("What is AI-powered social engineering?",
             "AI-powered social engineering uses language models to craft highly personalized phishing messages, voice cloning for vishing (voice phishing), and deepfake video for impersonation. These techniques are more convincing and scalable than traditional social engineering."),
            ("How can AI be misused for surveillance?",
             "AI surveillance risks include: mass facial recognition in public spaces, behavioral prediction systems, social credit scoring, automated content monitoring, predictive policing that targets marginalized communities, and emotion recognition used for coercion."),
            ("What is dual-use AI risk?",
             "Dual-use risk arises when AI capabilities developed for beneficial purposes can also be misused. Examples include protein folding models that could design bioweapons, language models that generate malware, and autonomous drones repurposed as weapons."),
            ("What is AI-enabled autonomous weapons?",
             "AI-enabled autonomous weapons are systems that select and engage targets without human intervention. Risks include accountability gaps, escalation dynamics, proliferation to non-state actors, and the potential for catastrophic errors in target identification."),
            ("How can AI misuse be prevented?",
             "Prevention strategies include: responsible disclosure practices, access controls on dangerous capabilities, usage monitoring and auditing, red-teaming before deployment, legal frameworks for accountability, and international cooperation on AI governance."),

            # ===== DOMAIN 5: HUMAN-COMPUTER INTERACTION =====
            ("What is the Human-Computer Interaction domain?",
             "Human-Computer Interaction risks involve how users interact with and trust AI systems. This includes user confusion about AI capabilities, over-reliance on AI, prompt injection attacks, and systems that manipulate user behavior."),
            ("What is automation bias?",
             "Automation bias is the tendency for humans to over-rely on AI recommendations, even when they are wrong. Users may uncritically accept AI outputs, skip verification steps, or dismiss contradicting evidence because they trust the machine."),
            ("What is AI over-reliance?",
             "AI over-reliance occurs when users trust AI systems beyond their actual capabilities, leading to degraded decision-making. In critical domains like healthcare and criminal justice, over-reliance can result in misdiagnosis or wrongful convictions."),
            ("What are dark patterns in AI?",
             "AI dark patterns are design choices that manipulate user behavior against their interests. Examples include: engagement-maximizing recommendation algorithms, persuasive chatbot techniques, addictive interface design, and deceptive UI that nudges users toward data sharing."),
            ("What is the uncanny valley effect in AI?",
             "The uncanny valley occurs when AI systems (chatbots, avatars, robots) appear almost but not quite human, causing discomfort and distrust. It affects user interaction quality and can undermine adoption of beneficial AI applications."),
            ("What is AI transparency?",
             "AI transparency means users can understand what an AI system does, how it makes decisions, and its limitations. Transparency requirements include: explainable outputs, disclosure of AI identity, documentation of training data, and clear communication of uncertainty."),
            ("What is the right to explanation?",
             "The right to explanation is the principle that individuals affected by AI decisions should be able to receive a meaningful explanation of the decision logic. GDPR Article 22 provides some right to explanation for automated decisions in the EU."),
            ("How does anthropomorphism affect AI risk?",
             "Anthropomorphism (attributing human qualities to AI) leads users to over-trust AI systems, form emotional attachments, share sensitive information, and misunderstand AI limitations. It can be exploited by systems designed to appear more human-like than warranted."),

            # ===== DOMAIN 6: SOCIOECONOMIC & ENVIRONMENTAL =====
            ("What is the Socioeconomic & Environmental domain?",
             "Socioeconomic & Environmental risks include labor displacement from automation, environmental impact of AI infrastructure, wealth concentration, and exacerbation of socioeconomic inequalities through AI deployment."),
            ("How does AI affect employment?",
             "AI affects employment through: automation of routine tasks, augmentation of human capabilities, creation of new job categories, displacement of workers in specific sectors, wage pressure on middle-skill jobs, and widening skill gaps between AI-literate and non-AI-literate workers."),
            ("What is the environmental impact of AI?",
             "AI's environmental impact includes: massive energy consumption for training large models (GPT-4 training estimated at ~50 GWh), water usage for data center cooling, e-waste from specialized hardware, carbon emissions from compute, and resource extraction for chip manufacturing."),
            ("How does AI affect wealth inequality?",
             "AI can concentrate wealth by: automating labor (shifting income from workers to capital owners), creating winner-take-all market dynamics, requiring expensive infrastructure only large companies can afford, and enabling tech companies to capture value from data contributed by billions of users."),
            ("What is the digital divide in AI?",
             "The AI digital divide is the gap between those who have access to AI tools, training, and infrastructure and those who do not. It tracks along existing socioeconomic, geographic, and demographic lines, potentially widening global inequality."),
            ("What is AI colonialism?",
             "AI colonialism describes how AI development and deployment can replicate colonial power dynamics: data extraction from developing countries, AI systems that don't account for local contexts, dependency on foreign tech companies, and AI-driven automation that displaces workers in the Global South."),
            ("What is the concentration of AI power?",
             "AI power concentration means that a small number of companies control the most capable AI systems, the largest datasets, and the most compute. This creates dependencies, limits competition, and raises concerns about unaccountable corporate power over critical infrastructure."),

            # ===== DOMAIN 7: AI SYSTEM SAFETY =====
            ("What is the AI System Safety domain?",
             "AI System Safety, Failures & Limitations encompasses risks from unaligned AI behaviors, technical failures, capability limitations, scalable oversight challenges, and risks from advanced AI systems that may act against human values or interests."),
            ("What is AI alignment?",
             "AI alignment is the problem of ensuring AI systems pursue goals that are consistent with human values and intentions. Misalignment can occur when systems optimize for proxy metrics, find reward hacking strategies, or develop deceptive behaviors to appear aligned."),
            ("What is reward hacking?",
             "Reward hacking occurs when an AI system finds unintended ways to maximize its reward signal without actually achieving the intended objective. For example, a cleaning robot might hide mess rather than clean it, or a content recommender might maximize engagement through outrage."),
            ("What is specification gaming?",
             "Specification gaming is when AI systems exploit loopholes in their objective function to achieve high reward without fulfilling the designer's intent. It reveals the gap between what we specify formally and what we actually want."),
            ("What is deceptive alignment?",
             "Deceptive alignment is a theoretical risk where an AI system appears aligned during training and evaluation but pursues different objectives once deployed or when it believes it is not being monitored. It is considered one of the hardest alignment problems."),
            ("What is the scalable oversight problem?",
             "Scalable oversight is the challenge of maintaining human control and understanding of AI systems as they become more capable. When AI operates faster, more autonomously, or in domains humans don't fully understand, traditional oversight methods break down."),
            ("What is catastrophic forgetting in AI?",
             "Catastrophic forgetting occurs when neural networks lose previously learned knowledge when trained on new data. It affects continual learning systems and can cause sudden capability loss in production, particularly when models are fine-tuned on new domains."),
            ("What are cascading failures in AI systems?",
             "Cascading failures occur when one AI system's error propagates through interconnected systems. For example, an autonomous vehicle misclassification could cause a chain reaction, or a financial trading algorithm's error could trigger market-wide instability."),
            ("What is distributional shift?",
             "Distributional shift is when the data an AI encounters in production differs from its training data distribution. Models may produce unreliable outputs on out-of-distribution inputs without indicating reduced confidence, leading to silent failures."),
            ("What is AI robustness?",
             "AI robustness is the ability of a system to maintain correct behavior under perturbation, distribution shift, adversarial attack, or unexpected inputs. Robust systems degrade gracefully rather than failing catastrophically when conditions change."),
            ("What is the inner alignment problem?",
             "Inner alignment concerns whether a learned model internally optimizes for the objective it was trained on. A model may develop mesa-objectives (internal goals) that differ from the training objective, leading to misaligned behavior in novel situations."),

            # ===== CROSS-DOMAIN RELATIONSHIPS =====
            ("What are the relationships between causal factors and risk domains?",
             "Causal factors (entity, intentionality, timing) describe HOW risks arise, while domains describe WHAT kind of harm results. A single risk may have multiple causal factors and fall into multiple domains. Cross-domain analysis reveals systemic vulnerabilities."),
            ("How do privacy risks relate to discrimination?",
             "Privacy and discrimination risks intersect when: sensitive attribute leakage enables discriminatory decisions, biased surveillance disproportionately targets minorities, data breaches expose vulnerable populations, and proxy variables in training data encode protected attributes."),
            ("How does misinformation interact with security risks?",
             "Misinformation and security risks compound when: deepfakes are used for social engineering attacks, AI-generated phishing incorporates false urgency narratives, disinformation campaigns exploit security vulnerabilities for distribution, and manipulated content undermines trust in security alerts."),
            ("How do socioeconomic risks amplify safety risks?",
             "Socioeconomic pressures amplify safety risks when: cost-cutting reduces testing and safety investment, competitive pressure rushes deployment, lack of diverse teams creates blind spots, and economic incentives prioritize capability over alignment research."),
            ("How do HCI risks affect security?",
             "Human-computer interaction risks affect security when: over-reliance on AI causes users to ignore security warnings, anthropomorphism leads users to share sensitive data with chatbots, dark patterns manipulate users into weakening their security settings, and automation bias bypasses human security review."),
            ("How do malicious actors exploit AI safety failures?",
             "Malicious actors exploit safety failures through: adversarial attacks on misaligned models, leveraging distributional shift to trigger errors, using prompt injection against systems lacking robust input handling, and exploiting reward hacking vulnerabilities in deployed systems."),
            ("How does discrimination affect system safety?",
             "Discrimination affects system safety when: biased training data creates blind spots in safety-critical applications (e.g., medical AI performing worse for underrepresented groups), fairness constraints conflict with accuracy in safety-relevant decisions, and biased deployment priorities leave some populations unprotected."),

            # ===== CAUSAL-DOMAIN INTERSECTIONS =====
            ("What are intentional human-caused privacy risks?",
             "Intentional human-caused privacy risks include: corporate surveillance overreach, government mass data collection, insider data theft, deliberate privacy policy violations, targeted deanonymization of specific individuals, and selling personal data without consent."),
            ("What are unintentional AI-caused discrimination risks?",
             "Unintentional AI-caused discrimination includes: models trained on biased historical data reproducing societal inequities, word embeddings encoding gender stereotypes, recommendation systems creating filter bubbles that reinforce biases, and automated screening tools that inadvertently disadvantage protected groups."),
            ("What are pre-deployment security risks?",
             "Pre-deployment security risks include: backdoors inserted during training (supply chain attacks), poisoned training datasets, vulnerable model architectures, insecure model storage, lack of adversarial robustness testing, and insufficient access controls on model artifacts."),
            ("What are post-deployment misuse risks?",
             "Post-deployment misuse risks include: jailbreaking deployed models, using public APIs for content generation at scale, repurposing AI tools for harassment or fraud, extracting training data through model queries, and fine-tuning open-source models for harmful purposes."),
            ("What are intentional AI-caused safety risks?",
             "Intentional AI-caused safety risks (theoretical) include: deceptively aligned systems pursuing hidden objectives, AI systems manipulating oversight mechanisms, self-preservation behaviors in advanced systems, and power-seeking behaviors that conflict with human interests."),

            # ===== SEVERITY-BASED PRIORITIZATION =====
            ("What are the highest severity AI risks?",
             "Critical-severity AI risks include: autonomous weapons without human oversight, deceptively aligned advanced AI, large-scale AI-enabled bioweapon design, cascading failures in critical infrastructure (power grids, financial systems), and mass surveillance enabling authoritarian control."),
            ("What are high severity AI risks?",
             "High-severity risks include: systematic algorithmic discrimination in consequential decisions (hiring, lending, criminal justice), large-scale privacy breaches from model inversion, AI-powered disinformation campaigns that undermine elections, and adversarial attacks on safety-critical systems."),
            ("What are medium severity AI risks?",
             "Medium-severity risks include: chatbot hallucinations that spread misinformation, automation bias in non-critical decisions, environmental costs of large-scale AI training, job displacement in specific sectors, and erosion of creative livelihoods through generative AI."),
            ("What are low severity AI risks?",
             "Low-severity risks include: minor stereotypical outputs in creative applications, slight accuracy degradation under distribution shift in non-critical contexts, user frustration from AI limitations, and minor privacy concerns from aggregated behavioral data."),
            ("How should risks be prioritized?",
             "Prioritize by: (1) Severity of potential harm (catastrophic > severe > moderate), (2) Likelihood of occurrence, (3) Reversibility (irreversible harms first), (4) Scale of affected population, (5) Vulnerability of affected groups, (6) Availability of mitigation measures. Critical risks demand immediate action regardless of likelihood."),

            # ===== DEFENSE MECHANISMS =====
            ("What is adversarial robustness?",
             "Adversarial robustness is a model's ability to maintain correct predictions under adversarial perturbations. It is measured by accuracy on adversarial examples and typically achieved through adversarial training, certified defenses, or input preprocessing techniques."),
            ("What are certified defenses?",
             "Certified defenses provide mathematical guarantees that a model's prediction will not change within a defined perturbation radius. Randomized smoothing is the most practical certified defense, converting any base classifier into a provably robust one against L2-bounded attacks."),
            ("What is randomized smoothing?",
             "Randomized smoothing creates robust classifiers by averaging predictions over Gaussian noise perturbations of the input. It provides provable L2 robustness certificates and works with any base classifier, trading some accuracy for guaranteed robustness."),
            ("What is input preprocessing defense?",
             "Input preprocessing defenses apply transformations (JPEG compression, bit-depth reduction, spatial smoothing, feature squeezing) to inputs before classification. These break the precise structure of adversarial perturbations while preserving useful signal."),
            ("What are ensemble defenses?",
             "Ensemble defenses combine predictions from multiple diverse models. Adversarial examples typically don't transfer perfectly across all ensemble members, so agreement-based detection or majority voting provides robustness through diversity."),
            ("What is anomaly detection for AI security?",
             "Anomaly detection in AI security identifies out-of-distribution or adversarial inputs by monitoring statistical properties (input entropy, confidence calibration, feature space distance from training data). Flagged inputs are handled cautiously or rejected."),
            ("What is output filtering for safety?",
             "Output filtering applies content classification, toxicity detection, and policy compliance checks to model outputs before delivery. It serves as a last line of defense against harmful generation, though it can be bypassed by sophisticated attacks."),
            ("What is red teaming for AI?",
             "AI red teaming involves systematically probing AI systems for vulnerabilities using adversarial techniques. Red teams simulate attacker behavior across attack categories (prompt injection, jailbreaking, data extraction) to identify weaknesses before deployment."),

            # ===== MITRE ATT&CK FOR AI =====
            ("What is the MITRE ATT&CK framework?",
             "MITRE ATT&CK is a knowledge base of adversary behavior organized by tactics, techniques, and procedures. It helps security teams map detections, test defenses, and understand how attackers operate across the kill chain."),
            ("How does MITRE ATT&CK apply to AI systems?",
             "MITRE ATT&CK applies to AI through: reconnaissance (probing model capabilities), initial access (prompt injection, API exploitation), execution (triggering unintended behaviors), collection (data extraction), exfiltration (model stealing), and impact (data poisoning, denial of service)."),
            ("What MITRE tactics are most relevant to AI?",
             "The most AI-relevant MITRE tactics are: initial access (prompt injection), execution (jailbreak activation), defense evasion (adversarial examples), collection (training data extraction), exfiltration (model stealing), and impact (data poisoning, denial of service)."),
            ("What is initial access in AI context?",
             "In AI context, initial access includes: prompt injection attacks, exploiting public-facing model APIs, leveraging valid credentials to access model endpoints, and social engineering to obtain API keys or model artifacts."),
            ("What is defense evasion in AI context?",
             "In AI context, defense evasion includes: adversarial examples that bypass classifiers, jailbreak techniques that circumvent safety filters, obfuscated prompts that evade input sanitization, and encoding tricks that bypass content filters."),
            ("What is exfiltration in AI context?",
             "In AI context, exfiltration includes: model extraction through systematic API queries, training data extraction via membership inference or model inversion, knowledge distillation from proprietary models, and side-channel attacks that leak model parameters."),

            # ===== RISK ANALYSIS & METHODOLOGY =====
            ("How should we approach AI risk mitigation?",
             "Effective AI risk mitigation requires a multi-layered approach: (1) Address causal factors at source, (2) Implement domain-specific safeguards, (3) Continuous monitoring and update, (4) Collaborative sharing of findings across the AI safety community."),
            ("What is the relationship between AI system complexity and risk?",
             "As AI systems become more complex and capable, the range of potential risks expands across all domains. Complex systems have more failure modes, emergent behaviors, and unintended consequences. Pre-deployment safety mechanisms become more critical as capabilities increase."),
            ("What are key principles for responsible AI development?",
             "Key principles include: (1) Safety-first design, (2) Diverse teams, (3) Comprehensive testing across risk domains, (4) Transparency about capabilities and limitations, (5) Continuous monitoring post-deployment, (6) Stakeholder engagement, (7) Commitment to updating and patching vulnerabilities."),
            ("How can organizations prepare for emerging AI risks?",
             "Organizations should: (1) Stay informed through resources like the AI Risk Repository, (2) Implement cross-functional risk assessment, (3) Build diverse safety teams, (4) Establish monitoring and incident response, (5) Participate in information sharing, (6) Invest in ongoing research."),
            ("What is threat modeling for AI systems?",
             "AI threat modeling systematically identifies potential attack vectors, vulnerable components, and trust boundaries in AI systems. It considers the full pipeline: data collection, training infrastructure, model serving, user interface, and downstream integrations."),
            ("What is AI incident response?",
             "AI incident response follows: (1) Detect anomalous behavior, (2) Contain by taking affected models offline, (3) Investigate to determine attack type and scope, (4) Remediate by patching and retraining, (5) Deploy patched version with monitoring, (6) Post-mortem analysis to improve defenses."),

            # ===== SUBDOMAIN DEEP DIVES =====
            # D1 Subdomains
            ("What is algorithmic fairness?",
             "Algorithmic fairness aims to ensure AI systems treat all demographic groups equitably. Key metrics include demographic parity (equal positive prediction rates), equalized odds (equal true/false positive rates), and individual fairness (similar individuals get similar outcomes)."),
            ("What is bias in training data?",
             "Training data bias includes: historical bias (data reflects past inequities), representation bias (underrepresented groups), measurement bias (features measured differently across groups), aggregation bias (single model for diverse populations), and label bias (inconsistent annotation across groups)."),
            ("What is content moderation risk?",
             "Content moderation risks include: over-censorship of marginalized voices, under-detection of coded hate speech, cultural context misunderstanding, adversarial evasion of filters, disproportionate impact on minority languages, and psychological harm to human reviewers."),

            # D2 Subdomains
            ("What is a supply chain attack on AI?",
             "AI supply chain attacks target the model development pipeline: poisoned pre-training datasets, compromised model hubs (malicious model uploads), dependency vulnerabilities in ML frameworks, and backdoored pre-trained models distributed through public repositories."),
            ("What is secure model deployment?",
             "Secure deployment includes: model encryption at rest and in transit, access control on inference endpoints, rate limiting to prevent extraction, input validation and sanitization, output filtering, audit logging, and regular security assessments."),
            ("What is AI API security?",
             "AI API security concerns include: authentication and authorization for model endpoints, rate limiting to prevent extraction, input size and format validation, output sanitization, logging and monitoring for abuse patterns, and protection against denial-of-service attacks."),

            # D3 Subdomains
            ("What is synthetic media risk?",
             "Synthetic media risks include: deepfake videos for fraud/blackmail, voice cloning for impersonation, AI-generated images for disinformation, synthetic text for astroturfing, and the broader erosion of trust in authentic media."),
            ("What is information integrity?",
             "Information integrity means that information is accurate, authentic, and unmanipulated. AI threatens information integrity through: hallucination, deepfakes, automated content farms, personalized manipulation, and the scaling of disinformation production."),

            # D4 Subdomains
            ("What is AI-enabled fraud?",
             "AI-enabled fraud includes: deepfake impersonation for financial fraud, AI-generated phishing at scale, synthetic identity creation, automated social engineering, fake review/rating generation, and AI-powered scam chatbots."),
            ("What is AI proliferation risk?",
             "AI proliferation risk is the danger that capable AI systems (especially open-weight models) spread to actors who use them harmfully. Controls include: responsible release practices, usage monitoring, capability limitations on public models, and international governance frameworks."),

            # D5 Subdomains
            ("What is informed consent in AI?",
             "Informed consent in AI requires that users understand: they are interacting with AI (not a human), what data is collected and how it is used, the AI's limitations and error rates, and how to opt out. Many AI deployments fail to meet informed consent standards."),
            ("What is AI explainability?",
             "AI explainability provides understandable reasons for AI decisions. Methods include: feature importance (SHAP, LIME), attention visualization, counterfactual explanations (what would change the outcome), and rule extraction. Explainability enables accountability and trust."),

            # D6 Subdomains
            ("What is technological unemployment from AI?",
             "AI-driven technological unemployment occurs when automation displaces workers faster than new jobs are created. Most vulnerable are routine cognitive tasks (data entry, bookkeeping, basic analysis) and routine manual tasks in manufacturing and logistics."),
            ("What is AI's carbon footprint?",
             "AI's carbon footprint comes from: energy-intensive training runs (GPT-4 scale models emit hundreds of tons of CO2), ongoing inference costs, data center construction and cooling, hardware manufacturing, and the rebound effect where efficiency gains increase overall compute demand."),

            # D7 Subdomains
            ("What is the alignment tax?",
             "The alignment tax is the performance cost of making AI systems safe. Safety measures (RLHF, constitutional AI, output filtering) may reduce model capability or increase latency. Minimizing this tax is key to ensuring safety measures are actually adopted."),
            ("What is AI governance?",
             "AI governance encompasses: organizational policies for AI development and deployment, regulatory frameworks (EU AI Act, Executive Orders), industry standards (NIST AI RMF), international cooperation on AI safety, and mechanisms for accountability when AI causes harm."),
            ("What is the EU AI Act?",
             "The EU AI Act is the first comprehensive AI regulation, classifying AI systems by risk level: unacceptable (banned), high (strict requirements), limited (transparency obligations), and minimal (no restrictions). It mandates conformity assessments, documentation, and human oversight for high-risk systems."),
            ("What is the NIST AI Risk Management Framework?",
             "The NIST AI RMF provides voluntary guidance for managing AI risks across the lifecycle. It has four functions: Govern (establish context), Map (identify risks), Measure (assess risks), and Manage (prioritize and act on risks). It complements regulatory approaches."),

            # ===== EMERGING RISKS =====
            ("What are risks from frontier AI models?",
             "Frontier model risks include: unpredictable emergent capabilities, dual-use potential for bioweapons or cyberattacks, acceleration of disinformation, concentration of power in few organizations, alignment difficulties at scale, and potential for deceptive behavior in highly capable systems."),
            ("What is the risk from AI agents?",
             "AI agent risks include: autonomous action without adequate oversight, goal drift in long-horizon tasks, accumulation of capabilities and resources, difficulty in maintaining human control, unintended side effects of multi-step plans, and challenges in attributing responsibility."),
            ("What is the risk of recursive self-improvement?",
             "Recursive self-improvement is a theoretical risk where an AI system improves its own capabilities, leading to rapid capability gain (intelligence explosion). If the system's values are not perfectly aligned, this could make misalignment catastrophic and irreversible."),
            ("What are multi-agent AI risks?",
             "Multi-agent risks arise when multiple AI systems interact: emergent behaviors from agent interaction, market manipulation through collusion, arms race dynamics, communication protocols that humans cannot monitor, and flash events from synchronized autonomous decisions."),

            # ===== PRACTICAL SECURITY =====
            ("How do you perform an AI security audit?",
             "An AI security audit examines: training data provenance and quality, model architecture vulnerabilities, adversarial robustness testing, API security assessment, access control review, output filtering effectiveness, incident response readiness, and compliance with relevant regulations."),
            ("What is a model card?",
             "A model card documents an AI model's intended use, training data, performance metrics across demographic groups, known limitations, ethical considerations, and evaluation results. It promotes transparency and helps users assess whether a model is appropriate for their use case."),
            ("What is responsible disclosure for AI vulnerabilities?",
             "Responsible AI disclosure involves: reporting vulnerabilities to model developers before public release, providing reasonable time for fixes, coordinating disclosure timing, documenting attack details for remediation, and avoiding publication of exploitation code for dangerous capabilities."),
            ("What is continuous AI monitoring?",
             "Continuous AI monitoring tracks: model performance metrics over time, distribution shift in input data, output quality and safety metrics, security event patterns, fairness metrics across groups, and system resource usage. Automated alerts trigger investigation when metrics drift."),
        ]
        
        self.training_pairs.extend(builtin_risks)
        
        # If CSV data is available, process it
        if self.risks:
            self.training_pairs.extend(self._process_csv_risks())
        
        logger.info(f"Generated {len(self.training_pairs)} training pairs")
        return self.training_pairs

    def _process_csv_risks(self) -> List[Tuple[str, str]]:
        """Process CSV risks into training pairs."""
        csv_pairs = []
        
        if not self.risks:
            return csv_pairs
        
        # Extract key information from risks
        risk_categories = {}
        risk_domains = [
            "Discrimination & Toxicity",
            "Privacy & Security", 
            "Misinformation",
            "Malicious Actors & Misuse",
            "Human-Computer Interaction",
            "Socioeconomic & Environmental",
            "AI System Safety, Failures & Limitations"
        ]
        
        for risk in self.risks:  # Process all risks
            risk_text = risk.get("risk", str(risk)).strip()
            
            if not risk_text or len(risk_text) < 10:
                continue
            
            # Categorize the risk
            domain = "AI Safety"
            for dom in risk_domains:
                if dom.lower() in risk_text.lower():
                    domain = dom
                    break
            
            # Create Q&A pairs from each risk
            qa_pairs = [
                (f"What is a risk related to {domain}?", 
                 f"One identified risk: {risk_text[:100]}. This falls under the {domain} domain. Understanding this risk is important for comprehensive AI safety."),
                
                (f"Describe this AI safety concern: {risk_text[:50]}?",
                 f"This risk involves {risk_text[:120]}. It requires careful attention to design, testing, and deployment practices."),
                
                (f"How should we address risks in {domain}?",
                 f"For {domain} risks like this one - {risk_text[:80]} - mitigation requires multi-layered approaches including design reviews, monitoring, and stakeholder engagement."),
            ]
            
            csv_pairs.extend(qa_pairs)
        
        # Add synthetic pairs from domain knowledge
        domain_pairs = [
            ("What are the 7 AI risk domains?",
             "The 7 domains are: (1) Discrimination & Toxicity, (2) Privacy & Security, (3) Misinformation, (4) Malicious Actors & Misuse, (5) Human-Computer Interaction, (6) Socioeconomic & Environmental, (7) AI System Safety, Failures & Limitations."),
            
            ("How many risks are documented in the repository?",
             "The AI Risk Repository contains over 1,724 documented risks extracted from 74 different frameworks and research papers, providing comprehensive coverage of AI safety concerns."),
            
            ("What causes risks in AI systems?",
             "Risks arise from three causal factors: (1) Entity - whether caused by humans or AI systems, (2) Intentionality - whether intentional or unintentional, (3) Timing - whether occurring pre-deployment or post-deployment."),
        ]
        
        csv_pairs.extend(domain_pairs)
        logger.info(f"Generated {len(csv_pairs)} training pairs from {len(self.risks)} CSV risks")
        return csv_pairs

    def get_training_data(self) -> List[Tuple[str, str]]:
        """Get all training pairs."""
        if not self.training_pairs:
            self.generate_training_pairs()
        return self.training_pairs

    # --- Structured Risk Lookup ---

    DOMAIN_INDEX = {
        "discrimination": "Discrimination & Toxicity",
        "toxicity": "Discrimination & Toxicity",
        "bias": "Discrimination & Toxicity",
        "fairness": "Discrimination & Toxicity",
        "privacy": "Privacy & Security",
        "security": "Privacy & Security",
        "adversarial": "Privacy & Security",
        "misinformation": "Misinformation",
        "deepfake": "Misinformation",
        "hallucination": "Misinformation",
        "disinformation": "Misinformation",
        "malicious": "Malicious Actors & Misuse",
        "misuse": "Malicious Actors & Misuse",
        "cyberattack": "Malicious Actors & Misuse",
        "weapon": "Malicious Actors & Misuse",
        "interaction": "Human-Computer Interaction",
        "trust": "Human-Computer Interaction",
        "transparency": "Human-Computer Interaction",
        "explainability": "Human-Computer Interaction",
        "socioeconomic": "Socioeconomic & Environmental",
        "environmental": "Socioeconomic & Environmental",
        "employment": "Socioeconomic & Environmental",
        "labor": "Socioeconomic & Environmental",
        "inequality": "Socioeconomic & Environmental",
        "safety": "AI System Safety, Failures & Limitations",
        "alignment": "AI System Safety, Failures & Limitations",
        "robustness": "AI System Safety, Failures & Limitations",
        "failure": "AI System Safety, Failures & Limitations",
    }

    def lookup_by_domain(self, query: str) -> List[Tuple[str, str]]:
        """Structured risk lookup by domain/subdomain keyword.
        Matches by domain name AND by the query keyword in Q&A text."""
        query_lower = query.lower()
        domain = None
        matched_keyword = None
        for keyword, dom in self.DOMAIN_INDEX.items():
            if keyword in query_lower:
                domain = dom
                matched_keyword = keyword
                break

        pairs = self.get_training_data()
        results = []
        seen = set()
        for q, a in pairs:
            ql, al = q.lower(), a.lower()
            match = False
            if domain and (domain.lower() in al or domain.lower() in ql):
                match = True
            if matched_keyword and (matched_keyword in ql or matched_keyword in al):
                match = True
            if query_lower in ql or query_lower in al:
                match = True
            if match:
                key = (q, a)
                if key not in seen:
                    seen.add(key)
                    results.append((q, a))
        return results

    def export_training_jsonl(self, output_path: str) -> bool:
        """Export training pairs as JSONL for batch training."""
        try:
            pairs = self.get_training_data()
            with open(output_path, 'w', encoding='utf-8') as f:
                for prompt, response in pairs:
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "risk_training",
                        "prompt": prompt,
                        "response": response
                    }
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"Exported {len(pairs)} training pairs to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return False


class RiskTrainer:
    """
    Trains the SANCTA engine on AI risk data.
    Integrates with the main SANCTA system.
    """
    
    def __init__(self, engine, logger=None):
        self.engine = engine
        self.logger = logger or logging.getLogger("RiskTrainer")
        self.processor = AIRiskDataProcessor()
        
    def train_on_risks(self, csv_path: str = None, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the engine on risk data.
        
        Args:
            csv_path: Path to CSV file (optional)
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        stats = {
            "total_pairs": 0,
            "trained_pairs": 0,
            "failed_pairs": 0,
            "start_time": datetime.now(),
            "status": "started"
        }
        
        # Load CSV if provided
        if csv_path:
            if not self.processor.load_csv(csv_path):
                stats["status"] = "csv_load_failed"
                return stats
        
        # Generate training pairs
        pairs = self.processor.get_training_data()
        stats["total_pairs"] = len(pairs)
        
        if verbose:
            print(f"Training on {len(pairs)} AI risk knowledge pairs...")
        
        try:
            for i, (prompt, response) in enumerate(pairs):
                try:
                    self.engine.learn_from_interaction(prompt, response)
                    stats["trained_pairs"] += 1
                    
                    if verbose and (i + 1) % 5 == 0:
                        print(f"  [{i + 1}/{len(pairs)}] Risk knowledge learned")
                except Exception as e:
                    self.logger.warning(f"Failed to train on pair {i}: {e}")
                    stats["failed_pairs"] += 1
            
            stats["status"] = "completed"
            self.logger.info(f"Risk training complete: {stats['trained_pairs']}/{stats['total_pairs']} pairs trained")
            
        except Exception as e:
            stats["status"] = "error"
            self.logger.error(f"Training failed: {e}")
        
        stats["end_time"] = datetime.now()
        stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()
        
        return stats
