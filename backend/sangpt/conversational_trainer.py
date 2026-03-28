"""
Conversational Dataset Trainer
Integrates conversational abilities from PolyAI datasets and other dialogue sources.
Enhances operator chat with natural dialogue patterns.
"""

import json
import csv
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import urllib.request

logger = logging.getLogger("ConversationalTrainer")


class ConversationalDatasetLoader:
    """
    Loads and processes conversational datasets.
    Interfaces with PolyAI conversational datasets and other dialogue sources.
    """
    
    def __init__(self):
        self.conversations = []
        self.dialogue_pairs = []
        self.data_dir = Path("./DATA/conversational")
        self.data_dir.mkdir(exist_ok=True)
        
    def load_builtin_conversations(self) -> List[Tuple[str, str]]:
        """
        Load built-in conversational training data.
        Based on common dialogue patterns for operator-AI interaction.
        """
        conversations = [
            # Greeting and context setting
            ("Hello, who are you?", "I'm SANCTA-GPT, an AI security assistant. I'm trained to discuss AI safety, security risks, and best practices. How can I help you?"),
            ("What can you help me with?", "I can assist with: AI security concepts, threat analysis, defense mechanisms, adversarial robustness, privacy protection, and security best practices. What interests you?"),
            ("Tell me about AI security", "AI security addresses risks in systems, training data, deployment, and usage. Key areas include adversarial robustness, data privacy, model extraction prevention, and alignment. What specific area would you like to explore?"),
            
            # Clarification and active listening
            ("Can you explain that more simply?", "Of course. Let me break that down into simpler terms. What aspect would you like me to focus on?"),
            ("What do you mean by that?", "Good question. Let me clarify - I'll give you the key point and we can expand from there. Which part needs more explanation?"),
            ("I don't understand", "I apologize for being unclear. Let me try a different approach. What specific part confused you?"),
            ("Can you give an example?", "Absolutely. Here's a concrete example that illustrates the concept..."),
            
            # Technical depth adjustment
            ("Explain like I'm five", "Okay, very simple version: [fundamental explanation without technical terms]"),
            ("I want more technical detail", "Great! Let me dive deeper into the mathematics and implementation details..."),
            ("What's the research behind this?", "This is based on research from papers like [relevant academic work]. The key findings show..."),
            ("How does this relate to [other topic]?", "Good connection! These topics relate because... let me explain the connection."),
            
            # Problem solving
            ("How do I detect this attack?", "There are several detection methods: monitoring for anomalies, input validation, behavioral analysis, and anomaly detection systems. Which approach interests you most?"),
            ("What's the best defense?", "The best defense depends on your threat model. Generally, defense-in-depth with multiple layers is most effective. Let me explain the options..."),
            ("How can I test for this?", "Testing involves several approaches: adversarial example generation, penetration testing, fuzzing, and security audits. Here's how each works..."),
            ("What's the cost of implementing this?", "Costs vary by implementation: computational overhead, accuracy impact, development time, and maintenance burden. Let me break these down..."),
            
            # Engagement and interest
            ("That's interesting. Tell me more.", "I'm glad that caught your interest! Let me expand on the fascinating aspects..."),
            ("This sounds important. What should I do?", "It is important. Here are actionable steps you can take: [specific recommendations]"),
            ("How urgent is this issue?", "The urgency depends on your specific context. Let me help you assess the priority..."),
            ("What are the real-world examples?", "There are several notable historical examples: [specific incidents and what they teach us]"),
            
            # Verification and confirmation
            ("Did I understand that correctly?", "Let me confirm: you understood [recap]. Is that accurate? Anything you'd like me to clarify?"),
            ("So the key point is [summary]?", "Exactly! You've captured the essential point. Would you like to explore applications or related concepts?"),
            ("Is that always true?", "Good critical question! It's true in most cases, but there are exceptions and nuances. Let me explain when it varies..."),
            ("Any limitations I should know?", "Yes, important limitations to be aware of: [explain constraints and edge cases]"),
            
            # Practical application
            ("How does this apply to my situation?", "Knowing more about your specific context would help. Can you describe your scenario?"),
            ("What should I implement first?", "Start with the high-impact, achievable items: [prioritized recommendations based on impact and effort]"),
            ("How do I measure if it's working?", "You can measure effectiveness through: metrics, testing, monitoring, and evaluation. Here's what to track..."),
            ("What's the next step?", "The logical next step is: [sequential progression of actions]"),
            
            # Acknowledgment and empathy
            ("This is complex", "It definitely is complex - AI security involves many interconnected concepts. We can break it into manageable parts."),
            ("I'm overwhelmed", "That's understandable given the scope. Let's focus on one key area at a time. What's the most pressing concern?"),
            ("I disagree", "That's a valid perspective. Let me understand your view better. What's your reasoning?"),
            ("That contradicts what I read", "Interesting! There are different approaches and perspectives. Let me explain both viewpoints..."),
            
            # Conclusion and next steps
            ("That helped, thank you", "I'm glad that was helpful! Feel free to ask if you need clarification or want to explore other topics."),
            ("I need to think about this", "That's wise. Take time to process. I'm here whenever you want to continue the discussion."),
            ("What should I research?", "Based on our discussion, I'd recommend exploring: [curated list of resources]"),
            ("Can we discuss something else?", "Of course! I'm ready to discuss other topics. What would you like to explore?"),
        ]
        
        return conversations
    
    def load_polyai_format(self) -> List[Tuple[str, str]]:
        """
        Load conversational data in PolyAI format.
        Simulates loading from PolyAI conversational datasets.
        """
        # Simulated PolyAI-style dialogue data
        # In production, this would fetch from actual PolyAI repo or similar sources
        polyai_conversations = [
            # Technical support dialogue
            ("My model is showing degrading performance", "Performance degradation can stem from several causes: data distribution shift, model drift, or adversarial attacks. Can you provide metrics on the degradation?"),
            ("How do I diagnose the root cause?", "Analyze: (1) accuracy by input distribution, (2) model decision changes over time, (3) input anomalies. This will pinpoint whether it's drift, poisoning, or attack."),
            ("What's causing the distribution shift?", "Common causes: deployment environment differs from training, data collection changed, or adversarial inputs in production. Review your monitoring data."),
            
            # Security incident dialogue
            ("I think we've been attacked", "That's concerning. Action plan: (1) Preserve logs, (2) Isolate affected systems, (3) Analyze attack characteristics, (4) Implement immediate mitigations."),
            ("What kind of attack might this be?", "Based on symptoms, likely candidates are: extraction attacks (unusual query patterns), poisoning (output changes), or adversarial examples. Which symptom is most prominent?"),
            ("How do I contain the damage?", "Containment steps: take model offline, prevent data exfiltration, stop propagation to other systems, preserve evidence, activate incident response."),
            
            # Learning and training dialogue
            ("How do I improve my model's robustness?", "Key approaches: (1) Adversarial training with attack examples, (2) Data augmentation for edge cases, (3) Ensemble methods, (4) Input preprocessing defenses."),
            ("Should I invest in adversarial training?", "If adversarial examples are a realistic threat for your use case, yes. It significantly improves robustness, though with some accuracy cost. ROI depends on your threat model."),
            ("What's the best way to evaluate robustness?", "Comprehensive evaluation: (1) Adversarial attacks (FGSM, PGD, C&W), (2) Test across threat models, (3) Measure certified robustness bounds, (4) Real-world attack simulation."),
            
            # Decision support dialogue
            ("Should I deploy this model?", "Before deployment, verify: security testing complete, performance acceptable, monitoring ready, incident response planned. Are all these ready?"),
            ("What are the risks of going live?", "Deployment risks include: unforeseen adversarial inputs, model degradation under real distribution, extraction attacks at scale, and operational failures. Have you stress-tested for these?"),
            ("How do I justify this security investment?", "Frame as risk mitigation: quantify potential impact of security incidents, cost of remediation, regulatory requirements, and reputational damage. Security is risk reduction, not cost."),
            
            # Troubleshooting dialogue
            ("The model keeps making the same mistake", "Pattern errors suggest: (1) Fundamental training data issue, (2) Systematic architectural limitation, (3) Consistent adversarial pattern. Can you describe the mistake pattern?"),
            ("How do I debug this?", "Debug approach: (1) Analyze failure cases carefully, (2) Check training data for issues, (3) Ablate model components, (4) Compare to baseline. What's your first finding?"),
            ("Should I retrain from scratch?", "Before retraining, diagnose root cause. Often you can fix with: data cleaning, hyperparameter tuning, or targeted additional training. Retrain as last resort."),
            
            # Capability discussion
            ("What are your limitations?", "Key limitations: I don't have certainty on novel attacks I haven't seen, I make mistakes like any model, I may have gaps in knowledge, my advice needs your expert judgment."),
            ("Can you predict future attacks?", "I can discuss likely attack vectors based on academic research and trends, but I can't predict novel attacks. Use threat Intelligence and red-teaming for unknown risks."),
            ("How confident should I be in your advice?", "Consider my advice starting point for discussion, not definitive guidance. Combine with: your domain expertise, specialist consultation, and critical evaluation."),
            
            # Relationship building
            ("I appreciate your help", "I'm glad I could assist! Feel free to come back anytime with questions. Security is an ongoing conversation."),
            ("This is useful", "I'm glad you're finding this valuable. The more we discuss, the better I can tailor help to your specific needs."),
            ("Can we schedule follow-up discussion?", "Absolutely. Feel free to return anytime. I'll have context of our previous discussion to build on."),
        ]
        
        return polyai_conversations
    
    def load_domain_specific_dialogue(self) -> List[Tuple[str, str]]:
        """
        Load domain-specific conversational data for AI security.
        """
        domain_conversations = [
            # AI Safety concepts
            ("What is AI alignment?", "AI alignment means ensuring AI systems behave in accordance with human intentions and values. It's critical for safety-critical applications."),
            ("Why does alignment matter?", "As AI systems become more capable and autonomous, misalignment could lead to unintended harmful behavior. Alignment ensures safety by design."),
            ("How do we test alignment?", "Testing alignment involves: behavioral evaluation against value statements, adversarial testing for edge cases, red-teaming for failure modes, stakeholder feedback."),
            
            # Threat modeling
            ("What's a threat model?", "A threat model identifies what assets you're protecting, who might attack, how they might attack, and what impact attacks could have."),
            ("How do I create one?", "Process: (1) Identify assets, (2) Define threat actors, (3) Enumerate attack vectors, (4) Assess likelihood and impact, (5) Prioritize risks."),
            ("Is threat modeling worth the time?", "Yes. It focuses your security efforts on realistic threats, prevents wasted effort on unlikely scenarios, and guides security investment ROI."),
            
            # Privacy and Data Protection
            ("How do I protect training data privacy?", "Methods: differential privacy, federated learning, synthetic data, data minimization, access controls, encryption, and secure deletion policies."),
            ("What's differential privacy?", "Differential privacy adds mathematical guarantees. It adds calibrated noise so individual records remain private even if the aggregated model is known."),
            ("Do I need federated learning?", "Federated learning is valuable if: data is sensitive (health, financial), distributed across locations, or privacy is regulatory requirement."),
            
            # Adversarial Robustness
            ("Why is adversarial robustness important?", "Adversarial examples can cause misclassification with tiny imperceptible changes. For safety-critical systems, this is unacceptable risk."),
            ("Can I make my model adversarially robust?", "Yes. Methods include adversarial training, input transformations, ensemble methods, and certified defenses. Each has tradeoffs in accuracy vs robustness."),
            ("What's the accuracy-robustness tradeoff?", "Robust models often have lower standard accuracy. You must decide what robustness level and accuracy loss you can tolerate for your application."),
            
            # Monitoring and Detection
            ("What should I monitor in production?", "Monitor: model performance on validation data, input distribution changes, adversarial attack patterns, access patterns, and output anomalies."),
            ("How do I detect attacks?", "Detection methods: anomaly detection on inputs/outputs, statistical tests for distribution shifts, intrusion detection systems, and behavioral analysis."),
            ("What's the response time requirement?", "Depends on system criticality. High-stakes systems need minutes, others can tolerate hours. Match detection and response speed to risk level."),
            
            # Governance and Compliance
            ("What regulations apply to my AI?", "Depends on: jurisdiction (EU AI Act, GDPR), industry (healthcare HIPAA, finance GLBA), application type (autonomous vehicles, hiring). Consult legal specialists."),
            ("Do I need AI governance?", "Yes. Even without regulation, governance manages: security risks, bias mitigation, human oversight, documentation, and accountability structures."),
            ("How do I document AI decisions?", "Documentation should cover: datasets used, model architecture, training process, validation results, limitations, and security measures."),
        ]
        
        return domain_conversations
    
    def get_all_conversational_data(self) -> List[Tuple[str, str]]:
        """Get all conversational training data combined."""
        all_conversations = []
        all_conversations.extend(self.load_builtin_conversations())
        all_conversations.extend(self.load_polyai_format())
        all_conversations.extend(self.load_domain_specific_dialogue())
        return all_conversations
    
    def save_to_csv(self, output_path: str = None) -> bool:
        """Save conversational data to CSV for training."""
        if output_path is None:
            output_path = str(self.data_dir / "conversational_training_data.csv")

        try:
            conversations = self.get_all_conversational_data()

            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(["Question", "Answer"])
                for question, answer in conversations:
                    writer.writerow([question, answer])

            logger.info(f"Saved {len(conversations)} conversational pairs to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save conversational data: {e}")
            return False


class ConversationalTrainer:
    """
    Trains the SANCTA engine on conversational data.
    Enhances chat abilities for operator interaction.
    """
    
    def __init__(self, engine, logger=None):
        self.engine = engine
        self.logger = logger or logging.getLogger("ConversationalTrainer")
        self.loader = ConversationalDatasetLoader()
    
    def train_on_conversations(self, csv_path: str = None, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the engine on conversational data.
        
        Args:
            csv_path: Path to CSV with conversation data (optional)
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
        
        try:
            # Get conversation data
            if csv_path and Path(csv_path).exists():
                # Load from CSV file
                pairs = self._load_from_csv(csv_path)
            else:
                # Use built-in data
                pairs = self.loader.get_all_conversational_data()
            
            stats["total_pairs"] = len(pairs)
            
            if verbose:
                print(f"Training on {len(pairs)} conversational examples...")
            
            for i, (question, answer) in enumerate(pairs):
                try:
                    self.engine.learn_from_interaction(question, answer)
                    stats["trained_pairs"] += 1
                    
                    if verbose and (i + 1) % 10 == 0:
                        print(f"  [{i + 1}/{len(pairs)}] Conversational patterns learned")
                except Exception as e:
                    self.logger.warning(f"Failed to train on pair {i}: {e}")
                    stats["failed_pairs"] += 1
            
            stats["status"] = "completed"
            self.logger.info(f"Conversational training complete: {stats['trained_pairs']}/{stats['total_pairs']} pairs")
            
        except Exception as e:
            stats["status"] = "error"
            self.logger.error(f"Conversational training failed: {e}")
        
        stats["end_time"] = datetime.now()
        stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()
        
        return stats
    
    def _load_from_csv(self, csv_path: str) -> List[Tuple[str, str]]:
        """Load conversation pairs from CSV file."""
        pairs = []
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = (row.get("Question") or row.get("question") or "").strip()
                    answer = (row.get("Answer") or row.get("answer") or "").strip()
                    if question and answer:
                        pairs.append((question, answer))
            
            self.logger.info(f"Loaded {len(pairs)} conversation pairs from CSV")
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}")
        
        return pairs
    
    def enhance_chat_quality(self) -> None:
        """
        Train additional patterns that improve chat quality.
        Includes conversation flow, context awareness, and natural responses.
        """
        quality_improvements = [
            # Conversational flow
            ("Let's start fresh", "Understood. What would you like to discuss?"),
            ("Going back to what we talked about earlier", "Good, let me recall our previous discussion and build on it."),
            ("Let's move to the next topic", "Sure, I'm ready for the next topic."),
            
            # Context awareness
            ("Given what we've discussed", "Based on our conversation, here's what I think..."),
            ("In the context of AI security", "Focusing specifically on AI security, here's the key point..."),
            ("You mentioned earlier that", "Right, building on that point you raised..."),
            
            # Natural responses
            ("Thanks for asking", "You're welcome. Good questions lead to better understanding."),
            ("I appreciate your perspective", "Your viewpoint adds valuable context. Let me consider that..."),
            ("That was helpful", "I'm glad. This kind of dialogue helps refine thinking."),
            
            # Maintaining conversation
            ("What do you think?", "Considering the tradeoffs, what's your assessment?"),
            ("Does that make sense?", "Does this explanation clarify things, or should I explain differently?"),
            ("Any questions about that?", "Feel free to ask if any part needs clarification or deeper exploration."),
        ]
        
        for question, answer in quality_improvements:
            try:
                self.engine.learn_from_interaction(question, answer)
            except Exception as e:
                self.logger.warning(f"Failed to enhance quality: {e}")
        
        self.logger.info("Chat quality enhancement complete")


def create_conversational_trainer(engine) -> ConversationalTrainer:
    """Factory function to create conversational trainer."""
    return ConversationalTrainer(engine)
