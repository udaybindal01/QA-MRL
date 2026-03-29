"""
Educational dataset: corpus builder, training pairs, PyTorch datasets, dataloaders.

All dataset classes and the corpus builder live here for single-import convenience.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ─────────────────────────── Data structures ──────────────────────────────

@dataclass
class Passage:
    id: str
    text: str
    subject: str
    topic: str
    bloom_level: int
    source: str
    difficulty: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class QueryPassagePair:
    query: str
    positive: Passage
    negatives: List[Passage]
    bloom_level: int
    subject: str
    query_type: str
    learner_profile: Optional[Dict] = None


SUBJECT_TO_IDX = {
    "biology": 0, "chemistry": 1, "physics": 2,
    "mathematics": 3, "computer_science": 4,
}

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply",
               4: "Analyze", 5: "Evaluate", 6: "Create"}


# ─────────────────────────── Corpus Builder ───────────────────────────────

class EducationalCorpusBuilder:
    """
    Builds a multi-level educational corpus.

    Same concept appears at different Bloom's / difficulty levels,
    enabling evaluation of level-appropriate retrieval.
    """

    BLOOM_DESCRIPTORS = {
        1: ["define", "list", "recall", "identify", "name"],
        2: ["explain", "describe", "summarize", "paraphrase", "classify"],
        3: ["solve", "calculate", "use", "demonstrate", "apply"],
        4: ["compare", "contrast", "examine", "differentiate", "analyse"],
        5: ["judge", "critique", "justify", "assess", "evaluate"],
        6: ["design", "construct", "develop", "formulate", "create"],
    }

    TOPICS = {
        "biology": [
            "photosynthesis", "cellular_respiration", "mitosis", "meiosis",
            "dna_replication", "protein_synthesis", "evolution", "ecology",
            "genetics", "cell_structure", "immune_system", "nervous_system",
        ],
        "chemistry": [
            "atomic_structure", "chemical_bonding", "thermodynamics",
            "reaction_kinetics", "acid_base", "electrochemistry",
            "organic_chemistry", "equilibrium", "periodic_table",
        ],
        "physics": [
            "newtons_laws", "thermodynamics", "electromagnetism",
            "quantum_mechanics", "relativity", "waves", "optics",
            "nuclear_physics", "fluid_dynamics",
        ],
        "mathematics": [
            "calculus", "linear_algebra", "probability", "statistics",
            "differential_equations", "number_theory", "graph_theory",
            "topology", "abstract_algebra",
        ],
        "computer_science": [
            "algorithms", "data_structures", "machine_learning",
            "neural_networks", "operating_systems", "databases",
            "compilers", "cryptography", "distributed_systems",
        ],
    }

    def __init__(self, config: dict):
        self.config = config
        self.corpus: List[Passage] = []
        self.topic_to_passages: Dict[str, List[Passage]] = {}

    def build_synthetic_corpus(self, num_passages_per_topic_level: int = 5) -> List[Passage]:
        """Build synthetic multi-level corpus for development."""
        pid = 0
        diff_map = {1: "beginner", 2: "beginner", 3: "intermediate",
                    4: "intermediate", 5: "advanced", 6: "expert"}
        src_map = {1: "simple_wiki", 2: "wikipedia", 3: "textbook",
                   4: "textbook", 5: "paper", 6: "arxiv"}
        templates = {
            1: "{topic_c} is a fundamental concept in {subject}. Key terms to {verb} "
               "include the basic definitions, components, and vocabulary used when "
               "discussing {topic}.",
            2: "To {verb} {topic}, one must consider how the process functions step by "
               "step. The underlying mechanism involves several stages that interact with "
               "each other. Grasping the relationships between components is essential "
               "for building a strong conceptual foundation in {subject}.",
            3: "Applying knowledge of {topic} allows one to {verb} real-world problems "
               "in {subject}. Practical exercises include working through calculations, "
               "following procedures, and interpreting experimental results related to "
               "{topic} in laboratory and field settings.",
            4: "Analyzing {topic} requires examining the underlying mechanisms in detail. "
               "To {verb} the system, one must compare different components, identify "
               "cause-effect relationships, and break the process into constituent parts. "
               "A deeper look reveals subtleties that introductory treatments omit.",
            5: "Critically evaluating research on {topic} demands assessing experimental "
               "methodology, statistical rigor, and potential confounds. To {verb} claims "
               "about {topic}, one must weigh evidence from multiple studies, consider "
               "alternative explanations, and understand the limits of current knowledge.",
            6: "Synthesizing novel approaches to {topic} requires integrating knowledge "
               "across {subject} sub-fields. To {verb} new experimental designs or "
               "theoretical frameworks, one must combine established principles in "
               "innovative ways and identify gaps in the current literature.",
        }

        for subject, topics in self.TOPICS.items():
            for topic in topics:
                for bloom in range(1, 7):
                    descs = self.BLOOM_DESCRIPTORS[bloom]
                    for i in range(num_passages_per_topic_level):
                        verb = descs[i % len(descs)]
                        text = templates[bloom].format(
                            topic=topic.replace("_", " "),
                            topic_c=topic.replace("_", " ").title(),
                            subject=subject,
                            verb=verb,
                        )
                        p = Passage(
                            id=f"p_{pid}",
                            text=text,
                            subject=subject,
                            topic=topic,
                            bloom_level=bloom,
                            source=src_map[bloom],
                            difficulty=diff_map[bloom],
                            metadata={"variant": i},
                        )
                        self.corpus.append(p)
                        key = f"{subject}_{topic}"
                        self.topic_to_passages.setdefault(key, []).append(p)
                        pid += 1
        return self.corpus

    def generate_training_pairs(self, num_pairs=10000, num_hard_negatives=7):
        """Generate query-passage training pairs with hard negatives."""
        query_templates = {
            "factual": ["What is {topic}?", "Define {topic} in {subject}.",
                        "List the components of {topic}."],
            "conceptual": ["Explain how {topic} works.",
                           "Describe the mechanism of {topic}.",
                           "Why is {topic} important in {subject}?"],
            "procedural": ["How do you solve problems involving {topic}?",
                           "Apply {topic} to a real-world scenario.",
                           "Calculate results related to {topic}."],
            "metacognitive": ["Compare different approaches to {topic}.",
                              "Evaluate the evidence for {topic}.",
                              "Design an experiment to test {topic}."],
        }
        qt_to_bloom = {
            "factual": [1, 2], "conceptual": [2, 3],
            "procedural": [3, 4], "metacognitive": [4, 5, 6],
        }

        pairs = []
        for _ in range(num_pairs):
            subject = random.choice(list(self.TOPICS.keys()))
            topic = random.choice(self.TOPICS[subject])
            key = f"{subject}_{topic}"
            if key not in self.topic_to_passages:
                continue

            passages = self.topic_to_passages[key]
            qt = random.choice(list(query_templates.keys()))
            bloom = random.choice(qt_to_bloom[qt])
            query = random.choice(query_templates[qt]).format(
                topic=topic.replace("_", " "), subject=subject
            )

            # Positive: matching Bloom level
            positives = [p for p in passages if p.bloom_level == bloom]
            if not positives:
                positives = [p for p in passages if abs(p.bloom_level - bloom) <= 1]
            if not positives:
                continue
            positive = random.choice(positives)

            # Hard negatives: same topic different Bloom + cross-topic
            hard_negs = [p for p in passages if p.bloom_level != bloom]
            cross = []
            for t in self.TOPICS[subject]:
                k2 = f"{subject}_{t}"
                if k2 != key and k2 in self.topic_to_passages:
                    cross.extend(self.topic_to_passages[k2])
            cross_sample = random.sample(cross, min(num_hard_negatives // 2, len(cross)))
            negatives = (hard_negs + cross_sample)[:num_hard_negatives]
            while len(negatives) < num_hard_negatives:
                negatives.append(random.choice(self.corpus))

            pairs.append(QueryPassagePair(
                query=query, positive=positive, negatives=negatives,
                bloom_level=bloom, subject=subject, query_type=qt,
            ))
        return pairs

    def save_corpus(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for p in self.corpus:
                f.write(json.dumps({
                    "id": p.id, "text": p.text, "subject": p.subject,
                    "topic": p.topic, "bloom_level": p.bloom_level,
                    "source": p.source, "difficulty": p.difficulty,
                }) + "\n")

    def save_pairs(self, pairs, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps({
                    "query": pair.query,
                    "positive_id": pair.positive.id,
                    "positive_text": pair.positive.text,
                    "negative_ids": [n.id for n in pair.negatives],
                    "negative_texts": [n.text for n in pair.negatives],
                    "bloom_level": pair.bloom_level,
                    "subject": pair.subject,
                    "query_type": pair.query_type,
                }) + "\n")


# ─────────────────────── PyTorch Datasets ─────────────────────────────────

class EducationalRetrievalDataset(Dataset):
    """Training dataset: query + positive + hard negatives + labels."""

    def __init__(self, data_path, tokenizer, max_query_length=128,
                 max_passage_length=256, num_hard_negatives=7):
        self.tokenizer = tokenizer
        self.max_q = max_query_length
        self.max_p = max_passage_length
        self.num_neg = num_hard_negatives

        self.samples = []
        with open(data_path) as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        q_enc = self.tokenizer(s["query"], max_length=self.max_q,
                               padding="max_length", truncation=True, return_tensors="pt")
        p_enc = self.tokenizer(s["positive_text"], max_length=self.max_p,
                               padding="max_length", truncation=True, return_tensors="pt")

        neg_texts = s["negative_texts"][:self.num_neg]
        while len(neg_texts) < self.num_neg:
            neg_texts.append(neg_texts[-1] if neg_texts else s["positive_text"])
        n_enc = self.tokenizer(neg_texts, max_length=self.max_p,
                               padding="max_length", truncation=True, return_tensors="pt")

        bloom = s["bloom_level"] - 1  # Query Bloom level (0-indexed)
        subject = SUBJECT_TO_IDX.get(s["subject"], 0)
        lf = torch.zeros(6)
        lf[bloom] = 1.0

        return {
            "query_input_ids": q_enc["input_ids"].squeeze(0),
            "query_attention_mask": q_enc["attention_mask"].squeeze(0),
            "positive_input_ids": p_enc["input_ids"].squeeze(0),
            "positive_attention_mask": p_enc["attention_mask"].squeeze(0),
            "negative_input_ids": n_enc["input_ids"],
            "negative_attention_mask": n_enc["attention_mask"],
            "bloom_label": torch.tensor(bloom, dtype=torch.long),  # Query-only
            "subject_label": torch.tensor(subject, dtype=torch.long),
            "learner_features": lf,
            # v3: No negative_blooms — docs have no Bloom labels
        }


class CorpusDataset(Dataset):
    """Dataset for encoding the full corpus at evaluation time."""

    def __init__(self, corpus_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.passages = []
        with open(corpus_path) as f:
            for line in f:
                self.passages.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        p = self.passages[idx]
        enc = self.tokenizer(p["text"], max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "passage_id": p["id"],
            "bloom_level": p.get("bloom_level", 0),  # Legacy compat, may not exist in v3 data
            "subject": p["subject"],
            "topic": p.get("topic", ""),
        }


# ─────────────────────── DataLoader Builder ───────────────────────────────

def build_dataloaders(config: dict, tokenizer) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders from config."""
    dc = config["data"]
    tc = config["training"]

    loaders = {}
    for split, path in [("train", dc["train_path"]),
                        ("val", dc["val_path"]),
                        ("test", dc["test_path"])]:
        if os.path.exists(path):
            ds = EducationalRetrievalDataset(
                data_path=path,
                tokenizer=tokenizer,
                max_query_length=dc["max_query_length"],
                max_passage_length=dc["max_passage_length"],
                num_hard_negatives=dc["num_hard_negatives"],
            )
            loaders[split] = DataLoader(
                ds,
                batch_size=tc["batch_size"],
                shuffle=(split == "train"),
                num_workers=4,
                pin_memory=True,
                drop_last=(split == "train"),
            )
    return loaders
