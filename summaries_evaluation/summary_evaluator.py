"""
summary_evaluator.py

A fully automated, text-type-aware evaluator for measuring how much
informational content a summary preserves from a source text.

Architecture
------------
Source + summary
    -> automatic text-type router (definition / scientific / non-scientific)
    -> type-specific claim extraction and evidence handling
    -> shared semantic, NLI and QA backbone
    -> type-specific diagnostics
    -> coverage + faithfulness
    -> harmonic-mean information score
    -> evaluator disagreement / uncertainty

The design deliberately keeps a set of common raw metrics identical across
text types. Type-specific components are added as diagnostics and as optional
inputs to the final score. The default aggregation weights are heuristics;
for research use, calibrate them against human judgments using --calibration.

English is the primary supported language because the default QG/QA/NLI
checkpoints are English models.

Example
-------
python revised_summary_evaluator.py \
    --source source.txt \
    --summary summary.txt \
    --output report.json \
    --text-type auto \
    --profile balanced

Batch CSV:
python revised_summary_evaluator.py \
    --batch input.csv \
    --batch-output metrics.csv \
    --details-output details.jsonl \
    --source-column source \
    --summary-column summary \
    --text-type-column text_type
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

TextType = Literal["definition", "scientific", "non_scientific"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    general_nli_model: str = "cross-encoder/nli-deberta-v3-base"
    scientific_nli_model: str = "tasksource/ModernBERT-base-nli"
    # scientific_nli_model: str = "shidey/deberta-v3-mednli-scifact-open-sentence-nli"
    generator_model: str = "google/flan-t5-base"
    qg_model: str = "valhalla/t5-small-qg-hl"
    qa_model: str = "deepset/roberta-base-squad2"

    # Fallback label orders are used only if a model config does not expose
    # meaningful id2label entries.
    general_nli_fallback_order: Tuple[str, str, str] = (
        "contradiction", "entailment", "neutral"
    )
    scientific_nli_fallback_order: Tuple[str, str, str] = (
        "entailment", "neutral", "contradiction"
    )


@dataclass
class EvaluationConfig:
    models: ModelConfig = field(default_factory=ModelConfig)

    profile: str = "balanced"
    text_type: str = "auto"
    seed: int = 13

    # Segmentation / retrieval
    paragraph_char_limit: int = 2200
    evidence_top_k: int = 3
    scientific_top_paragraphs: int = 3
    scientific_top_windows: int = 3
    window_radius: int = 1

    # Claim extraction
    generative_claim_extraction: bool = True
    validate_generated_claims: bool = True
    min_claim_validation_support: float = 0.20
    max_claims_source: int = 50
    max_claims_summary: int = 30
    generation_max_new_tokens: int = 192

    # QA
    qa_max_questions_definition: int = 5
    qa_max_questions_scientific: int = 10
    qa_max_questions_non_scientific: int = 8
    qa_min_source_confidence: float = 0.06
    qa_min_target_match: float = 0.15

    # Router
    router_use_zero_shot: bool = True
    router_heuristic_weight: float = 0.35
    router_model_weight: float = 0.65

    # Generic support score: entailment * (1 - contradiction)
    contradiction_penalty: float = 1.0

    # Common coverage weights. These are intentionally modest defaults, not
    # learned/calibrated values.
    common_coverage_nli_weight: float = 0.60
    common_coverage_qa_weight: float = 0.40

    # Final per-profile aggregation. Can be overridden by calibration.
    definition_coverage_weights: Dict[str, float] = field(default_factory=lambda: {
        "common_nli": 0.50,
        "qa": 0.20,
        "slot_coverage": 0.30,
    })
    definition_faithfulness_weights: Dict[str, float] = field(default_factory=lambda: {
        "common_nli": 0.70,
        "slot_consistency": 0.30,
    })

    scientific_coverage_weights: Dict[str, float] = field(default_factory=lambda: {
        "general_nli": 0.35,
        "scientific_nli": 0.30,
        "qa": 0.25,
        "semantic_recall": 0.10,
    })
    scientific_faithfulness_weights: Dict[str, float] = field(default_factory=lambda: {
        "general_nli": 0.35,
        "scientific_nli": 0.35,
        "numeric_consistency": 0.15,
        "qualifier_consistency": 0.15,
    })

    nonscientific_coverage_weights: Dict[str, float] = field(default_factory=lambda: {
        "common_nli": 0.45,
        "qa": 0.35,
        "discourse_coverage": 0.20,
    })
    nonscientific_faithfulness_weights: Dict[str, float] = field(default_factory=lambda: {
        "common_nli": 0.55,
        "discourse_consistency": 0.45,
    })

    calibration_path: Optional[str] = None

    @classmethod
    def from_profile(cls, name: str) -> "EvaluationConfig":
        name = name.lower().strip()
        cfg = cls(profile=name)
        if name == "fast":
            cfg.models.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            cfg.models.general_nli_model = "cross-encoder/nli-deberta-v3-xsmall"
            cfg.models.generator_model = "google/flan-t5-small"
            cfg.evidence_top_k = 2
            cfg.scientific_top_paragraphs = 2
            cfg.scientific_top_windows = 2
            cfg.max_claims_source = 30
            cfg.max_claims_summary = 20
            cfg.qa_max_questions_scientific = 6
            cfg.qa_max_questions_non_scientific = 6
            cfg.qa_max_questions_definition = 4
        elif name == "balanced":
            pass
        elif name == "research":
            cfg.models.embedding_model = "sentence-transformers/all-mpnet-base-v2"
            cfg.models.general_nli_model = "cross-encoder/nli-deberta-v3-large"
            cfg.models.generator_model = "google/flan-t5-large"
            cfg.evidence_top_k = 4
            cfg.scientific_top_paragraphs = 4
            cfg.scientific_top_windows = 4
            cfg.max_claims_source = 70
            cfg.max_claims_summary = 40
            cfg.qa_max_questions_scientific = 14
            cfg.qa_max_questions_non_scientific = 10
            cfg.qa_max_questions_definition = 6
        else:
            raise ValueError("profile must be one of: fast, balanced, research")
        return cfg


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clip01(x: float) -> float:
    if x is None or not np.isfinite(x):
        return float("nan")
    return float(max(0.0, min(1.0, x)))


def harmonic_mean(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    if a <= 0 or b <= 0:
        return 0.0
    return float(2 * a * b / (a + b))


def safe_mean(xs: Iterable[float], default: float = float("nan")) -> float:
    vals = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(vals)) if vals else default


def weighted_mean(values: Dict[str, float], weights: Dict[str, float]) -> float:
    pairs = [
        (float(values[k]), float(w))
        for k, w in weights.items()
        if k in values and values[k] is not None and np.isfinite(values[k]) and w > 0
    ]
    if not pairs:
        return float("nan")
    denom = sum(w for _, w in pairs)
    return float(sum(v * w for v, w in pairs) / denom)


def normalized_disagreement(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return 0.0
    diffs = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            diffs.append(abs(vals[i] - vals[j]))
    return clip01(float(np.mean(diffs))) if diffs else 0.0


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_token_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9%]+", (text or "").lower())


def token_f1(a: str, b: str) -> float:
    ta = normalize_token_text(a)
    tb = normalize_token_text(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(tb)
    r = overlap / len(ta)
    return 2 * p * r / (p + r)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# NLP segmentation
# ---------------------------------------------------------------------------

class TextProcessor:
    def __init__(self) -> None:
        self.nlp = None
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = spacy.blank("en")
                self.nlp.add_pipe("sentencizer")
        except Exception as exc:
            warnings.warn(f"spaCy unavailable; using regex sentence splitting: {exc}")

    def sentences(self, text: str) -> List[str]:
        text = normalize_space(text)
        if not text:
            return []
        if self.nlp is not None:
            try:
                return [
                    normalize_space(sent.text)
                    for sent in self.nlp(text).sents
                    if normalize_space(sent.text)
                ]
            except Exception:
                pass
        return [
            normalize_space(x)
            for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
            if normalize_space(x)
        ]

    def paragraphs(self, text: str, char_limit: int = 2200) -> List[str]:
        raw = [
            normalize_space(p)
            for p in re.split(r"\n\s*\n+", text or "")
            if normalize_space(p)
        ]
        if not raw:
            raw = self.sentences(text)

        paragraphs: List[str] = []
        for p in raw:
            if len(p) <= char_limit:
                paragraphs.append(p)
                continue
            sents = self.sentences(p)
            current: List[str] = []
            size = 0
            for s in sents:
                if current and size + len(s) + 1 > char_limit:
                    paragraphs.append(" ".join(current))
                    current, size = [], 0
                current.append(s)
                size += len(s) + 1
            if current:
                paragraphs.append(" ".join(current))
        return paragraphs

    def entities(self, text: str) -> List[str]:
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                ents = [normalize_space(e.text) for e in getattr(doc, "ents", [])]
                if ents:
                    return list(dict.fromkeys(ents))
            except Exception:
                pass
        # Conservative fallback: proper-name sequences.
        return list(dict.fromkeys(
            re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text or "")
        ))

    def noun_phrases(self, text: str) -> List[str]:
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                chunks = [normalize_space(c.text) for c in doc.noun_chunks]
                if chunks:
                    return list(dict.fromkeys(chunks))
            except Exception:
                pass
        # Fallback approximate noun-like spans.
        return list(dict.fromkeys(
            re.findall(
                r"\b(?:[A-Za-z][\w-]*\s+){0,3}(?:method|effect|result|study|system|"
                r"process|treatment|group|model|crop|machine|organization|decision|"
                r"policy|change|increase|decrease|definition|concept)\b",
                text or "",
                flags=re.I,
            )
        ))


# ---------------------------------------------------------------------------
# Lazy model hub
# ---------------------------------------------------------------------------

class NLIModel:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        fallback_order: Sequence[str],
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.fallback_order = [x.lower() for x in fallback_order]
        self.label_by_id = self._resolve_labels()

    def _resolve_labels(self) -> Dict[int, str]:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        resolved: Dict[int, str] = {}
        meaningful = False
        for idx, label in id2label.items():
            lab = str(label).lower().strip()
            if any(k in lab for k in ["entail", "neutral", "contrad"]):
                meaningful = True
            resolved[int(idx)] = lab
        if meaningful:
            return resolved
        return {i: label for i, label in enumerate(self.fallback_order)}

    def probabilities(self, premise: str, hypothesis: str) -> Dict[str, float]:
        premise = normalize_space(premise)
        hypothesis = normalize_space(hypothesis)
        if not premise or not hypothesis:
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

        max_length = getattr(self.tokenizer, "model_max_length", 512)
        if max_length is None or max_length > 4096:
            max_length = 512
        max_length = int(min(max_length, 2048))

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(**inputs).logits[0], dim=-1).detach().cpu().numpy()

        result = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for idx, p in enumerate(probs):
            label = self.label_by_id.get(idx, "").lower()
            if "entail" in label:
                result["entailment"] += float(p)
            elif "contrad" in label:
                result["contradiction"] += float(p)
            elif "neutral" in label:
                result["neutral"] += float(p)

        total = sum(result.values())
        if total <= 0:
            # Last-resort mapping.
            order = self.fallback_order
            result = {k: 0.0 for k in result}
            for idx, p in enumerate(probs[:len(order)]):
                key = order[idx]
                if key in result:
                    result[key] = float(p)
            total = sum(result.values())
        if total > 0:
            result = {k: float(v / total) for k, v in result.items()}
        return result


class ModelHub:
    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._embedding = None
        self._general_nli = None
        self._scientific_nli = None
        self._generator_tokenizer = None
        self._generator_model = None
        self._qg_tokenizer = None
        self._qg_model = None
        self._qa_pipeline = None
        self._router_pipeline = None

    @property
    def embedding(self) -> SentenceTransformer:
        if self._embedding is None:
            self._embedding = SentenceTransformer(
                self.cfg.models.embedding_model,
                device=str(self.device),
            )
        return self._embedding

    @property
    def general_nli(self) -> NLIModel:
        if self._general_nli is None:
            self._general_nli = NLIModel(
                self.cfg.models.general_nli_model,
                self.device,
                self.cfg.models.general_nli_fallback_order,
            )
        return self._general_nli

    @property
    def scientific_nli(self) -> NLIModel:
        if self._scientific_nli is None:
            self._scientific_nli = NLIModel(
                self.cfg.models.scientific_nli_model,
                self.device,
                self.cfg.models.scientific_nli_fallback_order,
            )
        return self._scientific_nli

    @property
    def generator(self):
        if self._generator_model is None:
            self._generator_tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.models.generator_model
            )
            self._generator_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.cfg.models.generator_model
            ).to(self.device)
            self._generator_model.eval()
        return self._generator_tokenizer, self._generator_model

    @property
    def qg(self):
        if self._qg_model is None:
            self._qg_tokenizer = AutoTokenizer.from_pretrained(self.cfg.models.qg_model)
            self._qg_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.cfg.models.qg_model
            ).to(self.device)
            self._qg_model.eval()
        return self._qg_tokenizer, self._qg_model

    @property
    def qa(self):
        if self._qa_pipeline is None:
            tok = AutoTokenizer.from_pretrained(self.cfg.models.qa_model)
            model = AutoModelForQuestionAnswering.from_pretrained(self.cfg.models.qa_model)
            self._qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tok,
                device=0 if self.device.type == "cuda" else -1,
                handle_impossible_answer=True,
            )
        return self._qa_pipeline

    @property
    def router(self):
        if self._router_pipeline is None:
            # Reuse the general NLI checkpoint for zero-shot classification.
            self._router_pipeline = pipeline(
                "zero-shot-classification",
                model=self.cfg.models.general_nli_model,
                device=0 if self.device.type == "cuda" else -1,
            )
        return self._router_pipeline

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        emb = self.embedding.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        emb = self.encode([a, b])
        return clip01(float(np.dot(emb[0], emb[1])))

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        tok, model = self.generator
        max_new_tokens = max_new_tokens or self.cfg.generation_max_new_tokens
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(getattr(tok, "model_max_length", 512), 1024),
        ).to(self.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
            )
        return normalize_space(tok.decode(out[0], skip_special_tokens=True))

    def generate_question(self, context: str, answer: str) -> str:
        tok, model = self.qg
        context = normalize_space(context)
        answer = normalize_space(answer)
        if not answer:
            return ""
        # Highlight the first case-insensitive occurrence.
        pattern = re.compile(re.escape(answer), flags=re.I)
        marked, n = pattern.subn(lambda m: f"<hl> {m.group(0)} <hl>", context, count=1)
        if n == 0:
            marked = f"<hl> {answer} <hl> {context}"
        prompt = f"{marked} </s>"
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(getattr(tok, "model_max_length", 512), 512),
        ).to(self.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=2,
            )
        q = normalize_space(tok.decode(out[0], skip_special_tokens=True))
        if q and not q.endswith("?"):
            q += "?"
        return q


# ---------------------------------------------------------------------------
# Text type routing
# ---------------------------------------------------------------------------

class TextTypeRouter:
    SCIENTIFIC_TERMS = {
        "study", "studies", "experiment", "experimental", "method", "methods",
        "result", "results", "analysis", "sample", "participants", "dataset",
        "statistical", "significant", "significance", "hypothesis", "model",
        "measured", "measurement", "treatment", "control", "observed",
        "confidence interval", "p-value", "p value", "correlation", "regression",
        "methodology", "findings", "conclusion",
    }

    def __init__(self, hub: ModelHub, processor: TextProcessor, cfg: EvaluationConfig):
        self.hub = hub
        self.processor = processor
        self.cfg = cfg

    def heuristic_scores(self, source: str) -> Dict[str, float]:
        text = normalize_space(source)
        low = text.lower()
        wc = max(word_count(text), 1)
        sents = self.processor.sentences(text)

        definition = 0.0
        scientific = 0.0

        # Definition cues
        if wc <= 120:
            definition += 0.25
        if wc <= 80:
            definition += 0.15
        first = sents[0] if sents else text
        if re.search(
            r"\b(?:is|are|means|refers to|is defined as|can be defined as)\b",
            first,
            flags=re.I,
        ):
            definition += 0.30
        if len(sents) <= 3:
            definition += 0.15
        if ":" in first and wc <= 120:
            definition += 0.05

        # Scientific cues
        term_hits = sum(1 for term in self.SCIENTIFIC_TERMS if term in low)
        scientific += min(0.40, 0.04 * term_hits)
        if re.search(r"\bp\s*[<=>]\s*0?\.\d+", low):
            scientific += 0.20
        if re.search(r"\b\d+(?:\.\d+)?\s*%", low):
            scientific += 0.08
        if re.search(r"\[[0-9,\s-]+\]|\([A-Z][A-Za-z-]+ et al\.,? \d{4}\)", text):
            scientific += 0.15
        if any(h in low for h in ["methods:", "results:", "discussion:", "abstract:"]):
            scientific += 0.20
        if wc >= 150:
            scientific += 0.05

        definition = clip01(definition)
        scientific = clip01(scientific)
        non_scientific = max(0.0, 1.0 - max(definition, scientific))

        scores = {
            "definition": definition,
            "scientific": scientific,
            "non_scientific": non_scientific,
        }
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def model_scores(self, source: str) -> Dict[str, float]:
        labels = [
            "a dictionary or ontology definition",
            "scientific or technical research prose",
            "general non-scientific prose",
        ]
        result = self.hub.router(
            normalize_space(source)[:8000],
            candidate_labels=labels,
            hypothesis_template="This text is {}.",
            multi_label=False,
        )
        mapping = {
            labels[0]: "definition",
            labels[1]: "scientific",
            labels[2]: "non_scientific",
        }
        scores = {k: 0.0 for k in mapping.values()}
        for label, score in zip(result["labels"], result["scores"]):
            scores[mapping[label]] = float(score)
        return scores

    def route(self, source: str, override: str = "auto") -> Dict[str, Any]:
        if override and override != "auto":
            norm = override.lower().replace("-", "_").replace(" ", "_")
            aliases = {
                "definition": "definition",
                "definitions": "definition",
                "scientific": "scientific",
                "scientific_text": "scientific",
                "scientific_texts": "scientific",
                "non_scientific": "non_scientific",
                "nonscientific": "non_scientific",
                "non_scientific_text": "non_scientific",
                "non_scientific_texts": "non_scientific",
            }
            if norm not in aliases:
                raise ValueError(f"Unknown text type: {override}")
            selected = aliases[norm]
            return {
                "selected": selected,
                "confidence": 1.0,
                "source": "manual_override",
                "scores": {selected: 1.0},
            }

        h = self.heuristic_scores(source)
        if not self.cfg.router_use_zero_shot:
            selected = max(h, key=h.get)
            return {
                "selected": selected,
                "confidence": h[selected],
                "source": "heuristics",
                "scores": h,
            }

        try:
            m = self.model_scores(source)
            combined = {}
            for k in h:
                combined[k] = (
                    self.cfg.router_heuristic_weight * h[k]
                    + self.cfg.router_model_weight * m[k]
                )
            total = sum(combined.values()) or 1.0
            combined = {k: v / total for k, v in combined.items()}
            selected = max(combined, key=combined.get)
            return {
                "selected": selected,
                "confidence": combined[selected],
                "source": "heuristic_plus_zero_shot",
                "scores": combined,
                "heuristic_scores": h,
                "model_scores": m,
            }
        except Exception as exc:
            selected = max(h, key=h.get)
            return {
                "selected": selected,
                "confidence": h[selected],
                "source": "heuristics_fallback",
                "scores": h,
                "warning": str(exc),
            }


# ---------------------------------------------------------------------------
# Claim extraction and salience
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    claim_id: str
    text: str
    source_index: int
    source_unit: str
    weight: float = 1.0


class ClaimExtractor:
    def __init__(self, hub: ModelHub, processor: TextProcessor, cfg: EvaluationConfig):
        self.hub = hub
        self.processor = processor
        self.cfg = cfg

    def _prompt(self, text_type: TextType, unit: str) -> str:
        if text_type == "definition":
            return (
                "Extract the minimal self-contained factual claims from this definition. "
                "Preserve the superordinate category, essential properties, function, "
                "conditions, restrictions, exclusions, quantities, and negation. "
                "Do not add facts. Return one claim per line, each beginning with '- '.\n\n"
                f"DEFINITION:\n{unit}"
            )
        if text_type == "scientific":
            return (
                "Extract minimal self-contained scientific claims from the passage. "
                "Every claim must preserve population/sample, intervention or exposure, "
                "comparator, outcome, numerical values and units, direction of effect, "
                "statistical significance, uncertainty/hedging, temporal conditions, "
                "and experimental conditions when present. Do not infer missing facts. "
                "Return one claim per line, each beginning with '- '.\n\n"
                f"PASSAGE:\n{unit}"
            )
        return (
            "Extract factual event claims from the passage. Make each claim self-contained. "
            "Resolve pronouns to named referents when the passage makes the referent clear. "
            "Preserve actor, action, object, time, place, attribution, modality, and causal "
            "or temporal relations. Do not split a causal, temporal, or contrastive relation "
            "if splitting would change its meaning. Do not add facts. "
            "Return one claim per line, each beginning with '- '.\n\n"
            f"PASSAGE:\n{unit}"
        )

    @staticmethod
    def _parse_lines(raw: str) -> List[str]:
        if not raw:
            return []
        # Normalize separators that instruction models commonly emit.
        text = raw.replace("•", "\n- ").replace(" | ", "\n- ")
        lines = []
        for line in text.splitlines():
            line = normalize_space(re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line))
            if len(line.split()) >= 3:
                lines.append(line)
        if not lines and len(raw.split()) >= 3:
            # Occasionally T5 emits multiple claims in one sentence-like string.
            lines = [
                normalize_space(x)
                for x in re.split(r"\s*;\s*|(?<=[.!?])\s+(?=[A-Z])", raw)
                if len(normalize_space(x).split()) >= 3
            ]
        return list(dict.fromkeys(lines))

    def extract(self, text: str, text_type: TextType, prefix: str, is_summary: bool) -> List[Claim]:
        if text_type == "definition":
            units = [normalize_space(text)]
        else:
            units = self.processor.paragraphs(text, self.cfg.paragraph_char_limit)

        max_claims = self.cfg.max_claims_summary if is_summary else self.cfg.max_claims_source
        claims: List[Claim] = []

        for idx, unit in enumerate(units):
            candidates: List[str] = []
            if self.cfg.generative_claim_extraction:
                try:
                    raw = self.hub.generate(self._prompt(text_type, unit))
                    candidates = self._parse_lines(raw)
                except Exception as exc:
                    warnings.warn(f"Claim generation failed; falling back to sentences: {exc}")

            if not candidates:
                candidates = self.processor.sentences(unit) or [unit]

            # Validate generated claims against the unit from which they were extracted.
            validated: List[str] = []
            for c in candidates:
                if not self.cfg.validate_generated_claims or not self.cfg.generative_claim_extraction:
                    validated.append(c)
                    continue
                try:
                    probs = self.hub.general_nli.probabilities(unit, c)
                    support = probs["entailment"] * (
                        1.0 - self.cfg.contradiction_penalty * probs["contradiction"]
                    )
                    if support >= self.cfg.min_claim_validation_support:
                        validated.append(c)
                except Exception:
                    validated.append(c)

            if not validated:
                validated = self.processor.sentences(unit) or [unit]

            for c in validated:
                if len(claims) >= max_claims:
                    break
                claims.append(Claim(
                    claim_id=f"{prefix}{len(claims)+1:04d}",
                    text=normalize_space(c),
                    source_index=idx,
                    source_unit=unit,
                ))
            if len(claims) >= max_claims:
                break

        return claims


class SalienceWeighter:
    def __init__(self, hub: ModelHub):
        self.hub = hub

    @staticmethod
    def _profile_multiplier(text: str, text_type: TextType) -> float:
        low = text.lower()
        mult = 1.0
        if text_type == "definition":
            if re.search(r"\b(is a|is an|are a|type of|kind of|refers to|defined as)\b", low):
                mult += 0.20
            if any(x in low for x in ["characterized by", "distinguished by", "excluding", "except"]):
                mult += 0.20
        elif text_type == "scientific":
            if any(x in low for x in ["result", "found", "observed", "increased", "decreased",
                                      "significant", "conclude", "conclusion"]):
                mult += 0.25
            if re.search(r"\b\d+(?:\.\d+)?\s*%|\bp\s*[<=>]\s*0?\.\d+", low):
                mult += 0.20
            if any(x in low for x in ["background", "previous studies", "has been widely"]):
                mult -= 0.10
        else:
            if any(x in low for x in ["because", "therefore", "as a result", "led to", "caused"]):
                mult += 0.20
            if any(x in low for x in ["main", "central", "decided", "announced", "approved", "rejected"]):
                mult += 0.10
        return max(0.5, mult)

    def apply(self, claims: List[Claim], text_type: TextType) -> None:
        if not claims:
            return
        texts = [c.text for c in claims]
        emb = self.hub.encode(texts)
        sim = np.clip(emb @ emb.T, -1.0, 1.0)
        if len(claims) == 1:
            centrality = np.array([1.0], dtype=float)
        else:
            centrality = (sim.sum(axis=1) - 1.0) / max(len(claims) - 1, 1)
            centrality = (centrality + 1.0) / 2.0

        raw = []
        n = len(claims)
        lengths = np.array([min(word_count(c.text), 40) / 40 for c in claims])
        for i, c in enumerate(claims):
            position = 1.0 - (i / max(n - 1, 1))
            base = 0.75 * centrality[i] + 0.10 * position + 0.15 * lengths[i]
            raw.append(max(1e-6, base * self._profile_multiplier(c.text, text_type)))

        denom = sum(raw) or 1.0
        for c, w in zip(claims, raw):
            c.weight = float(w / denom)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, hub: ModelHub, processor: TextProcessor, cfg: EvaluationConfig):
        self.hub = hub
        self.processor = processor
        self.cfg = cfg

    def summary_chunks(self, text: str) -> List[str]:
        sents = self.processor.sentences(text)
        if not sents:
            return [normalize_space(text)] if normalize_space(text) else []
        chunks = []
        for i in range(0, len(sents), 2):
            chunks.append(" ".join(sents[i:i+2]))
        return chunks

    def general_chunks(self, text: str) -> List[str]:
        paragraphs = self.processor.paragraphs(text, self.cfg.paragraph_char_limit)
        return paragraphs or self.summary_chunks(text)

    def retrieve(self, query: str, chunks: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
        chunks = [normalize_space(c) for c in chunks if normalize_space(c)]
        if not chunks:
            return []
        q = self.hub.encode([query])[0]
        c = self.hub.encode(chunks)
        scores = c @ q
        order = np.argsort(-scores)[:max(1, min(top_k, len(chunks)))]
        return [
            {"index": int(i), "text": chunks[int(i)], "similarity": float(scores[int(i)])}
            for i in order
        ]

    def scientific_evidence(self, query: str, text: str) -> List[Dict[str, Any]]:
        paragraphs = self.processor.paragraphs(text, self.cfg.paragraph_char_limit)
        if not paragraphs:
            return []
        top_paras = self.retrieve(query, paragraphs, self.cfg.scientific_top_paragraphs)

        windows: List[str] = []
        metadata: List[Tuple[int, int]] = []
        for para_item in top_paras:
            pi = para_item["index"]
            sents = self.processor.sentences(paragraphs[pi])
            if not sents:
                windows.append(paragraphs[pi])
                metadata.append((pi, 0))
                continue
            for si in range(len(sents)):
                lo = max(0, si - self.cfg.window_radius)
                hi = min(len(sents), si + self.cfg.window_radius + 1)
                windows.append(" ".join(sents[lo:hi]))
                metadata.append((pi, si))

        hits = self.retrieve(query, windows, self.cfg.scientific_top_windows)
        for h in hits:
            pi, si = metadata[h["index"]]
            h["paragraph_index"] = pi
            h["center_sentence_index"] = si
        return hits

    def evidence(
        self,
        query: str,
        text: str,
        text_type: TextType,
        text_is_summary: bool = False,
    ) -> List[Dict[str, Any]]:
        if text_type == "definition":
            full = normalize_space(text)
            return [{"index": 0, "text": full, "similarity": self.hub.similarity(query, full)}] if full else []
        if text_type == "scientific" and not text_is_summary:
            return self.scientific_evidence(query, text)
        chunks = self.summary_chunks(text) if text_is_summary else self.general_chunks(text)
        return self.retrieve(query, chunks, self.cfg.evidence_top_k)


# ---------------------------------------------------------------------------
# NLI scoring
# ---------------------------------------------------------------------------

def support_from_probs(probs: Dict[str, float], contradiction_penalty: float = 1.0) -> float:
    return clip01(
        probs.get("entailment", 0.0)
        * max(0.0, 1.0 - contradiction_penalty * probs.get("contradiction", 0.0))
    )


class NLIEvaluator:
    def __init__(self, hub: ModelHub, retriever: Retriever, cfg: EvaluationConfig):
        self.hub = hub
        self.retriever = retriever
        self.cfg = cfg

    def score_claims(
        self,
        claims: Sequence[Claim],
        evidence_text: str,
        text_type: TextType,
        evidence_is_summary: bool,
        scientific_model: bool = False,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        records = []
        scores = []
        weights = []

        model = self.hub.scientific_nli if scientific_model else self.hub.general_nli

        for claim in claims:
            hits = self.retriever.evidence(
                claim.text, evidence_text, text_type, text_is_summary=evidence_is_summary
            )
            if not hits:
                score = 0.0
                records.append({
                    "claim_id": claim.claim_id,
                    "claim": claim.text,
                    "support": score,
                    "evidence": "",
                    "nli": None,
                })
                scores.append(score)
                weights.append(claim.weight)
                continue

            # Test each retrieved span independently and keep the best supported span.
            candidates = []
            for hit in hits:
                probs = model.probabilities(hit["text"], claim.text)
                score = support_from_probs(probs, self.cfg.contradiction_penalty)
                candidates.append((score, probs, hit))
            best = max(candidates, key=lambda x: x[0])
            score, probs, hit = best
            records.append({
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "weight": claim.weight,
                "support": score,
                "evidence": hit["text"],
                "retrieval_similarity": hit.get("similarity"),
                "nli": probs,
            })
            scores.append(score)
            weights.append(claim.weight)

        if not scores:
            return float("nan"), records

        w = np.asarray(weights, dtype=float)
        s = np.asarray(scores, dtype=float)
        if w.sum() <= 0:
            aggregate = float(np.mean(s))
        else:
            aggregate = float(np.sum(w * s) / np.sum(w))
        return clip01(aggregate), records


# ---------------------------------------------------------------------------
# Definition-specific structured evaluation
# ---------------------------------------------------------------------------

DEFINITION_SLOT_WEIGHTS = {
    "superordinate_category": 3.0,
    "essential_properties": 3.0,
    "function_or_purpose": 2.0,
    "parts_or_materials": 1.5,
    "conditions_or_restrictions": 2.5,
    "exclusions": 2.5,
}


class DefinitionSlotEvaluator:
    def __init__(self, hub: ModelHub):
        self.hub = hub

    def _extract(self, text: str) -> Dict[str, List[str]]:
        prompt = (
            "Analyze the following definition and output ONLY valid JSON with these keys: "
            "superordinate_category, essential_properties, function_or_purpose, "
            "parts_or_materials, conditions_or_restrictions, exclusions. "
            "Every value must be a JSON list of short strings. Use [] when absent. "
            "Do not infer information that is not explicit.\n\n"
            f"DEFINITION:\n{text}"
        )
        try:
            raw = self.hub.generate(prompt, max_new_tokens=220)
            match = re.search(r"\{.*\}", raw, flags=re.S)
            if match:
                data = json.loads(match.group(0))
                out = {}
                for key in DEFINITION_SLOT_WEIGHTS:
                    val = data.get(key, [])
                    if isinstance(val, str):
                        val = [val] if val.strip() else []
                    if not isinstance(val, list):
                        val = []
                    out[key] = [normalize_space(str(x)) for x in val if normalize_space(str(x))]
                return out
        except Exception:
            pass

        # Conservative fallback: treat full text as an essential-property unit.
        return {
            "superordinate_category": [],
            "essential_properties": [normalize_space(text)] if text.strip() else [],
            "function_or_purpose": [],
            "parts_or_materials": [],
            "conditions_or_restrictions": [],
            "exclusions": [],
        }

    def evaluate(self, source: str, summary: str) -> Dict[str, Any]:
        src = self._extract(source)
        summ = self._extract(summary)

        per_slot = {}
        coverage_num = 0.0
        coverage_den = 0.0

        for slot, weight in DEFINITION_SLOT_WEIGHTS.items():
            src_vals = src.get(slot, [])
            sum_vals = summ.get(slot, [])
            if not src_vals:
                continue
            source_item_scores = []
            for sv in src_vals:
                if not sum_vals:
                    source_item_scores.append(0.0)
                else:
                    source_item_scores.append(max(self.hub.similarity(sv, tv) for tv in sum_vals))
            slot_coverage = safe_mean(source_item_scores, 0.0)
            per_slot[slot] = {
                "source": src_vals,
                "summary": sum_vals,
                "coverage": slot_coverage,
            }
            coverage_num += weight * slot_coverage
            coverage_den += weight

        coverage = coverage_num / coverage_den if coverage_den else float("nan")

        # Summary-to-source slot consistency. Every summary slot item should be
        # semantically aligned with some source item from the same slot.
        faith_scores = []
        faith_weights = []
        for slot, weight in DEFINITION_SLOT_WEIGHTS.items():
            src_vals = src.get(slot, [])
            sum_vals = summ.get(slot, [])
            for tv in sum_vals:
                if src_vals:
                    score = max(self.hub.similarity(tv, sv) for sv in src_vals)
                else:
                    score = 0.0
                faith_scores.append(score)
                faith_weights.append(weight)

        if faith_scores:
            consistency = float(np.average(faith_scores, weights=faith_weights))
        else:
            consistency = float("nan")

        return {
            "source_slots": src,
            "summary_slots": summ,
            "per_slot": per_slot,
            "slot_coverage": clip01(coverage) if np.isfinite(coverage) else float("nan"),
            "slot_consistency": clip01(consistency) if np.isfinite(consistency) else float("nan"),
        }


# ---------------------------------------------------------------------------
# Scientific diagnostics: numbers, units, direction, qualifiers
# ---------------------------------------------------------------------------

NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<sign>[+-]?)"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent|percentage|mg|g|kg|µg|ug|ml|mL|l|L|mm|cm|m|km|"
    r"°C|K|Hz|kHz|MHz|GHz|Pa|kPa|MPa|years?|months?|days?|hours?|minutes?|"
    r"seconds?|nM|mM|µM|uM|mol|mmol|cells?|participants?|subjects?|samples?)?"
    r"(?=$|[\s,.;:)\]])",
    flags=re.I,
)

P_VALUE_RE = re.compile(r"\bp\s*([<=>])\s*(0?\.\d+)", flags=re.I)

INCREASE_WORDS = {"increase", "increased", "increases", "higher", "rise", "rose", "greater", "improved", "improvement"}
DECREASE_WORDS = {"decrease", "decreased", "decreases", "lower", "decline", "declined", "reduced", "reduction", "worse"}
NEGATION_WORDS = {"no", "not", "neither", "nor", "without", "failed"}
HEDGE_LOW = {"may", "might", "could", "possibly", "potentially", "suggest", "suggests", "suggested"}
HEDGE_MED = {"likely", "probably", "appears", "indicates", "associated"}
CERTAIN_WORDS = {"demonstrates", "shows", "proves", "causes", "establishes", "confirmed"}


def _norm_unit(unit: Optional[str]) -> str:
    if not unit:
        return ""
    u = unit.lower()
    aliases = {
        "percent": "%", "percentage": "%", "ml": "ml", "l": "l",
        "ug": "µg", "um": "µm",
    }
    return aliases.get(u, u)


def extract_quantities(text: str) -> List[Dict[str, Any]]:
    out = []
    for m in NUMBER_RE.finditer(text or ""):
        raw_num = m.group("num").replace(",", "")
        try:
            value = float(raw_num)
        except Exception:
            continue
        out.append({
            "value": value,
            "unit": _norm_unit(m.group("unit")),
            "raw": m.group(0),
        })
    for m in P_VALUE_RE.finditer(text or ""):
        out.append({
            "value": float(m.group(2)),
            "unit": "p_value",
            "comparator": m.group(1),
            "raw": m.group(0),
        })
    return out


def quantity_match(q: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> float:
    best = 0.0
    for c in candidates:
        if q.get("unit") and c.get("unit") and q["unit"] != c["unit"]:
            continue
        a, b = q["value"], c["value"]
        tol = max(1e-8, abs(a) * 1e-4)
        numeric = 1.0 if abs(a - b) <= tol else max(0.0, 1.0 - abs(a - b) / max(abs(a), abs(b), 1.0))
        if q.get("unit") == "p_value" and c.get("unit") == "p_value":
            numeric *= 1.0 if q.get("comparator") == c.get("comparator") else 0.5
        best = max(best, numeric)
    return best


def direction_label(text: str) -> str:
    toks = set(normalize_token_text(text))
    inc = bool(toks & INCREASE_WORDS)
    dec = bool(toks & DECREASE_WORDS)
    neg = bool(toks & NEGATION_WORDS)
    if inc and not dec:
        return "increase_negated" if neg else "increase"
    if dec and not inc:
        return "decrease_negated" if neg else "decrease"
    return "unknown"


def certainty_level(text: str) -> int:
    toks = set(normalize_token_text(text))
    if toks & HEDGE_LOW:
        return 0
    if toks & HEDGE_MED:
        return 1
    if toks & CERTAIN_WORDS:
        return 3
    return 2


class ScientificDiagnostics:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def evaluate_summary_claims(self, claims: Sequence[Claim], source: str) -> Dict[str, Any]:
        numeric_records = []
        qualifier_records = []

        for claim in claims:
            hits = self.retriever.scientific_evidence(claim.text, source)
            evidence = " ".join(h["text"] for h in hits[:2])

            q_claim = extract_quantities(claim.text)
            q_ev = extract_quantities(evidence)
            if q_claim:
                per_q = [quantity_match(q, q_ev) for q in q_claim]
                numeric_records.append({
                    "claim_id": claim.claim_id,
                    "claim": claim.text,
                    "evidence": evidence,
                    "quantities": q_claim,
                    "evidence_quantities": q_ev,
                    "score": safe_mean(per_q, 0.0),
                })

            d_claim = direction_label(claim.text)
            d_ev = direction_label(evidence)
            direction_score = 1.0
            if d_claim != "unknown" and d_ev != "unknown":
                direction_score = 1.0 if d_claim == d_ev else 0.0

            c_claim = certainty_level(claim.text)
            c_ev = certainty_level(evidence)
            # Penalize strengthening: summary certainty higher than evidence.
            certainty_score = 1.0 if c_claim <= c_ev else max(0.0, 1.0 - 0.35 * (c_claim - c_ev))

            sig_claim = "significant" in claim.text.lower()
            sig_ev = "significant" in evidence.lower()
            nonsig_claim = bool(re.search(r"\bnot\s+(?:statistically\s+)?significant\b", claim.text, re.I))
            nonsig_ev = bool(re.search(r"\bnot\s+(?:statistically\s+)?significant\b", evidence, re.I))
            significance_score = 1.0
            if sig_claim:
                if nonsig_claim != nonsig_ev and (nonsig_claim or nonsig_ev):
                    significance_score = 0.0
                elif not sig_ev:
                    significance_score = 0.5

            qualifier_score = float(np.mean([direction_score, certainty_score, significance_score]))
            qualifier_records.append({
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "evidence": evidence,
                "direction_claim": d_claim,
                "direction_evidence": d_ev,
                "direction_score": direction_score,
                "certainty_claim": c_claim,
                "certainty_evidence": c_ev,
                "certainty_score": certainty_score,
                "significance_score": significance_score,
                "score": qualifier_score,
            })

        return {
            "numeric_consistency": safe_mean([r["score"] for r in numeric_records], 1.0),
            "qualifier_consistency": safe_mean([r["score"] for r in qualifier_records], 1.0),
            "numeric_records": numeric_records,
            "qualifier_records": qualifier_records,
        }


# ---------------------------------------------------------------------------
# Non-scientific discourse diagnostics
# ---------------------------------------------------------------------------

TEMPORAL_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,\s*(?:19|20)\d{2})?\b|"
    r"\b(?:yesterday|today|tomorrow|earlier|later|before|after|subsequently|previously)\b",
    flags=re.I,
)

CAUSAL_MARKERS = [
    "because", "due to", "caused", "causes", "led to", "leads to",
    "resulted in", "results in", "therefore", "as a result", "consequently",
]

ATTRIBUTION_MARKERS = [
    "said", "says", "stated", "reported", "claimed", "argued", "according to",
    "announced", "wrote",
]


class DiscourseDiagnostics:
    def __init__(self, hub: ModelHub, processor: TextProcessor, retriever: Retriever):
        self.hub = hub
        self.processor = processor
        self.retriever = retriever

    @staticmethod
    def _marker_presence(text: str, markers: Sequence[str]) -> bool:
        low = text.lower()
        return any(m in low for m in markers)

    def _claim_consistency(self, claim: Claim, source: str) -> Dict[str, Any]:
        hits = self.retriever.evidence(claim.text, source, "non_scientific", text_is_summary=False)
        evidence = " ".join(h["text"] for h in hits[:2])

        claim_entities = self.processor.entities(claim.text)
        ev_entities = self.processor.entities(evidence)
        entity_scores = []
        for ce in claim_entities:
            if not ev_entities:
                entity_scores.append(0.0)
            else:
                exact = max(
                    1.0 if ce.lower() == ee.lower() else self.hub.similarity(ce, ee)
                    for ee in ev_entities
                )
                entity_scores.append(exact)
        entity_score = safe_mean(entity_scores, 1.0)

        claim_times = [m.group(0).lower() for m in TEMPORAL_RE.finditer(claim.text)]
        ev_times = [m.group(0).lower() for m in TEMPORAL_RE.finditer(evidence)]
        if claim_times:
            temporal_score = safe_mean([
                1.0 if any(t == e for e in ev_times) else 0.0 for t in claim_times
            ], 0.0)
        else:
            temporal_score = 1.0

        claim_causal = self._marker_presence(claim.text, CAUSAL_MARKERS)
        ev_causal = self._marker_presence(evidence, CAUSAL_MARKERS)
        causal_score = 1.0 if not claim_causal or ev_causal else 0.35

        claim_attr = self._marker_presence(claim.text, ATTRIBUTION_MARKERS)
        ev_attr = self._marker_presence(evidence, ATTRIBUTION_MARKERS)
        attribution_score = 1.0 if not claim_attr or ev_attr else 0.50

        semantic_event_score = self.hub.similarity(claim.text, evidence)
        score = float(np.mean([
            entity_score,
            temporal_score,
            causal_score,
            attribution_score,
            semantic_event_score,
        ]))
        return {
            "claim_id": claim.claim_id,
            "claim": claim.text,
            "evidence": evidence,
            "entities_claim": claim_entities,
            "entities_evidence": ev_entities,
            "entity_score": entity_score,
            "temporal_score": temporal_score,
            "causal_score": causal_score,
            "attribution_score": attribution_score,
            "semantic_event_score": semantic_event_score,
            "score": score,
        }

    def evaluate_faithfulness(self, summary_claims: Sequence[Claim], source: str) -> Dict[str, Any]:
        records = [self._claim_consistency(c, source) for c in summary_claims]
        return {
            "discourse_consistency": safe_mean([r["score"] for r in records], float("nan")),
            "records": records,
        }

    def evaluate_coverage(self, source_claims: Sequence[Claim], summary: str) -> Dict[str, Any]:
        # Reverse the same logic: important source events should have matching
        # event/discourse evidence in the summary.
        records = []
        for claim in source_claims:
            hits = self.retriever.evidence(
                claim.text, summary, "non_scientific", text_is_summary=True
            )
            evidence = " ".join(h["text"] for h in hits[:2])
            sem = self.hub.similarity(claim.text, evidence) if evidence else 0.0
            temporal = 1.0
            times = [m.group(0).lower() for m in TEMPORAL_RE.finditer(claim.text)]
            if times:
                ev_times = [m.group(0).lower() for m in TEMPORAL_RE.finditer(evidence)]
                temporal = safe_mean([1.0 if t in ev_times else 0.0 for t in times], 0.0)
            causal = 1.0
            if self._marker_presence(claim.text, CAUSAL_MARKERS):
                causal = 1.0 if self._marker_presence(evidence, CAUSAL_MARKERS) else 0.5
            score = float(np.mean([sem, temporal, causal]))
            records.append({
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "weight": claim.weight,
                "summary_evidence": evidence,
                "score": score,
            })

        if not records:
            coverage = float("nan")
        else:
            weights = np.array([r["weight"] for r in records], dtype=float)
            scores = np.array([r["score"] for r in records], dtype=float)
            coverage = float(np.average(scores, weights=weights)) if weights.sum() else float(scores.mean())
        return {"discourse_coverage": clip01(coverage) if np.isfinite(coverage) else float("nan"),
                "records": records}


# ---------------------------------------------------------------------------
# Semantic alignment
# ---------------------------------------------------------------------------

class SemanticEvaluator:
    def __init__(self, hub: ModelHub):
        self.hub = hub

    def evaluate(self, source_claims: Sequence[Claim], summary_claims: Sequence[Claim]) -> Dict[str, Any]:
        if not source_claims or not summary_claims:
            return {
                "semantic_recall": 0.0,
                "semantic_precision": 0.0,
                "semantic_f1": 0.0,
                "matrix": [],
            }
        a = self.hub.encode([c.text for c in source_claims])
        b = self.hub.encode([c.text for c in summary_claims])
        sim = np.clip(a @ b.T, -1.0, 1.0)
        sim01 = (sim + 1.0) / 2.0

        src_best = sim01.max(axis=1)
        src_weights = np.array([c.weight for c in source_claims], dtype=float)
        recall = float(np.average(src_best, weights=src_weights)) if src_weights.sum() else float(src_best.mean())

        sum_best = sim01.max(axis=0)
        precision = float(sum_best.mean())
        return {
            "semantic_recall": clip01(recall),
            "semantic_precision": clip01(precision),
            "semantic_f1": harmonic_mean(recall, precision),
            "source_best_matches": src_best.tolist(),
            "summary_best_matches": sum_best.tolist(),
        }


# ---------------------------------------------------------------------------
# QA coverage
# ---------------------------------------------------------------------------

class QACoverageEvaluator:
    def __init__(
        self,
        hub: ModelHub,
        processor: TextProcessor,
        retriever: Retriever,
        cfg: EvaluationConfig,
    ):
        self.hub = hub
        self.processor = processor
        self.retriever = retriever
        self.cfg = cfg

    def _max_questions(self, text_type: TextType) -> int:
        if text_type == "definition":
            return self.cfg.qa_max_questions_definition
        if text_type == "scientific":
            return self.cfg.qa_max_questions_scientific
        return self.cfg.qa_max_questions_non_scientific

    def _targets(self, claim: Claim, text_type: TextType) -> List[str]:
        text = claim.text
        targets = []

        # Scientific: numbers are particularly important.
        if text_type == "scientific":
            targets.extend([q["raw"] for q in extract_quantities(text)])

        targets.extend(self.processor.entities(text))
        targets.extend(self.processor.noun_phrases(text))

        # Definition-specific category target.
        if text_type == "definition":
            m = re.search(
                r"\b(?:is|are)\s+(?:an?\s+|the\s+)?([^,.;]+)",
                text,
                flags=re.I,
            )
            if m:
                targets.insert(0, normalize_space(m.group(1)))

        # Dates/times for non-scientific narratives.
        if text_type == "non_scientific":
            targets.extend([m.group(0) for m in TEMPORAL_RE.finditer(text)])

        cleaned = []
        for t in targets:
            t = normalize_space(t)
            if 1 <= len(t.split()) <= 10 and len(t) >= 2 and t.lower() not in {
                "the study", "the result", "the results", "the text"
            }:
                cleaned.append(t)
        return list(dict.fromkeys(cleaned))

    def _answer(self, question: str, context: str) -> Dict[str, Any]:
        if not context.strip():
            return {"answer": "", "score": 0.0}
        try:
            out = self.hub.qa(question=question, context=context)
            return {"answer": normalize_space(out.get("answer", "")), "score": float(out.get("score", 0.0))}
        except Exception:
            return {"answer": "", "score": 0.0}

    def evaluate(
        self,
        source_claims: Sequence[Claim],
        source: str,
        summary: str,
        text_type: TextType,
    ) -> Dict[str, Any]:
        max_q = self._max_questions(text_type)
        ordered = sorted(source_claims, key=lambda c: c.weight, reverse=True)
        records = []

        for claim in ordered:
            if len(records) >= max_q:
                break
            targets = self._targets(claim, text_type)
            if not targets:
                continue

            # Usually one question per claim is enough; use a second target only
            # when the first cannot be validated.
            for target in targets[:3]:
                try:
                    question = self.hub.generate_question(claim.text, target)
                except Exception:
                    question = ""
                if not question:
                    continue

                src_hits = self.retriever.evidence(
                    question, source, text_type, text_is_summary=False
                )
                src_context = " ".join(h["text"] for h in src_hits[:2]) or claim.source_unit
                src_answer = self._answer(question, src_context)
                target_match = max(
                    token_f1(target, src_answer["answer"]),
                    self.hub.similarity(target, src_answer["answer"]) if src_answer["answer"] else 0.0,
                )

                if (
                    src_answer["score"] < self.cfg.qa_min_source_confidence
                    or target_match < self.cfg.qa_min_target_match
                ):
                    continue

                sum_hits = self.retriever.evidence(
                    question, summary, text_type, text_is_summary=True
                )
                sum_context = " ".join(h["text"] for h in sum_hits[:2])
                sum_answer = self._answer(question, sum_context)

                if not sum_answer["answer"]:
                    answer_match = 0.0
                else:
                    answer_match = (
                        0.55 * token_f1(src_answer["answer"], sum_answer["answer"])
                        + 0.45 * self.hub.similarity(src_answer["answer"], sum_answer["answer"])
                    )
                relative_conf = min(
                    1.0,
                    sum_answer["score"] / max(src_answer["score"], 1e-8),
                )
                score = clip01(0.85 * answer_match + 0.15 * relative_conf)

                records.append({
                    "claim_id": claim.claim_id,
                    "claim": claim.text,
                    "claim_weight": claim.weight,
                    "target": target,
                    "question": question,
                    "source_context": src_context,
                    "source_answer": src_answer,
                    "summary_context": sum_context,
                    "summary_answer": sum_answer,
                    "answer_match": answer_match,
                    "score": score,
                })
                break

        if records:
            weights = np.array([r["claim_weight"] for r in records], dtype=float)
            scores = np.array([r["score"] for r in records], dtype=float)
            coverage = float(np.average(scores, weights=weights)) if weights.sum() else float(scores.mean())
        else:
            coverage = float("nan")

        return {
            "coverage_qa": clip01(coverage) if np.isfinite(coverage) else float("nan"),
            "valid_generated_questions": len(records),
            "questions": records,
        }


# ---------------------------------------------------------------------------
# Optional calibration
# ---------------------------------------------------------------------------

class Calibrator:
    """
    Optional linear calibrator.

    JSON format:
    {
      "definition": {
        "coverage": {
          "intercept": 0.0,
          "coefficients": {"common_nli": 0.5, "qa": 0.2, "slot_coverage": 0.3}
        },
        "faithfulness": {...}
      },
      ...
    }

    The output is clipped to [0, 1].
    """

    def __init__(self, path: Optional[str]):
        self.data = None
        if path:
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def apply(
        self,
        text_type: TextType,
        target: str,
        features: Dict[str, float],
        fallback: float,
    ) -> float:
        if not self.data:
            return fallback
        spec = self.data.get(text_type, {}).get(target)
        if not spec:
            return fallback
        value = float(spec.get("intercept", 0.0))
        for name, coef in spec.get("coefficients", {}).items():
            x = features.get(name)
            if x is not None and np.isfinite(x):
                value += float(coef) * float(x)
        return clip01(value)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class TextTypeAwareSummaryEvaluator:
    def __init__(self, cfg: Optional[EvaluationConfig] = None):
        self.cfg = cfg or EvaluationConfig()
        set_seed(self.cfg.seed)

        self.processor = TextProcessor()
        self.hub = ModelHub(self.cfg)
        self.router = TextTypeRouter(self.hub, self.processor, self.cfg)
        self.claim_extractor = ClaimExtractor(self.hub, self.processor, self.cfg)
        self.salience = SalienceWeighter(self.hub)
        self.retriever = Retriever(self.hub, self.processor, self.cfg)
        self.semantic = SemanticEvaluator(self.hub)
        self.nli = NLIEvaluator(self.hub, self.retriever, self.cfg)
        self.qa = QACoverageEvaluator(self.hub, self.processor, self.retriever, self.cfg)
        self.definition_slots = DefinitionSlotEvaluator(self.hub)
        self.scientific_diagnostics = ScientificDiagnostics(self.retriever)
        self.discourse = DiscourseDiagnostics(self.hub, self.processor, self.retriever)
        self.calibrator = Calibrator(self.cfg.calibration_path)

    def _profile_scores(
        self,
        text_type: TextType,
        common: Dict[str, float],
        profile_diag: Dict[str, Any],
        scientific_extra: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
        if text_type == "definition":
            cov_features = {
                "common_nli": common["coverage_nli"],
                "qa": common["coverage_qa"],
                "slot_coverage": profile_diag.get("slot_coverage", float("nan")),
            }
            faith_features = {
                "common_nli": common["faithfulness_nli"],
                "slot_consistency": profile_diag.get("slot_consistency", float("nan")),
            }
            cov = weighted_mean(cov_features, self.cfg.definition_coverage_weights)
            faith = weighted_mean(faith_features, self.cfg.definition_faithfulness_weights)

        elif text_type == "scientific":
            sci = scientific_extra or {}
            cov_features = {
                "general_nli": common["coverage_nli"],
                "scientific_nli": sci.get("coverage_scientific_nli", float("nan")),
                "qa": common["coverage_qa"],
                "semantic_recall": common["semantic_recall"],
            }
            faith_features = {
                "general_nli": common["faithfulness_nli"],
                "scientific_nli": sci.get("faithfulness_scientific_nli", float("nan")),
                "numeric_consistency": profile_diag.get("numeric_consistency", float("nan")),
                "qualifier_consistency": profile_diag.get("qualifier_consistency", float("nan")),
            }
            cov = weighted_mean(cov_features, self.cfg.scientific_coverage_weights)
            faith = weighted_mean(faith_features, self.cfg.scientific_faithfulness_weights)

        else:
            cov_features = {
                "common_nli": common["coverage_nli"],
                "qa": common["coverage_qa"],
                "discourse_coverage": profile_diag.get("discourse_coverage", float("nan")),
            }
            faith_features = {
                "common_nli": common["faithfulness_nli"],
                "discourse_consistency": profile_diag.get("discourse_consistency", float("nan")),
            }
            cov = weighted_mean(cov_features, self.cfg.nonscientific_coverage_weights)
            faith = weighted_mean(faith_features, self.cfg.nonscientific_faithfulness_weights)

        cov = self.calibrator.apply(text_type, "coverage", cov_features, cov)
        faith = self.calibrator.apply(text_type, "faithfulness", faith_features, faith)
        return clip01(cov), clip01(faith), cov_features, faith_features

    def evaluate(
        self,
        source: str,
        summary: str,
        text_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        source = normalize_space(source)
        summary = normalize_space(summary)
        if not source:
            raise ValueError("source text is empty")
        if not summary:
            raise ValueError("summary text is empty")

        route = self.router.route(source, text_type or self.cfg.text_type)
        tt: TextType = route["selected"]

        source_claims = self.claim_extractor.extract(source, tt, "SRC-", is_summary=False)
        summary_claims = self.claim_extractor.extract(summary, tt, "SUM-", is_summary=True)

        self.salience.apply(source_claims, tt)
        if summary_claims:
            # Summary claim weights are equal for faithfulness; assigning them
            # here makes detailed output uniform.
            equal = 1.0 / len(summary_claims)
            for c in summary_claims:
                c.weight = equal

        semantic = self.semantic.evaluate(source_claims, summary_claims)

        # Common NLI directions:
        # coverage: summary evidence => source claim
        coverage_nli, coverage_nli_records = self.nli.score_claims(
            source_claims,
            summary,
            tt,
            evidence_is_summary=True,
            scientific_model=False,
        )
        # faithfulness: source evidence => summary claim
        faith_nli, faith_nli_records = self.nli.score_claims(
            summary_claims,
            source,
            tt,
            evidence_is_summary=False,
            scientific_model=False,
        )

        qa = self.qa.evaluate(source_claims, source, summary, tt)

        common = {
            "semantic_recall": semantic["semantic_recall"],
            "semantic_precision": semantic["semantic_precision"],
            "semantic_f1": semantic["semantic_f1"],
            "coverage_nli": coverage_nli,
            "coverage_qa": qa["coverage_qa"],
            "coverage_combined_generic": weighted_mean(
                {"nli": coverage_nli, "qa": qa["coverage_qa"]},
                {
                    "nli": self.cfg.common_coverage_nli_weight,
                    "qa": self.cfg.common_coverage_qa_weight,
                },
            ),
            "faithfulness_nli": faith_nli,
        }

        profile_diag: Dict[str, Any] = {}
        scientific_extra: Dict[str, float] = {}
        scientific_records: Dict[str, Any] = {}

        if tt == "definition":
            profile_diag = self.definition_slots.evaluate(source, summary)

        elif tt == "scientific":
            sci_cov, sci_cov_records = self.nli.score_claims(
                source_claims,
                summary,
                tt,
                evidence_is_summary=True,
                scientific_model=True,
            )
            sci_faith, sci_faith_records = self.nli.score_claims(
                summary_claims,
                source,
                tt,
                evidence_is_summary=False,
                scientific_model=True,
            )
            scientific_extra = {
                "coverage_scientific_nli": sci_cov,
                "faithfulness_scientific_nli": sci_faith,
            }
            scientific_records = {
                "coverage_scientific_nli_records": sci_cov_records,
                "faithfulness_scientific_nli_records": sci_faith_records,
            }
            profile_diag = self.scientific_diagnostics.evaluate_summary_claims(
                summary_claims, source
            )

        else:
            disc_faith = self.discourse.evaluate_faithfulness(summary_claims, source)
            disc_cov = self.discourse.evaluate_coverage(source_claims, summary)
            profile_diag = {**disc_faith, **disc_cov}

        coverage, faithfulness, coverage_features, faithfulness_features = self._profile_scores(
            tt, common, profile_diag, scientific_extra
        )
        overall = harmonic_mean(coverage, faithfulness)

        # Disagreement is computed from the components actually used in the
        # type-specific score, not from compression or word count.
        uncertainty = normalized_disagreement(
            list(coverage_features.values()) + list(faithfulness_features.values())
        )

        src_wc = word_count(source)
        sum_wc = word_count(summary)
        compression_ratio = sum_wc / src_wc if src_wc else float("nan")
        word_reduction = 1.0 - compression_ratio if np.isfinite(compression_ratio) else float("nan")

        metrics = {
            "source_words": src_wc,
            "summary_words": sum_wc,
            "compression_ratio": compression_ratio,
            "word_reduction": word_reduction,
            **common,
            **scientific_extra,
            "coverage_final": coverage,
            "faithfulness_final": faithfulness,
            "overall_information_score": overall,
            "evaluator_disagreement": uncertainty,
            "valid_generated_questions": qa["valid_generated_questions"],
        }

        # Surface profile-specific scalar diagnostics at metric level too.
        for key in [
            "slot_coverage", "slot_consistency",
            "numeric_consistency", "qualifier_consistency",
            "discourse_coverage", "discourse_consistency",
        ]:
            if key in profile_diag:
                metrics[key] = profile_diag[key]

        report = {
            "text_type": tt,
            "router": route,
            "configuration": {
                "profile": self.cfg.profile,
                "models": asdict(self.cfg.models),
                "calibration_path": self.cfg.calibration_path,
            },
            "metrics": metrics,
            "aggregation_features": {
                "coverage": coverage_features,
                "faithfulness": faithfulness_features,
            },
            "source_claims": [asdict(c) for c in source_claims],
            "summary_claims": [asdict(c) for c in summary_claims],
            "details": {
                "semantic": semantic,
                "coverage_nli": coverage_nli_records,
                "faithfulness_nli": faith_nli_records,
                "qa": qa,
                "profile": profile_diag,
                **scientific_records,
            },
            "interpretation_notes": [
                "Scores are model-based estimates, not probabilities of human correctness.",
                "Default final-score weights are heuristic; calibrate against human judgments for formal benchmarking.",
                "Raw common metrics remain comparable across text types; profile-specific final scores use different evidence representations.",
            ],
        }
        return json_safe(report)

    def evaluate_batch(
        self,
        rows: Sequence[Dict[str, Any]],
        source_column: str = "source",
        summary_column: str = "summary",
        text_type_column: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        flat_rows = []
        reports = []
        for i, row in enumerate(rows):
            source = str(row.get(source_column, "") or "")
            summary = str(row.get(summary_column, "") or "")
            override = None
            if text_type_column and row.get(text_type_column) not in (None, "", "auto"):
                override = str(row[text_type_column])
            report = self.evaluate(source, summary, override)
            reports.append(report)

            flat = {
                "id": row.get(id_column, i) if id_column else i,
                "text_type": report["text_type"],
                "router_confidence": report["router"].get("confidence"),
            }
            flat.update(report["metrics"])
            flat_rows.append(flat)
        return flat_rows, reports


# ---------------------------------------------------------------------------
# CSV / CLI helpers
# ---------------------------------------------------------------------------

def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in keys})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Text-type-aware source-summary informational-content evaluator."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", help="Path to original source text.")
    mode.add_argument("--batch", help="CSV containing source/summary rows.")

    p.add_argument("--summary", help="Path to summary text (required with --source).")
    p.add_argument("--output", default="evaluation_report.json", help="Detailed JSON output.")
    p.add_argument("--batch-output", default="evaluation_metrics.csv", help="Batch flat CSV output.")
    p.add_argument("--details-output", default="evaluation_details.jsonl", help="Batch detailed JSONL output.")

    p.add_argument("--profile", choices=["fast", "balanced", "research"], default="balanced")
    p.add_argument(
        "--text-type",
        default="auto",
        choices=["auto", "definition", "scientific", "non_scientific"],
        help="Automatic routing by default; manual override is useful for controlled experiments.",
    )
    p.add_argument("--calibration", default=None, help="Optional calibration JSON.")

    p.add_argument("--source-column", default="source")
    p.add_argument("--summary-column", default="summary")
    p.add_argument("--text-type-column", default=None)
    p.add_argument("--id-column", default=None)

    p.add_argument(
        "--sentence-claims",
        action="store_true",
        help="Disable generative claim decomposition and use sentence/paragraph units.",
    )
    p.add_argument(
        "--no-router-model",
        action="store_true",
        help="Use heuristic routing only.",
    )
    p.add_argument("--seed", type=int, default=13)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = EvaluationConfig.from_profile(args.profile)
    cfg.text_type = args.text_type
    cfg.seed = args.seed
    cfg.calibration_path = args.calibration
    if args.sentence_claims:
        cfg.generative_claim_extraction = False
        cfg.validate_generated_claims = False
    if args.no_router_model:
        cfg.router_use_zero_shot = False

    evaluator = TextTypeAwareSummaryEvaluator(cfg)

    if args.batch:
        rows = read_csv_rows(args.batch)
        flat, reports = evaluator.evaluate_batch(
            rows,
            source_column=args.source_column,
            summary_column=args.summary_column,
            text_type_column=args.text_type_column,
            id_column=args.id_column,
        )
        write_csv_rows(args.batch_output, flat)
        with open(args.details_output, "w", encoding="utf-8") as f:
            for report in reports:
                f.write(json.dumps(json_safe(report), ensure_ascii=False) + "\n")
        print(f"Wrote {len(flat)} rows to {args.batch_output}")
        print(f"Wrote detailed reports to {args.details_output}")
        return

    if not args.summary:
        raise SystemExit("--summary is required when --source is used")

    report = evaluator.evaluate(
        read_text(args.source),
        read_text(args.summary),
        text_type=args.text_type,
    )
    Path(args.output).write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    m = report["metrics"]
    print("\nEvaluation results")
    print("------------------")
    print(f"Text type:                   {report['text_type']}")
    print(f"Router confidence:           {report['router'].get('confidence', float('nan')):.3f}")
    print(f"Source words:                {m['source_words']}")
    print(f"Summary words:               {m['summary_words']}")
    print(f"Compression ratio:           {m['compression_ratio']:.3f}")
    print(f"Word reduction:              {m['word_reduction']:.3f}")
    print(f"Semantic F1:                 {m['semantic_f1']:.3f}")
    print(f"Generic NLI coverage:        {m['coverage_nli']:.3f}")
    if m.get("coverage_qa") is not None:
        print(f"QA coverage:                 {m['coverage_qa']:.3f}")
    print(f"Generic NLI faithfulness:    {m['faithfulness_nli']:.3f}")
    print(f"Final coverage:              {m['coverage_final']:.3f}")
    print(f"Final faithfulness:          {m['faithfulness_final']:.3f}")
    print(f"Overall information score:   {m['overall_information_score']:.3f}")
    print(f"Evaluator disagreement:      {m['evaluator_disagreement']:.3f}")
    print(f"Valid generated questions:   {m['valid_generated_questions']}")
    print(f"\nDetailed report: {args.output}")


if __name__ == "__main__":
    main()
