"""
raptor/eval_metrics.py — Official NLP and QA evaluation metrics for RAPTOR benchmark reproduction:
Token F1, BLEU-1, BLEU-4 (with Appendix H smoothing & uniform weights), ROUGE-L, METEOR, and Multiple-Choice Accuracy.
"""

import re
import string
from typing import List, Dict, Any, Union
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_token_f1(gold: str, pred: str) -> float:
    """
    Standard QA Token F1 (harmonic mean of token precision and recall).
    """
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()

    if not gold_toks or not pred_toks:
        return 1.0 if gold_toks == pred_toks else 0.0

    common = set(gold_toks) & set(pred_toks)
    if not common:
        return 0.0

    # Counts taking into account duplicates
    same_count = sum(min(gold_toks.count(w), pred_toks.count(w)) for w in common)
    prec = 1.0 * same_count / len(pred_toks)
    rec = 1.0 * same_count / len(gold_toks)
    f1 = (2 * prec * rec) / (prec + rec)
    return float(f1)


def compute_max_token_f1(gold_references: List[str], pred: str) -> float:
    """Compute max Token F1 across multiple reference answers."""
    if not gold_references:
        return 0.0
    return max(compute_token_f1(gold, pred) for gold in gold_references)


def compute_bleu_scores(gold_references: List[str], pred: str) -> Dict[str, float]:
    """
    Compute BLEU-1 and BLEU-4 adhering strictly to RAPTOR Appendix H:
    - Added Smoothing: SmoothingFunction().method1
    - Modified BLEU-4 Weighting: uniform (0.25, 0.25, 0.25, 0.25)
    """
    ref_tokens = [normalize_answer(r).split() for r in gold_references if r.strip()]
    pred_tokens = normalize_answer(pred).split()

    if not ref_tokens or not pred_tokens:
        return {"bleu_1": 0.0, "bleu_4": 0.0}

    smooth = SmoothingFunction().method1

    b1 = sentence_bleu(
        ref_tokens,
        pred_tokens,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smooth
    )
    b4 = sentence_bleu(
        ref_tokens,
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth
    )

    return {
        "bleu_1": round(float(b1) * 100, 2),
        "bleu_4": round(float(b4) * 100, 2)
    }


def compute_rouge_l(gold_references: List[str], pred: str) -> float:
    """
    Compute ROUGE-L F-measure (percentage).
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best_score = 0.0
    for ref in gold_references:
        if not ref.strip():
            continue
        scores = scorer.score(ref, pred)
        f_meas = scores["rougeL"].fmeasure
        if f_meas > best_score:
            best_score = f_meas

    return round(float(best_score) * 100, 2)


def compute_meteor(gold_references: List[str], pred: str) -> float:
    """
    Compute METEOR score adhering to Appendix H (tokenized before mapping).
    """
    tokenized_refs = [normalize_answer(r).split() for r in gold_references if r.strip()]
    tokenized_pred = normalize_answer(pred).split()

    if not tokenized_refs or not tokenized_pred:
        return 0.0

    try:
        score = meteor_score(tokenized_refs, tokenized_pred)
        return round(float(score) * 100, 2)
    except Exception:
        return 0.0


def evaluate_narrativeqa_metrics(gold_references: List[str], pred: str) -> Dict[str, float]:
    """
    Compute all 4 NarrativeQA benchmark metrics: ROUGE-L, BLEU-1, BLEU-4, METEOR (Table 1 & 6).
    """
    bleu = compute_bleu_scores(gold_references, pred)
    rouge_l = compute_rouge_l(gold_references, pred)
    meteor = compute_meteor(gold_references, pred)

    return {
        "rouge_l": rouge_l,
        "bleu_1": bleu["bleu_1"],
        "bleu_4": bleu["bleu_4"],
        "meteor": meteor,
    }


def evaluate_multiple_choice_answer(pred_text: str, gold_label: int, options: List[str]) -> bool:
    """
    Evaluate Multiple-Choice question (e.g. QuALITY benchmark).
    Matches predicted letter (A, B, C, D) or text against gold option index.
    """
    pred_clean = pred_text.strip()
    gold_letter = chr(65 + gold_label)
    gold_option_text = options[gold_label].strip().lower()

    # 1. Match explicit option letter at beginning of answer or in parens
    letter_match = re.search(r"\b([A-D])\b", pred_clean.upper())
    if letter_match and letter_match.group(1) == gold_letter:
        return True

    # 2. Check if the text matches the correct option text
    if gold_option_text in pred_clean.lower():
        # Make sure other option texts aren't also mentioned
        other_matches = sum(1 for i, opt in enumerate(options) if i != gold_label and opt.strip().lower() in pred_clean.lower())
        if other_matches == 0:
            return True

    return False
