import os
import sys
import json
import time
import numpy as np

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from raptor import (
    RAPTOR,
    BM25Retriever,
    OllamaEmbeddingModel,
    GemmaQAModel,
    cosine_similarity,
    compute_max_token_f1,
    compute_bleu_scores,
    compute_rouge_l,
    compute_meteor,
    evaluate_narrativeqa_metrics,
    evaluate_multiple_choice_answer
)

print("Imports succeeded!")
