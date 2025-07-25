"""
Evaluation metrics
"""

from collections import Counter


def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score"""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = pred_counter & ref_counter
    overlap_count = sum(overlap.values())
    
    precision = overlap_count / len(pred_tokens)
    recall = overlap_count / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match score"""
    return float(prediction.lower().strip() == reference.lower().strip()) 