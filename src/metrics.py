"""
PathSnap - Unified Evaluation Metrics Module
Co-owned by P2 (Sushanth), P3 (Mahalakshmi), and P4 (Sudha)

Provides standardized evaluation routines for:
1. Observation-level Top-1 Accuracy (Pointwise)
2. Topological Route Continuity (1-hop / 2-hop / Length Ratio)
3. Sequence-level Arc Set Precision, Recall, F1 & Jaccard
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Union, Optional, Any

try:
    from hmm import reachable
except ImportError:
    from src.hmm import reachable


# ==========================================
# Mahalakshmi (P3) Metric Functions
# ==========================================

def pointwise_top1(pred, ref_map):
    """pred: arc_id per obs (-1 = none). ref_map: {obs_idx: reference arc}."""
    tot = hit = 0
    for i, a in enumerate(pred):
        if i in ref_map:
            tot += 1
            hit += int(a == ref_map[i])
    return hit / tot if tot else float("nan")


def route_continuity(pred, network, hops=1):
    """Fraction of consecutive valid predictions reachable within `hops` in road network."""
    seq = [a for a in pred if a >= 0]
    if len(seq) < 2:
        return float("nan")
    ok = sum(1 for a, b in zip(seq[:-1], seq[1:]) if b in reachable(network, a, hops))
    return ok / (len(seq) - 1)


def _collapse(seq):
    """Compress consecutive identical arc predictions [a, a, a, b, b, c] -> [a, b, c]."""
    out = []
    for a in seq:
        if a >= 0 and (not out or out[-1] != a):
            out.append(a)
    return out


def sequence_scores(pred, ref_map):
    """Computes precision, recall, and Jaccard similarity over collapsed unique arcs."""
    p = set(_collapse(pred))
    if isinstance(ref_map, dict):
        r = set(_collapse([ref_map[i] for i in sorted(ref_map)]))
    else:
        r = set(_collapse(ref_map))
        
    if not p or not r:
        return {"precision": float("nan"), "recall": float("nan"), "jaccard": float("nan"), "f1": float("nan")}
    inter = len(p & r)
    prec = inter / len(p)
    rec = inter / len(r)
    jacc = inter / len(p | r)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "jaccard": jacc, "f1": f1}


# ==========================================
# P2 / P4 Pipeline Standard API Aliases
# ==========================================

compress_consecutive_arcs = _collapse


def compute_top1_accuracy(
    df: pd.DataFrame,
    pred_col: str = "prob",
    label_col: str = "label",
    group_cols: List[str] = ["trip_id", "point_idx"]
) -> float:
    """Computes observation-level Top-1 accuracy by selecting candidate with max prediction."""
    idx_max = df.groupby(group_cols)[pred_col].idxmax()
    return float(df.loc[idx_max][label_col].mean())


def compute_sequence_f1(
    predicted_arcs: List[int],
    reference_arcs: Union[List[int], np.ndarray, Set[int]]
) -> Dict[str, float]:
    """Computes precision, recall, f1, and jaccard between predicted and reference routes."""
    pred_set: Set[int] = set(predicted_arcs)
    ref_set: Set[int] = set(reference_arcs)
    
    inter = len(pred_set & ref_set)
    precision = inter / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = inter / len(ref_set) if len(ref_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = inter / len(pred_set | ref_set) if len(pred_set | ref_set) > 0 else 0.0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard)
    }


def compute_route_continuity(
    predicted_arcs: List[int],
    reference_or_network: Any,
    arc_lengths_or_hops: Any = None
) -> float:
    """
    Polymorphic continuity metric:
    - If reference_or_network is a network dict, computes topological graph continuity (hops).
    - If reference_or_network is a reference route and arc_lengths is passed, computes route length ratio (%).
    """
    if isinstance(reference_or_network, dict) and "succ" in reference_or_network:
        hops = arc_lengths_or_hops if isinstance(arc_lengths_or_hops, int) else 1
        return float(route_continuity(predicted_arcs, reference_or_network, hops=hops) * 100.0)
        
    reference_arcs = reference_or_network
    arc_lengths = arc_lengths_or_hops
    if arc_lengths is not None and isinstance(arc_lengths, dict):
        pred_len = sum(arc_lengths.get(a, 0.0) for a in predicted_arcs)
        ref_len = sum(arc_lengths.get(a, 0.0) for a in reference_arcs)
        if ref_len > 0:
            return float((pred_len / ref_len) * 100.0)
            
    if len(reference_arcs) > 0:
        return float((len(predicted_arcs) / len(reference_arcs)) * 100.0)
    return 100.0
