import numpy as np

try:
    from hmm import reachable
except ImportError:
    from src.hmm import reachable


def pointwise_top1(pred, ref_map):
    """pred: arc_id per obs (-1 = none). ref_map: {obs_idx: reference arc}."""
    tot = hit = 0
    for i, a in enumerate(pred):
        if i in ref_map:
            tot += 1
            hit += int(a == ref_map[i])
    return hit / tot if tot else float("nan")


def route_continuity(pred, network, hops=1):
    seq = [a for a in pred if a >= 0]
    if len(seq) < 2:
        return float("nan")
    ok = sum(1 for a, b in zip(seq[:-1], seq[1:]) if b in reachable(network, a, hops))
    return ok / (len(seq) - 1)


def _collapse(seq):
    out = []
    for a in seq:
        if a >= 0 and (not out or out[-1] != a):
            out.append(a)
    return out


def sequence_scores(pred, ref_map):
    p = set(_collapse(pred))
    r = set(_collapse([ref_map[i] for i in sorted(ref_map)]))
    if not p or not r:
        return {"precision": float("nan"), "recall": float("nan"), "jaccard": float("nan")}
    inter = len(p & r)
    return {"precision": inter / len(p), "recall": inter / len(r), "jaccard": inter / len(p | r)}