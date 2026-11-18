"""HMM + Viterbi route reconstruction over candidate arcs."""
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

DEFAULT_CONFIG = {
    "eps": 1e-6,         # probability clipping
    "alpha": 1.0,        # emission temperature (prob ** alpha)
    "use_distance": True,  # add Gaussian distance term to emission
    "sigma": 20.0,       # metres, distance term
    "p_stay": 0.5,       # same arc as previous point
    "p_next": 0.3,       # directly connected arc
    "beta": 1.5,         # decay per extra hop
    "p_floor": 1e-6,     # unreachable transition (never 0)
    "max_hops": 3,
    "gap_s": 10.0,       # time gap above which the chain is relaxed
}


def build_network(arcs_df):
    """arcs_df: columns from_node, to_node; row index is arc_id. Arcs are DIRECTED."""
    from_nodes = arcs_df["from_node"].to_numpy()
    to_nodes = arcs_df["to_node"].to_numpy()
    starts_at = defaultdict(list)
    for a, f in enumerate(from_nodes):
        starts_at[int(f)].append(a)
    succ = {a: starts_at.get(int(to_nodes[a]), []) for a in range(len(arcs_df))}
    return {"succ": succ, "n_arcs": len(arcs_df), "_cache": {}}


def reachable(network, arc, max_hops):
    """dict {arc: hops} reachable from `arc` within max_hops (0 = same arc). Cached."""
    key = (arc, max_hops)
    cache = network["_cache"]
    if key in cache:
        return cache[key]
    dist = {arc: 0}
    q = deque([arc])
    while q:
        u = q.popleft()
        if dist[u] == max_hops:
            continue
        for v in network["succ"][u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    cache[key] = dist
    return dist


def log_emission(dist_m, probs, cfg):
    """probs=None -> distance-only emission (baseline / before P2's model is ready)."""
    e = np.zeros(len(dist_m))
    if probs is not None:
        p = np.clip(probs, cfg["eps"], 1 - cfg["eps"])
        e += cfg["alpha"] * np.log(p)
    if probs is None or cfg["use_distance"]:
        e += -(dist_m ** 2) / (2 * cfg["sigma"] ** 2)
    return e


def log_transition(network, prev_arcs, cur_arcs, cfg, relax=False):
    """Matrix (len(prev), len(cur)) of log transition scores."""
    if relax:  # big gap: no information from the previous point, restart the chain
        return np.zeros((len(prev_arcs), len(cur_arcs)))
    floor = math.log(cfg["p_floor"])
    T = np.full((len(prev_arcs), len(cur_arcs)), floor)
    cur_pos = defaultdict(list)
    for j, b in enumerate(cur_arcs):
        cur_pos[b].append(j)
    for i, a in enumerate(prev_arcs):
        for b, h in reachable(network, a, cfg["max_hops"]).items():
            if b in cur_pos:
                if h == 0:
                    p = cfg["p_stay"]
                elif h == 1:
                    p = cfg["p_next"]
                else:
                    p = cfg["p_next"] * math.exp(-(h - 1) / cfg["beta"])
                for j in cur_pos[b]:
                    T[i, j] = math.log(max(p, cfg["p_floor"]))
    return T


def viterbi(emissions, transitions):
    """emissions: list of 1-D log arrays (one per step).
    transitions[t]: (n_{t-1}, n_t) log matrix for t >= 1 (transitions[0] unused)."""
    delta = emissions[0].copy()
    back = [None]
    for t in range(1, len(emissions)):
        scores = delta[:, None] + transitions[t]
        arg = scores.argmax(axis=0)
        delta = scores[arg, np.arange(scores.shape[1])] + emissions[t]
        back.append(arg)
    path = [int(delta.argmax())]
    for t in range(len(emissions) - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return path[::-1]


def decode(cands_df, probs, network, config=None):
    """Returns predicted arc_id for every obs_idx 0..max (-1 where the point had no candidates).
    cands_df: one trip, columns obs_idx, arc_id, dist_m, dt_prev_s.
    probs: array aligned with cands_df rows, or None for distance-only."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    df = cands_df.reset_index(drop=True)
    if probs is not None:
        df = df.assign(_p=np.asarray(probs, dtype=float))

    obs_ids, arcs_per_t, ems, gaps = [], [], [], []
    last_obs = None
    for o, g in df.groupby("obs_idx", sort=True):
        p = g["_p"].to_numpy() if probs is not None else None
        ems.append(log_emission(g["dist_m"].to_numpy(), p, cfg))
        arcs_per_t.append(g["arc_id"].astype(int).tolist())
        dt = g["dt_prev_s"].iloc[0] if "dt_prev_s" in g else 1.0
        skipped = last_obs is not None and o - last_obs > 1  # points with no candidates in between
        gaps.append(bool(skipped or (dt == dt and dt > cfg["gap_s"])))
        obs_ids.append(int(o))
        last_obs = o

    n_obs = int(df["obs_idx"].max()) + 1
    out = [-1] * n_obs
    if not obs_ids:
        return out

    trans = [None] + [
        log_transition(network, arcs_per_t[t - 1], arcs_per_t[t], cfg, relax=gaps[t])
        for t in range(1, len(obs_ids))
    ]
    path = viterbi(ems, trans)
    for t, o in enumerate(obs_ids):
        out[o] = arcs_per_t[t][path[t]]
    return out


def classifier_only(cands_df, probs):
    """Argmax per point, same output format as decode()."""
    df = cands_df.reset_index(drop=True).assign(_p=np.asarray(probs, dtype=float))
    best = df.loc[df.groupby("obs_idx")["_p"].idxmax()]
    out = [-1] * (int(df["obs_idx"].max()) + 1)
    for o, a in zip(best["obs_idx"], best["arc_id"]):
        out[int(o)] = int(a)
    return out