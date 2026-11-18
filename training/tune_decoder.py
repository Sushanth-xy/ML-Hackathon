"""Bayesian (Optuna/TPE) tuning of the HMM decoder parameters.
Run from the repo root:  python training/tune_decoder.py   (or: python -m training.tune_decoder)
Uses ONLY train + student_test trips. Evaluator trips are never loaded."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import optuna
import pandas as pd
from hmm import build_network, decode, reachable, DEFAULT_CONFIG

# ---- settings ----
N_TRIALS = 40
N_TUNE_TRIPS = 30
RAW_ROOT = ROOT / "Student folder"
PREDS_PATH = ROOT / "results" / "models" / "preds_best.csv"

ALIASES = {
    "trip_id": ["trip", "tid", "trip_name"],
    "obs_idx": ["point_idx", "obs", "idx", "gps_idx"],
    "arc_id": ["cand_arc_id", "arc", "cand_arc", "candidate_arc"],
    "dist_m": ["dist", "distance_m", "perp_dist_m", "dist_to_arc_m"],
    "dt_prev_s": ["dt_prev", "dt", "time_gap_s"],
    "label": ["y", "is_ref", "target"],
    "prob": ["proba", "p", "score", "y_prob", "pred_prob", "prob_1"],
}


def canon(df):
    for name, options in ALIASES.items():
        if name not in df.columns:
            for o in options:
                if o in df.columns:
                    df = df.rename(columns={o: name})
                    break
    df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
    return df


def load_candidates():
    parts = [("*train_candidates*.csv", "train"), ("*student*candidates*.csv", "student")]
    frames = []
    for pattern, split in parts:
        for f in sorted((ROOT / "data").glob(pattern)):
            d = canon(pd.read_csv(f, dtype={"trip_id": str}))
            d["split"] = split
            frames.append(d)
    if not frames:
        raise SystemExit("No candidate CSVs found in data/")
    return pd.concat(frames, ignore_index=True)


cands = load_candidates()
preds = canon(pd.read_csv(PREDS_PATH, dtype={"trip_id": str}))
need = {"trip_id", "obs_idx", "arc_id", "dist_m", "label"}
if not need <= set(cands.columns):
    raise SystemExit(f"Candidate CSV is missing {need - set(cands.columns)}. Columns: {list(cands.columns)}")
if "prob" not in preds.columns:
    raise SystemExit(f"No probability column in preds_best.csv. Columns: {list(preds.columns)}")

cands = cands.drop(columns=[c for c in ["prob"] if c in cands.columns])
df = cands.merge(preds[["trip_id", "obs_idx", "arc_id", "prob"]],
                 on=["trip_id", "obs_idx", "arc_id"], how="inner")
print("rows: candidates", len(cands), "| preds", len(preds), "| merged", len(df))
print("merged trips per split:", df.drop_duplicates("trip_id").split.value_counts().to_dict())

# reference arc per point = the candidate with label == 1
ref = {t: dict(zip(g["obs_idx"].astype(int), g["arc_id"].astype(int)))
       for t, g in df[df["label"] == 1].groupby("trip_id")}
groups = {t: g.sort_values("obs_idx").reset_index(drop=True) for t, g in df.groupby("trip_id")}
split_of = df.drop_duplicates("trip_id").set_index("trip_id")["split"].to_dict()
train = sorted(t for t, s in split_of.items() if s == "train")
student = sorted(t for t, s in split_of.items() if s == "student")

# ---- per-trip road network from the raw .arcs files ----
_arc_files = list(RAW_ROOT.rglob("*.arcs"))
_nets = {}


def get_network(trip_id):
    if trip_id not in _nets:
        hits = [f for f in _arc_files if trip_id in f.stem or trip_id in f.parent.name]
        if not hits:
            raise SystemExit(f"No .arcs file found for trip {trip_id} under '{RAW_ROOT}'. "
                             f"Example file seen: {_arc_files[:1]}")
        arcs = pd.read_csv(hits[0], sep=r"\s+", header=None, names=["from_node", "to_node"], dtype=int)
        _nets[trip_id] = build_network(arcs)
    return _nets[trip_id]


# ---- metrics ----
def top1(pred, ref_map):
    tot = hit = 0
    for i, a in enumerate(pred):
        if i in ref_map:
            tot += 1
            hit += int(a == ref_map[i])
    return hit / tot if tot else float("nan")


def continuity(pred, net):
    seq = [a for a in pred if a >= 0]
    if len(seq) < 2:
        return float("nan")
    ok = sum(1 for a, b in zip(seq[:-1], seq[1:]) if b in reachable(net, a, 1))
    return ok / (len(seq) - 1)


def evaluate(cfg, trips):
    acc, cont = [], []
    for t in trips:
        g, net = groups[t], get_network(t)
        pred = decode(g, g["prob"].to_numpy(), net, cfg)
        acc.append(top1(pred, ref.get(t, {})))
        cont.append(continuity(pred, net))
    return float(np.nanmean(acc)), float(np.nanmean(cont))


# ---- which trips to tune on ----
if len(train) >= 10:
    tune_trips = random.Random(42).sample(train, min(N_TUNE_TRIPS, len(train)))
    print(f"Tuning on {len(tune_trips)} TRAIN trips (ask P2: are train probabilities out-of-fold?)")
else:
    tune_trips = student
    print("WARNING: preds_best.csv has no train trips, tuning on the student trips instead. "
          "Ask P2 for out-of-fold train predictions.")


def objective(trial):
    cfg = {
        "sigma": trial.suggest_float("sigma", 5, 60),
        "alpha": trial.suggest_float("alpha", 0.3, 2.0),
        "p_stay": trial.suggest_float("p_stay", 0.1, 0.8),
        "p_next": trial.suggest_float("p_next", 0.05, 0.6),
        "beta": trial.suggest_float("beta", 0.5, 3.0),
        "p_floor": trial.suggest_float("p_floor", 1e-8, 1e-3, log=True),
        "max_hops": trial.suggest_int("max_hops", 1, 4),
        "gap_s": trial.suggest_float("gap_s", 3, 30),
    }
    return evaluate({**DEFAULT_CONFIG, **cfg}, tune_trips)[0]   # objective: pointwise top-1


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS)

best = {**DEFAULT_CONFIG, **study.best_params}
(ROOT / "results" / "hmm").mkdir(parents=True, exist_ok=True)
study.trials_dataframe().to_csv(ROOT / "results" / "hmm" / "tuning.csv", index=False)
json.dump(best, open(ROOT / "artifacts" / "hmm_config.json", "w"), indent=2)

print("\nbest tuning score:", study.best_value)
print("best params:", study.best_params)
if student:
    print("STUDENT default (top1, continuity):", evaluate(DEFAULT_CONFIG, student))
    print("STUDENT tuned   (top1, continuity):", evaluate(best, student))