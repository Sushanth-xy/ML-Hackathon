"""P3 loader: reads P1's candidate CSVs, P2's predictions and the raw per-trip files."""
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import build_network

ROOT = Path(__file__).resolve().parents[1]
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
    """train + student_test only. The evaluator partition is never loaded."""
    frames = []
    for pattern, split in [("*train_candidates*.csv", "train"), ("*student*candidates*.csv", "student")]:
        for f in sorted((ROOT / "data").glob(pattern)):
            d = canon(pd.read_csv(f, dtype={"trip_id": str}))
            d["split"] = split
            frames.append(d)
    if not frames:
        raise SystemExit("No candidate CSVs found in data/")
    return pd.concat(frames, ignore_index=True)


def merged():
    c = load_candidates()
    p = canon(pd.read_csv(PREDS_PATH, dtype={"trip_id": str}))
    c = c.drop(columns=[x for x in ["prob"] if x in c.columns])
    m = c.merge(p[["trip_id", "obs_idx", "arc_id", "prob"]],
                on=["trip_id", "obs_idx", "arc_id"], how="inner")
    return c, p, m


def reference_from_labels(df):
    r = df[df["label"] == 1].drop_duplicates(["trip_id", "obs_idx"])
    return {t: dict(zip(g["obs_idx"].astype(int), g["arc_id"].astype(int)))
            for t, g in r.groupby("trip_id")}


@lru_cache(maxsize=None)
def _files(ext):
    return tuple(RAW_ROOT.rglob(f"*.{ext}"))


def find(trip_id, ext):
    hits = [f for f in _files(ext) if trip_id in f.stem or trip_id in f.parent.name]
    if not hits:
        raise FileNotFoundError(f"No .{ext} file for trip {trip_id} under {RAW_ROOT}")
    return hits[0]


def read_track(trip_id):
    return pd.read_csv(find(trip_id, "track"), sep=r"\s+", header=None, names=["lon", "lat", "t"])


def read_nodes(trip_id):
    return pd.read_csv(find(trip_id, "nodes"), sep=r"\s+", header=None, names=["lon", "lat"])


def read_arcs(trip_id):
    return pd.read_csv(find(trip_id, "arcs"), sep=r"\s+", header=None,
                       names=["from_node", "to_node"], dtype=int)


def read_route(trip_id):
    return pd.read_csv(find(trip_id, "route"), sep=r"\s+", header=None)[0].astype(int).tolist()


_NETS = {}


def get_network(trip_id):
    if trip_id not in _NETS:
        _NETS[trip_id] = build_network(read_arcs(trip_id))
    return _NETS[trip_id]