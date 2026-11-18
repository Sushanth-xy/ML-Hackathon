import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import p3_io
from hmm import decode, classifier_only, DEFAULT_CONFIG
from metrics import pointwise_top1, route_continuity, sequence_scores

tuned = json.load(open(ROOT / "artifacts" / "hmm_config.json"))
_, _, df = p3_io.merged()
ref = p3_io.reference_from_labels(df)
groups = {t: g.sort_values("obs_idx").reset_index(drop=True) for t, g in df.groupby("trip_id")}
split_of = df.drop_duplicates("trip_id").set_index("trip_id")["split"].to_dict()

METHODS = {
    "classifier_only":   lambda g, net: classifier_only(g, g["prob"].to_numpy()),
    "distance_only_hmm": lambda g, net: decode(g, None, net, DEFAULT_CONFIG),
    "hmm_default":       lambda g, net: decode(g, g["prob"].to_numpy(), net, DEFAULT_CONFIG),
    "hmm_tuned":         lambda g, net: decode(g, g["prob"].to_numpy(), net, tuned),
}

rows = []
for t, g in groups.items():
    net, r = p3_io.get_network(t), ref.get(t, {})
    for name, fn in METHODS.items():
        t0 = time.perf_counter()
        pred = fn(g, net)
        sec = time.perf_counter() - t0
        row = dict(split=split_of[t], trip_id=t, method=name, n_obs=len(pred),
                   top1=pointwise_top1(pred, r),
                   cont_1hop=route_continuity(pred, net, 1),
                   cont_2hop=route_continuity(pred, net, 2),
                   seconds=sec)
        row.update(sequence_scores(pred, r))
        rows.append(row)

out = ROOT / "results" / "hmm"
out.mkdir(parents=True, exist_ok=True)
per = pd.DataFrame(rows)
per.to_csv(out / "per_trip.csv", index=False)
summary = per.groupby(["split", "method"]).mean(numeric_only=True).drop(columns="n_obs")
summary.to_csv(out / "comparison.csv")
print(summary.round(4).to_string())

wide = per[per.split == "student"].pivot(index="trip_id", columns="method", values="top1")
if not wide.empty:
    wide["gain"] = wide["hmm_tuned"] - wide["classifier_only"]
    wide.sort_values("gain").to_csv(out / "gain_per_student_trip.csv")
    print("\nHMM hurt most :", list(wide.sort_values("gain").index[:2]))
    print("HMM helped most:", list(wide.sort_values("gain").index[-2:]))