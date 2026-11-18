"""
PathSnap - EDA + data-quality evidence (run AFTER build_dataset.py).
Run:  python src/eda.py            (auto-finds the dataset folder)
  or: python src/eda.py "C:/path/to/Student folder"
Needs: pip install matplotlib folium
Outputs go to <project>/results/eda/ (every table/plot is labelled with its partition).
"""
import sys, os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import folium
sys.path.insert(0, os.path.dirname(__file__))
from build_dataset import load_trip, find_data_root, PROJECT_ROOT, DATA_DIR

OUT = str(PROJECT_ROOT / "results" / "eda"); os.makedirs(OUT, exist_ok=True)

def main(root=None):
    root = str(find_data_root(root))
    print('Using dataset folder:', root)
    rows, all_dt = [], {"train": [], "student_test": []}
    for split in ["train", "student_test"]:
        for f in sorted(glob.glob(os.path.join(root, split, "*"))):
            tid, trk, nodes, arcs, route = load_trip(f)
            n = len(nodes)
            g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
            ncomp, lab = connected_components(g, directed=False)
            deg = np.bincount(arcs.ravel(), minlength=n)
            dt = np.diff(trk[:, 2]); all_dt[split].append(dt)
            files = {e: os.path.getsize(os.path.join(f, f"{tid}.{e}")) for e in ["track", "nodes", "arcs", "route"]}
            rows.append(dict(
                partition=split, trip_id=tid, n_points=len(trk),
                nan_gps=int(np.isnan(trk).sum()),
                lon_out_of_range=int(((trk[:, 0] < -180) | (trk[:, 0] > 180)).sum()),
                lat_out_of_range=int(((trk[:, 1] < -90) | (trk[:, 1] > 90)).sum()),
                time_sorted=bool((dt > 0).all()), dup_timestamps=int((dt == 0).sum()),
                pct_step_1s=round(float((dt == 1).mean()) * 100, 2), max_gap_s=float(dt.max()),
                lon_min=trk[:, 0].min(), lon_max=trk[:, 0].max(), lat_min=trk[:, 1].min(), lat_max=trk[:, 1].max(),
                n_nodes=n, n_arcs=len(arcs), n_components=int(ncomp),
                largest_component_pct=round(float(np.bincount(lab).max() / n) * 100, 1),
                isolated_nodes=int((deg == 0).sum()),
                route_len=len(route), route_arc_ids_valid=bool(route.max() < len(arcs) and route.min() >= 0),
                route_connected_pct=round(float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0])) * 100, 1),
                **{f"size_{k}_kb": round(v / 1024, 1) for k, v in files.items()}))
    inv = pd.DataFrame(rows); inv.to_csv(f"{OUT}/inventory_per_trip.csv", index=False)

    # partition summary
    summ = inv.groupby("partition").agg(trips=("trip_id", "count"), points_total=("n_points", "sum"),
            points_median=("n_points", "median"), nodes_median=("n_nodes", "median"),
            arcs_median=("n_arcs", "median"), missing_values=("nan_gps", "sum"),
            trips_with_gaps_gt5s=("max_gap_s", lambda s: int((s > 5).sum())),
            trips_disconnected_network=("n_components", lambda s: int((s > 1).sum())))
    summ.to_csv(f"{OUT}/partition_summary.csv"); print(summ.T)

    # join with label-quality info from build_report.csv if present
    if (DATA_DIR / "build_report.csv").exists():
        br = pd.read_csv(DATA_DIR / "build_report.csv")
        br["trip_id"] = br["trip_id"].astype(str).str.zfill(8); inv["trip_id"] = inv["trip_id"].astype(str).str.zfill(8)
        q = br.merge(inv[["trip_id", "n_components"]], on="trip_id")
        key = "pct_unlabeled" if "pct_unlabeled" in q else "miss_rate"   # rerun build_dataset.py for the full report
        q = q.sort_values(key, ascending=False); q.to_csv(f"{OUT}/label_quality_per_trip.csv", index=False)
        show = [c for c in ["split", "trip_id", "n_points", "pct_unlabeled", "miss_rate", "route_km", "gps_path_km"] if c in q]
        print("\nWorst label-quality trips:\n", q[show].head(6))

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    tr = inv[inv.partition == "train"]
    ax[0].hist(tr.n_points, bins=25); ax[0].set(title="Points per trip (train, 80 trips)", xlabel="GPS points")
    d = np.concatenate(all_dt["train"]); vc = pd.Series(d).clip(upper=10).value_counts().sort_index()
    ax[1].bar(vc.index.astype(str), vc.values); ax[1].set_yscale("log")
    ax[1].set(title="Time gap between GPS points (train); 10 = >=10 s", xlabel="seconds", ylabel="count (log)")
    ax[2].scatter(tr.n_arcs, tr.n_points, s=12); ax[2].set(title="Network size vs trip length (train)", xlabel="arcs", ylabel="points")
    plt.tight_layout(); plt.savefig(f"{OUT}/eda_overview_train.png", dpi=130); plt.close()

    # trajectory overlay (train, 6 trips)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for a, tid in zip(ax.ravel(), sorted(tr.trip_id)[:6]):
        _, trk, *_ = load_trip(os.path.join(root, "train", tid)); a.plot(trk[:, 0], trk[:, 1], lw=.6); a.set_title(f"train {tid}")
    plt.tight_layout(); plt.savefig(f"{OUT}/trajectories_train.png", dpi=120); plt.close()

    # Folium map: GPS (blue) + network (grey) + reference route (green)  [train trip]
    def folium_trip(split, tid, path):
        _, trk, nodes, arcs, route = load_trip(os.path.join(root, split, tid))
        m = folium.Map(location=[trk[:, 1].mean(), trk[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
        net = folium.FeatureGroup("Road network (grey)", show=False)
        c = np.array([trk[:, 0].mean(), trk[:, 1].mean()])
        near = np.where((np.abs(nodes[arcs[:, 0], 0] - c[0]) < .02) & (np.abs(nodes[arcs[:, 0], 1] - c[1]) < .02))[0][:4000]
        for a in near:
            folium.PolyLine([nodes[arcs[a, 0], ::-1], nodes[arcs[a, 1], ::-1]], color="grey", weight=1).add_to(net)
        rt = folium.FeatureGroup("Reference route (green)")
        for a in route:
            folium.PolyLine([nodes[arcs[a, 0], ::-1], nodes[arcs[a, 1], ::-1]], color="green", weight=4).add_to(rt)
        gp = folium.FeatureGroup("GPS observations (blue)")
        for p in trk[::5]: folium.CircleMarker([p[1], p[0]], radius=2, color="blue", fill=True).add_to(gp)
        for g in (net, rt, gp): g.add_to(m)
        folium.LayerControl().add_to(m)
        m.save(path)
    folium_trip("train", "00000005", f"{OUT}/map_train_00000005.html")
    folium_trip("train", "00000033", f"{OUT}/map_train_00000033_problem_trip.html")
    print("\nSaved everything to", OUT)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)