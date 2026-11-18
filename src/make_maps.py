import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import folium
import numpy as np
import p3_io
from hmm import decode, classifier_only

ROOT = p3_io.ROOT
cfg = json.load(open(ROOT / "artifacts" / "hmm_config.json"))
_, _, df = p3_io.merged()      # train/student only; evaluator trips are never drawn

LEGEND = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
padding:8px;border:1px solid grey;font-size:12px"><b>Legend</b><br>
<span style="color:blue">&#9679;</span> Raw GPS<br><span style="color:grey">&#9473;</span> Road network<br>
<span style="color:green">&#9473;</span> Reference route<br>
<span style="color:orange">&#9473;</span> Classifier-only<br>
<span style="color:red">&#9473;</span> HMM route</div>"""


def build_map(trip_id):
    g = df[df.trip_id == trip_id].sort_values("obs_idx")
    if g.empty:
        raise SystemExit(f"{trip_id}: not a train/student trip with predictions")
    track, arcs, nodes = p3_io.read_track(trip_id), p3_io.read_arcs(trip_id), p3_io.read_nodes(trip_id)
    try:
        ref_route = p3_io.read_route(trip_id)
    except FileNotFoundError:
        ref_route = []
    net = p3_io.get_network(trip_id)
    probs = g["prob"].to_numpy()
    clf, hm = classifier_only(g, probs), decode(g, probs, net, cfg)

    F, T = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()

    def line(a):
        return [(float(lat[F[a]]), float(lon[F[a]])), (float(lat[T[a]]), float(lon[T[a]]))]

    pad = 0.003
    lo0, lo1 = track.lon.min() - pad, track.lon.max() + pad
    la0, la1 = track.lat.min() - pad, track.lat.max() + pad
    inside = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
    near = np.where(inside[F] | inside[T])[0]

    m = folium.Map(location=[float(track.lat.mean()), float(track.lon.mean())], zoom_start=16)
    names = ["Raw GPS (blue)", "Road network (grey)", "Reference route (green)",
             "Classifier-only (orange)", "HMM route (red)"]
    layers = {n: folium.FeatureGroup(n, show=(n != "Classifier-only (orange)")) for n in names}

    for i, r in track.iterrows():
        folium.CircleMarker((r.lat, r.lon), radius=3, color="blue", fill=True,
                            tooltip=f"obs {i}, t={r.t}").add_to(layers["Raw GPS (blue)"])
    for a in near:
        folium.PolyLine(line(int(a)), color="grey", weight=2).add_to(layers["Road network (grey)"])

    def draw(route, color, name):
        for a in dict.fromkeys(x for x in route if x >= 0):
            folium.PolyLine(line(a), color=color, weight=5, opacity=0.8,
                            tooltip=f"arc {a}").add_to(layers[name])

    draw(ref_route, "green", "Reference route (green)")
    draw(clf, "orange", "Classifier-only (orange)")
    draw(hm, "red", "HMM route (red)")
    for lyr in layers.values():
        lyr.add_to(m)
    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND))
    m.fit_bounds([[la0, lo0], [la1, lo1]])
    out = ROOT / "results" / "maps" / f"trip_{trip_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print("saved", out)


if __name__ == "__main__":
    for tid in sys.argv[1:]:
        build_map(tid)