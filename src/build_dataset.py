"""
PathSnap - Step 1: raw files -> candidate-ranking table.

One row = (trip_id, point_idx, candidate arc). label=1 if that arc is the
reference arc the GPS point was aligned to.

Run:  python src/build_dataset.py            (auto-finds the dataset folder)
  or: python src/build_dataset.py "C:/path/to/Student folder"
Out:  data/train_candidates.csv, data/student_test_candidates.csv, data/build_report.csv
      (always written inside the project root, whatever folder you run from)
"""
import sys, os, glob, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # folder that contains src/
DATA_DIR = PROJECT_ROOT / "data"

def _is_root(p):
    return (p / "train").is_dir() and (p / "student_test").is_dir()

def find_data_root(arg=None):
    """Locate the dataset folder (the one holding train/ and student_test/).
    Order: CLI argument -> common names in project -> search project -> unzip Student_folder.zip."""
    cands = []
    if arg: cands.append(Path(arg))
    cands += [PROJECT_ROOT / "Student folder", PROJECT_ROOT / "Student_folder",
              PROJECT_ROOT / "data-preprocessing" / "Student folder"]
    for c in cands:
        if _is_root(c): return c
    for p in PROJECT_ROOT.rglob("student_test"):
        if p.is_dir() and _is_root(p.parent): return p.parent
    for z in [PROJECT_ROOT / "Student_folder.zip", Path.home() / "Downloads" / "Student_folder.zip"]:
        if z.exists():
            print(f"Dataset folder not found - extracting {z} into project ...")
            zipfile.ZipFile(z).extractall(PROJECT_ROOT)
            for c in [PROJECT_ROOT / "Student folder", PROJECT_ROOT / "Student_folder"]:
                if _is_root(c): return c
    raise SystemExit("ERROR: could not find the dataset folder (needs train/ and student_test/ inside).\n"
                     f"Looked in: {PROJECT_ROOT}\nFix: pass the path, e.g.  python src/build_dataset.py \"C:/path/to/Student folder\"")

K = 12            # candidates per GPS point
MAX_ALIGN_M = 30.0  # points further than this from every route arc are 'unlabeled'
R_EARTH = 6371000.0

# ---------- loading ----------
def load_trip(folder):
    tid = os.path.basename(folder)
    p = lambda e: os.path.join(folder, f"{tid}.{e}")
    track = np.loadtxt(p("track"))                    # lon, lat, t
    nodes = np.loadtxt(p("nodes"))                    # lon, lat
    arcs = np.loadtxt(p("arcs"), dtype=int)           # from, to
    route = np.loadtxt(p("route"), dtype=int).ravel() # arc ids
    return tid, track, nodes, arcs, route

def to_xy(lonlat, lon0, lat0):
    """Local equirectangular projection (metres). Fine at city scale (<~50 km).
    Documented choice: avoids degree-distance errors without needing pyproj."""
    x = np.radians(lonlat[:, 0] - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lonlat[:, 1] - lat0) * R_EARTH
    return np.c_[x, y]

# ---------- geometry ----------
def point_seg(P, A, B):
    """P:(n,2) A,B:(n,2) -> dist, frac t in [0,1], arc bearing (rad)"""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    t = np.clip(((P - A) * AB).sum(1) / L2, 0, 1)
    proj = A + t[:, None] * AB
    d = np.linalg.norm(P - proj, axis=1)
    return d, t, np.arctan2(AB[:, 1], AB[:, 0])

def ang_diff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)

# ---------- label alignment (monotone along reference route) ----------
def align_points_to_route(P, A_r, B_r):
    """Assign each GPS point a route index, non-decreasing, minimising total
    distance (DP). Needed because route has far fewer arcs than GPS points."""
    n, m = len(P), len(A_r)
    D = np.empty((n, m))
    for j in range(m):
        D[:, j] = point_seg(P, np.repeat(A_r[j][None], n, 0), np.repeat(B_r[j][None], n, 0))[0]
    cost = np.full((n, m), np.inf); back = np.zeros((n, m), int)
    cost[0] = D[0]
    for i in range(1, n):
        prev = np.minimum.accumulate(cost[i - 1])           # best of j'<=j
        arg = np.zeros(m, int); best = np.inf; bj = 0
        for j in range(m):
            if cost[i - 1, j] < best: best, bj = cost[i - 1, j], j
            arg[j] = bj
        cost[i] = prev + D[i]; back[i] = arg
    idx = np.zeros(n, int); idx[-1] = int(np.argmin(cost[-1]))
    for i in range(n - 1, 0, -1): idx[i - 1] = back[i, idx[i]]
    return idx, D[np.arange(n), idx]

# ---------- per trip ----------
def process_trip(folder, split):
    tid, track, nodes, arcs, route = load_trip(folder)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    P = to_xy(track[:, :2], lon0, lat0)
    N = to_xy(nodes, lon0, lat0)
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]
    alen = np.linalg.norm(B - A, axis=1)
    n = len(P)
    t = track[:, 2]
    dt = np.r_[np.nan, np.diff(t)]

    # GPS motion features (heading from neighbouring points, speed)
    d_prev = np.r_[np.nan, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    spd = d_prev / np.where(dt > 0, dt, np.nan)
    hv = np.zeros(n)
    dxy = P[np.minimum(np.arange(n) + 1, n - 1)] - P[np.maximum(np.arange(n) - 1, 0)]
    hv = np.arctan2(dxy[:, 1], dxy[:, 0])
    moving = np.linalg.norm(dxy, axis=1) > 1.0

    # node degree + reverse-arc existence
    deg_out = np.bincount(arcs[:, 0], minlength=len(nodes))
    deg_in = np.bincount(arcs[:, 1], minlength=len(nodes))
    pair = set(map(tuple, arcs.tolist()))
    has_rev = np.array([(b, a) in pair for a, b in arcs])

    # reference alignment
    r_idx, r_dist = align_points_to_route(P, A[route], B[route])
    true_arc = route[r_idx]

    # candidate generation: K nearest arcs (midpoint tree -> exact distance)
    mid = (A + B) / 2
    tree = cKDTree(mid)
    kq = min(len(arcs), K * 4)
    _, cand = tree.query(P, k=kq)
    cand = np.atleast_2d(cand)
    # include the arcs of nearest *nodes* too is optional; keep simple
    rows = []
    miss = 0
    for i in range(n):
        c = cand[i]
        d, f, brg = point_seg(np.repeat(P[i][None], len(c), 0), A[c], B[c])
        o = np.argsort(d)[:K]
        c, d, f, brg = c[o], d[o], f[o], brg[o]
        if true_arc[i] not in c: miss += 1
        dh = ang_diff(brg, hv[i]) if moving[i] else np.full(len(c), np.nan)
        for r in range(len(c)):
            rows.append((tid, i, int(c[r]), int(c[r] == true_arc[i]), split,
                         d[r], d[r] - d[0], r, f[r], dh[r], np.cos(dh[r]) if moving[i] else 0.0,
                         alen[c[r]], spd[i], dt[i], d_prev[i],
                         deg_out[arcs[c[r], 1]], deg_in[arcs[c[r], 0]], int(has_rev[c[r]]),
                         int(moving[i]), r_dist[i], int(r_dist[i] <= MAX_ALIGN_M)))
    cols = ["trip_id", "point_idx", "cand_arc_id", "label", "split",
            "dist_m", "dist_minus_min", "dist_rank", "frac_along", "heading_diff", "heading_cos",
            "arc_len", "speed", "dt", "step_dist", "end_deg_out", "start_deg_in", "has_reverse",
            "is_moving", "align_dist", "labeled"]
    df = pd.DataFrame(rows, columns=cols)
    # missing-value handling (fit-free constants, so no leakage):
    #  first point has no previous step -> speed/step_dist 0, dt nominal 1 s
    #  heading undefined when (almost) stationary -> 0 rad diff, flagged by is_moving=0
    df["speed"] = df["speed"].fillna(0.0)
    df["step_dist"] = df["step_dist"].fillna(0.0)
    df["dt"] = df["dt"].fillna(1.0)
    df["heading_diff"] = df["heading_diff"].fillna(0.0)
    info = dict(trip_id=tid, split=split, n_points=n, n_nodes=len(nodes), n_arcs=len(arcs),
                route_len=len(route), miss_rate=miss / n, align_med_dist_m=float(np.median(r_dist)),
                pct_dt_1s=float(np.mean(dt[1:] == 1)), pct_unlabeled=float(np.mean(r_dist > MAX_ALIGN_M)),
                route_km=float(np.linalg.norm(B[route]-A[route],axis=1).sum()/1000), gps_path_km=float(d_prev[1:].sum()/1000), max_gap_s=float(np.nanmax(dt)))
    return df, info

def main(root=None):
    root = find_data_root(root)
    print("Using dataset folder:", root)
    DATA_DIR.mkdir(exist_ok=True)
    infos = []
    for split in ["train", "student_test"]:
        dfs = []
        folders = sorted(glob.glob(os.path.join(str(root), split, "*")))
        if not folders: raise SystemExit(f"ERROR: no trip folders in {os.path.join(str(root), split)}")
        for folder in folders:
            df, info = process_trip(folder, split)
            dfs.append(df); infos.append(info)
            print(split, info["trip_id"], f"pts={info['n_points']} miss={info['miss_rate']:.3f} unlabeled={info['pct_unlabeled']:.3f}")
        pd.concat(dfs).to_csv(DATA_DIR / f"{split}_candidates.csv", index=False)
    pd.DataFrame(infos).to_csv(DATA_DIR / "build_report.csv", index=False)
    print("Done. Files written to", DATA_DIR)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)