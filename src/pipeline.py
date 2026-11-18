"""
PathSnap - End-to-End Inference Pipeline
Role P4: Systems and Delivery (Sudha)

Loads model artifacts, scalers, HMM configuration, and road networks once at startup.
Processes trip inference requests: Raw Data -> Features -> Model Probs -> HMM Viterbi -> Server-side Metrics.
"""

import os, time, json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np, pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.hmm import decode as hmm_decode, build_network, DEFAULT_CONFIG as HMM_DEFAULT_CONFIG
from src.metrics import (
    compute_top1_accuracy,
    compute_sequence_f1,
    compute_route_continuity,
    compress_consecutive_arcs
)

FEATURES = [
    "dist_m", "dist_minus_min", "dist_rank", "frac_along",
    "heading_diff", "heading_cos", "arc_len", "speed",
    "dt", "step_dist", "end_deg_out", "start_deg_in", "has_reverse", "is_moving"
]

class Pipeline:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        t0 = time.time()
        self.artifacts_dir = artifacts_dir or (PROJECT_ROOT / "artifacts")
        
        print("[Pipeline] Loading serialized machine learning artifacts...")
        self.model_pt = joblib.load(self.artifacts_dir / "best_model_pointwise.joblib")
        self.model_prob = joblib.load(self.artifacts_dir / "best_model_probability.joblib")
        self.scaler = joblib.load(self.artifacts_dir / "scaler.joblib")
        self.imputer = joblib.load(self.artifacts_dir / "imputer.joblib")
        
        config_path = self.artifacts_dir / "hmm_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.hmm_config = json.load(f)
        else:
            self.hmm_config = HMM_DEFAULT_CONFIG
            
        print("[Pipeline] Loading student-test candidate table...")
        cands_path = PROJECT_ROOT / "data" / "student_test_candidates.csv"
        self.cands_df = pd.read_csv(cands_path)
        
        # Discover available trips
        trip_ints = sorted(self.cands_df["trip_id"].unique())
        self.trip_ids = [f"{t:08d}" for t in trip_ints]
        
        # Pre-cache trip candidate splits
        self.trip_cands_cache = {
            f"{t:08d}": self.cands_df[self.cands_df["trip_id"] == t].copy()
            for t in trip_ints
        }
        
        # Cache road network data
        self.network_cache = {}
        for tid in self.trip_ids:
            self._load_network_for_trip(tid)
            
        elapsed = time.time() - t0
        print(f"[Pipeline] Initialized in {elapsed:.2f}s. Loaded {len(self.trip_ids)} trips: {self.trip_ids}")

    def _load_network_for_trip(self, trip_id: str):
        folder = PROJECT_ROOT / "Student folder" / "student_test" / trip_id
        if not folder.exists():
            folder = PROJECT_ROOT / "Student folder" / "train" / trip_id
            
        if not folder.exists():
            return
            
        arcs = np.loadtxt(folder / f"{trip_id}.arcs", dtype=int)
        nodes = np.loadtxt(folder / f"{trip_id}.nodes") # lon, lat
        track = np.loadtxt(folder / f"{trip_id}.track") # lon, lat, time
        
        route_path = folder / f"{trip_id}.route"
        route = np.loadtxt(route_path, dtype=int).ravel() if route_path.exists() else np.array([], dtype=int)
        
        # Build network graph for HMM
        arcs_df = pd.DataFrame(arcs, columns=["from_node", "to_node"])
        network_graph = build_network(arcs_df)
        
        # Compute arc lengths in meters
        arc_lengths = {}
        for aid, (u, v) in enumerate(arcs):
            p1, p2 = nodes[u], nodes[v]
            dx = np.radians(p2[0] - p1[0]) * 6371000.0 * np.cos(np.radians(0.5 * (p1[1] + p2[1])))
            dy = np.radians(p2[1] - p1[1]) * 6371000.0
            arc_lengths[aid] = float(np.sqrt(dx**2 + dy**2))
            
        self.network_cache[trip_id] = {
            "arcs": arcs,
            "arcs_df": arcs_df,
            "network_graph": network_graph,
            "nodes": nodes,
            "track": track,
            "route": route,
            "arc_lengths": arc_lengths
        }

    def run(self, trip_id: str) -> Dict[str, Any]:
        """
        Executes end-to-end inference on one trip.
        Returns full payload required by FastAPI and UI.
        """
        t_start = time.perf_counter()
        
        # Normalize trip_id formatting (e.g. '0' or '00000000')
        trip_str = f"{int(trip_id):08d}" if str(trip_id).isdigit() else str(trip_id)
        if trip_str not in self.trip_cands_cache or trip_str not in self.network_cache:
            raise KeyError(f"Trip '{trip_id}' not found in pipeline index.")
            
        # 1. Feature Retrieval / Extraction Stage
        t0 = time.perf_counter()
        trip_cands = self.trip_cands_cache[trip_str].copy()
        net = self.network_cache[trip_str]
        
        X = trip_cands[FEATURES].copy()
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        t_feat = (time.perf_counter() - t0) * 1000.0
        
        # 2. Classifier Prediction Stage
        t0 = time.perf_counter()
        prob_pt = self.model_pt.predict_proba(X_imp)[:, 1]
        prob_cal = self.model_prob.predict_proba(X_scaled)[:, 1]
        
        trip_cands["prob"] = prob_cal
        trip_cands["prob_pt"] = prob_pt
        
        # Map point_idx to obs_idx and cand_arc_id to arc_id for HMM compatibility
        trip_cands["obs_idx"] = trip_cands["point_idx"]
        trip_cands["arc_id"] = trip_cands["cand_arc_id"]
        trip_cands["dt_prev_s"] = trip_cands["dt"] if "dt" in trip_cands.columns else 1.0
        
        idx_max = trip_cands.groupby("point_idx")["prob_pt"].idxmax()
        clf_pred_df = trip_cands.loc[idx_max].sort_values("point_idx")
        clf_pred_arcs = clf_pred_df["cand_arc_id"].values.astype(int).tolist()
        t_clf = (time.perf_counter() - t0) * 1000.0
        
        # 3. HMM Viterbi Route Reconstruction Stage
        t0 = time.perf_counter()
        try:
            hmm_pred_arcs = hmm_decode(
                cands_df=trip_cands[["obs_idx", "arc_id", "dist_m", "dt_prev_s"]],
                probs=prob_cal,
                network=net["network_graph"],
                config=self.hmm_config
            )
        except Exception as e:
            # Fallback to classifier argmax if graph discontinuity is encountered
            print(f"[Warning] HMM decode fallback: {e}")
            hmm_pred_arcs = clf_pred_arcs
            
        t_hmm = (time.perf_counter() - t0) * 1000.0
        
        # 4. Compute Server-Side Metrics
        pt_groups = trip_cands.groupby("point_idx")
        top1_clf = float(clf_pred_df["label"].mean()) if "label" in clf_pred_df.columns else 0.0
        
        # HMM Top-1
        if "label" in trip_cands.columns:
            labels_per_pt = []
            for i, (_, g) in enumerate(pt_groups):
                matched = g[g["cand_arc_id"] == hmm_pred_arcs[i]]
                if len(matched) > 0:
                    labels_per_pt.append(matched["label"].iloc[0])
                else:
                    labels_per_pt.append(0)
            top1_hmm = float(np.mean(labels_per_pt)) if labels_per_pt else top1_clf
        else:
            top1_hmm = 0.0
            
        ref_route = net["route"]
        compressed_hmm = compress_consecutive_arcs([a for a in hmm_pred_arcs if a >= 0])
        seq_metrics = compute_sequence_f1(compressed_hmm, ref_route)
        route_continuity = compute_route_continuity(hmm_pred_arcs, net["network_graph"], 1)
        
        # 5. Build Polylines for UI (Leaflet expects [lat, lon])
        track = net["track"]
        nodes = net["nodes"]
        arcs = net["arcs"]
        
        # GPS polyline / points (subsample by 2 for snappy UI transfer)
        gps_coords = [[float(pt[1]), float(pt[0])] for pt in track[::2]]
        
        # Predicted HMM Route Polyline
        hmm_coords = []
        for aid in compressed_hmm:
            if 0 <= aid < len(arcs):
                u, v = arcs[aid]
                hmm_coords.append([float(nodes[u][1]), float(nodes[u][0])])
                hmm_coords.append([float(nodes[v][1]), float(nodes[v][0])])
            
        # Reference Route Polyline (ground truth for display only, never exposed as feature)
        ref_coords = []
        for aid in ref_route:
            if 0 <= aid < len(arcs):
                u, v = arcs[aid]
                ref_coords.append([float(nodes[u][1]), float(nodes[u][0])])
                ref_coords.append([float(nodes[v][1]), float(nodes[v][0])])
                
        # Classifier-only Route Polyline
        compressed_clf = compress_consecutive_arcs([a for a in clf_pred_arcs if a >= 0])
        clf_coords = []
        for aid in compressed_clf:
            if 0 <= aid < len(arcs):
                u, v = arcs[aid]
                clf_coords.append([float(nodes[u][1]), float(nodes[u][0])])
                clf_coords.append([float(nodes[v][1]), float(nodes[v][0])])
                
        # Background Road Network Arcs (candidate segments)
        cand_arcs = set(trip_cands["cand_arc_id"].unique())
        network_segments = []
        for aid in cand_arcs:
            if 0 <= aid < len(arcs):
                u, v = arcs[aid]
                network_segments.append([
                    [float(nodes[u][1]), float(nodes[u][0])],
                    [float(nodes[v][1]), float(nodes[v][0])]
                ])
            
        total_latency = (time.perf_counter() - t_start) * 1000.0
        
        return {
            "trip_id": trip_str,
            "n_obs": len(track),
            "gps": gps_coords,
            "predicted_route": {
                "arc_ids": compressed_hmm,
                "polyline": hmm_coords
            },
            "classifier_route": {
                "arc_ids": compressed_clf,
                "polyline": clf_coords
            },
            "reference_route": {
                "polyline": ref_coords
            },
            "road_network": network_segments,
            "metrics": {
                "top1_classifier": round(top1_clf * 100.0, 2),
                "top1_hmm": round(top1_hmm * 100.0, 2),
                "route_continuity": round(route_continuity, 1),
                "sequence_score": round(seq_metrics["f1"], 4)
            },
            "stage_latencies_ms": {
                "features": round(t_feat, 1),
                "classifier": round(t_clf, 1),
                "hmm": round(t_hmm, 1)
            },
            "latency_ms": int(total_latency)
        }
