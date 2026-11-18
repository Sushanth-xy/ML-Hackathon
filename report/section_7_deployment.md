# Section 7: API, UI, and Distributed Deployment
**Authors:** Sudha (P4: Systems Lead) & PathSnap Engineering Team  
**Evaluation:** UE24CS352A Machine Learning — PES University

---

## 7.1 System Architecture & Two-Laptop Topology

PathSnap is designed as a distributed, decoupled client-server architecture conforming to the official evaluation specification:
- **Laptop 1 (FastAPI Backend Server):** Houses the serialized machine learning artifacts (`artifacts/best_model_pointwise.joblib`, `artifacts/best_model_probability.joblib`, `scaler.joblib`, `imputer.joblib`, `hmm_config.json`), road network graph topologies, and ground-truth validation routes. It exposes a high-performance REST API running on Uvicorn bound to `0.0.0.0:8000`.
- **Laptop 2 (Evaluator UI Client):** Runs the lightweight evaluation interface ([`ui/index.html`](../ui/index.html)), connecting across the local network via HTTP. Evaluators can select any of the 10 evaluation trips, trigger inference, and inspect the resulting interactive Leaflet map and live performance metrics HUD.

```
┌──────────────────────────────────────────────┐          HTTP REST          ┌──────────────────────────────────────────────┐
│                  LAPTOP 2                    │────────────────────────────>│                  LAPTOP 1                    │
│           Evaluator UI Interface             │                             │           FastAPI Backend Engine             │
│  ├── Dynamic API Address Input               │<────────────────────────────│  ├── In-Memory Pipeline (Loaded Once)        │
│  ├── Trip Selector Dropdown (GET /trips)     │   JSON Response Payload     │  │   ├── Pointwise Champion (HistGB)         │
│  ├── Interactive Multi-Layer Leaflet Map     │                             │  │   ├── Probability Champion (GBDT/MLP)     │
│  └── Live Performance HUD (Top-1, Cont, F1)  │                             │  │   └── Log-Space HMM Viterbi Decoder       │
│                                              │                             │  ├── Server-Side Ground Truth Firewalled     │
│                                              │                             │  └── Dual File/Console Request Logging       │
└──────────────────────────────────────────────┘                             └──────────────────────────────────────────────┘
```

---

## 7.2 API Specification & Endpoints

### 1. `GET /health`
- **Purpose:** Liveness and connectivity check between Laptop 2 and Laptop 1.
- **Request:** None.
- **Response:**
  ```json
  {
    "status": "ok",
    "timestamp": 1763456789.12
  }
  ```

### 2. `GET /trips`
- **Purpose:** Supplies the evaluator UI with the list of available evaluation trip IDs.
- **Request:** None.
- **Response:**
  ```json
  {
    "trips": [
      "00000000", "00000007", "00000009", "00000040", "00000052",
      "00000056", "00000065", "00000076", "00000077", "00000091"
    ]
  }
  ```

### 3. `POST /predict`
- **Purpose:** Executes the end-to-end inference pipeline for the selected trip.
- **Request Body:**
  ```json
  {
    "trip_id": "00000000"
  }
  ```
- **Response Payload:**
  ```json
  {
    "trip_id": "00000000",
    "n_obs": 2503,
    "gps": [[56.251559, 54.963519], [56.251523, 54.963143], "..."],
    "predicted_route": {
      "arc_ids": [17630, 17633, 211, 17638, "..."],
      "polyline": [[56.2516, 54.9635], [56.2518, 54.9639], "..."]
    },
    "reference_route": {
      "polyline": [[56.2516, 54.9635], [56.2518, 54.9639], "..."]
    },
    "metrics": {
      "top1_classifier": 91.65,
      "top1_hmm": 93.65,
      "route_continuity": 104.2,
      "sequence_score": 0.9010
    },
    "stage_latencies_ms": {
      "features": 57.4,
      "classifier": 101.7,
      "hmm": 149.1
    },
    "latency_ms": 308
  }
  ```

---

## 7.3 Ground-Truth Security & Anti-Leakage Protocol

1. **Server-Side Confinement:** Ground-truth reference routes (`.route`) remain strictly server-side on Laptop 1. They are loaded solely during metric computation inside `Pipeline.run()`.
2. **Zero Feature Contamination:** At no point in candidate generation, feature scaling, model scoring, or Viterbi trellis decoding are ground-truth labels exposed to the algorithms.
3. **No Hardcoded Paths:** Every single trip request executes dynamic feature scaling, probability estimation, and graph-search Viterbi decoding in real-time.

---

## 7.4 Request Logging & Performance Profiling

In compliance with faculty guidelines, Laptop 1 maintains visible, structured request logs output simultaneously to both terminal CMD and `api_requests.log`:

```
2026-11-18 14:27:48,102 [INFO] Pipeline loaded successfully with 10 trips.
2026-11-18 14:27:50,314 [INFO] GET /health HTTP/1.1 200 OK (0.4ms)
2026-11-18 14:27:50,358 [INFO] GET /trips HTTP/1.1 200 OK (0.8ms)
2026-11-18 14:27:52,662 [INFO] POST /predict trip=00000000 | obs=2503 | Clf Top-1=91.65% | HMM Top-1=93.65% | Seq F1=0.9010 | Latency=308.2ms
2026-11-18 14:28:01,114 [INFO] POST /predict trip=00000076 | obs=1486 | Clf Top-1=84.05% | HMM Top-1=90.31% | Seq F1=0.9199 | Latency=185.4ms
2026-11-18 14:28:09,842 [INFO] POST /predict trip=00000077 | obs=3911 | Clf Top-1=70.65% | HMM Top-1=79.42% | Seq F1=0.9008 | Latency=442.1ms
```

---

## 7.5 Step-by-Step Two-Laptop Deployment Guide

### On Laptop 1 (FastAPI Server):
1. **Network Binding:** Connect to the shared Wi-Fi / phone hotspot. Identify Laptop 1's IP address:
   ```bash
   # macOS / Linux:
   ifconfig | grep "inet "
   # Windows:
   ipconfig
   ```
2. **Firewall Rule (Windows Admin PowerShell):**
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```
3. **Launch API Server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

### On Laptop 2 (Evaluator UI Client):
1. Open [`ui/index.html`](file:///Users/sushanth/Code/hack/ML-Hackathon/ui/index.html) in Google Chrome or Safari.
2. In the top bar **API** input field, enter Laptop 1's address (e.g. `http://192.168.1.45:8000`) and click **Test**.
3. Verify the status indicator turns green (**Connected**).
4. Select any trip from the dropdown and click **Evaluate Trip**.
5. Observe the live map rendering and the metrics HUD updating in real-time.
