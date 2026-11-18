# P4: Systems and Delivery (API, UI, Deployment, Report Assembly)

**You make it work live.** The evaluator will use your system on two laptops, so reliability matters more than polish.

## Mission
Build the FastAPI backend (Laptop 1), the evaluator UI (Laptop 2), the two-laptop deployment, logging, the final README/repo structure and the consolidated PDF report.

## Inputs / Outputs
| Inputs | Outputs |
|---|---|
| P1's `src/features.py`, network files; P2's model artifacts + `predict_proba` wrapper; P3's `decode()` + `hmm_config.json` | `api/`, `ui/`, `src/pipeline.py`, run scripts, README, request logs, screenshots, final PDF, repo hygiene |

## You own (files)
`api/`, `ui/`, `src/pipeline.py`, `requirements.txt`, `README.md`, `.gitignore`, `report/`, `docs/ai_disclosure.md`

## Timeline
| When (12h example) | Task | Done when |
|---|---|---|
| 0:00-0:40 | Create the repo (public), add team, folders, `.gitignore`, common README | Everyone can clone and push (C1) |
| 0:40-2:30 | FastAPI skeleton with **dummy predictor**, UI skeleton listing 10 fake trips. Confirm both laptops can see each other on the network | UI on Laptop 2 gets a response from API on Laptop 1 |
| 2:30-4:20 | `/trips`, `/predict` wired to stub, request logging, CORS, configurable API URL | Dummy end-to-end works |
| 4:20-7:20 | UI map layers (GPS, reference, predicted), metrics panel. Define `pipeline.py` interface | UI looks demo-ready |
| 7:20-9:10 | Plug in the real feature code, model, decoder (checkpoint C5). Load ground truth server-side | Works on student-test trips |
| 9:10-10:20 | **Two-laptop dry run on all 10 trips** (C6). Fix firewall/IP problems. Capture screenshots | All 10 trips pass |
| 10:20-12:00 | Final README, secrets check, assemble PDF, push (C7) | Clean-clone test passes |

## Architecture
```
Laptop 2 (UI)  --HTTP-->  Laptop 1 (FastAPI)
                          ├─ loads model, scaler, hmm_config, network (once at startup)
                          ├─ loads ground truth (server-side only)
                          └─ POST /predict {trip_id} -> features -> model -> HMM -> route + metrics
```

## API spec
| Endpoint | Purpose | Response |
|---|---|---|
| `GET /health` | liveness check | `{"status":"ok"}` |
| `GET /trips` | the 10 evaluator trip IDs | `{"trips":["t01", ...]}` |
| `POST /predict` body `{"trip_id": "..."}` | full pipeline for one trip | see below |

Example `/predict` response:
```json
{
  "trip_id": "t01",
  "n_obs": 812,
  "gps": [[lat, lon], ...],
  "predicted_route": {"arc_ids": [..], "polyline": [[lat, lon], ...]},
  "reference_route": {"polyline": [[lat, lon], ...]},
  "metrics": {"top1_classifier": 0.0, "top1_hmm": 0.0, "route_continuity": 0.0, "sequence_score": 0.0},
  "latency_ms": 0
}
```
Rules:
- **Ground truth stays server-side.** It is read only inside the scoring step after prediction. The model and decoder must never receive labels. The reference polyline returned for display is permitted for the UI, but not the per-point labels as features.
- No hardcoded outputs. Everything goes through the real pipeline. A fake path through the API is prohibited.
- Load heavy objects once at startup (`@app.on_event("startup")` / lifespan), not per request.
- Return clean errors (404 for unknown trip, 500 with a message), never a stack trace to the UI.

Skeleton:
```python
# api/main.py
import time, logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline import Pipeline   # loads artifacts once

logging.basicConfig(filename="api_requests.log", level=logging.INFO,
                    format="%(asctime)s %(message)s")
app = FastAPI(title="PathSnap")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipe = Pipeline()

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/trips")
def trips(): return {"trips": pipe.trip_ids}

@app.post("/predict")
def predict(req: dict):
    t0 = time.time()
    tid = req.get("trip_id")
    if tid not in pipe.trip_ids:
        raise HTTPException(404, "unknown trip")
    out = pipe.run(tid)
    out["latency_ms"] = int((time.time() - t0) * 1000)
    logging.info("POST /predict trip=%s latency_ms=%d", tid, out["latency_ms"])
    return out
```

## Logging (evaluators will ask to see it)
- Use uvicorn's access log plus your own line per request: timestamp, trip ID, latency, status, stage timings (features, model, HMM).
- Keep the CMD window visible on Laptop 1 during the demo. Practise showing it.
- Screenshots of successful responses from the receiving laptop's CMD are required for the report.

## UI (Laptop 2)
- Lists the 10 trips (fetched from `GET /trips`), a "Run" button, loading indicator.
- Map (Folium via Streamlit, or Leaflet in HTML/JS) with layers: GPS (blue), reference route (green), predicted route (red), layer control, legend.
- Metrics panel: Top-1 (classifier vs HMM), route continuity, sequence score, latency.
- API address is configurable (env var or text box). No hardcoded `localhost`.
- Graceful error message if the API is unreachable.

## Two-laptop network setup
1. Same Wi-Fi or one laptop's hotspot (avoid captive/college Wi-Fi that isolates clients; a phone hotspot is usually safest).
2. Start API bound to all interfaces: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
3. Find Laptop 1's IP: `ipconfig` (Windows) / `ifconfig` or `ip a` (Linux/macOS).
4. Open the firewall (Windows, admin PowerShell):
   `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
5. From Laptop 2, test: `curl http://<laptop1-ip>:8000/health`, and then open the UI with `API_URL=http://<laptop1-ip>:8000`.
6. Write the exact working commands into the README. Rehearse twice.

## Dependency discipline
- One `requirements.txt` with pinned versions that installs on the Spark (ARM64), Laptop 1 and Laptop 2.
- Get P2's artifacts onto Laptop 1 **early** and test-load them (don't wait for the final model).
- Separate `requirements-ui.txt` if the UI needs fewer packages.

## README (final, repo root)
Must include: project summary, repo structure, installation, how to train (brief), how to run API, how to run UI, two-laptop instructions, how to regenerate maps, where results live, known limitations. Test it by following it on a clean clone.

## Report ownership
- Section 7: UI screenshots, API response screenshots (CMD on receiving laptop), deployment description, note that code matches what was demonstrated.
- Final assembly: merge P1-P3 sections into **one PDF** with a consistent format, check each figure is labelled with its partition, check all numbers against `results/`, add the AI-use disclosure.

## Pre-submission checklist
- [x] Repo public and accessible
- [x] Folders named clearly, README with install, execution, deployment steps
- [x] Model weights + config files present (size permitting)
- [x] No secrets: `git grep -iE "password|secret|api_key|token"`; check `.env` not committed
- [x] All code used for training, evaluation, UI, API committed
- [x] Clean clone -> install -> run works on a fresh venv
- [x] All 10 trips work through the API with visible logs
- [x] PDF is a single consolidated file and follows the required structure

## Pitfalls
- Firewall/hotspot blocking the connection, discovered late
- CORS missing, so the UI fails silently
- Loading the model inside each request (slow demo)
- Exposing labels to the UI or model
- Different library versions on Laptop 1 vs Spark, so pickles fail to load
- Hardcoded `localhost` in UI
- Waiting for the final model before testing integration (use stubs, then swap)

## Definition of done
- [x] Demo works from cold start in under 2 minutes using only README steps
- [x] All 10 trips succeed, latency acceptable
- [x] Logs, screenshots captured
- [x] Final PDF and repo pushed before the deadline
