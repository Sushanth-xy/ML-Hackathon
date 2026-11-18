"""
PathSnap - FastAPI Evaluation Backend (Laptop 1)
Role P4: Systems and Delivery (Sudha)

Endpoints:
- GET  /health   -> Liveness check: {"status": "ok"}
- GET  /trips    -> Returns list of 10 evaluation trip IDs
- POST /predict  -> Executes pipeline: features -> classifier -> HMM -> server-side metrics
"""

import time, os, sys, logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline

# Configure robust dual logging (file + terminal)
log_file = PROJECT_ROOT / "api_requests.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pathsnap.api")

app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching & Route Reconstruction Backend (Laptop 1)",
    version="1.0.0"
)

# Enable CORS for cross-laptop communication from Laptop 2 UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def add_universal_cors(request: Request, call_next):
    if request.method == "OPTIONS":
        from starlette.responses import Response
        res = Response(status_code=204)
    else:
        res = await call_next(request)
    res.headers["Access-Control-Allow-Origin"] = "*"
    res.headers["Access-Control-Allow-Methods"] = "*"
    res.headers["Access-Control-Allow-Headers"] = "*"
    res.headers["Access-Control-Allow-Private-Network"] = "true"
    return res

# Pre-load pipeline once at module load
print("[API] Initializing PathSnap pipeline (Laptop 1)...")
pipeline = Pipeline()
logger.info("Pipeline loaded successfully with %d trips.", len(pipeline.trip_ids))

class PredictRequest(BaseModel):
    trip_id: str

@app.get("/")
def serve_ui():
    """Serves the evaluator UI directly to Laptop 2 browser."""
    ui_path = PROJECT_ROOT / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return {"message": "PathSnap API is live. Access /trips or /health."}

@app.get("/health")
def health():
    """Liveness probe to confirm Laptop 1 server is reachable."""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/trips")
def get_trips():
    """Returns the list of 10 evaluation trip IDs for the evaluator UI dropdown."""
    return {"trips": pipeline.trip_ids}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Executes full map-matching and route reconstruction pipeline for one trip.
    Ground-truth labels stay server-side and are used strictly for accuracy scoring.
    """
    t0 = time.perf_counter()
    trip_id = req.trip_id.strip()
    
    # Normalize trip ID
    norm_id = f"{int(trip_id):08d}" if trip_id.isdigit() else trip_id
    
    if norm_id not in pipeline.trip_ids:
        logger.warning("404 Not Found: trip_id='%s' (available: %s)", trip_id, pipeline.trip_ids)
        raise HTTPException(
            status_code=404, 
            detail=f"Trip '{trip_id}' not found. Available trips: {pipeline.trip_ids}"
        )
        
    try:
        response_payload = pipeline.run(norm_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response_payload["total_api_latency_ms"] = round(elapsed_ms, 1)
        
        logger.info(
            "POST /predict trip=%s | obs=%d | Clf Top-1=%.2f%% | HMM Top-1=%.2f%% | Seq F1=%.4f | Latency=%.1fms",
            norm_id,
            response_payload["n_obs"],
            response_payload["metrics"]["top1_classifier"],
            response_payload["metrics"]["top1_hmm"],
            response_payload["metrics"]["sequence_score"],
            elapsed_ms
        )
        return response_payload
        
    except Exception as e:
        logger.exception("Inference failure for trip %s: %s", norm_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal inference failure on trip {norm_id}: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Bound to 0.0.0.0 so Laptop 2 can connect over Wi-Fi/hotspot
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
