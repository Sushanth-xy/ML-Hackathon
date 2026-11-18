# PathSnap: Distributed Map-Matching & Route Reconstruction

Team project for UE24CS352A Machine Learning (PES University). PathSnap reconstructs a vehicle's road-level route from noisy GPS points using a **candidate-ranking classifier + HMM/Viterbi decoder**, served by a **FastAPI backend (Laptop 1)** and an **evaluator UI (Laptop 2)**.

> This is the **team-wide README**. Each member also has a personal README in `docs/roles/`. Read this file fully first, then your own.

---

## 1. Team and ownership

| Member | Role | Personal README | Compute |
|---|---|---|---|
| **P1** | Data and Candidates | `docs/roles/P1_data_candidates.md` | Laptop (Spark optional) |
| **P2** | Modelling Lead | `docs/roles/P2_modelling_spark.md` | DGX Spark |
| **P3** | Decoder and Evaluation | `docs/roles/P3_decoder_evaluation.md` | Laptop |
| **P4** | Systems and Delivery | `docs/roles/P4_systems_delivery.md` | Laptop 1 (API), Laptop 2 (UI) |

Fill in names, GitHub handles and contact details here:

| Role | Name | GitHub | Phone/Discord |
|---|---|---|---|
| P1 | | | |
| P2 | | | |
| P3 | | | |
| P4 | | | |

---

## 2. System overview

```
 raw files (.track .nodes .arcs .route)
          |
          v
 [P1] load -> clean -> project to metric CRS -> spatial index
          |
          v
 [P1] candidate generation (arcs within R metres of each GPS point)
          |
          v
 [P1] features + labels  ---> candidates.parquet, folds.json
          |
          v
 [P2] train/compare >=5 models, Optuna on best ---> preds_<model>.parquet, artifacts/
          |
          v
 [P3] HMM + Viterbi over candidates (uses P2 probabilities + road connectivity)
          |
          v
 [P4] FastAPI (Laptop 1)  <--HTTP-->  UI (Laptop 2)
          ^
          |-- server-side ground truth (never sent to model or UI)
```

**Problem framing (locked):** binary classification over *(GPS point, candidate arc)* rows. `label = 1` if the candidate is the reference arc for that point. The model outputs a probability per row. The HMM turns those into a coherent route.

---

## 3. Repository layout

```
pathsnap/
├── README.md                 # this file
├── requirements.txt          # must install on Spark (ARM64) AND Laptop 1
├── .gitignore                # data/raw, *.db (large), .env
├── data/
│   ├── raw/                  # NOT committed (too large); see section 6
│   └── processed/            # candidates.parquet, folds.json, network files (small samples committed)
├── src/
│   ├── io.py                 # P1: readers for .track/.nodes/.arcs/.route
│   ├── geo.py                # P1: projection, distances, bearings
│   ├── candidates.py         # P1: spatial index + candidate generation
│   ├── labels.py             # P1: route -> per-point labels
│   ├── features.py           # P1: feature engineering (single source of truth)
│   ├── metrics.py            # SHARED: all metric definitions (P2 + P3 co-own)
│   ├── hmm.py                # P3: emission, transition, Viterbi
│   └── pipeline.py           # P4: raw trip -> features -> model -> HMM -> route
├── notebooks/                # eda.ipynb (P1), experiments (P2), hmm analysis (P3)
├── training/
│   ├── run_model.py          # P2: harness
│   ├── configs/              # P2: one YAML per model
│   └── bayes_opt.py          # P2: Optuna
├── results/
│   ├── eda/  models/  optuna/  hmm/  maps/    # CSV, JSON, PNG, Folium HTML
├── artifacts/                # model file, scaler, feature_list.json, hmm_config.json
├── api/                      # P4: FastAPI app
├── ui/                       # P4: evaluator UI
├── report/                   # P4 assembles the final PDF
└── docs/
    ├── roles/                # personal READMEs
    └── decisions.md          # running log of data-quality issues and decisions (everyone appends)
```

---

## 4. Locked conventions (do not deviate without telling everyone)

### 4.1 Coordinates
- Files are `longitude latitude` (check order on the first load and record it in `docs/decisions.md`).
- All distances are computed in a **projected metric CRS** (pick the UTM zone from the data's extent using `pyproj`). Never use degree-space Euclidean distance.
- The same CRS is used everywhere. P1 defines it in `src/geo.py`; everyone else imports it.

### 4.2 Partitions and leakage
| Partition | Trips | Who may use | For what |
|---|---|---|---|
| Training | 80 | everyone | fitting, CV, Optuna, decoder tuning |
| Student-test | 10 | everyone | development comparison, decoder tuning |
| Evaluator-test | 10 | **P4's API only, server-side** | final demo only |

- No evaluator-test labels in any notebook, script, feature selection or tuning step.
- Scalers or any learned preprocessing: fit on the training folds only.
- Never use `.route` at prediction time. It exists only for training labels and server-side scoring.

### 4.3 Folds
- `data/processed/folds.json` maps `trip_id -> fold` (GroupKFold by trip, 5 folds). P1 creates it once. **Nobody regenerates it.**
- `data/processed/splits.json` lists the train, student-test and evaluator-test trip IDs.
- Global seed: `SEED = 42` (defined in `src/config.py`).

### 4.4 Candidate table schema (`candidates.parquet`)
| Column | Meaning |
|---|---|
| `trip_id` | trip identifier |
| `obs_idx` | 0-based GPS row in the trip |
| `arc_id` | 0-based row index in `.arcs` |
| `dist_m` | perpendicular distance from the point to the arc |
| `frac_along` | projection position along the arc, 0 to 1 |
| `heading_diff_deg` | GPS movement bearing vs arc bearing |
| `speed_mps` | speed at the point |
| `dt_prev_s` | time since the previous point |
| `arc_len_m` | arc length |
| `dist_rank` | rank of this candidate by distance at this point |
| `n_cands` | number of candidates at this point |
| `label` | 1 if the candidate is the reference arc, else 0 (NaN if no reference route) |
| `split` | `train` / `student` / `evaluator` |

P1 may add columns, but only by appending, and must announce them. Names above are fixed.

### 4.5 Metric definitions (in `src/metrics.py`; import, never reimplement)
- **Macro F1:** macro average of F1 over the two classes (0 and 1) on candidate rows. State this in the report.
- **Observation-level Top-1 Accuracy:** per GPS point, take the highest-scoring candidate and compare it with the reference arc. Points whose reference arc is absent from the candidate set count as **wrong**, and the miss rate is reported separately.
- **Candidate Miss Rate:** fraction of points whose reference arc is not in the candidate set.
- **Log Loss:** binary log loss on candidate rows (clip probabilities).
- **Route Continuity:** share of consecutive predicted-arc transitions that are valid (same arc, or `to_node` of A equals `from_node` of B). Also report a variant allowing up to 2 hops.
- **Sequence-level metric:** fraction of a trip's reference arcs that appear in the predicted route (recall) and precision of predicted arcs, plus an edit-based or Jaccard score. P3 finalises this definition in week one.

### 4.6 Handoff contracts
| From -> To | Artifact | Format |
|---|---|---|
| P1 -> all | Candidate/feature table | `candidates.parquet` (schema above) |
| P1 -> all | Folds, splits | `folds.json`, `splits.json` |
| P1 -> P3, P4 | Network | `arcs.parquet`, `nodes.parquet` (projected), adjacency as `adjacency.pkl` or `.npz` |
| P2 -> P3 | Probabilities | `preds_<model>.parquet`: `trip_id, obs_idx, arc_id, prob`. **Out-of-fold for training trips**, plain predictions for student-test |
| P2 -> P4 | Model package | `artifacts/model.*`, `scaler.*`, `feature_list.json`, `predict_proba(df) -> np.ndarray` |
| P3 -> P4 | Decoder | `decode(cands_df, probs, network) -> list[arc_id]`, `artifacts/hmm_config.json` |

---

## 5. Environment setup

### 5.1 Everyone (laptops)
```bash
git clone <REPO_URL> pathsnap && cd pathsnap
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Use **Python 3.10 or 3.11** on every machine so pickles load everywhere. Record `python --version` in `docs/decisions.md`.

### 5.2 DGX Spark (P2)
See `docs/roles/P2_modelling_spark.md`. The Spark is ARM64: use a container or confirmed ARM wheels, and keep pinned versions compatible with the laptops.

### 5.3 Pinned core libraries (suggested; confirm and freeze in week one)
`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `xgboost` or `lightgbm`, `optuna`, `pyproj`, `shapely`, `scipy`, `networkx`, `folium`, `matplotlib`, `fastapi`, `uvicorn`, `pydantic`, `requests`, `streamlit` (or your UI stack).
Rule: **the version used to save a model must equal the version used to load it on Laptop 1.**

---

## 6. Data handling
- Raw data is placed in `data/raw/` locally by each member (do not commit it unless it is tiny).
- Processed outputs go to `data/processed/`. Commit only small samples. Large files: share via drive/`rsync`/`scp`, and document where they live.
- Evaluator-test **labels** are stored only on Laptop 1, outside the UI's reach.

---

## 7. Git workflow
- `main` is always runnable. Nobody pushes broken code to `main`.
- Branches: `p1/<topic>`, `p2/<topic>`, `p3/<topic>`, `p4/<topic>`. Open PRs, and get one teammate to glance at it before merging.
- Commit small, commit often. Pull before you start working.
- Only the owner edits their `src/` modules. Others request changes via an issue or message.
- **Never commit secrets** (`.env`, tokens, passwords). Run `git status` and a final secrets grep before each push.
- Results committed to the repo are the only numbers allowed in the report.

---

## 8. Master timeline

Scale the hours to your actual event length. The example column assumes a 12-hour working window; phases are what matter.

| Phase | Budget | 12h example | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|---|
| **0. Kickoff** | 5% | 0:00-0:40 | Read PDFs, load data | Read PDFs, Spark access | Read PDFs, plan HMM | Read PDFs, create repo |
| **1. Setup and EDA** | 15% | 0:40-2:30 | EDA, projection, `src/io.py`, `geo.py` | Spark env, harness skeleton | State space and metrics design | API/UI skeleton with dummy predictor |
| **2. Data ready** | 15% | 2:30-4:20 | Candidates, labels, features v1, **10-trip sample first**, then full table, `folds.json` | Test harness on sample | Distance-only HMM prototype | `/trips`, `/predict` wired to stub |
| **3. Modelling** | 25% | 4:20-7:20 | Candidate-miss analysis, features v2, EDA maps | Train all 6-7 models, comparison plots, `preds_*.parquet` | HMM with real probabilities, metrics code | UI map layers, logging, CORS, config |
| **4. Tuning** | 15% | 7:20-9:10 | Folium EDA maps, edge cases | Optuna, tuned vs untuned, export artifacts | Decoder tuning, classifier vs HMM, failure cases | Load real artifacts into API |
| **5. Integration** | 10% | 9:10-10:20 | Check API vs local results | Laptop-1 load test, inference timing | Final Folium prediction maps | Two-laptop dry run, all 10 trips |
| **6. Report and freeze** | 15% | 10:20-12:00 | Section 2.2 | Sections 3, 4 | Sections 5, 6 | Section 7, README, PDF, push |

### Hard checkpoints
| # | Checkpoint | Owner | Must be true |
|---|---|---|---|
| C1 | Repo + env works for everyone | P4 | all four can run `pip install -r requirements.txt` |
| C2 | **10-trip candidate sample released** | P1 | schema matches section 4.4, labels present |
| C3 | First model's predictions delivered | P2 | `preds_<model>.parquet` in agreed format |
| C4 | HMM runs end to end on a few trips | P3 | outputs a route and a continuity score |
| C5 | Real pipeline in the API | P4 | `/predict` works on student-test trips |
| C6 | Two-laptop dry run passes on all 10 trips | P4 | logs visible, no crashes |
| C7 | Code freeze | all | only report/README changes after this |

First-phase evaluation happens before lunch. Confirm with organisers exactly what is expected, and make sure C2 and a prototype demo are ready by then.

---

## 9. Report ownership

| Section | Owner | Notes |
|---|---|---|
| 2.2 Dataset and EDA | P1 | label partition on every figure |
| 3 Model comparison + per-model Q&A | P2 | 9 compulsory questions for every model |
| 4 Bayesian optimisation | P2 | tuned vs untuned table |
| 5 HMM and Viterbi | P3 | include cases where HMM did not help |
| 6 Folium visualisation | P3 | screenshot in PDF, code in repo |
| 7 API, UI, deployment | P4 | UI screenshots, API logs from the receiving laptop |
| Final assembly | P4 | one consolidated PDF |

Every number in the PDF must be reproducible from a file in `results/`.

---

## 10. Integration checklist (final 90 minutes)
- [ ] API starts with one command on Laptop 1 and prints its IP and port
- [ ] UI on Laptop 2 lists all 10 trips and reaches the API over the network
- [ ] All 10 trips return a route + metrics without errors
- [ ] API logs show every request (trip ID, time, latency, status)
- [ ] No ground truth appears in any API response field consumed by the model, or in the UI
- [ ] Models load with the pinned versions on Laptop 1
- [ ] README instructions followed from a clean clone work
- [ ] Repo is public, no secrets, model weights present (size permitting)
- [ ] Report numbers match `results/` files

---

## 11. Academic integrity
- Everyone must be able to explain every part of the code, experiments and results.
- Do not copy from other teams. Do not share models, configs or results across teams.
- AI tools may assist, but the experiments and analysis must be ours. Disclose AI assistance per the event policy.
- Log which AI tools were used and for what in `docs/ai_disclosure.md`.

---

## 12. Running the system (fill in as components land)
```bash
# P1: build candidates
python -m src.candidates --out data/processed/candidates.parquet

# P2: train one model
python training/run_model.py --model xgb --config training/configs/xgb.yaml

# P2: Bayesian optimisation
python training/bayes_opt.py --model xgb --trials 60

# P4: start API (Laptop 1)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# P4: start UI (Laptop 2)
streamlit run ui/app.py   # set API_URL=http://<laptop1-ip>:8000
```
