# PathSnap: Distributed Map-Matching & Route Reconstruction

Team project for UE24CS352A Machine Learning (PES University). PathSnap reconstructs a vehicle's road-level route from noisy GPS points using a **candidate-ranking classifier + HMM/Viterbi decoder**, served by a **FastAPI backend (Laptop 1)** and an **evaluator UI (Laptop 2)**.

> This is the **team-wide master README**. Each member has their personal role guide linked below. Read this file fully first, then review your individual role guide.

---

## 1. Team and Ownership

| Role | Member | Focus Area | Role Guide | Primary Compute |
|---|---|---|---|---|
| **P1** | **Sumukh** | Data Preprocessing, Spatial Indexing & Candidates | [Sumukh_data_candidates.md](./Sumukh_data_candidates.md) | Laptop |
| **P2** | **Sushanth** | Modelling Lead, Multi-Model Zoo & Tuning | [Sushanth_modelling_spark.md](./Sushanth_modelling_spark.md) | DGX Spark (ARM64) / Laptop |
| **P3** | **Mahalakshmi** | HMM + Viterbi Decoder & Sequence Evaluation | [Mahalakshmi_decoder_evaluation.md](./Mahalakshmi_decoder_evaluation.md) | Laptop |
| **P4** | **Sudha** | Systems, Two-Laptop FastAPI/UI & Final Report | [Sudha_systems_delivery.md](./Sudha_systems_delivery.md) | Laptop 1 (API) & Laptop 2 (UI) |

---

## 2. Live Experimental Benchmark & Results

> **Status: Phase 1 Preprocessing & Multi-Model Training Complete.** All metrics below are empirically measured and reproducible using `src/build_dataset.py` and `training/train_models.py`.

### 2.1 Preprocessing & Candidate Generation Statistics
*(Generated from 90 trips in `Student folder/`, logged in [`data/build_report.csv`](./data/build_report.csv))*

| Metric | Training Partition (80 Trips) | Student-Test Partition (10 Trips) | Overall / Notes |
|---|---|---|---|
| **Total Candidates Evaluated** | **2,501,004 rows** | **229,920 rows** | $K=12$ candidate arcs per observation |
| **Average GPS Points / Trip** | 2,605 points | 1,916 points | Trajectories range from 262 to 19,313 points |
| **Candidate Miss Rate ($K=12$)** | **3.48%** | **1.35%** | True road segment absent from candidates |
| **Candidate Recall Ceiling** | **96.52%** | **98.65%** | Hard upper bound on downstream ML accuracy |
| **Median GPS-to-Arc Alignment** | 8.16 meters | 5.08 meters | Within civilian GPS urban noise bounds |
| **1-Second Step Consistency (`dt=1s`)**| 93.8% | 98.2% | ~2-6% jitter/dropouts handled via kinematic $dt$ |

---

### 2.2 Comprehensive Model Zoo Benchmark Comparison (8 Models)
*(Evaluated on the held-out 10 Student-Test trips, 229,920 candidate rows; logged in [`results/models/comparison.csv`](./results/models/comparison.csv))*

| Model | Algorithm Family | Top-1 Pointwise Accuracy | Student-Test Macro F1 | Log Loss (Cross-Entropy) | Training Time | CPU Inference Latency (per 1k rows) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear | 80.23% | 0.7463 | 0.2517 | 0.7s | **0.02 ms** |
| **Gaussian Naive Bayes** | Probabilistic | 81.39% | 0.6288 | 1.1779 | **0.0s** | 0.07 ms |
| **Random Forest** | Ensemble (Bagging) | 84.05% | 0.8762 | 0.1327 | 6.0s | 0.47 ms |
| **HistGradientBoosting** *(Top Pick)*| Boosting (Histogram Tree) | **84.70%** | 0.8635 | 0.1353 | 9.0s | 0.95 ms |
| **GradientBoosting (GBDT)** | Boosting (Sequential Tree) | 84.09% | **0.9153** | **0.0676** | 100.7s | 0.89 ms |
| **AdaBoost** | Boosting (Adaptive) | 81.98% | 0.4786 | 0.2703 | 19.9s | 1.89 ms |
| **Multi-Layer Perceptron (MLP)** | Neural Network (Dense) | 84.14% | 0.9129 | 0.0702 | 7.4s | 0.27 ms |
| **1D-CNN (PyTorch)** | Deep Learning (Convolutional) | 82.88% | 0.8258 | 0.1769 | 103.8s | 21.91 ms |

#### Key Empirical Takeaways:
1. **HistGradientBoosting Leads Pointwise Accuracy (84.70%):** Histogram binning enables fast, highly accurate decision splits across continuous distance and heading boundaries.
2. **GBDT and MLP Excel in Probability Calibration & Macro F1 (F1 > 0.91, Loss < 0.07):** Both models effectively optimize cross-entropy loss under class imbalance, producing low-loss calibrated likelihoods optimal for HMM emissions.
3. **1D-CNN Validation:** Demonstrates that local 1D convolutional feature extractors over kinematic/geometric feature channels learn valid discriminative representations (82.88% Top-1), though standard GBDT remains more computationally efficient for tabular inference.
4. **Gaussian Naive Bayes Calibration Breakdown:** Shows an inflated Log Loss (1.1779) due to conditional dependence between features (e.g., `dist_m` and `dist_rank`), serving as a textbook illustration of why generative models with independence assumptions struggle on geometric tables.
5. **Real-Time Edge Readiness:** GBDT, HistGB, RF, and MLP easily meet the latency budgets (< 1 ms per 1,000 rows on CPU), ensuring Laptop 1 meets real-time evaluation constraints.

---

## 3. Evaluation Schedule & Panel Viva Defense Guide

> **All team members must read and master this section.** Evaluators may direct questions to any member regardless of role to test individual understanding and authenticity.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  ⏰ 11:30 AM — PHASE 1 EVALUATION (10 Marks Total)                               │
│  ├── Data Preprocessing & Feature Engineering (5 Marks)                         │
│  └── Model Selection & Training (5 Marks)                                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│  ⏰ 03:00 PM — PHASE 2 & FINAL EVALUATION                                        │
│  ├── Bayesian Hyperparameter Optimization (Optuna)                               │
│  ├── HMM + Viterbi Route Reconstruction & Continuity                             │
│  ├── Two-Laptop Distributed Deployment (FastAPI on L1, UI on L2)                 │
│  └── Consolidated Technical Report & Code Viva                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### ⏰ 11:30 AM — Evaluation Phase 1 (10 Marks)

#### A. Data Preprocessing & Feature Engineering (5 Marks) — Led by Sumukh (P1)
**Key Concepts Every Member Must Explain:**
1. **Raw Data Inspection & Formatting:**
   - 4 file types: `.track` (lon, lat, timestamp at ~1 Hz), `.nodes` (lon, lat with 0-indexed `node_id`), `.arcs` (from_node, to_node with 0-indexed `arc_id`), `.route` (chronological sequence of traversed `arc_id`s).
   - Coordinate order: verified as `longitude, latitude` (WGS84).
2. **Geographical Coordinate Handling & Projection:**
   - **Why Euclidean distance in degrees is invalid:** Degrees of longitude shrink towards the poles ($\Delta x = \Delta \lambda \cos \phi$). Euclidean distance on degrees creates severe directional distortion.
   - **Solution:** Project all coordinates using `pyproj` into a local metric **UTM projection** (meters) before calculating distances.
3. **Data Cleaning & Anomaly Handling:**
   - Identified missing coordinates, timestamp jitter (~3% of intervals are not exactly 1 second), and disconnected road network components.
4. **Spatial Candidate Generation:**
   - Arc geometry is represented as line segments between `from_node` and `to_node`.
   - Built a spatial index (STRtree / KD-tree) over road segments.
   - For every GPS observation, extracted all candidate road arcs within radius $R$ (evaluated across 30 m, 50 m, 75 m, 100 m).
   - Tracked and reported **Candidate Miss Rate** (when the true arc is not among the candidates, capping downstream accuracy).
5. **Feature Engineering (Single Source of Truth in `src/features.py`):**
   - `dist_m`: Perpendicular physical distance from GPS point to candidate arc.
   - `frac_along`: Fractional projection position along the arc segment ($0 \le x \le 1$).
   - `heading_diff_deg`: Angular difference between vehicle trajectory heading and arc orientation ($\Delta \theta$).
   - `speed_mps`: Instantaneous speed derived from $\Delta d / \Delta t$.
   - `arc_len_m`: Physical length of the candidate road segment.
   - `dist_rank` & `n_cands`: Relative distance rank and local candidate density.
6. **Label Alignment & Strict Anti-Leakage:**
   - `.route` provides trip-level traversed arcs. Per-point labels are derived by walking monotonically along the reference route, assigning each point to the closest route arc with a maximum distance threshold.
   - **Leakage Firewall:** 80 training trips strictly partitioned from 10 student-test trips and 10 evaluator-test trips. Evaluator ground truth never enters feature extraction or model training.
- **Evidence / Artifacts to show:** `candidates.parquet`, `folds.json`, `results/eda/` plots, and interactive Folium EDA map.

---

#### B. Model Selection & Training (5 Marks) — Led by Sushanth (P2)
**Key Concepts Every Member Must Explain:**
1. **Problem Framing:**
   - Framed as **binary classification** over `(GPS observation, candidate arc)` pairs.
   - Label = $1$ if the arc is the true matched segment; $0$ otherwise.
   - The classifier outputs calibrated probabilities $\hat{p} \in [0, 1]$ used directly as emission likelihoods in downstream HMM.
2. **Class Imbalance Management:**
   - For each GPS point, typically 1 candidate is positive and 5–15 are negative.
   - Handled using `scale_pos_weight` / `class_weight='balanced'` and probability threshold tuning.
3. **Multi-Model Zoo ($\ge 5$ Distinct Algorithm Families):**
   - **Linear:** Logistic Regression (fast linear baseline; requires standardized features).
   - **Probabilistic:** Gaussian Naive Bayes (evaluates independence assumption; shows fast inference).
   - **Instance-Based:** K-Nearest Neighbors (KNN with distance weighting; tests metric similarity).
   - **Tree / Ensemble:** Random Forest / Extra Trees (non-linear decision boundaries, robust to feature scaling).
   - **Boosting:** XGBoost / LightGBM / HistGradientBoosting (sequential error correction, handles tabular interactions best).
   - **Neural:** Multi-Layer Perceptron (MLP with PyTorch/sklearn; provides epoch loss curves).
4. **Group-Aware Cross-Validation:**
   - Used **GroupKFold (5-fold) grouped by `trip_id`** on the 80 training trips.
   - Prevents observations from the same trip from appearing in both training and validation folds (preventing spatial/temporal data leakage).
5. **Evaluation Metrics:**
   - **CV Macro F1**: primary validation metric across imbalanced classes.
   - **Observation-level Top-1 Accuracy**: taking the highest-scoring candidate per GPS point against reference arc.
   - **Log Loss**: evaluates confidence and calibration quality of predicted probabilities (critical for HMM emission).
   - **Latency**: training time and inference latency (ms/1000 observations).
- **Evidence / Artifacts to show:** `results/models/comparison.csv`, confusion matrices, out-of-fold probability files (`preds_*.parquet`).

---

### ⏰ 03:00 PM — Evaluation Phase 2 & Final Deployment

#### 1. Bayesian Hyperparameter Optimization — Led by Sushanth (P2)
- Applied **Optuna** on the top-performing model (XGBoost/LightGBM) using `TPESampler(seed=42)`.
- Search space: `max_depth` (3–12), `learning_rate` (0.01–0.3), `n_estimators` (100–800), `subsample`, `colsample_bytree`, `reg_lambda`.
- Objective: Maximize 5-fold GroupKFold Macro F1 / minimize Log Loss on training trips only.
- Compared **tuned vs. untuned** performance table on student-test data.

#### 2. HMM + Viterbi Route Reconstruction — Led by Mahalakshmi (P3)
- **Why Classifier-Only Fails:** Pointwise argmax predictions jump across disconnected streets and parallel roads.
- **Hidden Markov Model Formulation:**
  - **Hidden States ($S_t$):** The set of candidate road arcs for GPS observation $t$.
  - **Emission Probability $e_t(s)$:** Model probability $\hat{p}$ combined with Gaussian spatial distance penalty $e^{-d^2 / (2\sigma^2)}$, temperature-scaled.
  - **Transition Probability $T(a, b)$:** Road network graph topology:
    - Same arc ($a = b$): high probability $p_{stay}$.
    - Directly connected in road graph (`to_node(a) == from_node(b)`): moderate probability $p_{next}$.
    - $k$-hop reachable: exponential decay $\exp(-hops / \beta)$.
    - Disconnected: floor probability $p_{floor} > 0$ to prevent Viterbi dead-ends.
- **Viterbi Decoding:** Solves for the globally optimal, continuous sequence of road segments in log-space.
- **Edge Cases Handled:** Points with zero candidates (radius expansion / bridging), large GPS dropouts (relaxing transition constraint).
- **Evidence:** Table comparing Classifier-Only vs. Classifier + HMM on Top-1 Accuracy, Route Continuity, and Sequence Precision/Recall.

#### 3. Two-Laptop Deployment Architecture — Led by Sudha (P4)
- **Laptop 1 (FastAPI Model Server):**
  - Loads trained classifier weights, scaler, road network graph, and HMM config **once at startup**.
  - Stores evaluator ground truth server-side only; computes metrics securely without leaking labels.
  - Runs with live CMD logs displaying request timestamps, trip ID, latency, and status.
- **Laptop 2 (Evaluator Client UI - Streamlit / Web):**
  - Queries `GET /trips`, displays 10 evaluation trip buttons.
  - Submits selected `trip_id` via `POST /predict` over LAN / mobile hotspot (configurable IP, no hardcoded localhost).
  - Renders Folium map with togglable layers: GPS observations (blue), ground truth reference (green), reconstructed HMM route (red).

---

### 💡 Panel Viva "Quick Defense" Cheat Sheet (Every Member Must Know)

| Question from Panel | Rock-Solid Answer |
|---|---|
| **Q1: Why can't we use Euclidean distance directly on GPS coordinates?** | Latitude and longitude are angular coordinates on an oblate spheroid. A degree of longitude shrinks as latitude increases ($\cos \phi$), distorting East-West distances relative to North-South. We project to a local UTM metric coordinate system using `pyproj` to calculate true physical distances in meters. |
| **Q2: How do you prevent data leakage in cross-validation?** | GPS points within the same vehicle trajectory are strongly autocorrelated in space and time. Standard K-Fold splits points from the same trip into both train and test. We strictly use **GroupKFold grouped by `trip_id`**, ensuring entire trajectories are either in train or validation. |
| **Q3: Why isn't a high classifier accuracy enough on its own?** | A classifier predicts each GPS observation independently. Even with 90% pointwise accuracy, 10% erroneous predictions cause impossible physical jumps to parallel roads or dead-end segments. The HMM enforces topological continuity across the road network graph. |
| **Q4: How does the HMM calculate transition probabilities?** | We inspect the road network graph directed adjacency (`from_node` $\rightarrow$ `to_node`). If consecutive candidate segments are connected, they receive a high transition probability. Multi-hop reachability decays exponentially with shortest path distance, and disconnected candidates receive a small floor probability ($p_{floor}$) so the trellis never breaks. |
| **Q5: What happens if a GPS observation has no road candidates within the search radius?** | Silently dropping the point would corrupt timestamp continuity. We implement an adaptive radius expansion (or bridge across missing observations in the HMM trellis) and explicitly report the **Candidate Miss Rate** in our metrics. |
| **Q6: How do you ensure ground truth does not leak to the UI or model during evaluation?** | On Laptop 1, the prediction pipeline processes only raw `.track`, `.nodes`, and `.arcs`. Reference `.route` files remain strictly server-side and are loaded solely by the scoring module after the predicted route is generated. |

---

## 3. System Overview

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

## 4. Repository Layout

```
pathsnap/
├── README.md                 # this file
├── requirements.txt          # must install on Spark (ARM64) AND Laptop 1
├── .gitignore                # data/raw, *.db (large), .env
├── data/
│   ├── raw/                  # NOT committed (too large); see section 7
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

## 5. Locked Conventions (Do not deviate without team consensus)

### 5.1 Coordinates
- Files are `longitude latitude` (check order on the first load and record it in `docs/decisions.md`).
- All distances are computed in a **projected metric CRS** (pick the UTM zone from the data's extent using `pyproj`). Never use degree-space Euclidean distance.
- The same CRS is used everywhere. P1 defines it in `src/geo.py`; everyone else imports it.

### 5.2 Partitions and Leakage
| Partition | Trips | Who may use | For what |
|---|---|---|---|
| Training | 80 | everyone | fitting, CV, Optuna, decoder tuning |
| Student-test | 10 | everyone | development comparison, decoder tuning |
| Evaluator-test | 10 | **P4's API only, server-side** | final demo only |

- No evaluator-test labels in any notebook, script, feature selection or tuning step.
- Scalers or any learned preprocessing: fit on the training folds only.
- Never use `.route` at prediction time. It exists only for training labels and server-side scoring.

### 5.3 Folds
- `data/processed/folds.json` maps `trip_id -> fold` (GroupKFold by trip, 5 folds). P1 creates it once. **Nobody regenerates it.**
- `data/processed/splits.json` lists the train, student-test and evaluator-test trip IDs.
- Global seed: `SEED = 42` (defined in `src/config.py`).

### 5.4 Candidate Table Schema (`candidates.parquet`)
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

### 5.5 Metric Definitions (in `src/metrics.py`; import, never reimplement)
- **Macro F1:** macro average of F1 over the two classes (0 and 1) on candidate rows. State this in the report.
- **Observation-level Top-1 Accuracy:** per GPS point, take the highest-scoring candidate and compare it with the reference arc. Points whose reference arc is absent from the candidate set count as **wrong**, and the miss rate is reported separately.
- **Candidate Miss Rate:** fraction of points whose reference arc is not in the candidate set.
- **Log Loss:** binary log loss on candidate rows (clip probabilities).
- **Route Continuity:** share of consecutive predicted-arc transitions that are valid (same arc, or `to_node` of A equals `from_node` of B). Also report a variant allowing up to 2 hops.
- **Sequence-level metric:** fraction of a trip's reference arcs that appear in the predicted route (recall) and precision of predicted arcs, plus an edit-based or Jaccard score. P3 finalises this definition in week one.

### 5.6 Handoff Contracts
| From -> To | Artifact | Format |
|---|---|---|
| P1 -> all | Candidate/feature table | `candidates.parquet` (schema above) |
| P1 -> all | Folds, splits | `folds.json`, `splits.json` |
| P1 -> P3, P4 | Network | `arcs.parquet`, `nodes.parquet` (projected), adjacency as `adjacency.pkl` or `.npz` |
| P2 -> P3 | Probabilities | `preds_<model>.parquet`: `trip_id, obs_idx, arc_id, prob`. **Out-of-fold for training trips**, plain predictions for student-test |
| P2 -> P4 | Model package | `artifacts/model.*`, `scaler.*`, `feature_list.json`, `predict_proba(df) -> np.ndarray` |
| P3 -> P4 | Decoder | `decode(cands_df, probs, network) -> list[arc_id]`, `artifacts/hmm_config.json` |

---

## 6. Environment Setup

### 6.1 Everyone (Laptops)
```bash
git clone <REPO_URL> pathsnap && cd pathsnap
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Use **Python 3.10 or 3.11** on every machine so pickles load everywhere. Record `python --version` in `docs/decisions.md`.

### 6.2 DGX Spark (P2)
See [Sushanth_modelling_spark.md](./Sushanth_modelling_spark.md). The Spark is ARM64: use a container or confirmed ARM wheels, and keep pinned versions compatible with the laptops.

### 6.3 Pinned Core Libraries (Suggested)
`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `xgboost` or `lightgbm`, `optuna`, `pyproj`, `shapely`, `scipy`, `networkx`, `folium`, `matplotlib`, `fastapi`, `uvicorn`, `pydantic`, `requests`, `streamlit` (or your UI stack).
Rule: **the version used to save a model must equal the version used to load it on Laptop 1.**

---

## 7. Data Handling
- Raw data is placed in `data/raw/` locally by each member (do not commit it unless it is tiny).
- Processed outputs go to `data/processed/`. Commit only small samples. Large files: share via drive/`rsync`/`scp`, and document where they live.
- Evaluator-test **labels** are stored only on Laptop 1, outside the UI's reach.

---

## 8. Git Workflow
- `main` is always runnable. Nobody pushes broken code to `main`.
- Branches: `p1/<topic>`, `p2/<topic>`, `p3/<topic>`, `p4/<topic>`. Open PRs, and get one teammate to glance at it before merging.
- Commit small, commit often. Pull before you start working.
- Only the owner edits their `src/` modules. Others request changes via an issue or message.
- **Never commit secrets** (`.env`, tokens, passwords). Run `git status` and a final secrets grep before each push.
- Results committed to the repo are the only numbers allowed in the report.

---

## 9. Master Timeline

Scale the hours to your actual event length. The example column assumes a 12-hour working window; phases are what matter.

| Phase | Budget | 12h example | P1 (Sumukh) | P2 (Sushanth) | P3 (Mahalakshmi) | P4 (Sudha) |
|---|---|---|---|---|---|---|
| **0. Kickoff** | 5% | 0:00-0:40 | Read PDFs, load data | Read PDFs, Spark access | Read PDFs, plan HMM | Read PDFs, create repo |
| **1. Setup and EDA** | 15% | 0:40-2:30 | EDA, projection, `src/io.py`, `geo.py` | Spark env, harness skeleton | State space and metrics design | API/UI skeleton with dummy predictor |
| **2. Data ready** | 15% | 2:30-4:20 | Candidates, labels, features v1, **10-trip sample first**, then full table, `folds.json` | Test harness on sample | Distance-only HMM prototype | `/trips`, `/predict` wired to stub |
| **3. Modelling** | 25% | 4:20-7:20 | Candidate-miss analysis, features v2, EDA maps | Train all 6-7 models, comparison plots, `preds_*.parquet` | HMM with real probabilities, metrics code | UI map layers, logging, CORS, config |
| **4. Tuning** | 15% | 7:20-9:10 | Folium EDA maps, edge cases | Optuna, tuned vs untuned, export artifacts | Decoder tuning, classifier vs HMM, failure cases | Load real artifacts into API |
| **5. Integration** | 10% | 9:10-10:20 | Check API vs local results | Laptop-1 load test, inference timing | Final Folium prediction maps | Two-laptop dry run, all 10 trips |
| **6. Report and freeze** | 15% | 10:20-12:00 | Section 2.2 | Sections 3, 4 | Sections 5, 6 | Section 7, README, PDF, push |

### Hard Checkpoints
| # | Checkpoint | Owner | Must be true |
|---|---|---|---|
| C1 | Repo + env works for everyone | P4 | all four can run `pip install -r requirements.txt` |
| C2 | **10-trip candidate sample released** | P1 | schema matches section 5.4, labels present |
| C3 | First model's predictions delivered | P2 | `preds_<model>.parquet` in agreed format |
| C4 | HMM runs end to end on a few trips | P3 | outputs a route and a continuity score |
| C5 | Real pipeline in the API | P4 | `/predict` works on student-test trips |
| C6 | Two-laptop dry run passes on all 10 trips | P4 | logs visible, no crashes |
| C7 | Code freeze | all | only report/README changes after this |

---

## 10. Report Ownership

| Section | Owner | Notes |
|---|---|---|
| 2.2 Dataset and EDA | P1 (Sumukh) | label partition on every figure |
| 3 Model comparison + per-model Q&A | P2 (Sushanth) | 9 compulsory questions for every model |
| 4 Bayesian optimisation | P2 (Sushanth) | tuned vs untuned table |
| 5 HMM and Viterbi | P3 (Mahalakshmi) | include cases where HMM did not help |
| 6 Folium visualisation | P3 (Mahalakshmi) | screenshot in PDF, code in repo |
| 7 API, UI, deployment | P4 (Sudha) | UI screenshots, API logs from the receiving laptop |
| Final assembly | P4 (Sudha) | one consolidated PDF |

Every number in the PDF must be reproducible from a file in `results/`.

---

## 11. Integration Checklist (Final 90 Minutes)
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

## 12. Academic Integrity
- Everyone must be able to explain every part of the code, experiments and results.
- Do not copy from other teams. Do not share models, configs or results across teams.
- AI tools may assist, but the experiments and analysis must be ours. Disclose AI assistance per the event policy.
- Log which AI tools were used and for what in `docs/ai_disclosure.md`.

---

## 13. Running the System (Fill in as components land)
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
