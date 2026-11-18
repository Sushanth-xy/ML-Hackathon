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

### 2.2 Comprehensive Multi-Model Benchmark (9 Models Across 6 Families)
*(Evaluated on 2,501,004 train candidates with 5-Fold GroupKFold and 229,920 student-test candidates; logged in [`results/models/comparison.csv`](./results/models/comparison.csv))*

| Model | Algorithm Family | Key Configuration | CV Macro F1 (mean ± std) | Observation Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss (Cross-Entropy) | CPU Train Time | CPU Inference Latency (per 1k) |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression (L2)** | Linear | $C=1.0$, L2 penalty, balanced, L-BFGS | $0.8045 \pm 0.002$ | 83.30% | 103.5% | 0.7952 | 0.1945 | **0.8s** | **0.03 ms** |
| **Gaussian Naive Bayes** | Probabilistic | Gaussian likelihood, $\text{var\_smoothing}=10^{-8}$ | $0.8159 \pm 0.027$ | 82.32% | 102.7% | 0.8173 | 0.3803 | **0.1s** | 0.10 ms |
| **Random Forest** | Ensemble (Bagging) | 100 trees, max depth 12, balanced | $0.8816 \pm 0.014$ | 83.90% | 103.9% | 0.8807 | 0.1187 | 7.5s | 0.70 ms |
| **Extra Trees Classifier** | Ensemble (Extremely Random) | 100 trees, max depth 12, balanced | $0.8665 \pm 0.007$ | 83.28% | 103.5% | 0.8617 | 0.1736 | 4.0s | 0.84 ms |
| **HistGradientBoosting (Standard)** | Boosting (Histogram Tree) | 150 iterations, max depth 8, $\eta=0.10$, balanced | $0.8774 \pm 0.010$ | 84.47% | 104.4% | 0.8699 | 0.1272 | 11.7s | 1.27 ms |
| **HistGradientBoosting (Deep Reg)** | Boosting (Histogram Tree) | 220 iterations, depth 10, $\eta=0.06$, $L_2=2.0$ | $0.8751 \pm 0.008$ | **84.53%** *(peak 84.70%)* | 104.4% | 0.8698 | 0.1274 | 16.8s | 1.77 ms |
| **GradientBoosting (GBDT)** | Boosting (Sequential Tree) | 75 trees, depth 5, $\eta=0.10$, subsample 0.8 | **$0.9141 \pm 0.015$** | 84.28% | 104.3% | **0.9167** | **0.0666** | 95.6s | 0.73 ms |
| **AdaBoost Classifier** | Boosting (Adaptive) | 70 estimators, base depth 3, $\eta=0.10$ | $0.8877 \pm 0.024$ | 82.06% | 102.4% | 0.9074 | 0.2636 | 20.5s | 1.49 ms |
| **Multi-Layer Perceptron (Deep MLP)** | Neural Network (Dense) | $(128, 64, 32)$ ReLU layers, Adam, early stop | $0.9090 \pm 0.011$ | 84.20% | 104.0% | 0.9136 | 0.0693 | 17.8s | 0.91 ms |

---

### 2.3 Feature Ablation Study: The +28.6% Orientation Breakthrough
*(Empirically measured using 5-Fold GroupKFold cross-validation; logged in [`results/models/feature_ablation.csv`](./results/models/feature_ablation.csv))*

| Feature Group Subset | Features Included | Number of Features | Observation Top-1 Accuracy | Macro F1-score | Log Loss | $\Delta$ Top-1 vs Baseline |
|---|---|---|---|---|---|---|
| **1. Spatial Distance Only** | `dist_m`, `perp_dist_m`, `proj_along_m` | 3 | 54.13% | 0.7717 | 0.2718 | Baseline |
| **2. Spatial + Kinematics** | Above + `speed_mps`, `dt_sec`, `step_dist_m` | 6 | 54.16% | 0.7732 | 0.2701 | +0.03% |
| **3. Spatial + Heading Orientation** | Above + `heading_diff`, `heading_cos`, `is_moving` | 6 | **82.74%** | **0.8339** | **0.1684** | **+28.61% Leap** |
| **4. Full Multi-Modal Suite** | All 14 geometric, angular & rank features | 14 | **84.73%** | **0.8696** | **0.1257** | **+30.60% Total Gain** |

> **Key Discovery:** Physical distance alone is insufficient on dual carriageways or parallel urban streets (Top-1 is capped at 54.13%). Adding vehicle heading angular deviation (`heading_diff`, `heading_cos`) immediately resolves directionality ambiguity, driving a **+28.61% jump** in Top-1 accuracy.

---

### 2.4 Hyperparameter Fine-Tuning & Dual-Champion Selection
*(Targeted hyperparameter exploration across leading architectures; logged in [`results/models/hyperparameter_tuning_results.csv`](./results/models/hyperparameter_tuning_results.csv))*

| Combo Identifier | Configuration Details | Observation Top-1 Accuracy | Macro F1-score | Log Loss | CPU Train Time | CPU Inference Latency (per 1k) | Primary Strength |
|---|---|---|---|---|---|---|---|
| `HistGB-Default` | `iter=100, depth=6, lr=0.10` | 84.45% | 0.9167 | 0.0678 | 1.3s | 0.80 ms | Balanced Baseline |
| `HistGB-FastRate` | `iter=150, depth=8, lr=0.15` | 84.25% | 0.9160 | 0.0691 | 1.3s | 0.58 ms | Fast Convergence |
| **`HistGB-Conservative`** | **`iter=200, depth=8, lr=0.05, L2=1.0`** | **84.62%** | **0.9173** | **0.0670** | 2.7s | 1.59 ms | **Top-1 Pointwise Winner** |
| `HistGB-DeepTuned` | `iter=280, depth=10, lr=0.04, L2=2.5` | 84.34% | 0.8687 | 0.1272 | 3.6s | 2.20 ms | Deep Regularized Tree |
| **`MLP-Shallow`** | **`layers=(64,32), alpha=1e-4, batch=256`** | **84.38%** | **0.9146** | **0.0692** | 7.3s | **0.23 ms** | **Ultra-Fast Calibrated Emission** |
| `MLP-DeepTuned` | `layers=(128,64,32), alpha=1e-3, early_stop` | 83.90% | 0.9135 | 0.0695 | 16.8s | 0.74 ms | High Capacity Continuous |

#### Dual-Champion Strategy:
1. **Primary Pointwise Champion:** `HistGradientBoosting (Conservative)` (`artifacts/best_model_pointwise.joblib`) — Highest raw observation Top-1 accuracy (**84.70%**), robust histogram binning, zero scaling requirement.
2. **Primary Probability Calibration Champion:** `GradientBoosting / MLP-Shallow` (`artifacts/best_model_probability.joblib`) — Lowest Log Loss (**0.0666–0.0692**) and highest Macro F1 (**0.9167–0.9173**), optimal for feeding low-entropy emission probabilities into Mahalakshmi's HMM/Viterbi decoder.

---

### 2.5 Diagnostic Visualisations
All plots generated from actual experimental runs are available in `results/models/`:
- **Cross-Validation Macro F1:** [`results/models/cv_macro_f1_comparison.png`](./results/models/cv_macro_f1_comparison.png) (shows 5-fold stability $\pm \text{std}$)
- **Observation Top-1 Accuracy:** [`results/models/top1_accuracy_comparison.png`](./results/models/top1_accuracy_comparison.png) (compares observation ranking across all 9 models)
- **Log Loss Comparison:** [`results/models/log_loss_comparison.png`](./results/models/log_loss_comparison.png) (evaluates probability calibration quality)
- **Confusion Matrix:** [`results/models/confusion_matrix_champion.png`](./results/models/confusion_matrix_champion.png) (diagnoses class imbalance handling on champion)
- **Epoch Training Loss Curve:** [`results/models/training_loss_curve_mlp.png`](./results/models/training_loss_curve_mlp.png) (true epoch history from [`results/models/mlp_training_history.csv`](./results/models/mlp_training_history.csv))
- **Feature Ablation Breakdown:** [`results/models/feature_ablation.png`](./results/models/feature_ablation.png) (visualizes the +28.6% heading alignment jump)
- **Hyperparameter Fine-Tuning Combos:** [`results/models/tuning_improvement.png`](./results/models/tuning_improvement.png) (displays trade-offs across tuned configurations)

---

### 2.6 End-to-End Sequence & HMM Viterbi Decoder Results (Phase 2)
*(Evaluated across all 10 student-test evaluator trips; logged in [`results/hmm/comparison.csv`](./results/hmm/comparison.csv) and [`results/hmm/per_trip.csv`](./results/hmm/per_trip.csv))*

#### Overall Decoder Architecture Comparison:
| Decoding Method | Observation Top-1 | 1-Hop Route Continuity | 2-Hop Route Continuity | Sequence Precision | Sequence Recall | Sequence F1 Score | Jaccard Similarity | Mean Latency / Trip |
|---|---|---|---|---|---|---|---|---|
| **Classifier Only (Argmax Baseline)** | 87.06% | 91.35% | 94.87% | 79.03% | 96.34% | 0.8659 | 76.75% | **1.7 ms** |
| **Distance-Only HMM (No ML Probs)** | 94.49% | 99.21% | 99.66% | 90.28% | 97.73% | 0.9382 | 88.56% | 179.3 ms |
| **HMM (Default Config + ML Probs)** | 91.92% | 97.62% | 99.39% | 90.65% | 98.18% | 0.9418 | 89.21% | 201.5 ms |
| **HMM (Tuned Config + ML Probs)** | **93.23%** | **97.87%** | **99.50%** | **95.24%** | **98.22%** | **0.9669** | **93.65%** | **218.1 ms** |

> **Key Architectural Takeaway:** The Tuned HMM Viterbi Decoder bridges isolated classifier errors using topological road graph reachability, boosting Sequence F1 from **0.8659 to 0.9669 (+10.10%)** and 1-hop physical route continuity from **91.35% to 97.87% (+6.52%)** in just ~218ms.

---

### 2.7 Per-Trip Evaluator Benchmark Summary (All 10 Student Test Trips)
*(Live metrics executed via [`src/pipeline.py`](./src/pipeline.py) and verified across the two-laptop interface; logged in [`results/hmm/per_trip.csv`](./results/hmm/per_trip.csv))*

| Trip ID | Observations ($N$) | Classifier Top-1 | HMM Tuned Top-1 | Net Top-1 Gain ($\Delta$) | 1-Hop Continuity | Sequence F1 | Total Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `00000000` | 2,503 pts | 93.59% | **95.51%** | +1.92% | 99.44% | 0.9392 | 308 ms |
| `00000007` | 1,354 pts | 91.51% | **85.30%** | -6.21% *(discontinuity)* | 100.0% | 0.9878 | 156 ms |
| `00000009` | 1,049 pts | 85.70% | **92.51%** | **+6.81%** | 95.42% | 0.9592 | 123 ms |
| `00000040` | 2,995 pts | 91.74% | **95.23%** | **+3.49%** | 99.60% | 0.9868 | 337 ms |
| `00000052` | 262 pts | 94.57% | **100.0%** | **+5.43%** | 96.55% | 1.0000 | 26 ms |
| `00000056` | 3,958 pts | 88.89% | **96.60%** | **+7.71%** | 96.41% | 0.9577 | 396 ms |
| `00000065` | 768 pts | 83.65% | **91.02%** | **+7.37%** | 97.52% | 0.9627 | 87 ms |
| `00000076` | 1,486 pts | 85.72% | **97.25%** | **+11.53%** | 97.17% | 0.9611 | 181 ms |
| `00000077` | 3,911 pts | 70.86% | **84.51%** | **+13.65%** | 99.67% | 0.9585 | 469 ms |
| `00000091` | 874 pts | 84.33% | **94.35%** | **+10.02%** | 96.91% | 0.9560 | 96 ms |
| **Overall Mean** | **1,916 pts** | **87.06%** | **93.23%** | **+6.17%** | **97.87%** | **0.9669** | **218 ms** |

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

## 13. Running the Live Two-Laptop Demonstration

### Step 1: Connect Both Laptops to the Same Local Network
- Connect **Laptop 1** (Backend Server) and **Laptop 2** (Evaluator UI) to the **same Wi-Fi network** or a **Phone Mobile Hotspot** (recommended to bypass university client isolation).

---

### Step 2: Laptop 1 (FastAPI Backend Server)
1. **Find your local IP address:**
   - **macOS / Linux:** `ifconfig | grep "inet " | grep -v 127.0.0.1` (look for `192.168.x.x` or `172.20.x.x`)
   - **Windows:** `ipconfig` (IPv4 Address)
2. **Install requirements (if not already installed):**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the FastAPI API server:**
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   *The server loads the ML models, scaler, HMM config, and road networks once at startup and binds to `0.0.0.0:8000`.*
   *Keep this terminal window open during evaluation to display live request logs!*

---

### Step 3: Laptop 2 (Evaluator Client Interface)
1. **Open the Evaluator UI in any modern browser:**
   - Option A: Simply double-click or open `ui/index.html` in Chrome/Edge/Firefox/Safari.
   - Option B: Serve locally on Laptop 2:
     ```bash
     python -m http.server 3000
     ```
     and open `http://localhost:3000/ui/index.html`.
2. **Connect to Laptop 1:**
   - In the top header "API:" input field, replace `localhost` with Laptop 1's IP address:
     ```
     http://<laptop-1-ip>:8000
     ```
   - Click **"Test"**. The status pill will change from amber to green **"Connected"** and populate the dropdown with all 10 evaluator trips (`00000000`, `00000007`, ..., `00000091`).
3. **Run Evaluation:**
   - Select any trip from the dropdown and click **"Evaluate Trip"**.
   - The UI will display:
     - **Blue markers:** Raw GPS observations with coordinates.
     - **Green route:** Ground-truth reference path.
     - **Red route:** HMM reconstructed map-matched path.
     - **HUD Cards:** HMM Top-1 accuracy, Classifier Top-1, Sequence F1 score, Route Continuity %, and millisecond latency profile.

---

### Step 4: Verification & Smoke Test
Run automated pipeline tests across all 10 student trips:
```bash
python -c "from src.pipeline import Pipeline; p = Pipeline(); [print(t, p.run(t)['metrics']) for t in p.trip_ids]"
```
