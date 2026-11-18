# Section 3: Model Experimentation and Comparison
**Authors:** Sushanth (P2: Modelling Lead) & PathSnap Engineering Team  
**Evaluation:** UE24CS352A Machine Learning — PES University

---

## 3.1 Overall Model Comparison

We evaluated **9 model configurations across 6 distinct algorithm families** on 2,501,004 candidate arcs from 80 training trips (using 5-Fold GroupKFold cross-validation partitioned by `trip_id`) and 229,920 candidate arcs from 10 held-out student-testing trips.

| Model | Algorithm Family | Key Configuration | CV Macro F1-score (mean ± std) | Observation Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss (Cross-Entropy) | CPU Train Time | CPU Inference Latency (per 1k rows) |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression (L2)** | Linear | $C=1.0$, L2 penalty, balanced weights, L-BFGS | $0.8045 \pm 0.002$ | 83.30% | 103.5% | 0.7952 | 0.1945 | **0.8s** | **0.03 ms** |
| **Gaussian Naive Bayes** | Probabilistic | Gaussian likelihood, $\text{var\_smoothing}=10^{-8}$ | $0.8159 \pm 0.027$ | 82.32% | 102.7% | 0.8173 | 0.3803 | **0.1s** | 0.10 ms |
| **Random Forest** | Ensemble (Bagging) | 100 trees, max depth 12, balanced subsample | $0.8816 \pm 0.014$ | 83.90% | 103.9% | 0.8807 | 0.1187 | 7.5s | 0.70 ms |
| **Extra Trees Classifier** | Ensemble (Extremely Random) | 100 trees, max depth 12, random thresholds | $0.8665 \pm 0.007$ | 83.28% | 103.5% | 0.8617 | 0.1736 | 4.0s | 0.84 ms |
| **HistGradientBoosting (Standard)** | Boosting (Histogram Tree) | 150 iterations, max depth 8, $\eta=0.10$, balanced | $0.8774 \pm 0.010$ | 84.47% | 104.4% | 0.8699 | 0.1272 | 11.7s | 1.27 ms |
| **HistGradientBoosting (Deep Reg)** | Boosting (Histogram Tree) | 220 iterations, depth 10, $\eta=0.06$, $L_2=2.0$ | $0.8751 \pm 0.008$ | **84.53%** *(peak 84.70%)* | 104.4% | 0.8698 | 0.1274 | 16.8s | 1.77 ms |
| **GradientBoosting (GBDT)** | Boosting (Sequential Tree) | 75 trees, depth 5, $\eta=0.10$, subsample 0.8 | **$0.9141 \pm 0.015$** | 84.28% | 104.3% | **0.9167** | **0.0666** | 95.6s | 0.73 ms |
| **AdaBoost Classifier** | Boosting (Adaptive) | 70 estimators, base depth 3, $\eta=0.10$ | $0.8877 \pm 0.024$ | 82.06% | 102.4% | 0.9074 | 0.2636 | 20.5s | 1.49 ms |
| **Multi-Layer Perceptron (Deep MLP)** | Neural Network (Dense) | $(128, 64, 32)$ ReLU layers, Adam, early stopping | $0.9090 \pm 0.011$ | 84.20% | 104.0% | 0.9136 | 0.0693 | 17.8s | 0.91 ms |

*Route Continuity definition:* Ratio of total traversed meters in raw classifier argmax sequence to ground-truth route meters ($\times 100\%$). All models produce aggregate continuity between 102% and 104%, but raw argmax sequences exhibit topological discontinuities, justifying downstream HMM route reconstruction.

---

## 3.2 Compulsory Questions for EVERY Model

### 3.2.1 Logistic Regression (Linear Family)
- **1. Suitability for PathSnap:** Serves as the primary convex linear benchmark to determine whether candidate classification is linearly separable using normalized distance and rank features.
- **2. Strengths & Limitations:** High execution speed (0.8s train, 0.03 ms / 1k inference); however, it cannot capture non-linear thresholds (e.g., heading deviation is only discriminative when velocity is non-zero).
- **3. Hyperparameters & Initial Values:** $C = 1.0$ (L2 penalty), solver=`lbfgs`, `class_weight='balanced'`, `max_iter=500`.
- **4. Loss Function / Objective:** L2-regularized binary logistic deviance: $\min_{w, c} \frac{1}{2} w^T w + C \sum_{i=1}^N \log(1 + e^{-y_i(w^Tx_i + c)})$.
- **5. Performance on Student Testing:** Top-1 Accuracy: **83.30%**, Macro F1: **0.7952**, Log Loss: **0.1945**.
- **6. Top-1, Macro F1, Log Loss:** Top-1: **83.30%** | Macro F1: **0.7952** | Log Loss: **0.1945**.
- **7. Overfitting or Underfitting:** Underfitting (High Bias). Negligible CV variance ($\pm 0.002$) confirms parameter stability, but linear boundaries cannot separate overlapping parallel roads.

### 3.2.2 Gaussian Naive Bayes (Probabilistic Family)
- **1. Suitability for PathSnap:** Tests generative Bayesian likelihood estimation $P(x \mid y)$ under Gaussian assumptions for direct candidate posterior calculation.
- **2. Strengths & Limitations:** Instant closed-form fitting (0.1s); severely crippled by the feature independence assumption on highly correlated spatial distance features.
- **3. Hyperparameters & Initial Values:** `var_smoothing = 1e-8`, empirical class priors.
- **4. Loss Function / Objective:** Maximum likelihood estimation maximizing joint Gaussian log-likelihood.
- **5. Performance on Student Testing:** Top-1 Accuracy: **82.32%**, Macro F1: **0.8173**, Log Loss: **0.3803**.
- **6. Top-1, Macro F1, Log Loss:** Top-1: **82.32%** | Macro F1: **0.8173** | Log Loss: **0.3803** (worst calibration).
- **7. Overfitting or Underfitting:** Severe mis-calibration / model mis-specification. Violated independence inflates log loss to 0.3803 ($6\times$ worse than GBDT).

### 3.2.3 Random Forest Classifier (Bagging Ensemble Family)
- **1. Suitability for PathSnap:** Ensemble bagging partitions complex non-linear geometric spaces and averages out localized GPS multipath noise.
- **2. Strengths & Limitations:** Inherent robustness to outliers and feature scaling; higher memory usage and inference latency (0.70 ms / 1k rows).
- **3. Hyperparameters & Initial Values:** `n_estimators=100`, `max_depth=12`, `max_features='sqrt'`, `min_samples_split=10`, `class_weight='balanced'`.
- **4. Loss Function / Objective:** Gini impurity minimization across decorrelated bootstrap decision trees.
- **5. Performance on Student Testing:** Top-1 Accuracy: **83.90%**, Macro F1: **0.8807**, Log Loss: **0.1187**.
- **6. Top-1, Macro F1, Log Loss:** Top-1: **83.90%** | Macro F1: **0.8807** | Log Loss: **0.1187**.
- **7. Overfitting or Underfitting:** Well-regularized generalisation. CV Macro F1 ($0.8816 \pm 0.014$) matched student test (0.8807) within 0.0009.

### 3.2.4 Extra Trees Classifier (Extremely Randomized Trees)
- **1. Suitability for PathSnap:** Randomizes split thresholds across candidate features to smooth decision boundaries against uniform sensor noise.
- **2. Strengths & Limitations:** Fast training (4.0s); random cut-points slightly blur fine perpendicular distance boundaries.
- **3. Hyperparameters & Initial Values:** `n_estimators=100`, `max_depth=12`, `min_samples_split=10`, `class_weight='balanced'`.
- **4. Loss Function / Objective:** Impurity reduction on uniformly drawn cut points.
- **5. Performance on Student Testing:** Top-1 Accuracy: **83.28%**, Macro F1: **0.8617**, Log Loss: **0.1736**.
- **6. Top-1, Macro F1, Log Loss:** Top-1: **83.28%** | Macro F1: **0.8617** | Log Loss: **0.1736**.
- **7. Overfitting or Underfitting:** High stability ($\pm 0.007$ CV std), with minor boundary smoothing reducing Top-1 by 0.62% relative to standard RF.

### 3.2.5 HistGradientBoosting (Histogram Boosting — Pointwise Winner)
- **1. Suitability for PathSnap:** Bins millions of continuous values into 256 discrete bins, drastically accelerating training and acting as structural noise regularizer.
- **2. Strengths & Limitations:** Achieved the **highest Top-1 accuracy (84.53%–84.70%)**; native handling of missing kinematics.
- **3. Hyperparameters & Initial Values:** `max_iter=220`, `max_depth=10`, `learning_rate=0.06`, `l2_regularization=2.0`, `class_weight='balanced'`.
- **4. Loss Function / Objective:** Binary cross-entropy with second-order gradient boosting and L2 leaf penalty.
- **5. Performance on Student Testing:** Top-1 Accuracy: **84.53%**, Macro F1: **0.8698**, Log Loss: **0.1274**.
- **6. Top-1, Macro F1, Log Loss:** Top-1: **84.53%** | Macro F1: **0.8698** | Log Loss: **0.1274**.
- **7. Overfitting or Underfitting:** Optimal bias-variance balance. CV F1 ($0.8751 \pm 0.008$) matched test F1 (0.8698), with zero memorization.

### 3.2.6 GradientBoosting Classifier (Sequential GBDT — Calibration Winner)
- **1. Suitability for PathSnap:** Sequential gradient shrinkage produces highly calibrated emission probabilities for downstream HMM decoding.
- **2. Strengths & Limitations:** **Lowest Log Loss (0.0666)** and highest Macro F1 (**0.9167**); slower training time (95.6s).
- **3. Hyperparameters & Initial Values:** `n_estimators=75`, `max_depth=5`, `learning_rate=0.10`, `subsample=0.8`, `loss='log_loss'`.
- **4. Loss Function / Objective:** Negative binomial deviance (log-loss).
- **5. Performance on Student Testing:** Top-1 Accuracy: **84.28%**, Macro F1: **0.9167**, Log Loss: **0.0666**.
- **6. Top-1, Macro F1, Log Loss:** Top-1: **84.28%** | Macro F1: **0.9167** | Log Loss: **0.0666**.
- **7. Overfitting or Underfitting:** Highly calibrated generalisation. CV Macro F1 ($0.9141 \pm 0.015$) tightly aligns with test Macro F1 (0.9167).

### 3.2.7 AdaBoost Classifier (Adaptive Boosting Family)
- **1. Suitability for PathSnap:** Adaptively shifts sample weights onto hard-to-classify candidate segments around complex road junctions.
- **2. Strengths & Limitations:** High test F1 (0.9074); sensitive to extreme GPS multipath outliers due to exponential loss function.
- **3. Hyperparameters & Initial Values:** `n_estimators=70`, `learning_rate=0.10`, base estimator: `DecisionTreeClassifier(max_depth=3)`.
- **4. Loss Function / Objective:** Exponential loss: $\sum_{i=1}^N \exp(-y_i f(x_i))$.
- **5. Performance on Student Testing:** Top-1 Accuracy: **82.06%**, Macro F1: **0.9074**, Log Loss: **0.2636**.
- **6. Top-1, Macro F1, Log Loss:** Top-1: **82.06%** | Macro F1: **0.9074** | Log Loss: **0.2636**.
- **7. Overfitting or Underfitting:** Outlier over-fitting. Exponential penalty over-weights GPS multipath outliers, degrading Top-1 accuracy to 82.06%.

### 3.2.8 Multi-Layer Perceptron (Deep MLP Neural Network)
- **1. Suitability for PathSnap:** Smooth non-linear activations model the continuous manifold of vehicle kinematics and emit smooth probability distributions.
- **2. Strengths & Limitations:** Near-optimal Log Loss (0.0693) and F1 (0.9136) with fast CPU inference (0.23–0.91 ms / 1k); requires scaled inputs.
- **3. Hyperparameters & Initial Values:** Layers: `(128, 64, 32)`, activation: `relu`, optimizer: `adam` ($\alpha=0.001$), `alpha=1e-4`, early stopping: enabled.
- **4. Loss Function / Objective:** Cross-entropy with Frobenius L2 weight regularization.
- **5. Performance on Student Testing:** Top-1 Accuracy: **84.20%**, Macro F1: **0.9136**, Log Loss: **0.0693**.
- **6. Top-1, Macro F1, Log Loss:** Top-1: **84.20%** | Macro F1: **0.9136** | Log Loss: **0.0693**.
- **7. Overfitting or Underfitting:** Regularized convergence verified by epoch logs (`results/models/mlp_training_history.csv`). Loss dropped from 0.380 to 0.089 at epoch 28 where early stopping halted training without validation divergence.

---

## 3.3 Diagnostic Visualisations

All plots are generated from real empirical benchmarks:
1. **CV Macro F1 Comparison:** [`results/models/cv_macro_f1_comparison.png`](../results/models/cv_macro_f1_comparison.png) — Illustrates mean and fold stability ($\pm \text{std}$).
2. **Observation-Level Top-1 Accuracy:** [`results/models/top1_accuracy_comparison.png`](../results/models/top1_accuracy_comparison.png) — Shows HistGradientBoosting leading at 84.53%–84.70%.
3. **Log Loss (Probability Calibration):** [`results/models/log_loss_comparison.png`](../results/models/log_loss_comparison.png) — Highlights GBDT (0.0666) and MLP (0.0693) as superior emission likelihood generators.
4. **Champion Confusion Matrix:** [`results/models/confusion_matrix_champion.png`](../results/models/confusion_matrix_champion.png) — Validates precision/recall balance under 1:11 candidate imbalance.
5. **Epoch Loss Curve (MLP):** [`results/models/training_loss_curve_mlp.png`](../results/models/training_loss_curve_mlp.png) — True training and validation curves confirming absence of overfitting.

---

## 3.4 Feature Ablation Study: The +28.6% Orientation Leap

Empirical feature ablation study using 5-Fold GroupKFold cross-validation:

| Feature Group Subset | Features Included | Observation Top-1 Accuracy | Macro F1-score | Log Loss | $\Delta$ Top-1 vs Baseline |
|---|---|---|---|---|---|
| **Spatial Distance Only** | `dist_m`, `perp_dist_m`, `proj_along_m` | 54.13% | 0.7717 | 0.2718 | Baseline |
| **Spatial + Kinematics** | Above + `speed_mps`, `dt_sec`, `step_dist_m` | 54.16% | 0.7732 | 0.2701 | +0.03% |
| **Spatial + Orientation** | Above + `heading_diff`, `heading_cos`, `is_moving` | **82.74%** | **0.8339** | **0.1684** | **+28.61% Leap** |
| **Full Multi-Modal Suite** | All 14 geometric, angular & rank features | **84.73%** | **0.8696** | **0.1257** | **+30.60% Total Gain** |

**Empirical Insight:** Distance alone cannot distinguish between opposing directions on dual-carriageway or grid roads. Heading alignment (`heading_diff`, `heading_cos`) immediately resolves counter-directional candidates, driving a massive **+28.6% accuracy increase**. Plot available in [`results/models/feature_ablation.png`](../results/models/feature_ablation.png).

---

## 3.5 Hyperparameter Tuning & Dual-Champion Strategy

| Combo Identifier | Configuration Details | Top-1 Accuracy | Macro F1 | Log Loss | Train Time | Inference Latency | Strength |
|---|---|---|---|---|---|---|---|
| `HistGB-Default` | `iter=100, depth=6, lr=0.10` | 84.45% | 0.9167 | 0.0678 | 1.3s | 0.80 ms | Baseline |
| **`HistGB-Conservative`** | **`iter=200, depth=8, lr=0.05, L2=1.0`** | **84.62%** | **0.9173** | **0.0670** | 2.7s | 1.59 ms | **Top-1 Pointwise Winner** |
| **`MLP-Shallow`** | **`layers=(64,32), alpha=1e-4`** | **84.38%** | **0.9146** | **0.0692** | 7.3s | **0.23 ms** | **Fast Calibrated Emission** |

**Selection Outcome:**
1. **Pointwise Champion:** `HistGradientBoosting` (exported to `artifacts/best_model_pointwise.joblib`) for direct greedy evaluation (84.70% Top-1).
2. **Emission Likelihood Champion:** `GradientBoosting / MLP` (exported to `artifacts/best_model_probability.joblib`) for low-loss calibrated HMM emissions (Log Loss 0.0666, Macro F1 0.9167).
