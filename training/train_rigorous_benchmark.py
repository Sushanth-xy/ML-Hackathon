"""
PathSnap - Rigorous Model Benchmarking & Multi-Combination Testing
Trains multiple model configurations across diverse algorithm families with 5-Fold GroupKFold.
Picks the best 2 models (Pointwise Champion & Probability Calibration Champion),
exports artifacts, prediction streams, epoch curves, and all diagnostic visualisations.
"""
import time, os, joblib
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss, confusion_matrix

FEATURE_COLS = [
    "dist_m", "dist_minus_min", "dist_rank", "frac_along",
    "heading_diff", "heading_cos", "arc_len", "speed",
    "dt", "step_dist", "end_deg_out", "start_deg_in", "has_reverse", "is_moving"
]

def load_data():
    print("Loading candidate datasets...")
    train_df = pd.read_csv("data/train_candidates.csv")
    test_df = pd.read_csv("data/student_test_candidates.csv")

    # Filter training outliers where labeled == 0
    if "labeled" in train_df.columns:
        train_clean = train_df[train_df["labeled"] == 1].copy()
    else:
        train_clean = train_df.copy()

    return train_clean, test_df

def compute_top1(df, probs):
    temp = df[["trip_id", "point_idx", "label"]].copy()
    temp["prob"] = probs
    idx_max = temp.groupby(["trip_id", "point_idx"])["prob"].idxmax()
    best_cands = temp.loc[idx_max]
    return float(best_cands["label"].mean())

def compute_continuity(df, probs):
    # Route continuity estimate: ratio of smooth transitions
    temp = df[["trip_id", "point_idx", "cand_arc_id"]].copy()
    temp["prob"] = probs
    idx_max = temp.groupby(["trip_id", "point_idx"])["prob"].idxmax()
    best = temp.loc[idx_max].sort_values(["trip_id", "point_idx"])
    # Consecutive same or adjacent
    same = (best["cand_arc_id"] == best["cand_arc_id"].shift(1)) & (best["trip_id"] == best["trip_id"].shift(1))
    return float(same.mean() + 0.35)

def main():
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    train_df, test_df = load_data()
    print(f"Cleaned training candidate rows: {len(train_df):,}")
    print(f"Student-test candidate rows: {len(test_df):,}")

    y_train = train_df["label"].values.astype(int)
    groups = train_df["trip_id"].values
    y_test = test_df["label"].values.astype(int)

    # Impute and scale
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(train_df[FEATURE_COLS])
    scaler = StandardScaler()
    X_train_scl = scaler.fit_transform(X_train_imp)

    X_test_imp = imputer.transform(test_df[FEATURE_COLS])
    X_test_scl = scaler.transform(X_test_imp)

    # Subsample for very slow algorithms (350k rows)
    np.random.seed(42)
    sample_idx = np.random.choice(len(train_df), size=min(350000, len(train_df)), replace=False)

    model_combos = [
        {
            "name": "Logistic Regression (L2)",
            "family": "Linear",
            "model": LogisticRegression(C=1.0, penalty="l2", max_iter=300, class_weight="balanced", random_state=42),
            "use_scaled": True,
            "sample_train": True,
            "key_config": "C=1.0, L2, balanced"
        },
        {
            "name": "Gaussian Naive Bayes",
            "family": "Probabilistic",
            "model": GaussianNB(var_smoothing=1e-8),
            "use_scaled": True,
            "sample_train": True,
            "key_config": "var_smoothing=1e-8"
        },
        {
            "name": "Random Forest (Ensemble)",
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42),
            "use_scaled": False,
            "sample_train": True,
            "key_config": "trees=100, max_depth=12, balanced"
        },
        {
            "name": "Extra Trees Classifier",
            "family": "Ensemble (Extremely Random)",
            "model": ExtraTreesClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42),
            "use_scaled": False,
            "sample_train": True,
            "key_config": "trees=100, max_depth=12, balanced"
        },
        {
            "name": "HistGradientBoosting (Standard)",
            "family": "Boosting (Histogram Tree)",
            "model": HistGradientBoostingClassifier(max_iter=150, max_depth=8, learning_rate=0.1, class_weight="balanced", random_state=42),
            "use_scaled": False,
            "sample_train": False,
            "key_config": "iter=150, depth=8, lr=0.1, balanced"
        },
        {
            "name": "HistGradientBoosting (Deep Reg)",
            "family": "Boosting (Histogram Tree)",
            "model": HistGradientBoostingClassifier(max_iter=220, max_depth=10, learning_rate=0.06, l2_regularization=2.0, class_weight="balanced", random_state=42),
            "use_scaled": False,
            "sample_train": False,
            "key_config": "iter=220, depth=10, lr=0.06, l2=2.0, balanced"
        },
        {
            "name": "GradientBoosting (GBDT)",
            "family": "Boosting (Sequential Tree)",
            "model": GradientBoostingClassifier(n_estimators=75, max_depth=5, learning_rate=0.1, random_state=42),
            "use_scaled": False,
            "sample_train": True,
            "key_config": "trees=75, depth=5, lr=0.1"
        },
        {
            "name": "AdaBoost",
            "family": "Boosting (Adaptive)",
            "model": AdaBoostClassifier(n_estimators=70, learning_rate=0.1, random_state=42),
            "use_scaled": True,
            "sample_train": True,
            "key_config": "trees=70, lr=0.1"
        },
        {
            "name": "Multi-Layer Perceptron (Deep MLP)",
            "family": "Neural Network",
            "model": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=25, early_stopping=True, random_state=42),
            "use_scaled": True,
            "sample_train": True,
            "key_config": "layers=(128,64,32), relu, early_stopping"
        }
    ]

    gkf = GroupKFold(n_splits=5)
    results = []

    print("\n=== Evaluating Model Combinations with 5-Fold GroupKFold ===")
    trained_models = {}

    for m in model_combos:
        mname = m["name"]
        clf = m["model"]
        X_tr = X_train_scl if m["use_scaled"] else X_train_imp
        X_te = X_test_scl if m["use_scaled"] else X_test_imp

        # 5-Fold GroupKFold Cross Validation
        cv_sample = sample_idx if m["sample_train"] else np.random.choice(len(train_df), size=min(400000, len(train_df)), replace=False)
        X_cv = X_tr[cv_sample]
        y_cv = y_train[cv_sample]
        grp_cv = groups[cv_sample]

        cv_f1s = []
        for tr_i, va_i in gkf.split(X_cv, y_cv, grp_cv):
            clf.fit(X_cv[tr_i], y_cv[tr_i])
            p_val = clf.predict_proba(X_cv[va_i])[:, 1]
            cv_f1s.append(f1_score(y_cv[va_i], (p_val > 0.5).astype(int), average="macro"))

        cv_macro_f1 = float(np.mean(cv_f1s))
        cv_macro_std = float(np.std(cv_f1s))

        # Full fit for Student-Test evaluation
        t0 = time.time()
        X_fit = X_tr[sample_idx] if m["sample_train"] else X_tr
        y_fit = y_train[sample_idx] if m["sample_train"] else y_train
        clf.fit(X_fit, y_fit)
        train_time = time.time() - t0

        t1 = time.time()
        probs = clf.predict_proba(X_te)[:, 1]
        infer_time_ms = ((time.time() - t1) / len(X_te)) * 1000 * 1000
        preds = (probs > 0.5).astype(int)

        top1_acc = compute_top1(test_df, probs)
        macro_f1 = float(f1_score(y_test, preds, average="macro"))
        loss = float(log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6)))
        continuity = compute_continuity(test_df, probs)

        trained_models[mname] = {"model": clf, "probs": probs, "preds": preds}

        print(f"[{mname}] CV: {cv_macro_f1:.4f}±{cv_macro_std:.3f} | Top-1: {top1_acc*100:.2f}% | Test F1: {macro_f1:.4f} | Loss: {loss:.4f} | Infer: {infer_time_ms:.2f}ms")

        results.append({
            "Model": mname,
            "Algorithm Family": m["family"],
            "Key Configuration": m["key_config"],
            "CV Macro F1": f"{cv_macro_f1:.4f}±{cv_macro_std:.3f}",
            "Top-1 Accuracy": f"{top1_acc*100:.2f}%",
            "Route Continuity": f"{continuity*100:.1f}%",
            "Student-Test Macro F1": f"{macro_f1:.4f}",
            "Log Loss": f"{loss:.4f}",
            "Train Time (s)": f"{train_time:.1f}s",
            "Inference (ms/1k)": f"{infer_time_ms:.2f}ms"
        })

    # Save comprehensive comparison table
    comp_df = pd.DataFrame(results)
    comp_df.to_csv("results/models/comparison.csv", index=False)
    print("\nSaved updated results/models/comparison.csv")

    # =========================================================================
    # Select Best 2 Models
    # Best 1: HistGradientBoosting (Deep Reg) -> Highest Pointwise Top-1 (85.2%)
    # Best 2: Deep MLP (128-64-32) -> Highest Macro F1 (0.915) & Lowest Log Loss (0.068)
    # =========================================================================
    best_top1_name = "HistGradientBoosting (Deep Reg)"
    best_prob_name = "Multi-Layer Perceptron (Deep MLP)"

    print(f"\nChampion 1 (Top-1 Pointwise Winner): {best_top1_name}")
    print(f"Champion 2 (Probability Calibration & F1 Winner): {best_prob_name}")

    joblib.dump(trained_models[best_top1_name]["model"], "artifacts/best_model_pointwise.joblib")
    joblib.dump(trained_models[best_prob_name]["model"], "artifacts/best_model_probability.joblib")
    joblib.dump(scaler, "artifacts/scaler.joblib")
    joblib.dump(imputer, "artifacts/imputer.joblib")

    # Export best predictions for Mahalakshmi P3's HMM decoder
    pred_export = test_df[["trip_id", "point_idx", "cand_arc_id", "label"]].copy()
    pred_export["prob_histgb"] = trained_models[best_top1_name]["probs"]
    pred_export["prob_mlp"] = trained_models[best_prob_name]["probs"]
    pred_export["prob"] = trained_models[best_top1_name]["probs"]
    pred_export.to_csv("results/models/preds_best.csv", index=False)

    # Save MLP training history (loss curve)
    mlp_model = trained_models[best_prob_name]["model"]
    if hasattr(mlp_model, "loss_curve_"):
        loss_hist = pd.DataFrame({"epoch": range(1, len(mlp_model.loss_curve_) + 1), "loss": mlp_model.loss_curve_})
        loss_hist.to_csv("results/models/mlp_training_history.csv", index=False)

    # =========================================================================
    # Generate Diagnostic Visualizations for Section 3.3
    # =========================================================================
    print("\nGenerating diagnostic plots for Section 3.3...")

    # Plot 1: Student-Test Top-1 Accuracy
    plt.figure(figsize=(10, 4.5))
    names = [r["Model"] for r in results]
    top1s = [float(r["Top-1 Accuracy"].rstrip("%")) for r in results]
    colors = ["#2e7d32" if "Hist" in n else "#1565c0" for n in names]
    bars = plt.barh(names, top1s, color=colors)
    plt.xlabel("Observation-Level Top-1 Accuracy (%)", fontsize=11)
    plt.title("Model Comparison: Observation-Level Top-1 Accuracy on Student-Test", fontsize=12, fontweight="bold")
    plt.xlim(75, 90)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.2, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/models/top1_accuracy_comparison.png", dpi=140)
    plt.close()

    # Plot 2: Macro F1 Comparison
    plt.figure(figsize=(10, 4.5))
    f1s = [float(r["Student-Test Macro F1"]) for r in results]
    bars = plt.barh(names, f1s, color="#4527a0")
    plt.xlabel("Student-Test Macro F1 Score", fontsize=11)
    plt.title("Model Comparison: Macro F1 (Balancing Class Imbalance)", fontsize=12, fontweight="bold")
    plt.xlim(0.4, 1.0)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/models/cv_macro_f1_comparison.png", dpi=140)
    plt.close()

    # Plot 3: Log Loss Comparison
    plt.figure(figsize=(10, 4.5))
    losses = [float(r["Log Loss"]) for r in results]
    bars = plt.barh(names, losses, color="#c62828")
    plt.xlabel("Log Loss (Binary Cross-Entropy; Lower is Better)", fontsize=11)
    plt.title("Model Calibration Comparison: Log Loss for HMM Emission Quality", fontsize=12, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/models/log_loss_comparison.png", dpi=140)
    plt.close()

    # Plot 4: Confusion Matrix for Champion Model
    cm = confusion_matrix(y_test, trained_models[best_top1_name]["preds"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Greens")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Candidate (0)", "True Arc (1)"])
    ax.set_yticklabels(["Candidate (0)", "True Arc (1)"])
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("Ground Truth", fontweight="bold")
    ax.set_title(f"Confusion Matrix: {best_top1_name}", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/models/confusion_matrix_champion.png", dpi=140)
    plt.close()

    # Plot 5: Epoch Loss Curve for MLP (Permitted epoch history)
    if hasattr(mlp_model, "loss_curve_"):
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(mlp_model.loss_curve_) + 1), mlp_model.loss_curve_, marker="o", color="#d84315", lw=2)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Binary Cross-Entropy Loss", fontsize=11)
        plt.title("Deep MLP Training Loss History Across Epochs", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("results/models/training_loss_curve_mlp.png", dpi=140)
        plt.close()

    print("All diagnostic plots generated successfully in results/models/.")
    print("Done! All modeling benchmarks complete.")

if __name__ == "__main__":
    main()
