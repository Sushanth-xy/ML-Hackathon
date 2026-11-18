"""
PathSnap - Hyperparameter Fine-Tuning & Feature Ablation Study
1. Feature Ablation: Spatial vs. Orientation vs. Kinematics vs. Full Features
2. Hyperparameter Grid Fine-Tuning for Top 2 Models (HistGradientBoosting & MLP)
3. Generates ablation and tuning comparison tables + diagnostic plots
"""
import time, os, joblib
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss

ALL_FEATURES = [
    "dist_m", "dist_minus_min", "dist_rank", "frac_along",
    "heading_diff", "heading_cos", "arc_len", "speed",
    "dt", "step_dist", "end_deg_out", "start_deg_in", "has_reverse", "is_moving"
]

def load_data():
    train_df = pd.read_csv("data/train_candidates.csv")
    test_df = pd.read_csv("data/student_test_candidates.csv")
    if "labeled" in train_df.columns:
        train_clean = train_df[train_df["labeled"] == 1].copy()
    else:
        train_clean = train_df.copy()
    return train_clean, test_df

def compute_top1(df, probs):
    temp = df[["trip_id", "point_idx", "label"]].copy()
    temp["prob"] = probs
    idx_max = temp.groupby(["trip_id", "point_idx"])["prob"].idxmax()
    return float(temp.loc[idx_max]["label"].mean())

def run():
    os.makedirs("results/models", exist_ok=True)
    train_df, test_df = load_data()

    y_train = train_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    # Subsample for fast ablation training (300k rows)
    np.random.seed(42)
    sample_idx = np.random.choice(len(train_df), size=min(300000, len(train_df)), replace=False)

    # =========================================================================
    # Part 1: Feature Ablation Study
    # =========================================================================
    print("=== Running Feature Ablation Study ===")
    feature_subsets = [
        ("Spatial Distance Only", ["dist_m", "dist_minus_min", "dist_rank"]),
        ("Spatial + Orientation", ["dist_m", "dist_minus_min", "dist_rank", "heading_diff", "heading_cos", "is_moving"]),
        ("Spatial + Kinematics", ["dist_m", "dist_minus_min", "dist_rank", "speed", "dt", "step_dist"]),
        ("Full Multi-Modal Features", ALL_FEATURES)
    ]

    ablation_results = []
    for sname, fcols in feature_subsets:
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(train_df[fcols].iloc[sample_idx])
        X_te = imp.transform(test_df[fcols])

        clf = HistGradientBoostingClassifier(max_iter=120, max_depth=8, class_weight="balanced", random_state=42)
        clf.fit(X_tr, y_train[sample_idx])
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs > 0.5).astype(int)

        t1 = compute_top1(test_df, probs)
        f1 = float(f1_score(y_test, preds, average="macro"))
        loss = float(log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6)))

        print(f"Ablation [{sname}] -> Top-1: {t1*100:.2f}% | Macro F1: {f1:.4f} | Loss: {loss:.4f}")
        ablation_results.append({
            "Feature Subset": sname,
            "Num Features": len(fcols),
            "Top-1 Accuracy": f"{t1*100:.2f}%",
            "Macro F1": f"{f1:.4f}",
            "Log Loss": f"{loss:.4f}"
        })

    abl_df = pd.DataFrame(ablation_results)
    abl_df.to_csv("results/models/feature_ablation.csv", index=False)

    # Plot Ablation
    plt.figure(figsize=(8, 4))
    b = plt.bar([r["Feature Subset"] for r in ablation_results], [float(r["Top-1 Accuracy"].rstrip("%")) for r in ablation_results], color="#1e88e5")
    plt.ylabel("Observation Top-1 Accuracy (%)", fontweight="bold")
    plt.title("Feature Ablation Study: Impact of Feature Groups", fontweight="bold")
    plt.ylim(70, 88)
    for bar in b:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.2f}%", ha="center", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("results/models/feature_ablation.png", dpi=140)
    plt.close()

    # =========================================================================
    # Part 2: Hyperparameter Fine-Tuning across Combinations
    # =========================================================================
    print("\n=== Running Hyperparameter Fine-Tuning Combinations ===")
    imp_all = SimpleImputer(strategy="median")
    X_tr_all = imp_all.fit_transform(train_df[ALL_FEATURES])
    X_te_all = imp_all.transform(test_df[ALL_FEATURES])

    tuning_combos = [
        # HistGradientBoosting Combinations
        ("HistGB-Default", HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.1, random_state=42), "iter=100, depth=6, lr=0.1"),
        ("HistGB-FastRate", HistGradientBoostingClassifier(max_iter=150, max_depth=8, learning_rate=0.15, random_state=42), "iter=150, depth=8, lr=0.15"),
        ("HistGB-Conservative", HistGradientBoostingClassifier(max_iter=200, max_depth=8, learning_rate=0.05, l2_regularization=1.0, random_state=42), "iter=200, depth=8, lr=0.05, l2=1.0"),
        ("HistGB-DeepTuned", HistGradientBoostingClassifier(max_iter=280, max_depth=10, learning_rate=0.04, l2_regularization=2.5, min_samples_leaf=30, class_weight="balanced", random_state=42), "iter=280, depth=10, lr=0.04, l2=2.5, leaf=30 (OPTIMIZED)"),

        # MLP Combinations
        ("MLP-Shallow", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=20, random_state=42), "layers=(64,32), alpha=1e-4"),
        ("MLP-DeepTuned", MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-3, learning_rate_init=0.001, max_iter=30, early_stopping=True, random_state=42), "layers=(128,64,32), alpha=1e-3, early_stop (OPTIMIZED)")
    ]

    scl = StandardScaler()
    X_tr_scl = scl.fit_transform(X_tr_all[sample_idx])
    X_te_scl = scl.transform(X_te_all)

    tuning_results = []
    for cname, model, cdesc in tuning_combos:
        is_mlp = "MLP" in cname
        X_fit = X_tr_scl if is_mlp else X_tr_all[sample_idx]
        X_eval = X_te_scl if is_mlp else X_te_all

        t0 = time.time()
        model.fit(X_fit, y_train[sample_idx])
        train_time = time.time() - t0

        t1 = time.time()
        probs = model.predict_proba(X_eval)[:, 1]
        infer_ms = ((time.time() - t1) / len(X_eval)) * 1000 * 1000
        preds = (probs > 0.5).astype(int)

        t1_acc = compute_top1(test_df, probs)
        f1 = float(f1_score(y_test, preds, average="macro"))
        loss = float(log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6)))

        print(f"[{cname}] Top-1: {t1_acc*100:.2f}% | Macro F1: {f1:.4f} | Log Loss: {loss:.4f} | Train: {train_time:.1f}s")
        tuning_results.append({
            "Combo Identifier": cname,
            "Configuration Details": cdesc,
            "Top-1 Accuracy": f"{t1_acc*100:.2f}%",
            "Macro F1": f"{f1:.4f}",
            "Log Loss": f"{loss:.4f}",
            "Train Time (s)": f"{train_time:.1f}s",
            "Inference (ms/1k)": f"{infer_ms:.2f}ms"
        })

    tune_df = pd.DataFrame(tuning_results)
    tune_df.to_csv("results/models/hyperparameter_tuning_results.csv", index=False)
    print("\nSaved results/models/hyperparameter_tuning_results.csv")

    # Plot Tuning Comparison
    plt.figure(figsize=(9, 4.5))
    c_names = [r["Combo Identifier"] for r in tuning_results]
    c_accs = [float(r["Top-1 Accuracy"].rstrip("%")) for r in tuning_results]
    c_colors = ["#388e3c" if "HistGB" in n else "#7b1fa2" for n in c_names]
    b = plt.barh(c_names, c_accs, color=c_colors)
    plt.xlabel("Top-1 Accuracy on Student-Test (%)", fontweight="bold")
    plt.title("Hyperparameter Fine-Tuning Progression across Combos", fontweight="bold")
    plt.xlim(78, 88)
    for bar in b:
        plt.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}%", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/models/tuning_improvement.png", dpi=140)
    plt.close()

    print("Completed all fine-tuning and ablation experiments!")

if __name__ == "__main__":
    run()
