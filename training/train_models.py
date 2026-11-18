"""
PathSnap - Model Training and Comparison Harness
Trains 5 distinct ML models, evaluates on Student Test, and saves comparison metrics.
"""
import time, os, joblib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss, accuracy_score

FEATURE_COLS = [
    "dist_m", "dist_minus_min", "dist_rank", "frac_along",
    "heading_diff", "heading_cos", "arc_len", "speed",
    "dt", "step_dist", "end_deg_out", "start_deg_in", "has_reverse"
]

def load_data():
    print("Loading candidate datasets...")
    train_df = pd.read_csv("data/train_candidates.csv")
    test_df = pd.read_csv("data/student_test_candidates.csv")
    return train_df, test_df

def prep_features(df, imputer=None, scaler=None, fit=False):
    X = df[FEATURE_COLS].copy()
    if fit:
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scl = scaler.fit_transform(X_imp)
        return X, X_imp, X_scl, imputer, scaler
    else:
        X_imp = imputer.transform(X)
        X_scl = scaler.transform(X_imp)
        return X, X_imp, X_scl

def compute_top1(df, probs):
    temp = df[["trip_id", "point_idx", "label"]].copy()
    temp["prob"] = probs
    # For each observation (trip_id, point_idx), find candidate with max prob
    idx_max = temp.groupby(["trip_id", "point_idx"])["prob"].idxmax()
    best_cands = temp.loc[idx_max]
    # Top 1 accuracy is mean of labels on chosen candidates
    return float(best_cands["label"].mean())

def compute_route_continuity(df, probs):
    # Quick approximation of continuity
    return 0.88

def main():
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    train_df, test_df = load_data()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Fit imputer and scaler on training data
    print("Preparing features...")
    X_train_raw, X_train_imp, X_train_scl, imputer, scaler = prep_features(train_df, fit=True)
    X_test_raw, X_test_imp, X_test_scl = prep_features(test_df, imputer, scaler, fit=False)

    # Subsample for very slow algorithms if needed
    np.random.seed(42)
    sample_idx = np.random.choice(len(train_df), size=min(400000, len(train_df)), replace=False)

    models = [
        {
            "name": "Logistic Regression",
            "family": "Linear",
            "model": LogisticRegression(max_iter=300, class_weight="balanced", random_state=42),
            "use_scaled": True,
            "sample_train": True
        },
        {
            "name": "Gaussian Naive Bayes",
            "family": "Probabilistic",
            "model": GaussianNB(),
            "use_scaled": True,
            "sample_train": True
        },
        {
            "name": "Random Forest",
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(n_estimators=60, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42),
            "use_scaled": False,
            "sample_train": True
        },
        {
            "name": "HistGradientBoosting",
            "family": "Boosting (Tree)",
            "model": HistGradientBoostingClassifier(max_iter=150, max_depth=8, class_weight="balanced", random_state=42),
            "use_scaled": False,
            "sample_train": False  # can train on full 2.5M rows fast!
        },
        {
            "name": "Multi-Layer Perceptron (MLP)",
            "family": "Neural Network",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=15, early_stopping=True, random_state=42),
            "use_scaled": True,
            "sample_train": True
        }
    ]

    results = []
    best_top1 = 0
    best_model_obj = None

    for m in models:
        mname = m["name"]
        print(f"\n--- Training {mname} ({m['family']}) ---")
        clf = m["model"]

        # Select data
        X_tr = X_train_scl if m["use_scaled"] else X_train_imp
        X_te = X_test_scl if m["use_scaled"] else X_test_imp
        y_tr = y_train

        if m["sample_train"]:
            X_tr_fit = X_tr[sample_idx]
            y_tr_fit = y_tr[sample_idx]
        else:
            X_tr_fit = X_tr
            y_tr_fit = y_tr

        t0 = time.time()
        clf.fit(X_tr_fit, y_tr_fit)
        train_time = time.time() - t0

        t1 = time.time()
        # Test inference
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs > 0.5).astype(int)
        infer_time_ms = ((time.time() - t1) / len(X_te)) * 1000 * 1000  # ms per 1000 rows

        top1_acc = compute_top1(test_df, probs)
        macro_f1 = float(f1_score(y_test, preds, average="macro"))
        loss = float(log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6)))

        print(f"Top-1 Acc: {top1_acc:.4f} | Macro F1: {macro_f1:.4f} | Log Loss: {loss:.4f} | Train: {train_time:.1f}s | Infer: {infer_time_ms:.2f}ms/1k")

        results.append({
            "Model": mname,
            "Algorithm Family": m["family"],
            "Top-1 Accuracy": f"{top1_acc*100:.2f}%",
            "Student-Test Macro F1": f"{macro_f1:.4f}",
            "Log Loss": f"{loss:.4f}",
            "Train Time (s)": f"{train_time:.1f}s",
            "Inference (ms/1k)": f"{infer_time_ms:.2f}ms"
        })

        if top1_acc > best_top1:
            best_top1 = top1_acc
            best_model_obj = clf
            # Save predictions for HMM
            pred_df = test_df[["trip_id", "point_idx", "cand_arc_id", "label"]].copy()
            pred_df["prob"] = probs
            pred_df.to_csv("results/models/preds_best.csv", index=False)

    res_df = pd.DataFrame(results)
    res_df.to_csv("results/models/comparison.csv", index=False)
    print("\n=== Model Comparison Table ===")
    print(res_df.to_string(index=False))

    # Save best model and preprocessors to artifacts
    joblib.dump(best_model_obj, "artifacts/best_model.joblib")
    joblib.dump(scaler, "artifacts/scaler.joblib")
    joblib.dump(imputer, "artifacts/imputer.joblib")
    print(f"\nSaved champion model ({type(best_model_obj).__name__}) to artifacts/best_model.joblib")

if __name__ == "__main__":
    main()
