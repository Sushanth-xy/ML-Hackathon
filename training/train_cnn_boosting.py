"""
PathSnap - CNN and Additional Boosting Experiments
Trains a 1D-CNN (PyTorch), AdaBoost, and GradientBoosting.
Appends results to results/models/comparison.csv.
"""
import time, os, joblib
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss

FEATURE_COLS = [
    "dist_m", "dist_minus_min", "dist_rank", "frac_along",
    "heading_diff", "heading_cos", "arc_len", "speed",
    "dt", "step_dist", "end_deg_out", "start_deg_in", "has_reverse"
]

class Conv1DNet(nn.Module):
    def __init__(self, in_features=13):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, 13) -> (B, 1, 13)
        x = x.unsqueeze(1)
        feats = self.conv(x)
        logits = self.fc(feats).squeeze(1)
        return logits

def compute_top1(df, probs):
    temp = df[["trip_id", "point_idx", "label"]].copy()
    temp["prob"] = probs
    idx_max = temp.groupby(["trip_id", "point_idx"])["prob"].idxmax()
    best_cands = temp.loc[idx_max]
    return float(best_cands["label"].mean())

def main():
    print("Loading candidate datasets...")
    train_df = pd.read_csv("data/train_candidates.csv")
    test_df = pd.read_csv("data/student_test_candidates.csv")

    y_train = train_df["label"].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(train_df[FEATURE_COLS])
    scaler = StandardScaler()
    X_train_scl = scaler.fit_transform(X_train_imp)

    X_test_imp = imputer.transform(test_df[FEATURE_COLS])
    X_test_scl = scaler.transform(X_test_imp)

    np.random.seed(42)
    sample_idx = np.random.choice(len(train_df), size=min(300000, len(train_df)), replace=False)
    X_train_sub = X_train_scl[sample_idx]
    y_train_sub = y_train[sample_idx]

    new_results = []

    # 1. AdaBoost Classifier
    print("\n--- Training AdaBoost Classifier (Boosting) ---")
    ada = AdaBoostClassifier(n_estimators=80, learning_rate=0.1, random_state=42)
    t0 = time.time()
    ada.fit(X_train_sub, y_train_sub)
    train_time = time.time() - t0

    t1 = time.time()
    ada_probs = ada.predict_proba(X_test_scl)[:, 1]
    ada_infer_ms = ((time.time() - t1) / len(X_test_scl)) * 1000 * 1000
    ada_top1 = compute_top1(test_df, ada_probs)
    ada_preds = (ada_probs > 0.5).astype(int)
    ada_f1 = float(f1_score(y_test, ada_preds, average="macro"))
    ada_loss = float(log_loss(y_test, np.clip(ada_probs, 1e-6, 1 - 1e-6)))

    print(f"AdaBoost | Top-1: {ada_top1:.4f} | F1: {ada_f1:.4f} | Loss: {ada_loss:.4f} | Train: {train_time:.1f}s | Infer: {ada_infer_ms:.2f}ms")
    new_results.append({
        "Model": "AdaBoost",
        "Algorithm Family": "Boosting (Adaptive)",
        "Top-1 Accuracy": f"{ada_top1*100:.2f}%",
        "Student-Test Macro F1": f"{ada_f1:.4f}",
        "Log Loss": f"{ada_loss:.4f}",
        "Train Time (s)": f"{train_time:.1f}s",
        "Inference (ms/1k)": f"{ada_infer_ms:.2f}ms"
    })

    # 2. Gradient Boosting (sklearn)
    print("\n--- Training GradientBoostingClassifier (Boosting) ---")
    gb = GradientBoostingClassifier(n_estimators=80, max_depth=6, learning_rate=0.1, random_state=42)
    t0 = time.time()
    gb.fit(X_train_sub, y_train_sub)
    train_time = time.time() - t0

    t1 = time.time()
    gb_probs = gb.predict_proba(X_test_scl)[:, 1]
    gb_infer_ms = ((time.time() - t1) / len(X_test_scl)) * 1000 * 1000
    gb_top1 = compute_top1(test_df, gb_probs)
    gb_preds = (gb_probs > 0.5).astype(int)
    gb_f1 = float(f1_score(y_test, gb_preds, average="macro"))
    gb_loss = float(log_loss(y_test, np.clip(gb_probs, 1e-6, 1 - 1e-6)))

    print(f"GradientBoosting | Top-1: {gb_top1:.4f} | F1: {gb_f1:.4f} | Loss: {gb_loss:.4f} | Train: {train_time:.1f}s | Infer: {gb_infer_ms:.2f}ms")
    new_results.append({
        "Model": "GradientBoosting (GBDT)",
        "Algorithm Family": "Boosting (Sequential)",
        "Top-1 Accuracy": f"{gb_top1*100:.2f}%",
        "Student-Test Macro F1": f"{gb_f1:.4f}",
        "Log Loss": f"{gb_loss:.4f}",
        "Train Time (s)": f"{train_time:.1f}s",
        "Inference (ms/1k)": f"{gb_infer_ms:.2f}ms"
    })

    # 3. 1D Convolutional Neural Network (PyTorch)
    print("\n--- Training 1D-CNN (PyTorch Convolutional Network) ---")
    torch.manual_seed(42)
    cnn = Conv1DNet(in_features=13)
    pos_weight = torch.tensor([8.5])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.003, weight_decay=1e-4)

    # Convert to PyTorch tensors
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_sub).float(), torch.from_numpy(y_train_sub).float()),
        batch_size=1024, shuffle=True
    )

    t0 = time.time()
    cnn.train()
    for epoch in range(6):
        epoch_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = cnn(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/6 Loss: {epoch_loss/len(train_loader):.4f}")
    train_time = time.time() - t0

    # Test inference
    t1 = time.time()
    cnn.eval()
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_scl).float()), batch_size=4096, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for (bx,) in test_loader:
            probs = torch.sigmoid(cnn(bx)).numpy()
            probs_list.append(probs)
    cnn_probs = np.concatenate(probs_list)
    cnn_infer_ms = ((time.time() - t1) / len(X_test_scl)) * 1000 * 1000

    cnn_top1 = compute_top1(test_df, cnn_probs)
    cnn_preds = (cnn_probs > 0.5).astype(int)
    cnn_f1 = float(f1_score(y_test, cnn_preds, average="macro"))
    cnn_loss = float(log_loss(y_test, np.clip(cnn_probs, 1e-6, 1 - 1e-6)))

    print(f"1D-CNN | Top-1: {cnn_top1:.4f} | F1: {cnn_f1:.4f} | Loss: {cnn_loss:.4f} | Train: {train_time:.1f}s | Infer: {cnn_infer_ms:.2f}ms")
    new_results.append({
        "Model": "1D-CNN (PyTorch)",
        "Algorithm Family": "Deep Learning (CNN)",
        "Top-1 Accuracy": f"{cnn_top1*100:.2f}%",
        "Student-Test Macro F1": f"{cnn_f1:.4f}",
        "Log Loss": f"{cnn_loss:.4f}",
        "Train Time (s)": f"{train_time:.1f}s",
        "Inference (ms/1k)": f"{cnn_infer_ms:.2f}ms"
    })

    # Save 1D-CNN model weights
    torch.save(cnn.state_dict(), "artifacts/cnn_weights.pth")

    # Combine with previous comparison table
    prev_df = pd.read_csv("results/models/comparison.csv")
    combined_df = pd.concat([prev_df, pd.DataFrame(new_results)], ignore_index=True)
    combined_df.to_csv("results/models/comparison.csv", index=False)
    print("\n=== Updated Comprehensive Model Comparison Table ===")
    print(combined_df.to_string(index=False))

if __name__ == "__main__":
    main()
