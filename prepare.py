"""
FROZEN -- Do not modify this file.
Handles data loading, evaluation, matrix logging, performance plotting, 
and error taxonomy for the AutoResearch Agent.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import csv
import os
import time
import traceback

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED = 42
VAL_FRACTION = 0.2
RESULTS_FILE = "results.tsv"
ERROR_FILE = "errors.log"
DATA_PATH = "online_shoppers_WORK.csv"

# ── Data ───────────────────────────────────────────────────
def load_data():
    """Load and split the Shopper Intent WORK dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Ensure you ran the vault split script.")
    
    df = pd.read_csv(DATA_PATH)
    X = df.drop(['Revenue', 'Month'], axis=1)
    y = df['Revenue'].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_FRACTION, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, y_train, X_val, y_val, X.columns.tolist()

# ── Evaluation ─────────────────────────────────────────────
def evaluate(model, X_val, y_val):
    """Compute validation metrics."""
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    auc = float(roc_auc_score(y_val, y_prob))
    f1 = float(f1_score(y_val, y_pred))
    return auc, f1

# ── Logging: Experiment-Result Matrix ──────────────────────
def log_result(run_number, auc, f1, num_features, run_time, status, description):
    """Update the results.tsv matrix with comprehensive metadata."""
    file_exists = os.path.exists(RESULTS_FILE)
    
    # Extract model/hyperparameter info from description for the matrix
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow([
                "run_number", "auc_score", "f1_score", "num_features", 
                "run_time_sec", "status", "model_metadata"
            ])
        writer.writerow([
            run_number, f"{auc:.6f}", f"{f1:.6f}", num_features, 
            f"{run_time:.2f}", status, description
        ])

# ── Logging: Error Taxonomy ────────────────────────────────
def log_error(run_number, error_msg):
    """Log detailed error information to errors.log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_FILE, "a") as f:
        f.write(f"--- RUN #{run_number} | {timestamp} ---\n")
        f.write(f"ERROR: {error_msg}\n")
        f.write(f"TRACEBACK:\n{traceback.format_exc()}\n\n")

# ── Plotting: Metric-Over-Time Plot ────────────────────────
def plot_results(save_path="performance.png"):
    """Generate the AUC-over-time graph."""
    if not os.path.exists(RESULTS_FILE):
        return

    runs, aucs = [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            runs.append(int(row["run_number"]))
            aucs.append(float(row["auc_score"]))

    plt.figure(figsize=(10, 6))
    plt.plot(runs, aucs, marker='o', linestyle='-', color='#3498db', linewidth=2, label="Validation AUC")
    
    # Plot 'Best so far' line
    best_aucs = np.maximum.accumulate(aucs)
    plt.step(runs, best_aucs, where='post', color='#2ecc71', linewidth=2, label="Best Score")
    
    plt.title("Metric Over Time: AutoResearch Progress", fontsize=14)
    plt.xlabel("Experiment Run Number", fontsize=12)
    plt.ylabel("Validation AUC Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    plot_results()
