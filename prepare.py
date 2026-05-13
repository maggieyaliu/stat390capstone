"""
UPDATED -- Data loading, evaluation, experiment matrix logging,
AUC-over-time plotting, and error taxonomy utilities.
"""
import csv
import os
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
VAL_FRACTION = 0.2
DATA_PATH = "online_shoppers_WORK.csv"
RESULTS_FILE = "results.tsv"
ERROR_FILE = "errors.log"
PLOT_FILE = "metric_over_time.png"


def load_data():
    """Load and split the Shopper Intent WORK dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Ensure the dataset is present.")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(["Revenue", "Month"], axis=1)
    y = df["Revenue"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return X_train, y_train, X_val, y_val, X.columns.tolist()


def evaluate(model, X_val, y_val):
    """Compute validation ROC AUC and F1-score."""
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    auc = float(roc_auc_score(y_val, y_prob))
    f1 = float(f1_score(y_val, y_pred))
    return auc, f1


def next_run_number(results_file=RESULTS_FILE):
    """Return the next run number based on results.tsv."""
    if not os.path.exists(results_file):
        return 1

    max_run = 0
    with open(results_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                max_run = max(max_run, int(row["run_number"]))
            except (KeyError, ValueError, TypeError):
                continue
    return max_run + 1


def count_features(model, X_train=None):
    """Estimate feature count after preprocessing for logging."""
    try:
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            preprocessor = model.named_steps["preprocessor"]
            if hasattr(preprocessor, "get_feature_names_out"):
                return int(len(preprocessor.get_feature_names_out()))
        if X_train is not None:
            return int(X_train.shape[1])
    except Exception:
        pass
    return -1


def extract_model_metadata(model, description=""):
    """Build concise model metadata string for results.tsv."""
    try:
        estimator = model.named_steps.get("model", model)
    except Exception:
        estimator = model

    model_name = estimator.__class__.__name__
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        params = {}

    whitelist = {
        "n_estimators", "criterion", "max_depth", "max_features",
        "min_samples_leaf", "min_samples_split", "class_weight",
        "learning_rate", "max_iter", "l2_regularization",
        "C", "solver", "penalty", "ccp_alpha", "random_state"
    }

    parts = []
    for key in sorted(whitelist):
        if key in params:
            val = params[key]
            if isinstance(val, (str, int, float, bool)) or val is None:
                parts.append(f"{key}={val}")

    param_text = ", ".join(parts)
    metadata = f"model={model_name}; {param_text}" if param_text else f"model={model_name}"
    if description:
        metadata = f"{description} | {metadata}"
    return metadata


def log_result(run_number, auc, f1, num_features, run_time_sec, status, description, results_file=RESULTS_FILE):
    """Append one experiment row to the experiment-result matrix."""
    file_exists = os.path.exists(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(
                [
                    "run_number",
                    "val_auc",
                    "val_f1",
                    "num_features",
                    "run_time_sec",
                    "status",
                    "description",
                ]
            )
        writer.writerow(
            [
                run_number,
                f"{auc:.6f}",
                f"{f1:.6f}",
                num_features,
                f"{run_time_sec:.3f}",
                status,
                description,
            ]
        )


def log_error(run_number, error_msg, error_file=ERROR_FILE):
    """Append one error entry to the taxonomy file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(error_file, "a") as f:
        f.write(f"--- RUN #{run_number} | {timestamp} ---\n")
        f.write(f"ERROR: {error_msg}\n")
        f.write("TRACEBACK:\n")
        f.write(traceback.format_exc())
        f.write("\n")


def plot_results(results_file=RESULTS_FILE, save_path=PLOT_FILE):
    """Create/update AUC-over-time plot from results.tsv."""
    if not os.path.exists(results_file):
        return

    runs, aucs = [], []
    with open(results_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                runs.append(int(row["run_number"]))
                aucs.append(float(row["val_auc"]))
            except (KeyError, ValueError, TypeError):
                continue

    if not runs:
        return

    order = np.argsort(runs)
    runs = [runs[i] for i in order]
    aucs = [aucs[i] for i in order]

    best_aucs = np.maximum.accumulate(aucs)
    baseline_auc = aucs[0]

    plt.figure(figsize=(10, 6))
    plt.plot(runs, aucs, marker="o", linewidth=2, color="#1f77b4", label="Validation AUC")
    plt.step(runs, best_aucs, where="post", linewidth=2, color="#2ca02c", label="Best so far")
    plt.axhline(baseline_auc, color="#d62728", linestyle="--", linewidth=2, label=f"Baseline AUC ({baseline_auc:.4f})")
    plt.title("Metric Over Time")
    plt.xlabel("Experiment Number")
    plt.ylabel("Validation AUC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def record_success(model, X_train, auc, f1, run_time_sec, description, status="keep"):
    """Convenience helper to log successful run + refresh plot."""
    run_number = next_run_number()
    num_features = count_features(model, X_train)
    meta = extract_model_metadata(model, description)
    log_result(run_number, auc, f1, num_features, run_time_sec, status, meta)
    plot_results()
    return run_number


def record_failure(run_number, error_msg):
    """Convenience helper to log failed run in taxonomy file."""
    log_error(run_number, error_msg)


if __name__ == "__main__":
    plot_results()
