"""
evaluate_model.py
------------------
Module 5: Evaluation

Compares the ground-truth labels (present in the synthetic dataset) with
the predictions produced by the Isolation Forest (anomaly detection) and,
in parallel, with predictions from the Random Forest classifier. Computes
the standard binary-classification metrics, exports them to
data/evaluation_metrics.csv, and saves three PNG charts under
static/charts/ which are also displayed in the Flask dashboard.

Run:
    python evaluate_model.py
"""

import os
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
matplotlib.use("Agg")           # non-interactive backend, safe for headless boxes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DETECTION_FILE = os.path.join("data", "detection_results.csv")
DATA_FILE = os.path.join("data", "cloud_activity_logs.csv")
METRICS_FILE = os.path.join("data", "evaluation_metrics.csv")
CHARTS_DIR = os.path.join("static", "charts")
MODELS_DIR = "models"

CATEGORICAL_FEATURES = ["country", "resource_accessed", "action_type", "admin_action"]
NUMERICAL_FEATURES = ["failed_logins", "session_duration", "data_download_mb", "risk_score"]

# Consistent colour palette across all charts (works well on light & dark themes)
THREAT_COLOURS = {
    "Low":      "#3b82f6",    # blue
    "Medium":   "#f59e0b",    # amber
    "High":     "#ef4444",    # red
    "Critical": "#7c2d12",    # dark red / maroon
}

# Distinct colour per attack family for the attack_type chart
ATTACK_COLOURS = {
    "normal":                "#10b981",   # emerald (benign)
    "brute_force":           "#ef4444",   # red
    "data_exfiltration":     "#8b5cf6",   # violet
    "privilege_escalation":  "#f59e0b",   # amber
    "geographic_anomaly":    "#0ea5e9",   # sky blue
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Return a dict of standard binary metrics for the supplied predictions."""
    return {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def get_random_forest_predictions() -> np.ndarray:
    """Apply the saved Random Forest to the same dataset to get a second
    set of predictions for side-by-side comparison.
    """
    rf_clf = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))

    df = pd.read_csv(DATA_FILE)

    # Encode categorical features safely (handle unseen values defensively)
    for column in CATEGORICAL_FEATURES:
        enc: LabelEncoder = encoders[column]
        known = set(enc.classes_)
        df[column] = df[column].astype(str).apply(
            lambda v: enc.transform([v])[0] if v in known else -1
        )

    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])
    return rf_clf.predict(df[feature_columns])


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def plot_threat_level_distribution(detection_df: pd.DataFrame, out_path: str) -> None:
    """Bar chart of how many events fell into each threat level."""
    order = ["Low", "Medium", "High", "Critical"]
    counts = (
        detection_df["threat_level"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[THREAT_COLOURS[level] for level in counts.index],
        edgecolor="white",
        linewidth=1.2,
    )

    # Value labels above each bar
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#111827",
        )

    ax.set_title("Threat Level Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Threat Level", fontsize=11)
    ax.set_ylabel("Number of Events", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_attack_type_distribution(detection_df: pd.DataFrame, out_path: str) -> None:
    """Horizontal bar chart of the five attack_type categories.

    A horizontal layout is used because 'normal' dwarfs the four attack
    families; horizontal bars keep all category labels easy to read and
    let the value labels sit cleanly to the right of each bar.
    """
    order = [
        "normal",
        "brute_force",
        "data_exfiltration",
        "privilege_escalation",
        "geographic_anomaly",
    ]
    counts = (
        detection_df["attack_type"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))

    # Horizontal bars, largest at the top
    y_positions = np.arange(len(order))[::-1]
    bars = ax.barh(
        y_positions,
        counts.values,
        color=[ATTACK_COLOURS[a] for a in counts.index],
        edgecolor="white",
        linewidth=1.2,
    )

    # Pretty labels with no underscores
    pretty_labels = [a.replace("_", " ").title() for a in counts.index]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pretty_labels, fontsize=11)

    # Value labels just to the right of each bar
    x_max = counts.max()
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_width() + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#111827",
        )

    ax.set_xlim(0, x_max * 1.12)
    ax.set_title("Attack Type Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Events", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    """Confusion matrix heatmap for the Isolation Forest predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
        xticklabels=["Normal (0)", "Suspicious (1)"],
        yticklabels=["Normal (0)", "Suspicious (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        "Confusion Matrix - Isolation Forest",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_metrics(metrics_df: pd.DataFrame, out_path: str) -> None:
    """Side-by-side comparison of the two models on Accuracy / Precision /
    Recall / F1-Score."""
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    pretty_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    iso_row = metrics_df[metrics_df["model"] == "Isolation Forest"].iloc[0]
    rf_row = metrics_df[metrics_df["model"] == "Random Forest"].iloc[0]

    iso_values = [iso_row[m] for m in metric_names]
    rf_values = [rf_row[m] for m in metric_names]

    bars_iso = ax.bar(x - width / 2, iso_values, width, label="Isolation Forest", color="#2563eb")
    bars_rf  = ax.bar(x + width / 2, rf_values,  width, label="Random Forest",    color="#10b981")

    # Value labels
    for bars, values in [(bars_iso, iso_values), (bars_rf, rf_values)]:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#111827",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print(" Cloud Threat Detection - Model Evaluation")
    print("=" * 60)

    # ---- 1. Load --------------------------------------------------------
    if not os.path.exists(DETECTION_FILE):
        raise FileNotFoundError(
            f"{DETECTION_FILE} not found. Please run detect_threats.py first."
        )

    print(f"\n[1/4] Loading detection results from {DETECTION_FILE} ...")
    detection_df = pd.read_csv(DETECTION_FILE)

    y_true = detection_df["label"].values
    iso_pred = detection_df["prediction"].values

    # Also get Random Forest predictions on the same data, for a fair side-by-side
    print("      Computing Random Forest predictions for comparison ...")
    rf_pred = get_random_forest_predictions()

    # ---- 2. Metrics -----------------------------------------------------
    print("\n[2/4] Computing metrics ...")
    iso_metrics = compute_metrics(y_true, iso_pred, "Isolation Forest")
    rf_metrics = compute_metrics(y_true, rf_pred, "Random Forest")
    metrics_df = pd.DataFrame([iso_metrics, rf_metrics])

    # Confusion matrix values (saved to the metrics CSV as well, for completeness)
    cm_iso = confusion_matrix(y_true, iso_pred, labels=[0, 1])
    metrics_df["TN"] = [cm_iso[0, 0], confusion_matrix(y_true, rf_pred, labels=[0, 1])[0, 0]]
    metrics_df["FP"] = [cm_iso[0, 1], confusion_matrix(y_true, rf_pred, labels=[0, 1])[0, 1]]
    metrics_df["FN"] = [cm_iso[1, 0], confusion_matrix(y_true, rf_pred, labels=[0, 1])[1, 0]]
    metrics_df["TP"] = [cm_iso[1, 1], confusion_matrix(y_true, rf_pred, labels=[0, 1])[1, 1]]

    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    metrics_df.to_csv(METRICS_FILE, index=False)
    print(f"      Metrics saved to {METRICS_FILE}")

    # ---- 3. Charts ------------------------------------------------------
    print("\n[3/4] Generating charts ...")
    os.makedirs(CHARTS_DIR, exist_ok=True)

    plot_threat_level_distribution(
        detection_df,
        os.path.join(CHARTS_DIR, "threat_level_distribution.png"),
    )
    plot_attack_type_distribution(
        detection_df,
        os.path.join(CHARTS_DIR, "attack_type_distribution.png"),
    )
    plot_confusion_matrix(
        y_true,
        iso_pred,
        os.path.join(CHARTS_DIR, "confusion_matrix.png"),
    )
    plot_model_metrics(
        metrics_df,
        os.path.join(CHARTS_DIR, "model_metrics.png"),
    )
    print(f"      Charts saved to {CHARTS_DIR}/")

    # ---- 4. Pretty print ------------------------------------------------
    print("\n[4/4] Classification report - Isolation Forest")
    print("-" * 60)
    print(classification_report(y_true, iso_pred, target_names=["Normal", "Suspicious"]))

    print("Classification report - Random Forest")
    print("-" * 60)
    print(classification_report(y_true, rf_pred, target_names=["Normal", "Suspicious"]))

    print("\nMetric summary table")
    print("-" * 60)
    print(metrics_df[["model", "accuracy", "precision", "recall", "f1_score"]].to_string(index=False))

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
