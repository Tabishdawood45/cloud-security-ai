"""
detect_threats.py
------------------
Module 3: Threat Detection

Applies the trained Isolation Forest model to every event in the dataset,
converts its raw output into a clean Normal / Suspicious flag, derives a
human-readable threat level (Low, Medium, High, Critical) and a
recommended action, and saves the enriched dataset to
data/detection_results.csv.

Run:
    python detect_threats.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join("data", "cloud_activity_logs.csv")
OUTPUT_FILE = os.path.join("data", "detection_results.csv")
MODELS_DIR = "models"

CATEGORICAL_FEATURES = ["country", "resource_accessed", "action_type", "admin_action"]
NUMERICAL_FEATURES = ["failed_logins", "session_duration", "data_download_mb", "risk_score"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_artefacts() -> tuple:
    """Load the Isolation Forest model along with encoders, scaler and
    the ordered list of feature columns used at training time.
    """
    paths = {
        "model":    os.path.join(MODELS_DIR, "isolation_forest.joblib"),
        "scaler":   os.path.join(MODELS_DIR, "scaler.joblib"),
        "encoders": os.path.join(MODELS_DIR, "encoders.joblib"),
        "columns":  os.path.join(MODELS_DIR, "feature_columns.joblib"),
    }
    for label, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artefact missing: {path}. Please run train_model.py first."
            )

    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])
    encoders = joblib.load(paths["encoders"])
    feature_columns = joblib.load(paths["columns"])
    return model, scaler, encoders, feature_columns


def preprocess(df: pd.DataFrame, encoders: dict, scaler, feature_columns: list) -> pd.DataFrame:
    """Apply the same encoding + scaling as in train_model.py."""
    work = df.copy()

    # Encode categorical features. Any value unseen at training time falls back
    # to a safe default (-1) so we never crash at inference time.
    for column in CATEGORICAL_FEATURES:
        encoder = encoders[column]
        known_classes = set(encoder.classes_)
        work[column] = work[column].astype(str).apply(
            lambda v: encoder.transform([v])[0] if v in known_classes else -1
        )

    # Scale numerical features using the previously fit scaler
    work[NUMERICAL_FEATURES] = scaler.transform(work[NUMERICAL_FEATURES])

    return work[feature_columns]


def classify_threat_level(row: pd.Series) -> str:
    """Derive an overall threat level for the event.

    The logic combines four signals:
        * Was the event flagged as an anomaly?
        * Number of failed logins
        * Volume of data downloaded
        * The original risk_score from the log itself
    """
    is_anomaly = row["prediction"] == 1
    failed = row["failed_logins"]
    download = row["data_download_mb"]
    risk = row["risk_score"]

    # Strong individual indicators always escalate severity
    if failed >= 10 or download >= 2000 or risk >= 90:
        return "Critical"

    if is_anomaly and (failed >= 5 or download >= 1000 or risk >= 75):
        return "High"

    if is_anomaly and (failed >= 3 or download >= 500 or risk >= 60):
        return "Medium"

    if is_anomaly:
        return "Low"

    # Normal events get the lowest classification
    return "Low"


def recommend_action(row: pd.Series) -> str:
    """Map (threat_level, prediction) -> recommended SOC response."""
    level = row["threat_level"]
    is_anomaly = row["prediction"] == 1

    if not is_anomaly:
        return "No Action Required"

    if level == "Critical":
        # Multiple drastic signals -> immediate containment
        if row["failed_logins"] >= 10:
            return "Block IP address"
        if row["data_download_mb"] >= 2000:
            return "Terminate session"
        return "Temporarily lock account"

    if level == "High":
        return "Temporarily lock account"

    if level == "Medium":
        return "Require MFA verification"

    # Low level anomaly
    return "Monitor activity"


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print(" Cloud Threat Detection - Anomaly Inference")
    print("=" * 60)

    # ---- 1. Load model + data -----------------------------------------
    print("\n[1/4] Loading model artefacts ...")
    model, scaler, encoders, feature_columns = load_artefacts()

    print(f"[2/4] Loading dataset from {DATA_FILE} ...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Run generate_logs.py first."
        )
    df = pd.read_csv(DATA_FILE)
    print(f"      Loaded {len(df):,} events.")

    # ---- 2. Predict ----------------------------------------------------
    print("\n[3/4] Running Isolation Forest inference ...")
    X = preprocess(df, encoders, scaler, feature_columns)

    raw_pred = model.predict(X)                  # -1 = anomaly, 1 = normal
    # Anomaly score: more negative = more anomalous.
    # Invert so that higher value = higher anomaly likelihood (easier to read).
    anomaly_score = -model.score_samples(X)

    df["prediction"] = np.where(raw_pred == -1, 1, 0)
    df["anomaly_score"] = np.round(anomaly_score, 4)

    # ---- 3. Threat level + recommended action --------------------------
    df["threat_level"] = df.apply(classify_threat_level, axis=1)
    df["recommended_action"] = df.apply(recommend_action, axis=1)

    # ---- 4. Save -------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[4/4] Detection results saved to: {OUTPUT_FILE}")

    # ---- Summary print --------------------------------------------------
    total = len(df)
    suspicious = int((df["prediction"] == 1).sum())
    level_counts = df["threat_level"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"]
    ).fillna(0).astype(int)

    print("\nDetection summary")
    print("-" * 40)
    print(f"Total events       : {total:,}")
    print(f"Flagged suspicious : {suspicious:,} ({suspicious / total:.1%})")
    print("\nThreat level distribution:")
    for level, count in level_counts.items():
        print(f"  {level:<8} : {count:,}")
    print("\nThreat detection complete.")


if __name__ == "__main__":
    main()
