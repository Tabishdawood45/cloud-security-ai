"""
response_engine.py
-------------------
Module 4: Automated Response Engine

Reads detection_results.csv, filters out the events that were flagged as
suspicious, and generates a corresponding alert for each one. Alerts are
written both to a CSV file (data/alerts.csv) and to an SQLite database
(cloud_security.db) so the same data can be queried easily by the Flask
dashboard.

Run:
    python response_engine.py
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DETECTION_FILE = os.path.join("data", "detection_results.csv")
ALERTS_CSV = os.path.join("data", "alerts.csv")
SQLITE_DB = "cloud_security.db"

# Order matches CSV columns so screenshots look tidy
ALERT_COLUMNS = [
    "alert_id",
    "event_id",
    "timestamp",
    "user_id",
    "ip_address",
    "country",
    "resource_accessed",
    "action_type",
    "attack_type",
    "failed_logins",
    "data_download_mb",
    "risk_score",
    "anomaly_score",
    "threat_level",
    "recommended_action",
    "status",
    "created_at",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Build the alert DataFrame from the detection results."""
    # Only suspicious events become alerts
    suspicious = df[df["prediction"] == 1].copy()
    suspicious = suspicious.reset_index(drop=True)

    suspicious["alert_id"] = [f"A{i:06d}" for i in range(1, len(suspicious) + 1)]
    suspicious["status"] = "OPEN"
    suspicious["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Re-order columns to the canonical order
    return suspicious[ALERT_COLUMNS]


def save_alerts_to_csv(alerts: pd.DataFrame, path: str) -> None:
    """Persist the alerts table to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    alerts.to_csv(path, index=False)


def save_alerts_to_sqlite(alerts: pd.DataFrame, db_path: str) -> None:
    """Persist the alerts table to an SQLite database.

    The database is overwritten on every run so screenshots taken during a
    thesis demo always reflect the latest detection results.
    """
    # Connect (creates the file if it does not exist)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Drop and recreate the alerts table for a clean slate
        cursor.execute("DROP TABLE IF EXISTS alerts;")
        cursor.execute(
            """
            CREATE TABLE alerts (
                alert_id            TEXT PRIMARY KEY,
                event_id            TEXT,
                timestamp           TEXT,
                user_id             TEXT,
                ip_address          TEXT,
                country             TEXT,
                resource_accessed   TEXT,
                action_type         TEXT,
                attack_type         TEXT,
                failed_logins       INTEGER,
                data_download_mb    REAL,
                risk_score          INTEGER,
                anomaly_score       REAL,
                threat_level        TEXT,
                recommended_action  TEXT,
                status              TEXT,
                created_at          TEXT
            );
            """
        )
        # Use pandas to_sql with append since we just created the schema
        alerts.to_sql("alerts", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print(" Cloud Threat Detection - Automated Response Engine")
    print("=" * 60)

    if not os.path.exists(DETECTION_FILE):
        raise FileNotFoundError(
            f"{DETECTION_FILE} not found. Please run detect_threats.py first."
        )

    print(f"\n[1/3] Reading detection results from {DETECTION_FILE} ...")
    df = pd.read_csv(DETECTION_FILE)
    total_events = len(df)
    suspicious_events = int((df["prediction"] == 1).sum())

    print(f"      Total events      : {total_events:,}")
    print(f"      Suspicious events : {suspicious_events:,}")

    print("\n[2/3] Generating alerts ...")
    alerts = build_alerts(df)
    save_alerts_to_csv(alerts, ALERTS_CSV)
    save_alerts_to_sqlite(alerts, SQLITE_DB)

    critical_alerts = int((alerts["threat_level"] == "Critical").sum())
    high_alerts = int((alerts["threat_level"] == "High").sum())
    medium_alerts = int((alerts["threat_level"] == "Medium").sum())
    low_alerts = int((alerts["threat_level"] == "Low").sum())

    print(f"      Alerts written to : {ALERTS_CSV}")
    print(f"      Alerts written to : {SQLITE_DB} (table: alerts)")

    print("\n[3/3] Response summary")
    print("-" * 40)
    print(f"Total events       : {total_events:,}")
    print(f"Suspicious events  : {suspicious_events:,}")
    print(f"Alerts generated   : {len(alerts):,}")
    print(f"  Critical         : {critical_alerts:,}")
    print(f"  High             : {high_alerts:,}")
    print(f"  Medium           : {medium_alerts:,}")
    print(f"  Low              : {low_alerts:,}")

    # Show breakdown by recommended action for the screenshot
    print("\nRecommended action breakdown:")
    for action, count in alerts["recommended_action"].value_counts().items():
        print(f"  {action:<28} : {count:,}")

    # Show breakdown by attack type (only suspicious rows make it into alerts)
    print("\nAlerts by attack type:")
    for attack_type, count in alerts["attack_type"].value_counts().items():
        print(f"  {attack_type:<22} : {count:,}")

    print("\nAutomated response engine complete.")


if __name__ == "__main__":
    main()
