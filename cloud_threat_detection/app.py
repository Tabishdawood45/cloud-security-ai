import os
import sqlite3

import pandas as pd
from flask import Flask, render_template


DETECTION_FILE = os.path.join("data", "detection_results.csv")
METRICS_FILE = os.path.join("data", "evaluation_metrics.csv")
SQLITE_DB = "cloud_security.db"

app = Flask(__name__)




def load_detection_summary() -> dict:
    """Compute the headline counters from detection_results.csv."""
    if not os.path.exists(DETECTION_FILE):
        return {
            "total_events": 0,
            "normal_events": 0,
            "suspicious_events": 0,
        }

    df = pd.read_csv(DETECTION_FILE)
    total = len(df)
    normal = int((df["prediction"] == 0).sum())
    suspicious = int((df["prediction"] == 1).sum())
    return {
        "total_events": total,
        "normal_events": normal,
        "suspicious_events": suspicious,
    }


def load_alerts_from_db() -> pd.DataFrame:
    """Return the full alerts table from the SQLite database. Empty
    DataFrame if the database has not been generated yet.
    """
    if not os.path.exists(SQLITE_DB):
        return pd.DataFrame()
    conn = sqlite3.connect(SQLITE_DB)
    try:
       
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alerts';"
        )
        if cur.fetchone() is None:
            return pd.DataFrame()
        return pd.read_sql_query("SELECT * FROM alerts;", conn)
    finally:
        conn.close()


def load_metrics() -> list[dict]:
    """Return the metric rows so the template can render a small table."""
    if not os.path.exists(METRICS_FILE):
        return []
    df = pd.read_csv(METRICS_FILE)
    keep = ["model", "accuracy", "precision", "recall", "f1_score"]
    df = df[keep].copy()
    for col in ["accuracy", "precision", "recall", "f1_score"]:
        df[col] = df[col].apply(lambda v: f"{v:.4f}")
    return df.to_dict(orient="records")



@app.route("/")
def dashboard():
    """Main dashboard route - renders templates/dashboard.html."""
    summary = load_detection_summary()
    alerts_df = load_alerts_from_db()


    total_alerts = len(alerts_df)
    critical_alerts = int((alerts_df["threat_level"] == "Critical").sum()) if total_alerts else 0
    high_alerts = int((alerts_df["threat_level"] == "High").sum()) if total_alerts else 0
    medium_alerts = int((alerts_df["threat_level"] == "Medium").sum()) if total_alerts else 0
    low_alerts = int((alerts_df["threat_level"] == "Low").sum()) if total_alerts else 0

   
    if not alerts_df.empty:
        latest_alerts = (
            alerts_df.sort_values("timestamp", ascending=False)
            .head(20)
            .to_dict(orient="records")
        )
    else:
        latest_alerts = []

    
    if not alerts_df.empty:
        top_users_df = (
            alerts_df.groupby("user_id")
            .agg(
                alert_count=("alert_id", "count"),
                critical_count=("threat_level", lambda x: int((x == "Critical").sum())),
                total_downloaded=("data_download_mb", "sum"),
                max_risk_score=("risk_score", "max"),
            )
            .sort_values("alert_count", ascending=False)
            .head(10)
            .reset_index()
        )
        top_users_df["total_downloaded"] = top_users_df["total_downloaded"].round(2)
        top_users = top_users_df.to_dict(orient="records")
    else:
        top_users = []

    metrics_rows = load_metrics()

    context = {
        "total_events":      summary["total_events"],
        "normal_events":     summary["normal_events"],
        "suspicious_events": summary["suspicious_events"],
        "total_alerts":      total_alerts,
        "critical_alerts":   critical_alerts,
        "high_alerts":       high_alerts,
        "medium_alerts":     medium_alerts,
        "low_alerts":        low_alerts,
        "latest_alerts":     latest_alerts,
        "top_users":         top_users,
        "metrics_rows":      metrics_rows,
    }
    return render_template("dashboard.html", **context)




if __name__ == "__main__":
    
    app.run(host="127.0.0.1", port=5000, debug=False)

